# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — sharded CI test runs with a complete-union gate
"""Run the CI test suite in duration-balanced shards and prove that their union is the suite.

Every shard takes its module list from one deterministic discovery of ``tests/`` and one
timing snapshot tracked at the same commit, so all shards of a run agree on the partition.
Timing data only orders the modules: membership always comes from discovery. A module
without a measurement receives the median measured duration, and a missing snapshot falls
back to round-robin over the sorted discovery.

Each shard runs the real pytest with the one admitted selection (``SELECTION``) and records
a manifest. The manifest holds the shard's modules, the node ids that pytest collected for
them, the pytest exit code and the number of JUnit test cases. The union gate refuses a run
in any of these cases:

- a shard is missing, duplicated, or reports a different commit, discovery, selection or
  snapshot;
- the modules of the shards overlap, or their union is not the discovery;
- the collected node ids overlap, or their union differs from a full collection of the
  same lane;
- a shard failed or timed out, or its JUnit report does not hold exactly one test case
  per collected test. JUnit entries of modules skipped during collection carry no
  classname and are counted separately.

The coverage union combines exactly one data file per shard before the original thresholds
are applied once.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import fnmatch
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import IO, Any, Final, TextIO, cast
import xml.etree.ElementTree as ET

SHARD_COUNT: Final[int] = 4
TEST_ROOT: Final[str] = "tests"
# The frozen exact-byte evidence records a specific Python 3.12.3, JAX 0.7.1 and CUDA host.
# Its reconstruction test stays on that matching evidence host; hosted runners run every
# portable test and the complete authenticated CVGC2 contract surface.
SELECTION: Final[tuple[str, ...]] = (
    "-m",
    "not experimental",
    "--deselect",
    "tests/test_diagnose_ida_fixed_reference_operator.py::"
    "test_real_diiid_fixed_reference_operator_decomposition",
)
MANIFEST_SCHEMA: Final[str] = "scpn-fusion-core.ci-test-shard.v1"
INVENTORY_SCHEMA: Final[str] = "scpn-fusion-core.ci-test-inventory.v1"
TIMINGS_SCHEMA: Final[str] = "scpn-fusion-core.ci-test-timings.v1"
TIMEOUT_RETURNCODE: Final[int] = 124
NO_TESTS_COLLECTED: Final[int] = 5
# pytest from this interpreter. The repository-root conftest already puts the root on
# ``sys.path``, so ``-m`` adds no import location that the ``pytest`` script lacks.
PYTEST: Final[tuple[str, ...]] = (sys.executable, "-m", "pytest")
# Seconds pytest gets to finish after SIGTERM before it is killed.
TERMINATE_GRACE_S: float = 60.0
# pytest's default ``norecursedirs``; the repository does not override it.
_IGNORED_DIRS: Final[tuple[str, ...]] = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
    "__pycache__",
)
_NODEID = re.compile(r"[^\s:][^:]*\.py::\S.*")


def discover(root: Path) -> list[str]:
    """List every module pytest collects under ``tests/``, sorted and repository-relative.

    Parameters
    ----------
    root : pathlib.Path
        Repository root holding the ``tests`` directory.

    Returns
    -------
    list of str
        POSIX paths of ``test_*.py`` and ``*_test.py`` files outside pytest's ignored
        directories.

    Raises
    ------
    ValueError
        No test module was found.
    """
    found: list[str] = []
    for path in sorted((root / TEST_ROOT).rglob("*.py")):
        relative = path.relative_to(root)
        directories = relative.parts[1:-1]
        if any(fnmatch.fnmatch(part, pattern) for part in directories for pattern in _IGNORED_DIRS):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            found.append(relative.as_posix())
    if not found:
        raise ValueError(f"no test modules under {TEST_ROOT}/")
    return found


def _digest(lines: Iterable[str]) -> str:
    """Return the SHA-256 of newline-terminated lines, identifying one exact ordered list.

    Parameters
    ----------
    lines : iterable of str
        Lines in their significant order.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()


def read_timings(path: Path | None) -> dict[str, float]:
    """Read the tracked timing snapshot; refuse malformed, nonfinite or nonpositive entries.

    Parameters
    ----------
    path : pathlib.Path or None
        Snapshot file. A missing file means a cold run without measurements.

    Returns
    -------
    dict of str to float
        Seconds keyed by repository-relative module path.

    Raises
    ------
    ValueError
        The snapshot has the wrong schema or an invalid duration.
    """
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != TIMINGS_SCHEMA:
        raise ValueError(f"unsupported timing snapshot: {path}")
    durations = data.get("durations")
    if not isinstance(durations, dict):
        raise ValueError("timing durations must be an object")
    result: dict[str, float] = {}
    for name, seconds in cast(dict[str, object], durations).items():
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"timing for {name} must be positive finite seconds")
        result[name] = float(seconds)
    return result


def partition(files: Sequence[str], count: int, timings: dict[str, float]) -> list[list[str]]:
    """Assign every discovered module to exactly one shard, longest measured modules first.

    Parameters
    ----------
    files : sequence of str
        Complete discovery list. Timing data never adds or removes a module.
    count : int
        Number of shards; each one receives at least one module.
    timings : dict of str to float
        Measured seconds. Unmeasured modules receive the median measurement.

    Returns
    -------
    list of list of str
        Sorted module lists, one per shard, in shard order.

    Raises
    ------
    ValueError
        The count does not fit the list, or the list has duplicates or empty names.
    """
    if not 1 <= count <= len(files) or len(set(files)) != len(files) or not all(files):
        raise ValueError("the shard count must fit a unique, nonempty module list")
    if not timings:
        ordered = sorted(files)
        return [ordered[index::count] for index in range(count)]
    default = statistics.median(timings.values())
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [(0.0, index) for index in range(count)]
    for name in sorted(files, key=lambda module: (-timings.get(module, default), module)):
        load, index = heapq.heappop(loads)
        shards[index].append(name)
        heapq.heappush(loads, (load + timings.get(name, default), index))
    return [sorted(shard) for shard in shards]


def parse_collected(output: str) -> list[str]:
    """Read node ids from ``pytest --collect-only -q`` output.

    Parameters
    ----------
    output : str
        Standard output of the collection; node ids end at its first blank line.

    Returns
    -------
    list of str
        Collected node ids in collection order.

    Raises
    ------
    ValueError
        A line before the blank line is not a node id, or a node id repeats.
    """
    nodeids: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            break
        if _NODEID.fullmatch(line) is None:
            raise ValueError(f"unexpected collection line: {line!r}")
        nodeids.append(line)
    if len(set(nodeids)) != len(nodeids):
        raise ValueError("pytest collected a node id twice")
    return nodeids


def _commit(root: Path) -> str:
    """Return the commit under test: ``GITHUB_SHA`` on runners, otherwise ``git rev-parse``.

    Parameters
    ----------
    root : pathlib.Path
        Repository checkout.

    Returns
    -------
    str
        Full commit id.

    Raises
    ------
    ValueError
        Neither ``GITHUB_SHA`` nor git is available.
    """
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha
    git = shutil.which("git")
    if git is None:
        raise ValueError("cannot identify the commit: no GITHUB_SHA and no git")
    result = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _collect(
    root: Path, pytest: Sequence[str], targets: Sequence[str]
) -> tuple[int, list[str], str]:
    """Collect the admitted selection of ``targets`` through the real runner.

    Parameters
    ----------
    root : pathlib.Path
        Directory pytest runs in.
    pytest : sequence of str
        Command that starts pytest.
    targets : sequence of str
        Modules or directories to collect.

    Returns
    -------
    tuple of (int, list of str, str)
        Exit code, collected node ids (empty unless collection succeeded) and the output.
    """
    result = subprocess.run(
        [*pytest, "--collect-only", "-q", *SELECTION, *targets],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    nodeids = parse_collected(result.stdout) if result.returncode == 0 else []
    return result.returncode, nodeids, result.stdout + result.stderr


def _junit_cases(directory: Path) -> list[ET.Element]:
    """Return the test cases of every JUnit report below ``directory``.

    Parameters
    ----------
    directory : pathlib.Path
        Directory searched recursively for ``junit*.xml``.

    Returns
    -------
    list of xml.etree.ElementTree.Element
        ``testcase`` elements in report order.

    Raises
    ------
    ValueError
        A report declares a document type or an entity; such reports are not expanded.
    """
    cases: list[ET.Element] = []
    for report in sorted(directory.rglob("junit*.xml")):
        raw = report.read_bytes()
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
            raise ValueError(f"JUnit report with a document type declaration: {report}")
        cases.extend(ET.fromstring(raw).iter("testcase"))
    return cases


def _pump(source: IO[str], log: TextIO) -> None:
    """Copy every line of pytest output to the console and the log until end of file.

    Parameters
    ----------
    source : IO of str
        The pytest process's combined output pipe.
    log : TextIO
        Log file kept with the shard's artifacts.
    """
    for line in source:
        sys.stdout.write(line)
        log.write(line)


def _wait(process: subprocess.Popen[str], log: TextIO, timeout_s: float) -> bool:
    """Stream pytest output while waiting for it; stop it when the deadline expires.

    Parameters
    ----------
    process : subprocess.Popen of str
        Running pytest process with a text output pipe.
    log : TextIO
        Log file kept with the shard's artifacts.
    timeout_s : float
        Deadline for the run, in seconds.

    Returns
    -------
    bool
        True when the deadline expired and pytest was stopped.
    """
    pump = threading.Thread(target=_pump, args=(process.stdout, log), daemon=True)
    pump.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    pump.join(timeout=TERMINATE_GRACE_S)
    return timed_out


def run_shard(
    root: Path,
    *,
    lane: str,
    index: int,
    count: int,
    timings_path: Path | None,
    out_dir: Path,
    lane_args: Sequence[str],
    timeout_minutes: float,
    pytest: Sequence[str] = PYTEST,
) -> int:
    """Run one shard through the real runner and write its manifest.

    Parameters
    ----------
    root : pathlib.Path
        Repository root; pytest runs there.
    lane : str
        Interpreter lane label, for example ``"3.12"``.
    index : int
        One-based shard index.
    count : int
        Number of shards.
    timings_path : pathlib.Path or None
        Timing snapshot shared by every shard of the run.
    out_dir : pathlib.Path
        Destination of ``manifest.json``, ``junit.xml`` and ``pytest.log``.
    lane_args : sequence of str
        Lane-specific pytest options, such as verbosity or coverage.
    timeout_minutes : float
        Bound for the test run; pytest receives SIGTERM when it expires.
    pytest : sequence of str
        Command that starts pytest.

    Returns
    -------
    int
        The pytest exit code; the collection exit code if collection failed; 0 for a shard
        whose modules hold no admitted test; 124 when the run timed out.

    Raises
    ------
    ValueError
        The shard index is outside ``1..count`` or the inputs are invalid.
    """
    if not 1 <= index <= count:
        raise ValueError("the shard must be K/N with 1 <= K <= N")
    files = discover(root)
    timings = read_timings(timings_path)
    selected = partition(files, count, timings)[index - 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    junit = out_dir / "junit.xml"
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "lane": lane,
        "shard": index,
        "shards": count,
        "commit": _commit(root),
        "discovery_sha256": _digest(files),
        "timings_sha256": _digest(sorted(f"{name}={value!r}" for name, value in timings.items())),
        "selection": list(SELECTION),
        "files": selected,
        "timeout_minutes": timeout_minutes,
        "timed_out": False,
    }
    started = time.monotonic()
    collect_rc, nodeids, collect_output = _collect(root, pytest, selected)
    manifest["nodeids"] = nodeids
    with (out_dir / "pytest.log").open("w", encoding="utf-8") as log:
        if collect_rc == NO_TESTS_COLLECTED:
            # Every module of this shard holds only deselected or experimental tests.
            log.write(collect_output)
            returncode = 0
        elif collect_rc != 0:
            sys.stdout.write(collect_output)
            log.write(collect_output)
            returncode = collect_rc
        else:
            process = subprocess.Popen(
                [*pytest, *SELECTION, *lane_args, f"--junitxml={junit}", *selected],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            manifest["timed_out"] = _wait(process, log, timeout_minutes * 60.0)
            returncode = TIMEOUT_RETURNCODE if manifest["timed_out"] else process.returncode
    # A module skipped while it is collected (``allow_module_level`` skips, ``importorskip``)
    # yields no node id but one JUnit entry with an empty ``classname``; every test item has
    # a classname. Only test items are compared with the collected node ids.
    cases = _junit_cases(out_dir) if junit.exists() else []
    manifest["junit_testcases"] = sum(1 for case in cases if case.get("classname"))
    manifest["junit_collection_reports"] = len(cases) - manifest["junit_testcases"]
    manifest["pytest_returncode"] = returncode
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest["timed_out"]:
        print(
            f"::error::Python {lane} shard {index}/{count} timed out after {timeout_minutes:g} min."
        )
    return returncode


def write_inventory(root: Path, *, lane: str, output: Path, pytest: Sequence[str] = PYTEST) -> int:
    """Collect the whole admitted suite of one lane through the real runner.

    Parameters
    ----------
    root : pathlib.Path
        Repository root; pytest runs there.
    lane : str
        Interpreter lane label.
    output : pathlib.Path
        Destination of the inventory JSON.
    pytest : sequence of str
        Command that starts pytest.

    Returns
    -------
    int
        The collection exit code.
    """
    returncode, nodeids, collect_output = _collect(root, pytest, [f"{TEST_ROOT}/"])
    if returncode != 0:
        sys.stdout.write(collect_output)
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "lane": lane,
        "commit": _commit(root),
        "selection": list(SELECTION),
        "returncode": returncode,
        "nodeids": nodeids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Python {lane}: {len(nodeids)} admitted tests in {len(discover(root))} modules")
    return returncode


def _load(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Parameters
    ----------
    path : pathlib.Path
        JSON file.

    Returns
    -------
    dict of str to Any
        The decoded object.

    Raises
    ------
    ValueError
        The file does not hold a JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], data)


def _repeated(groups: Iterable[Iterable[str]]) -> list[str]:
    """Return, sorted, every item that occurs more than once across ``groups``.

    Parameters
    ----------
    groups : iterable of iterable of str
        Groups whose items must be unique overall.

    Returns
    -------
    list of str
        The repeated items.
    """
    seen: set[str] = set()
    repeated: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                repeated.add(item)
            seen.add(item)
    return sorted(repeated)


def _shard_errors(
    manifest: dict[str, Any], *, lane: str, expect: int, commit: str, discovery: str
) -> list[str]:
    """Check one shard manifest against the run it must belong to.

    Parameters
    ----------
    manifest : dict of str to Any
        Shard manifest.
    lane : str
        Interpreter lane label.
    expect : int
        Number of shards the workflow declares.
    commit : str
        Commit the run tests.
    discovery : str
        Digest of the module discovery at that commit.

    Returns
    -------
    list of str
        Violations of this shard.
    """
    label = f"Python {lane} shard {manifest.get('shard')}"
    nodeids = manifest.get("nodeids", [])
    checks = (
        (manifest.get("schema") == MANIFEST_SCHEMA, "has an unexpected manifest schema"),
        (
            manifest.get("shards") == expect,
            f"ran as one of {manifest.get('shards')} shards, not {expect}",
        ),
        (manifest.get("commit") == commit, f"tested {manifest.get('commit')}, not {commit}"),
        (manifest.get("discovery_sha256") == discovery, "discovered a different module list"),
        (manifest.get("selection") == list(SELECTION), "ran a different test selection"),
        (
            manifest.get("pytest_returncode") == 0,
            f"pytest exited {manifest.get('pytest_returncode')}",
        ),
        (not manifest.get("timed_out"), f"timed out after {manifest.get('timeout_minutes')} min"),
        (
            manifest.get("junit_testcases") == len(nodeids),
            f"JUnit reports {manifest.get('junit_testcases')} cases for {len(nodeids)} collected tests",
        ),
    )
    return [f"{label}: {message}" for passed, message in checks if not passed]


def verify_union(
    root: Path, *, lane: str, expect: int, shards_dir: Path, inventory_path: Path, commit: str
) -> list[str]:
    """Check that the shards of one lane ran the complete admitted suite exactly once.

    Parameters
    ----------
    root : pathlib.Path
        Checkout of the commit under test, used for an independent discovery.
    lane : str
        Interpreter lane label.
    expect : int
        Number of shards the workflow declares.
    shards_dir : pathlib.Path
        Directory holding the downloaded shard artifacts.
    inventory_path : pathlib.Path
        Full collection of the lane.
    commit : str
        Commit the run tests.

    Returns
    -------
    list of str
        Every violation found; empty when the union is complete.
    """
    files = discover(root)
    manifests = [
        manifest
        for manifest in (_load(path) for path in sorted(shards_dir.rglob("manifest.json")))
        if manifest.get("lane") == lane
    ]
    errors: list[str] = []
    for manifest in sorted(manifests, key=lambda item: int(item.get("shard", 0))):
        print(
            f"Python {lane} shard {manifest.get('shard')}/{manifest.get('shards')}: "
            f"{len(manifest.get('files', []))} modules, {len(manifest.get('nodeids', []))} tests, "
            f"{manifest.get('elapsed_seconds')} s, exit {manifest.get('pytest_returncode')}"
        )
    indices = sorted(int(manifest.get("shard", 0)) for manifest in manifests)
    if indices != list(range(1, expect + 1)):
        errors.append(f"Python {lane}: expected shards 1..{expect}, found {indices}")
    for manifest in manifests:
        errors += _shard_errors(
            manifest, lane=lane, expect=expect, commit=commit, discovery=_digest(files)
        )
    if len({manifest.get("timings_sha256") for manifest in manifests}) > 1:
        errors.append(f"Python {lane}: shards used different timing snapshots")
    shard_files = [cast(list[str], manifest.get("files", [])) for manifest in manifests]
    if _repeated(shard_files):
        errors.append(
            f"Python {lane}: modules run by more than one shard: {_repeated(shard_files)[:10]}"
        )
    ran = {name for group in shard_files for name in group}
    if ran != set(files):
        missing, unknown = sorted(set(files) - ran), sorted(ran - set(files))
        errors.append(
            f"Python {lane}: modules not run {missing[:10]}, unknown modules {unknown[:10]}"
        )
    inventory = _load(inventory_path)
    if (inventory.get("schema"), inventory.get("lane"), inventory.get("commit")) != (
        INVENTORY_SCHEMA,
        lane,
        commit,
    ):
        errors.append(f"Python {lane}: the inventory is not this lane's collection of {commit}")
    if inventory.get("returncode") != 0 or inventory.get("selection") != list(SELECTION):
        errors.append(f"Python {lane}: the inventory collection failed or used another selection")
    shard_nodeids = [cast(list[str], manifest.get("nodeids", [])) for manifest in manifests]
    if _repeated(shard_nodeids):
        errors.append(
            f"Python {lane}: tests collected by more than one shard: {_repeated(shard_nodeids)[:10]}"
        )
    collected = {nodeid for group in shard_nodeids for nodeid in group}
    expected = set(cast(list[str], inventory.get("nodeids", [])))
    if collected != expected:
        missing, unknown = sorted(expected - collected), sorted(collected - expected)
        errors.append(
            f"Python {lane}: tests not run {missing[:10]}, tests outside the inventory {unknown[:10]}"
        )
    if not errors:
        print(
            f"Python {lane}: {len(collected)} tests in {len(files)} modules ran once across {expect} shards"
        )
    return errors


def combine_coverage(*, lane: str, expect: int, shards_dir: Path, output: Path) -> None:
    """Combine exactly one coverage data file per shard of ``lane`` into ``output``.

    Parameters
    ----------
    lane : str
        Interpreter lane label, used in diagnostics.
    expect : int
        Number of shards the workflow declares.
    shards_dir : pathlib.Path
        Directory holding the downloaded shard artifacts.
    output : pathlib.Path
        Combined coverage data file.

    Raises
    ------
    ValueError
        A shard's data file is missing, or an unexpected data file is present.
    subprocess.CalledProcessError
        ``coverage combine`` failed.
    """
    found = sorted(shards_dir.rglob(".coverage.shard-*"))
    names = sorted(path.name for path in found)
    wanted = sorted(f".coverage.shard-{index}" for index in range(1, expect + 1))
    if names != wanted:
        raise ValueError(f"Python {lane}: expected coverage data {wanted}, found {names}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            "--keep",
            f"--data-file={output}",
            *map(str, found),
        ],
        check=True,
    )


def record_timings(
    root: Path, *, junit_dir: Path, previous: Path | None, output: Path, source: dict[str, str]
) -> int:
    """Write the next timing snapshot from this run's JUnit durations.

    Parameters
    ----------
    root : pathlib.Path
        Checkout used to identify live modules.
    junit_dir : pathlib.Path
        Directory holding the downloaded shard artifacts with JUnit reports.
    previous : pathlib.Path or None
        Current snapshot. Live modules that this run did not measure keep their entry;
        removed modules are dropped.
    output : pathlib.Path
        Destination of the new snapshot.
    source : dict of str to str
        Provenance of the measurement, such as the run id and commit.

    Returns
    -------
    int
        Number of modules measured in this run.

    Raises
    ------
    ValueError
        No live module was measured, or a duration is negative or nonfinite.
    """
    modules = set(discover(root))
    live = {path[: -len(".py")].replace("/", "."): path for path in modules}
    durations = {
        name: seconds for name, seconds in read_timings(previous).items() if name in modules
    }
    measured: dict[str, float] = {}
    for case in _junit_cases(junit_dir):
        name = case.get("classname", "")
        while name and name not in live:
            name = name.rpartition(".")[0]
        if not name:
            continue
        seconds = float(case.get("time", "0"))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("JUnit durations must be finite and nonnegative")
        measured[live[name]] = measured.get(live[name], 0.0) + seconds
    if not measured:
        raise ValueError("the JUnit reports measure no live module")
    durations.update({name: round(max(seconds, 0.001), 3) for name, seconds in measured.items()})
    snapshot = {
        "schema": TIMINGS_SCHEMA,
        "source": source,
        "durations": dict(sorted(durations.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
    return len(measured)


def _shard(value: str) -> tuple[int, int]:
    """Parse ``K/N`` into its index and count.

    Parameters
    ----------
    value : str
        Shard designation.

    Returns
    -------
    tuple of (int, int)
        One-based index and count.

    Raises
    ------
    ValueError
        The value is not ``K/N`` with positive integers.
    """
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", value)
    if match is None:
        raise ValueError(f"the shard must be K/N: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``run``, ``inventory``, ``verify``, ``combine`` and ``record``
        commands.
    """
    parser = argparse.ArgumentParser(description="Run and verify sharded CI test runs.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one shard")
    run.add_argument("--lane", required=True)
    run.add_argument("--shard", required=True)
    run.add_argument("--timings", type=Path)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--timeout-minutes", type=float, default=75.0)
    run.add_argument("lane_args", nargs=argparse.REMAINDER)
    inventory = commands.add_parser("inventory", help="collect the whole lane")
    inventory.add_argument("--lane", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    for name in ("verify", "combine"):
        command = commands.add_parser(name, help=f"{name} the shards of one lane")
        command.add_argument("--lane", required=True)
        command.add_argument("--expect-shards", type=int, required=True)
        command.add_argument("--shards-dir", type=Path, required=True)
        if name == "verify":
            command.add_argument("--inventory", type=Path, required=True)
            command.add_argument("--commit", required=True)
        else:
            command.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record", help="write the next timing snapshot")
    record.add_argument("--junit-dir", type=Path, required=True)
    record.add_argument("--previous", type=Path)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--source", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shard, inventory, verify, combine or record command.

    Parameters
    ----------
    argv : sequence of str or None
        Command arguments; the process arguments when omitted.

    Returns
    -------
    int
        The pytest exit code for ``run`` and ``inventory``; 0 or 1 for ``verify``; 0 for a
        completed ``combine`` or ``record``; 2 for invalid inputs.
    """
    args = _parser().parse_args(argv)
    root = cast(Path, args.root).resolve()
    try:
        if args.command == "run":
            index, count = _shard(args.shard)
            return run_shard(
                root,
                lane=args.lane,
                index=index,
                count=count,
                timings_path=args.timings,
                out_dir=args.out_dir,
                lane_args=[arg for arg in args.lane_args if arg != "--"],
                timeout_minutes=args.timeout_minutes,
            )
        if args.command == "inventory":
            return write_inventory(root, lane=args.lane, output=args.output)
        if args.command == "verify":
            errors = verify_union(
                root,
                lane=args.lane,
                expect=args.expect_shards,
                shards_dir=args.shards_dir,
                inventory_path=args.inventory,
                commit=args.commit,
            )
            for error in errors:
                print(f"::error::{error}")
            return 1 if errors else 0
        if args.command == "combine":
            combine_coverage(
                lane=args.lane,
                expect=args.expect_shards,
                shards_dir=args.shards_dir,
                output=args.output,
            )
            return 0
        source: dict[str, str] = {}
        for item in args.source:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise ValueError(f"--source must be KEY=VALUE: {item!r}")
            source[key] = value
        measured = record_timings(
            root,
            junit_dir=args.junit_dir,
            previous=args.previous,
            output=args.output,
            source=source,
        )
        print(f"Measured {measured} modules into {args.output}")
        return 0
    except (OSError, ValueError, ET.ParseError, subprocess.CalledProcessError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
