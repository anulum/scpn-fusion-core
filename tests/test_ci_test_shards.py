# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — sharded CI test runs
"""Real-runner tests for the CI shard scheduler, its union gate and its coverage union."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import coverage
import pytest

from tools import ci_test_shards as shards

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c0ffee" * 6 + "c0ff"
PYTEST_INI = "[pytest]\naddopts = -p no:nengo\nmarkers =\n    experimental: research-only lane\n"
MODULES = {
    "tests/test_alpha.py": (
        "import pytest\n\n\n@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_square(value):\n    assert value * value >= value\n\n\n"
        "def test_plain():\n    assert True\n"
    ),
    "tests/test_beta.py": (
        "import pytest\n\n\nclass TestBeta:\n    def test_one(self):\n        assert 1\n\n"
        "    @pytest.mark.experimental\n    def test_research(self):\n        assert 1\n"
    ),
    "tests/test_gamma.py": (
        "import pytest\n\n\n@pytest.mark.experimental\ndef test_only_research():\n    assert 1\n"
    ),
    # Skipped while it is collected: no node id, one JUnit entry with an empty classname.
    "tests/test_epsilon.py": (
        "import pytest\n\npytest.skip('needs hardware', allow_module_level=True)\n\n\n"
        "def test_never():\n    assert 0\n"
    ),
    "tests/sub/test_delta.py": "def test_nested():\n    assert 1\n",
    "tests/sub/thing_test.py": "def test_suffix():\n    assert 1\n",
    "tests/helper.py": "VALUE = 1\n",
    "tests/.hidden/test_hidden.py": "def test_hidden():\n    assert 0\n",
    "tests/build/test_built.py": "def test_built():\n    assert 0\n",
}
DISCOVERED = [
    "tests/sub/test_delta.py",
    "tests/sub/thing_test.py",
    "tests/test_alpha.py",
    "tests/test_beta.py",
    "tests/test_epsilon.py",
    "tests/test_gamma.py",
]
ADMITTED = {
    "tests/sub/test_delta.py::test_nested",
    "tests/sub/thing_test.py::test_suffix",
    "tests/test_alpha.py::test_square[1]",
    "tests/test_alpha.py::test_square[2]",
    "tests/test_alpha.py::test_plain",
    "tests/test_beta.py::TestBeta::test_one",
}


def _project(root: Path, extra: dict[str, str] | None = None) -> Path:
    """Write a small pytest project with nested, ignored and experimental-only modules."""
    for relative, text in {**MODULES, **(extra or {})}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    return root


def _run(root: Path, index: int, count: int, out: Path, **options: Any) -> int:
    """Run one shard of ``root`` with the real pytest of this interpreter."""
    arguments: dict[str, Any] = {
        "lane": "3.12",
        "index": index,
        "count": count,
        "timings_path": None,
        "out_dir": out,
        "lane_args": ["-q"],
        "timeout_minutes": 5.0,
    }
    arguments.update(options)
    return shards.run_shard(root, **arguments)


@pytest.fixture()
def commit(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the commit the way a runner provides it."""
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    return COMMIT


@pytest.fixture(scope="module")
def _baseline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Run both shards and the inventory of a project once, through the real runner."""
    base = tmp_path_factory.mktemp("baseline")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GITHUB_SHA", COMMIT)
        project = _project(base / "project")
        for index in (1, 2):
            assert _run(project, index, 2, base / "shards" / f"python-shard-3.12-{index}") == 0
        inventory = base / "inventory" / "inventory.json"
        assert shards.write_inventory(project, lane="3.12", output=inventory) == 0
    yield base


@pytest.fixture()
def finished_run(tmp_path: Path, _baseline: Path) -> tuple[Path, Path, Path]:
    """Copy the finished run, so each test may alter it; return project, shards and inventory."""
    copy = tmp_path / "run"
    shutil.copytree(_baseline, copy)
    return copy / "project", copy / "shards", copy / "inventory" / "inventory.json"


def _manifest(shard_dir: Path, index: int) -> tuple[Path, dict[str, Any]]:
    """Return the path and content of one shard manifest."""
    path = shard_dir / f"python-shard-3.12-{index}" / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_discovery_follows_pytest_file_rules(tmp_path: Path) -> None:
    """Prefix and suffix modules are found in subdirectories; ignored directories are not."""
    assert shards.discover(_project(tmp_path)) == DISCOVERED


def test_discovery_refuses_a_tree_without_test_modules(tmp_path: Path) -> None:
    """An empty suite cannot be partitioned into shards."""
    (tmp_path / "tests").mkdir()
    with pytest.raises(ValueError, match="no test modules"):
        shards.discover(tmp_path)


def test_discovery_of_this_repository_is_every_tracked_test_module() -> None:
    """The scheduler's discovery equals the modules pytest's file rules select from the tree.

    Git lists tracked and untracked files that are not ignored, which on a runner is the
    checkout itself.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "tests"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    tracked = sorted(
        name
        for name in listed
        if name.endswith(".py")
        and (Path(name).name.startswith("test_") or name.endswith("_test.py"))
    )
    assert shards.discover(ROOT) == tracked


def test_timing_snapshot_reading(tmp_path: Path) -> None:
    """A missing snapshot is a cold run; a valid one may carry provenance next to durations."""
    assert shards.read_timings(None) == {}
    assert shards.read_timings(tmp_path / "absent.json") == {}
    path = tmp_path / "timings.json"
    snapshot = {"schema": shards.TIMINGS_SCHEMA, "source": {"run_id": "1"}, "durations": {"a": 2}}
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    assert shards.read_timings(path) == {"a": 2.0}


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        ([], "unsupported timing snapshot"),
        ({"schema": "other", "durations": {}}, "unsupported timing snapshot"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": []}, "must be an object"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": {"a": 0}}, "positive finite"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": {"a": -1.5}}, "positive finite"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": {"a": math.inf}}, "positive finite"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": {"a": True}}, "positive finite"),
        ({"schema": shards.TIMINGS_SCHEMA, "durations": {"a": "1"}}, "positive finite"),
    ],
)
def test_malformed_timing_snapshots_are_refused(
    tmp_path: Path, snapshot: object, message: str
) -> None:
    """Weights are checked before any test starts."""
    path = tmp_path / "timings.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        shards.read_timings(path)


def test_cold_partition_is_round_robin_over_the_sorted_discovery() -> None:
    """Without measurements every K-th sorted module goes to shard K."""
    assert shards.partition(["d", "b", "a", "c", "e"], 2, {}) == [["a", "c", "e"], ["b", "d"]]


def test_measured_partition_balances_longest_modules_first() -> None:
    """Longest modules go first to the least-loaded shard; unmeasured modules weigh the median."""
    timings = {"a": 9.0, "b": 5.0, "c": 4.0, "d": 1.0}
    result = shards.partition(["a", "b", "c", "d", "new"], 2, timings)
    # Order a(9), b(5), new(4.5 = median), c(4), d(1): a->0, b->1, new->1, c->0, d->1.
    assert result == [["a", "c"], ["b", "d", "new"]]
    loads = [
        sum(timings.get(name, statistics.median(timings.values())) for name in s) for s in result
    ]
    assert loads == [13.0, 10.5]


@pytest.mark.parametrize(
    ("files", "count"),
    [(["a"], 0), (["a"], 2), (["a", "a"], 1), (["a", ""], 1)],
)
def test_partition_refuses_counts_and_lists_that_cannot_be_split(
    files: list[str], count: int
) -> None:
    """Each shard needs a module, and every module is unique and named."""
    with pytest.raises(ValueError, match="must fit"):
        shards.partition(files, count, {})


def test_tracked_snapshot_balances_this_repository() -> None:
    """With the tracked snapshot every module lands in exactly one of the CI shards."""
    files = shards.discover(ROOT)
    timings = shards.read_timings(ROOT / "tools" / "ci_test_timings.json")
    result = shards.partition(files, shards.SHARD_COUNT, timings)
    assert sorted(name for shard in result for name in shard) == files
    default = statistics.median(timings.values())
    loads = [sum(timings.get(name, default) for name in shard) for shard in result]
    bound = sum(loads) / len(loads) + max(timings.values())
    assert max(loads) <= bound


def test_collection_output_parsing() -> None:
    """Node ids end at the first blank line; anything else before it is refused."""
    output = "tests/test_a.py::test_x\ntests/test_a.py::test_y[a b]\n\n2 tests collected\n"
    assert shards.parse_collected(output) == [
        "tests/test_a.py::test_x",
        "tests/test_a.py::test_y[a b]",
    ]
    with pytest.raises(ValueError, match="unexpected collection line"):
        shards.parse_collected("collected 1 item\ntests/test_a.py::test_x\n")
    with pytest.raises(ValueError, match="twice"):
        shards.parse_collected("tests/test_a.py::test_x\ntests/test_a.py::test_x\n")


def test_shards_and_inventory_form_a_complete_union(finished_run: tuple[Path, Path, Path]) -> None:
    """Two real shards run every admitted test once, which the union gate confirms."""
    project, shard_dir, inventory = finished_run
    first, second = _manifest(shard_dir, 1)[1], _manifest(shard_dir, 2)[1]
    assert first["files"] == [
        "tests/sub/test_delta.py",
        "tests/test_alpha.py",
        "tests/test_epsilon.py",
    ]
    assert second["files"] == [
        "tests/sub/thing_test.py",
        "tests/test_beta.py",
        "tests/test_gamma.py",
    ]
    assert set(first["nodeids"]) | set(second["nodeids"]) == ADMITTED
    assert (first["junit_testcases"], second["junit_testcases"]) == (4, 2)
    # The module skipped during collection is reported, but not as a test case.
    assert (first["junit_collection_reports"], second["junit_collection_reports"]) == (1, 0)
    assert first["selection"] == list(shards.SELECTION) and first["commit"] == COMMIT
    assert set(json.loads(inventory.read_text())["nodeids"]) == ADMITTED
    assert (shard_dir / "python-shard-3.12-1" / "pytest.log").read_text().strip()
    errors = shards.verify_union(
        project,
        lane="3.12",
        expect=2,
        shards_dir=shard_dir,
        inventory_path=inventory,
        commit=COMMIT,
    )
    assert errors == []


def _set(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    """Return a manifest mutation that sets one field."""
    return lambda manifest: manifest.__setitem__(key, value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("schema", "other"), "unexpected manifest schema"),
        (_set("shards", 3), "ran as one of 3 shards"),
        (_set("commit", "0" * 40), "not " + COMMIT),
        (_set("discovery_sha256", "0" * 64), "different module list"),
        (_set("selection", []), "different test selection"),
        (_set("pytest_returncode", 1), "pytest exited 1"),
        (_set("timed_out", True), "timed out"),
        (_set("junit_testcases", 0), "JUnit reports 0 cases"),
        (_set("timings_sha256", "0" * 64), "different timing snapshots"),
        (_set("files", ["tests/test_beta.py"]), "modules run by more than one shard"),
        (_set("files", []), "modules not run"),
        (
            _set("nodeids", ["tests/test_beta.py::TestBeta::test_one"]),
            "collected by more than one shard",
        ),
        (_set("nodeids", []), "tests not run"),
        (_set("shard", 2), "expected shards 1..2, found [2, 2]"),
    ],
)
def test_union_gate_refuses_every_incomplete_or_foreign_shard(
    finished_run: tuple[Path, Path, Path], mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    """A wrong, failed, overlapping or incomplete shard fails the union."""
    project, shard_dir, inventory = finished_run
    path, manifest = _manifest(shard_dir, 1)
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    errors = shards.verify_union(
        project,
        lane="3.12",
        expect=2,
        shards_dir=shard_dir,
        inventory_path=inventory,
        commit=COMMIT,
    )
    assert any(message in error for error in errors), errors


def test_union_gate_refuses_a_missing_shard_and_ignores_other_lanes(
    finished_run: tuple[Path, Path, Path],
) -> None:
    """A lane's shards are counted by lane, so another lane's manifest cannot fill a gap."""
    project, shard_dir, inventory = finished_run
    path, manifest = _manifest(shard_dir, 2)
    manifest["lane"] = "3.11"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    errors = shards.verify_union(
        project,
        lane="3.12",
        expect=2,
        shards_dir=shard_dir,
        inventory_path=inventory,
        commit=COMMIT,
    )
    assert "Python 3.12: expected shards 1..2, found [1]" in errors
    assert any("modules not run" in error for error in errors)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("lane", "3.11", "not this lane's collection"),
        ("commit", "0" * 40, "not this lane's collection"),
        ("schema", "other", "not this lane's collection"),
        ("returncode", 2, "collection failed"),
        ("selection", [], "collection failed"),
        ("nodeids", sorted(ADMITTED | {"tests/test_alpha.py::test_gone"}), "tests not run"),
    ],
)
def test_union_gate_refuses_a_foreign_or_failed_inventory(
    finished_run: tuple[Path, Path, Path], key: str, value: object, message: str
) -> None:
    """The node-id union is compared with a successful collection of the same lane and commit."""
    project, shard_dir, inventory = finished_run
    data = json.loads(inventory.read_text(encoding="utf-8"))
    data[key] = value
    inventory.write_text(json.dumps(data), encoding="utf-8")
    errors = shards.verify_union(
        project,
        lane="3.12",
        expect=2,
        shards_dir=shard_dir,
        inventory_path=inventory,
        commit=COMMIT,
    )
    assert any(message in error for error in errors), errors


def test_failing_test_fails_its_shard_and_the_union(tmp_path: Path, commit: str) -> None:
    """A real failing test returns pytest's exit code and is reported by the union gate."""
    failing = {"tests/test_alpha.py": "def test_broken():\n    assert 1 == 2\n"}
    project = _project(tmp_path / "project", failing)
    out = tmp_path / "shards" / "python-shard-3.12-1"
    assert _run(project, 1, 1, out, lane_args=["-v"]) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["pytest_returncode"] == 1 and manifest["junit_testcases"] == 4
    assert "test_broken FAILED" in (out / "pytest.log").read_text()
    inventory = tmp_path / "inventory.json"
    assert shards.write_inventory(project, lane="3.12", output=inventory) == 0
    errors = shards.verify_union(
        project,
        lane="3.12",
        expect=1,
        shards_dir=tmp_path / "shards",
        inventory_path=inventory,
        commit=COMMIT,
    )
    assert errors == ["Python 3.12 shard 1: pytest exited 1"]


def test_collection_error_fails_the_shard_and_the_inventory(tmp_path: Path, commit: str) -> None:
    """A module that cannot be imported stops the shard before its tests run."""
    project = _project(tmp_path / "project", {"tests/test_alpha.py": "def broken(:\n"})
    out = tmp_path / "shard"
    returncode = _run(project, 1, 1, out)
    assert returncode not in (0, shards.NO_TESTS_COLLECTED)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["nodeids"] == [] and manifest["junit_testcases"] == 0
    assert "test_alpha.py" in (out / "pytest.log").read_text()
    inventory = tmp_path / "inventory.json"
    assert shards.write_inventory(project, lane="3.12", output=inventory) != 0
    assert json.loads(inventory.read_text())["nodeids"] == []


@pytest.mark.parametrize(
    ("index", "module"), [(5, "tests/test_epsilon.py"), (6, "tests/test_gamma.py")]
)
def test_shard_without_admitted_tests_is_empty_not_failed(
    tmp_path: Path, commit: str, index: int, module: str
) -> None:
    """A shard whose only module is skipped or experimental runs nothing and succeeds."""
    project = _project(tmp_path / "project")
    out = tmp_path / "shard"
    assert _run(project, index, 6, out) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"] == [module]
    assert (manifest["nodeids"], manifest["junit_testcases"], manifest["pytest_returncode"]) == (
        [],
        0,
        0,
    )


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_stops_the_run_and_marks_the_shard(
    tmp_path: Path,
    commit: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ignore_term: bool,
) -> None:
    """At the deadline pytest gets SIGTERM, and is killed if it does not stop."""
    preamble = (
        "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else ""
    )
    slow = {
        "tests/test_alpha.py": f"import time\n{preamble}\n\ndef test_slow():\n    time.sleep(60)\n"
    }
    project = _project(tmp_path / "project", slow)
    monkeypatch.setattr(shards, "TERMINATE_GRACE_S", 0.5)
    out = tmp_path / "shard"
    assert _run(project, 1, 1, out, timeout_minutes=0.05) == shards.TIMEOUT_RETURNCODE
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["timed_out"] is True and manifest["pytest_returncode"] == shards.TIMEOUT_RETURNCODE
    )
    assert "timed out after 0.05 min" in capsys.readouterr().out


def test_shard_index_must_fit_the_count(tmp_path: Path) -> None:
    """``K/N`` requires ``1 <= K <= N``."""
    with pytest.raises(ValueError, match="1 <= K <= N"):
        _run(_project(tmp_path), 3, 2, tmp_path / "out")


def test_commit_comes_from_git_without_a_runner_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a runner the checked-out commit identifies the run; without git it cannot."""
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "seed",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert shards._commit(tmp_path) == head
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="cannot identify the commit"):
        shards._commit(tmp_path)


SOURCE = "/runner/work/src/scpn_fusion/branchy.py"
LEFT = {(-1, 1), (1, 2), (2, -1)}
RIGHT = {(-1, 1), (1, 4), (4, -1)}


def _coverage_data(directory: Path, name: str, arcs: set[tuple[int, int]]) -> Path:
    """Write a real branch-coverage data file holding the given arcs of one source file."""
    data = coverage.CoverageData(basename=str(directory / name))
    data.add_arcs({SOURCE: arcs})
    data.write()
    return directory / name


def test_coverage_union_combines_exactly_one_file_per_shard(tmp_path: Path) -> None:
    """Branches measured by different shards are all present in the union."""
    first = tmp_path / "python-shard-3.12-1"
    second = tmp_path / "python-shard-3.12-2"
    first.mkdir()
    second.mkdir()
    _coverage_data(first, ".coverage.shard-1", LEFT)
    _coverage_data(second, ".coverage.shard-2", RIGHT)
    combined = tmp_path / ".coverage"
    shards.combine_coverage(lane="3.12", expect=2, shards_dir=tmp_path, output=combined)
    union = coverage.CoverageData(basename=str(combined))
    union.read()
    assert union.has_arcs() and set(union.arcs(SOURCE) or ()) == LEFT | RIGHT
    assert (first / ".coverage.shard-1").exists(), "inputs are kept"


def test_coverage_union_refuses_missing_or_extra_shard_data(tmp_path: Path) -> None:
    """A union without every shard, or with an unexpected one, is never combined."""
    first = tmp_path / "python-shard-3.12-1"
    first.mkdir()
    _coverage_data(first, ".coverage.shard-1", LEFT)
    with pytest.raises(ValueError, match="expected coverage data"):
        shards.combine_coverage(lane="3.12", expect=2, shards_dir=tmp_path, output=tmp_path / "c")
    _coverage_data(first, ".coverage.shard-3", RIGHT)
    with pytest.raises(ValueError, match="expected coverage data"):
        shards.combine_coverage(lane="3.12", expect=1, shards_dir=tmp_path, output=tmp_path / "c")


def test_timing_record_measures_live_modules_and_drops_removed_ones(
    finished_run: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Durations come from real JUnit reports; unmeasured live modules keep their entry."""
    project, shard_dir, _ = finished_run
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "schema": shards.TIMINGS_SCHEMA,
                "durations": {"tests/test_gamma.py": 7.0, "tests/test_removed.py": 3.0},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "next" / "timings.json"
    source = {"run_id": "1", "commit": COMMIT}
    measured = shards.record_timings(
        project, junit_dir=shard_dir, previous=previous, output=output, source=source
    )
    snapshot = json.loads(output.read_text())
    assert measured == 4 and snapshot["source"] == source
    # The module skipped during collection has no test-case duration and no previous entry.
    assert set(snapshot["durations"]) == set(DISCOVERED) - {"tests/test_epsilon.py"}
    assert snapshot["durations"]["tests/test_gamma.py"] == 7.0
    assert all(value >= 0.001 for value in snapshot["durations"].values())
    assert shards.read_timings(output) == snapshot["durations"]


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ('<!DOCTYPE t [<!ENTITY e "x">]><testsuites/>', "document type declaration"),
        ('<testsuites><testcase classname="tests.test_alpha" time="-1"/></testsuites>', "finite"),
        ('<testsuites><testcase classname="tests.test_alpha" time="nan"/></testsuites>', "finite"),
        (
            '<testsuites><testcase classname="other.module" time="1"/></testsuites>',
            "no live module",
        ),
    ],
)
def test_timing_record_refuses_unsafe_or_useless_reports(
    tmp_path: Path, report: str, message: str
) -> None:
    """Reports with a DTD, impossible durations or no live module produce no snapshot."""
    project = _project(tmp_path / "project")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "junit.xml").write_text(report, encoding="utf-8")
    output = tmp_path / "timings.json"
    with pytest.raises(ValueError, match=message):
        shards.record_timings(project, junit_dir=reports, previous=None, output=output, source={})
    assert not output.exists()


def test_command_line_runs_every_command(
    tmp_path: Path, commit: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run``, ``inventory``, ``verify`` and ``record`` work through the CLI."""
    project = _project(tmp_path / "project")
    root = ["--root", str(project)]
    for index in (1, 2):
        out = tmp_path / "shards" / f"python-shard-3.12-{index}"
        command = [*root, "run", "--lane", "3.12", "--shard", f"{index}/2", "--out-dir", str(out)]
        assert shards.main([*command, "--", "-q"]) == 0
    inventory = tmp_path / "inventory.json"
    assert shards.main([*root, "inventory", "--lane", "3.12", "--output", str(inventory)]) == 0
    verify = [*root, "verify", "--lane", "3.12", "--shards-dir", str(tmp_path / "shards")]
    verify += ["--inventory", str(inventory), "--commit", COMMIT]
    assert shards.main([*verify, "--expect-shards", "2"]) == 0
    assert shards.main([*verify, "--expect-shards", "3"]) == 1
    assert "::error::Python 3.12: expected shards 1..3" in capsys.readouterr().out
    timings = tmp_path / "timings.json"
    record = [*root, "record", "--junit-dir", str(tmp_path / "shards"), "--output", str(timings)]
    assert shards.main([*record, "--source", "run_id=7"]) == 0
    assert json.loads(timings.read_text())["source"] == {"run_id": "7"}
    assert shards.main([*record, "--source", "no-separator"]) == 2
    assert "--source must be KEY=VALUE" in capsys.readouterr().err


def test_command_line_combines_coverage_and_reports_invalid_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``combine`` works through the CLI; invalid inputs exit 2 with a diagnostic."""
    data = tmp_path / "python-shard-3.12-1"
    data.mkdir()
    _coverage_data(data, ".coverage.shard-1", LEFT)
    combine = [
        "combine",
        "--lane",
        "3.12",
        "--shards-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "c"),
    ]
    assert shards.main([*combine, "--expect-shards", "1"]) == 0
    assert shards.main([*combine, "--expect-shards", "2"]) == 2
    project = _project(tmp_path / "project")
    assert (
        shards.main(
            ["--root", str(project), "run", "--lane", "3.12", "--shard", "0/2", "--out-dir", "x"]
        )
        == 2
    )
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("[]", encoding="utf-8")
    verify = ["--root", str(project), "verify", "--lane", "3.12", "--expect-shards", "1"]
    verify += [
        "--shards-dir",
        str(tmp_path / "bad"),
        "--inventory",
        str(tmp_path / "x"),
        "--commit",
        "c",
    ]
    assert shards.main(verify) == 2
    assert "expected a JSON object" in capsys.readouterr().err


def test_the_cli_module_runs_as_a_script(tmp_path: Path) -> None:
    """The workflow invokes the tool as ``python tools/ci_test_shards.py``."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "ci_test_shards.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{run,inventory,verify,combine,record}" in result.stdout
