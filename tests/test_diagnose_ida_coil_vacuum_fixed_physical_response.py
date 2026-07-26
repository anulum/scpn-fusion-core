# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Diagnostic Tests
"""Binding, writer, and CLI tests for the CVGC2 diagnostic."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.ida_coil_vacuum_fixed_physical_fixtures import (
    FROZEN_SOURCE_COMMIT,
    ROOT,
    report_fixture,
    source_artifacts_fixture,
)
from validation import diagnose_ida_coil_vacuum_grid_convergence as grid_diagnostic
from validation import diagnose_ida_coil_vacuum_fixed_physical_response as diagnostic
from validation import ida_coil_vacuum_fixed_physical_contract as contract
from validation import ida_coil_vacuum_grid_convergence as convergence

_REAL_SOURCE_ONLY_REQUIRE = diagnostic._require_source_only_bootstrap


@pytest.fixture(autouse=True)
def _clean_execution_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep writer tests independent of the author's deliberately staged diff."""
    artifacts = source_artifacts_fixture()
    monkeypatch.setattr(
        contract,
        "execution_source_artifacts",
        lambda root: artifacts,
    )
    monkeypatch.setattr(
        contract,
        "execution_source_snapshot",
        lambda root: (artifacts, {"fixture": (1,)}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_IMPORTED_SOURCE_SHA256",
        {name: str(artifacts[name]["sha256"]) for name in contract.SOURCE_PATHS},
    )
    monkeypatch.setattr(diagnostic, "_require_source_only_bootstrap", lambda: None)


def test_fixture_uses_the_source_commit_stored_in_cvgc2_evidence() -> None:
    """Later repository commits must not rewrite the frozen historical test binding."""
    stored = json.loads((ROOT / contract.REPORT_PATH).read_text(encoding="utf-8"))
    assert stored["source_artifacts"]["repository"]["git_commit"] == FROZEN_SOURCE_COMMIT
    assert source_artifacts_fixture()["repository"]["git_commit"] == FROZEN_SOURCE_COMMIT


def _grid(row: dict[str, Any]) -> convergence.GridResult:
    """Bind one upstream public row to minimal unused private arrays."""
    resolution = int(row["resolution"])
    array = np.zeros((1, 1), dtype=np.float64)
    mask = np.ones((1, 1), dtype=np.bool_)
    return convergence.GridResult(
        resolution=resolution,
        report=copy.deepcopy(row),
        total_forcing_zr=array,
        source_forcing_zr=array,
        source_free_forcing_zr=array,
        total_response_zr=array,
        source_response_zr=array,
        source_free_response_zr=array,
        interior_mask=mask,
        primary_source_mask=mask,
        fixed_source_free_mask=mask,
        plasma_support_mask=mask,
    )


def _execution(upstream: dict[str, Any]) -> grid_diagnostic.GridLadderExecution:
    """Return an execution binding over the exact upstream public rows."""
    return grid_diagnostic.GridLadderExecution(
        environment={"backend": "gpu"},
        source_artifacts={},
        bindings={},
        anchor=copy.deepcopy(upstream["anchor"]),
        coil_manifest=copy.deepcopy(upstream["coil_manifest"]),
        results=tuple(_grid(row) for row in upstream["grids"]),
    )


def test_verify_cvgc1_arrays_accepts_exact_upstream_rows() -> None:
    """Every CVGC2 total field and fixed radius must bind to CVGC1 exactly."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    diagnostic._verify_cvgc1_arrays(_execution(upstream), upstream)


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("anchor", "129 anchor drifted"),
        ("manifest", "coil manifest drifted"),
        ("forcing", "33 total forcing drifted"),
        ("response", "33 total response drifted"),
        ("radius", "fixed physical radius drifted"),
    ],
)
def test_verify_cvgc1_arrays_rejects_every_binding_drift(
    surface: str,
    message: str,
) -> None:
    """Anchor, manifest, field, and mask-radius drift must fail independently."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    execution = _execution(upstream)
    if surface == "anchor":
        execution.anchor["forcing_sha256"] = "0" * 64
    elif surface == "manifest":
        execution.coil_manifest["parent_count"] = 17
    elif surface == "forcing":
        execution.results[0].report["forcing_partition"]["total"]["field_sha256"] = "0" * 64
    elif surface == "response":
        execution.results[0].report["response_partition"]["total"]["field_sha256"] = "0" * 64
    elif surface == "radius":
        execution.results[0].report["masks"]["fixed_physical_radius_m"] = 0.2
    else:
        raise AssertionError(f"unhandled surface {surface}")
    with pytest.raises(ValueError, match=message):
        diagnostic._verify_cvgc1_arrays(execution, upstream)


def test_source_artifacts_bind_real_files_and_repository_probe() -> None:
    """Executed source provenance must name and hash every CVGC2 module."""
    assert diagnostic._source_artifacts() == source_artifacts_fixture()


def test_source_only_bootstrap_requires_empty_isolated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The numerical entry point accepts only the direct source-only loader state."""
    monkeypatch.setattr(diagnostic, "_SOURCE_ONLY_BOOTSTRAP", True)
    monkeypatch.setattr(diagnostic, "_SANITIZED_INTERPRETER", True)
    monkeypatch.setattr(diagnostic, "_SOURCE_ONLY_CACHE_PREFIX", tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path))
    for name, path in contract.SOURCE_PATHS.items():
        if name == "fixed_physical_diagnostic":
            continue
        module = sys.modules[f"validation.{Path(path).stem}"]
        monkeypatch.setattr(module, "__cached__", str(tmp_path / f"{name}.pyc"))
    _REAL_SOURCE_ONLY_REQUIRE()

    (tmp_path / "unexpected.pyc").write_bytes(b"bytecode")
    with pytest.raises(ValueError, match="direct source-only diagnostic launcher"):
        _REAL_SOURCE_ONLY_REQUIRE()


@pytest.mark.parametrize(
    "surface",
    [
        "cache_prefix",
        "missing_module",
        "loader",
        "missing_file",
        "wrong_file",
        "external_cache",
    ],
)
def test_source_only_bootstrap_rejects_noncanonical_project_loaders(
    surface: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime guard must reject every noncanonical project-loader surface."""
    monkeypatch.setattr(diagnostic, "_SOURCE_ONLY_BOOTSTRAP", True)
    monkeypatch.setattr(diagnostic, "_SANITIZED_INTERPRETER", True)
    monkeypatch.setattr(diagnostic, "_SOURCE_ONLY_CACHE_PREFIX", tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path))
    modules = {}
    for name, path in contract.SOURCE_PATHS.items():
        if name == "fixed_physical_diagnostic":
            continue
        module_name = f"validation.{Path(path).stem}"
        module = sys.modules[module_name]
        modules[name] = (module_name, module)
        monkeypatch.setattr(module, "__cached__", str(tmp_path / f"{name}.pyc"))

    target_name, target = modules["fixed_physical_response"]
    if surface == "cache_prefix":
        monkeypatch.setattr(diagnostic, "_SOURCE_ONLY_CACHE_PREFIX", None)
    elif surface == "missing_module":
        monkeypatch.delitem(sys.modules, target_name)
    elif surface == "loader":
        monkeypatch.setattr(target, "__loader__", object())
    elif surface == "missing_file":
        monkeypatch.setattr(target, "__file__", None)
    elif surface == "wrong_file":
        monkeypatch.setattr(target, "__file__", str(tmp_path / "external.py"))
    elif surface == "external_cache":
        monkeypatch.setattr(target, "__cached__", str(tmp_path.parent / "external.pyc"))
    else:
        raise AssertionError(f"unhandled surface {surface}")

    with pytest.raises(ValueError, match="direct source-only diagnostic launcher"):
        _REAL_SOURCE_ONLY_REQUIRE()


def test_source_only_prefix_ignores_unchecked_hash_bytecode(tmp_path: Path) -> None:
    """An ignored malicious unchecked-hash pyc cannot replace executed source."""
    package = tmp_path / "validation"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "ida_coil_vacuum_fixed_physical_response.py"
    module.write_text('LOADED = "malicious"\n', encoding="utf-8")
    py_compile.compile(
        str(module),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    module.write_text('LOADED = "clean-source"\n', encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    probe = "from validation.ida_coil_vacuum_fixed_physical_response import LOADED; print(LOADED)"
    unchecked = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert unchecked.stdout.strip() == "malicious"

    isolated_cache = tmp_path / "isolated-cache"
    isolated_cache.mkdir()
    source_only_probe = (
        "import sys; "
        "sys.dont_write_bytecode = True; "
        f"sys.pycache_prefix = {str(isolated_cache)!r}; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        f"{probe}"
    )
    source_only = subprocess.run(
        [sys.executable, "-c", source_only_probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert source_only.stdout.strip() == "clean-source"
    assert not any(isolated_cache.iterdir())

    real_source = diagnostic.ROOT / contract.SOURCE_PATHS["fixed_physical_response"]
    malicious_prefix = tmp_path / "malicious-prefix"
    previous_prefix = sys.pycache_prefix
    try:
        sys.pycache_prefix = str(malicious_prefix)
        malicious_pyc = Path(importlib.util.cache_from_source(str(real_source)))
    finally:
        sys.pycache_prefix = previous_prefix
    malicious_pyc.parent.mkdir(parents=True)
    malicious_source = tmp_path / "malicious_real_module.py"
    malicious_source.write_text(
        'raise RuntimeError("unchecked CVGC2 pyc executed")\n', encoding="utf-8"
    )
    py_compile.compile(
        str(malicious_source),
        cfile=str(malicious_pyc),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    real_environment = os.environ.copy()
    real_environment["PYTHONPYCACHEPREFIX"] = str(malicious_prefix)
    baseline = subprocess.run(
        [sys.executable, "-c", "import validation.ida_coil_vacuum_fixed_physical_response"],
        capture_output=True,
        text=True,
        env=real_environment,
        cwd=diagnostic.ROOT,
    )
    assert baseline.returncode != 0
    assert "unchecked CVGC2 pyc executed" in baseline.stderr

    direct_source = subprocess.run(
        [
            sys.executable,
            str(diagnostic.ROOT / contract.SOURCE_PATHS["fixed_physical_diagnostic"]),
            "--help",
        ],
        capture_output=True,
        text=True,
        env=real_environment,
        cwd=diagnostic.ROOT,
    )
    assert direct_source.returncode == 0
    assert "unchecked CVGC2 pyc executed" not in direct_source.stderr


def test_sanitized_reexec_discards_sitecustomize_preload_and_finder(tmp_path: Path) -> None:
    """PYTHONPATH startup hooks cannot survive the isolated pre-import re-exec."""
    marker = tmp_path / "sitecustomize-loaded.txt"
    injected_module = "validation.ida_coil_vacuum_fixed_physical_response"
    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import sys",
                "import types",
                "from pathlib import Path",
                f"with Path({str(marker)!r}).open('a', encoding='utf-8') as stream:",
                "    stream.write('loaded\\n')",
                f"module = types.ModuleType({injected_module!r})",
                "module.__file__ = '/external/injected.py'",
                f"sys.modules[{injected_module!r}] = module",
                "class InjectedFinder:",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        return None",
                "sys.meta_path.insert(0, InjectedFinder())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    contaminated = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; print({injected_module!r} in sys.modules); "
            "print(type(sys.meta_path[0]).__name__)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=diagnostic.ROOT,
    )
    assert contaminated.stdout.splitlines() == ["True", "InjectedFinder"]

    sanitized = subprocess.run(
        [
            sys.executable,
            str(diagnostic.ROOT / contract.SOURCE_PATHS["fixed_physical_diagnostic"]),
            "--help",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=diagnostic.ROOT,
    )
    assert sanitized.returncode == 0
    assert "usage:" in sanitized.stdout
    assert marker.read_text(encoding="utf-8").splitlines() == ["loaded", "loaded"]


@pytest.mark.parametrize(
    ("version", "safe_path", "expected"),
    [
        ((3, 10), None, True),
        ((3, 10), False, False),
        ((3, 11), None, False),
        ((3, 11), True, True),
    ],
)
def test_sanitized_interpreter_flags_are_version_bound(
    version: tuple[int, int],
    safe_path: bool | None,
    expected: bool,
) -> None:
    """Only Python 3.10 may lack the post-3.10 ``safe_path`` flag."""
    flags = SimpleNamespace(isolated=1, no_site=1, ignore_environment=1)
    if safe_path is not None:
        flags.safe_path = safe_path
    assert diagnostic._sanitized_interpreter_flags_are_valid(flags, version=version) is expected


@pytest.mark.parametrize(
    "environment",
    [
        {
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/forbidden/native-loader",
            "CVGC2_FORBIDDEN_SENTINEL": "present",
        },
        {
            "HOME": "/caller-controlled",
            "PATH": "/usr/bin:/bin",
        },
    ],
    ids=["forbidden-native-loader-environment", "forged-fixed-environment"],
)
def test_sanitized_child_flag_rejects_caller_forged_environment(
    environment: dict[str, str],
) -> None:
    """Caller-selected child mode cannot bypass exact environment sanitization."""
    forged_child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(diagnostic.ROOT / contract.SOURCE_PATHS["fixed_physical_diagnostic"]),
            diagnostic._CHILD_FLAG,
            "--help",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=diagnostic.ROOT,
    )
    assert forged_child.returncode != 0
    assert "CVGC2 sanitized child environment is invalid" in forged_child.stderr


def test_run_diagnostic_routes_exact_execution_into_fixed_partition_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must preserve execution bindings through the CVGC2 builder."""
    upstream = contract.load_upstream_report(diagnostic.ROOT)
    execution = _execution(upstream)
    fixed_rows = [
        SimpleNamespace(report={"resolution": row.resolution}) for row in execution.results
    ]
    captured: dict[str, Any] = {}

    monkeypatch.setattr(contract, "load_upstream_report", lambda root: upstream)
    monkeypatch.setattr(grid_diagnostic, "execute_grid_ladder", lambda: execution)
    monkeypatch.setattr(diagnostic, "build_fixed_physical_grid", lambda row: fixed_rows.pop(0))
    monkeypatch.setattr(
        diagnostic,
        "build_fixed_physical_convergence",
        lambda rows: {"count": len(rows)},
    )
    monkeypatch.setattr(diagnostic, "_source_artifacts", lambda: {"repository": {}})
    monkeypatch.setattr(
        contract,
        "build_report",
        lambda **kwargs: captured.update(kwargs) or {"result": "built"},
    )

    assert diagnostic.run_diagnostic(generated_at="2026-07-26T03:30:00Z") == {"result": "built"}
    assert captured["generated_at"] == "2026-07-26T03:30:00Z"
    assert captured["grids"] == [
        {"resolution": 33},
        {"resolution": 65},
        {"resolution": 129},
        {"resolution": 257},
    ]
    assert captured["convergence"] == {"count": 4}
    assert set(captured["execution_binding"]) == {
        "anchor_sha256",
        "coil_manifest_sha256",
        "source_artifacts_sha256",
    }


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("import", "imported source bytes"),
        ("runtime", "drifted during numerical execution"),
    ],
)
def test_run_diagnostic_rejects_import_or_transient_execution_drift(
    mode: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-time and mutate-then-restore execution drift both fail closed."""
    artifacts = source_artifacts_fixture()
    snapshots = iter(
        [
            (artifacts, {"source": (1,)}),
            (artifacts, {"source": (2,)}),
        ]
    )
    monkeypatch.setattr(contract, "execution_source_snapshot", lambda root: next(snapshots))
    if mode == "import":
        imported = {name: str(artifacts[name]["sha256"]) for name in contract.SOURCE_PATHS}
        imported[next(iter(contract.SOURCE_PATHS))] = "0" * 64
        monkeypatch.setattr(diagnostic, "_IMPORTED_SOURCE_SHA256", imported)
    else:
        upstream = contract.load_upstream_report(diagnostic.ROOT)
        execution = _execution(upstream)
        monkeypatch.setattr(contract, "load_upstream_report", lambda root: upstream)
        monkeypatch.setattr(grid_diagnostic, "execute_grid_ladder", lambda: execution)
        monkeypatch.setattr(
            diagnostic,
            "build_fixed_physical_grid",
            lambda row: SimpleNamespace(report={"resolution": row.resolution}),
        )
    with pytest.raises(ValueError, match=message):
        diagnostic.run_diagnostic(generated_at="2026-07-26T06:00:00Z")


def test_writer_and_cli_round_trip_validated_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must emit self-validating evidence and exact output digests."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)

    assert (
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
                "--check",
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    contract.validate_report(written)
    assert markdown.read_text(encoding="utf-8") == contract.render_markdown(report)
    result = json.loads(capsys.readouterr().out)
    assert result["payload_sha256"] == report["payload_sha256"]
    assert result["json_sha256"] == diagnostic._file_sha256(output)


def test_cli_without_check_writes_and_reports_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The normal CLI path must not require the optional re-read check."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)
    assert (
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["payload_sha256"] == report["payload_sha256"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonobject", "must remain a JSON object"),
        ("payload", "payload drifted"),
        ("markdown", "must end with a newline"),
    ],
)
def test_cli_check_rejects_written_output_drift(
    case: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-write object, payload, and Markdown corruption must fail independently."""
    report = report_fixture()
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *, generated_at: report)

    def drifted_writer(
        value: dict[str, Any],
        *,
        output: Path,
        markdown_output: Path,
    ) -> None:
        written: object = value
        markdown_text = "evidence\n"
        if case == "nonobject":
            written = []
        elif case == "payload":
            written = copy.deepcopy(value)
            cast_written = written
            if not isinstance(cast_written, dict):
                raise AssertionError("payload case must remain an object")
            cast_written["payload_sha256"] = "0" * 64
        elif case == "markdown":
            markdown_text = "evidence"
        else:
            raise AssertionError(f"unhandled case {case}")
        output.write_text(json.dumps(written), encoding="utf-8")
        markdown_output.write_text(markdown_text, encoding="utf-8")

    monkeypatch.setattr(diagnostic, "write_report", drifted_writer)
    if case in {"payload", "markdown"}:
        monkeypatch.setattr(contract, "validate_report", lambda value: None)
    with pytest.raises(ValueError, match=message):
        diagnostic.main(
            [
                "--generated-at",
                "2026-07-26T03:30:00Z",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
                "--check",
            ]
        )
