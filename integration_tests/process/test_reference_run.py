# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — real PROCESS source/runtime contract tests
"""Verify runtime custody and refusal behaviour against actual upstream execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import json
import venv
from pathlib import Path
from typing import Any

import pytest

from validation.process_reference_run import run_process_reference, verify_process_source


def test_installed_source_matches_pinned_snapshot() -> None:
    """Check installed source and packaged physical data before using the model."""
    source = verify_process_source()
    assert source["commit"] == "620d1e9a38f1b3c6d2597956c8556e9ab6c17037"
    assert len(source["manifest_sha256"]) == 64


def test_real_run_records_custody_without_physical_admission(
    reference_run: tuple[Path, dict[str, Any]],
) -> None:
    """A completed real run retains source/file hashes and its negative diagnostics."""
    directory, report = reference_run
    assert json.loads((directory / "power-report.json").read_text()) == report
    assert report["provenance_verified"]
    assert report["execution"]["source_verified_locally"]
    assert not report["execution"]["third_party_attested"]
    for name, digest in report["execution"]["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert report["scans"][0]["checks"]["solver_converged"]
    assert not report["scans"][0]["checks"]["upstream_has_no_errors"]
    assert not report["actionable"] and not report["evidence_claimed"]


def test_existing_output_directory_is_preserved(process_case: Path, tmp_path: Path) -> None:
    """Never overwrite an earlier run, even with identical input and parameters."""
    marker = tmp_path / "retain.txt"
    marker.write_text("previous run")
    with pytest.raises(FileExistsError):
        run_process_reference(process_case, tmp_path, equality_tolerance=1e-8)
    assert marker.read_text() == "previous run"


def test_changed_case_is_refused_before_execution(process_case: Path, tmp_path: Path) -> None:
    """A modified actual upstream case cannot claim the pinned input identity."""
    changed = tmp_path / "IN.DAT"
    changed.write_bytes(process_case.read_bytes() + b"\n* changed\n")
    with pytest.raises(ValueError, match="pinned case"):
        run_process_reference(changed, tmp_path / "run", equality_tolerance=1e-8)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_tolerance_refuses_before_creating_outputs(
    process_case: Path, tmp_path: Path, tolerance: float
) -> None:
    """An unusable diagnostic tolerance must not start an expensive solver run."""
    with pytest.raises(ValueError, match="equality_tolerance"):
        run_process_reference(process_case, tmp_path / "run", equality_tolerance=tolerance)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "relative_path", ["models/power.py", "data/lz_non_corona_14_elements/Ar_lz_tau.dat"]
)
def test_changed_installed_source_is_refused(tmp_path: Path, relative_path: str) -> None:
    """An altered real package cannot retain the pinned source identity."""
    spec = importlib.util.find_spec("process")
    assert spec is not None and spec.origin is not None
    package = Path(spec.origin).parent
    copied = tmp_path / "process"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))
    target = copied / relative_path
    target.write_bytes(target.read_bytes() + b"\n# altered source\n")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from validation.process_reference_run import verify_process_source; verify_process_source()",
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(tmp_path) + os.pathsep + str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "source/data do not match" in result.stderr


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_cannot_start_run(
    process_case: Path, tmp_path: Path, timeout: float
) -> None:
    """Reject unusable execution limits before creating any output directory."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_process_reference(
            process_case, tmp_path / "run", equality_tolerance=1e-8, timeout_seconds=timeout
        )
    assert not (tmp_path / "run").exists()


def test_actual_process_timeout_retains_input_and_log(process_case: Path, tmp_path: Path) -> None:
    """Kill and reap an actual child under a short deadline, retaining run evidence."""
    output = tmp_path / "run"
    with pytest.raises(subprocess.TimeoutExpired):
        run_process_reference(process_case, output, equality_tolerance=1e-8, timeout_seconds=0.001)
    assert (output / "IN.DAT").read_bytes() == process_case.read_bytes()
    assert (output / "execution.log").is_file()
    assert not (output / "power-report.json").exists()


@pytest.mark.parametrize("install_source", [False, True])
def test_incomplete_runtime_refuses_without_claiming_success(
    process_case: Path, tmp_path: Path, install_source: bool
) -> None:
    """Exercise a genuinely empty environment and a real package missing dependencies.

    The latter passes source custody then fails inside the upstream child import.
    No replacement modules or patched subprocess functions participate.
    """
    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(runtime)
    site_packages = next((runtime / "lib").glob("python*/site-packages"))
    if "COVERAGE_PROCESS_CONFIG" in os.environ:
        # Instrument the otherwise incomplete runtime with the actual measurement
        # tool. This supplies no PROCESS or numerical/model dependencies.
        coverage_distribution = importlib.metadata.distribution("coverage")
        assert coverage_distribution.files is not None
        for entry in coverage_distribution.files:
            if entry.parts[0] == "coverage" or (len(entry.parts) == 1 and entry.suffix == ".pth"):
                destination = site_packages.joinpath(*entry.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(coverage_distribution.locate_file(entry)), destination)
    if install_source:
        spec = importlib.util.find_spec("process")
        assert spec is not None and spec.origin is not None
        shutil.copytree(
            Path(spec.origin).parent,
            site_packages / "process",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        distribution = importlib.metadata.distribution("process")
        assert distribution.files is not None
        metadata = next(
            p
            for p in distribution.files
            if p.name == "METADATA" and p.parent.name.endswith(".dist-info")
        )
        source_metadata = Path(str(distribution.locate_file(metadata))).parent
        shutil.copytree(source_metadata, site_packages / source_metadata.name)
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            str(runtime / "bin/python"),
            "-m",
            "validation.process_reference_run",
            str(process_case),
            str(output),
            "--equality-tolerance",
            "1e-8",
        ],
        env=os.environ | {"PYTHONPATH": str(root)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert not (output / "power-report.json").exists()
    if install_source:
        assert "CalledProcessError" in result.stderr
        assert (output / "IN.DAT").read_bytes() == process_case.read_bytes()
        assert "ModuleNotFoundError" in (output / "execution.log").read_text()
    else:
        assert "Install the pinned PROCESS optional runtime" in result.stderr
        assert not output.exists()


def test_real_cli_preserves_execution_limit_and_diagnostics(
    process_case: Path, tmp_path: Path
) -> None:
    """Run the supported CLI to completion and inspect its actual JSON artifact."""
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.process_reference_run",
            str(process_case),
            str(output),
            "--equality-tolerance",
            "1e-8",
            "--timeout-seconds",
            "120",
        ],
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "power-report.json").read_text())
    assert report["execution"]["timeout_seconds"] == 120.0
    assert (
        report["execution"]["input_sha256"] == hashlib.sha256(process_case.read_bytes()).hexdigest()
    )
    assert report["provenance_verified"]
    assert not report["actionable"] and not report["federated"] and not report["evidence_claimed"]
    assert not report["scans"][0]["checks"]["upstream_has_no_errors"]


def test_case_changed_during_real_execution_cannot_claim_custody(
    process_case: Path, tmp_path: Path
) -> None:
    """Change the retained input while the actual CLI and upstream child are live."""
    output = tmp_path / "run"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.process_reference_run",
            str(process_case),
            str(output),
            "--equality-tolerance",
            "1e-8",
            "--timeout-seconds",
            "120",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            deadline = time.monotonic() + 30
            while not (output / "execution.log").exists():
                assert child.poll() is None, "CLI exited before creating its execution log"
                assert time.monotonic() < deadline, "CLI did not reach upstream execution"
                time.sleep(0.01)
            assert child.poll() is None
            with (output / "IN.DAT").open("ab") as case:
                case.write(b"\n* changed during execution\n")
            _, stderr = child.communicate(timeout=150)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert child.returncode != 0
    assert "PROCESS source or case changed during execution" in stderr
    assert (output / "MFILE.DAT").is_file()
    assert not (output / "power-report.json").exists()
