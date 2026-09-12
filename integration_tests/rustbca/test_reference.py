# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RustBCA real runtime contract tests
"""Test actual isolated RustBCA execution through the public adapter and CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import venv
from pathlib import Path
from typing import Any

import pytest

from validation.rustbca_reference import run_rustbca_reference

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runtime() -> tuple[Path, str]:
    """Identify the real separately installed optional binary used by this lane."""
    python = Path(os.environ["RUSTBCA_PYTHON"])
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import libRustBCA,hashlib; from pathlib import Path; print(hashlib.sha256(Path(libRustBCA.__file__).read_bytes()).hexdigest())",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return python, result.stdout.strip()


@pytest.fixture()
def request_case() -> dict[str, Any]:
    """Load the preserved upstream-example D/W energy-angle request."""
    value = json.loads((ROOT / "validation/reference_data/rustbca_request.json").read_bytes())
    assert isinstance(value, dict)
    return value


def test_actual_grid_preserves_seed_batches_and_unresolved_zero_dispersion(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Retain real reflected/sputtered counts without upgrading zero events to certainty."""
    report = run_rustbca_reference(
        request_case, tmp_path / "run", python=runtime[0], expected_binary_sha256=runtime[1]
    )
    assert len(report["rows"]) == 4
    assert json.loads((tmp_path / "run/report.json").read_bytes()) == report
    assert report["request"] == request_case
    assert not report["provenance_verified"] and not report["actionable"]
    assert not report["material_evidence_verified"] and not report["federated"]
    for name, digest in report["files"].items():
        assert hashlib.sha256((tmp_path / "run" / name).read_bytes()).hexdigest() == digest
    for row in report["rows"]:
        assert [batch["seed"] for batch in row["batches"]] == request_case["seeds"]
        for batch in row["batches"]:
            assert batch["binary_sha256"] == runtime[1]
            assert batch["sputtering_yield"] == batch["sputtered_count"] / 512
            assert batch["number_reflection"] == batch["reflected_count"] / 512
        if row["energy_eV"] == 100:
            summary = row["summary"]["sputtering_yield"]
            assert summary["mean"] == 0
            assert summary["batch_standard_error"] is None
            assert summary["uncertainty_status"] == "unresolved_zero_observed_dispersion"
        else:
            assert row["summary"]["sputtering_yield"]["batch_standard_error"] > 0


def test_wrong_binary_digest_refuses_with_retained_log(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The real runtime cannot claim another binary identity or write a success report."""
    with pytest.raises(subprocess.CalledProcessError):
        run_rustbca_reference(
            request_case, tmp_path / "run", python=runtime[0], expected_binary_sha256="0" * 64
        )
    assert "binary digest mismatch" in (tmp_path / "run/batch-0000.log").read_text()
    assert not (tmp_path / "run/report.json").exists()


def test_expired_total_budget_preserves_request_without_success(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A total deadline cannot be extended separately for each batch."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_rustbca_reference(
            request_case,
            tmp_path / "run",
            python=runtime[0],
            expected_binary_sha256=runtime[1],
            timeout_seconds=1e-12,
        )
    assert (tmp_path / "run/request.json").is_file()
    assert not (tmp_path / "run/report.json").exists()


def test_existing_output_is_not_overwritten(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Preserve earlier output directories rather than mixing execution receipts."""
    with pytest.raises(FileExistsError):
        run_rustbca_reference(
            request_case, tmp_path, python=runtime[0], expected_binary_sha256=runtime[1]
        )
    assert list(tmp_path.iterdir()) == []


def test_actual_cli_executes_small_request(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Exercise the real user entry point and its child module dispatch."""
    request_case.update(energies_eV=[1000], angles_deg=[0], seeds=[0, 1], samples_per_seed=32)
    source = tmp_path / "input.json"
    source.write_text(json.dumps(request_case))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(source),
            str(tmp_path / "run"),
            "--python",
            str(runtime[0]),
            "--binary-sha256",
            runtime[1],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "run/report.json").read_bytes())
    assert len(report["rows"]) == 1
    assert len(report["rows"][0]["batches"]) == 2
    assert not report["evidence_claimed"]


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 10**1000])
def test_invalid_deadline_refuses_before_creating_files(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
    timeout: float,
) -> None:
    """Invalid and oversized execution budgets never reach the child process."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_rustbca_reference(
            request_case,
            tmp_path / "run",
            python=runtime[0],
            expected_binary_sha256=runtime[1],
            timeout_seconds=timeout,
        )
    assert not (tmp_path / "run").exists()


@pytest.fixture()
def isolated_runtime(tmp_path: Path) -> Path:
    """Create a test-owned interpreter, adding instrumentation only when requested."""
    venv.EnvBuilder(with_pip=False).create(tmp_path / "empty")
    python = tmp_path / "empty/bin/python"
    if os.environ.get("COVERAGE_PROCESS_START"):
        site = subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
            timeout=10,
        ).strip()
        shutil.copyfile(
            Path(sysconfig.get_path("purelib")) / "rustbca_coverage.pth",
            Path(site) / "rustbca_coverage.pth",
        )
    return python


def test_missing_optional_binary_is_reported_by_actual_empty_runtime(
    runtime: tuple[Path, str],
    isolated_runtime: Path,
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """An actual environment without the optional binary refuses execution."""
    with pytest.raises(subprocess.CalledProcessError):
        run_rustbca_reference(
            request_case,
            tmp_path / "run",
            python=isolated_runtime,
            expected_binary_sha256=runtime[1],
        )
    assert "Install the optional RustBCA runtime" in (tmp_path / "run/batch-0000.log").read_text()
    assert not (tmp_path / "run/report.json").exists()


def test_binary_changed_after_loading_refuses_success_report(
    runtime: tuple[Path, str],
    isolated_runtime: Path,
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Detect a changed scratch binary after the real worker has mapped the original."""
    receipt = json.loads(Path(os.environ["RUSTBCA_BUILD_RECEIPT"]).read_bytes())
    wheel = Path(receipt["wheel"])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["wheel_sha256"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(isolated_runtime),
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    location = subprocess.check_output(
        [
            str(isolated_runtime),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('libRustBCA').origin)",
        ],
        text=True,
        timeout=10,
    ).strip()
    binary = Path(location).resolve()
    assert binary.is_relative_to(tmp_path.resolve())
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == runtime[1]
    request_case.update(energies_eV=[1000], angles_deg=[0], seeds=[0, 1], samples_per_seed=10_000)
    source = tmp_path / "input.json"
    source.write_text(json.dumps(request_case))
    output = tmp_path / "run"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(source),
            str(output),
            "--python",
            str(isolated_runtime),
            "--binary-sha256",
            runtime[1],
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            deadline = time.monotonic() + 30
            mapped_pid = None
            while mapped_pid is None:
                assert child.poll() is None, "reference exited before loading the scratch binary"
                assert time.monotonic() < deadline, "worker did not map the scratch binary"
                children = Path(f"/proc/{child.pid}/task/{child.pid}/children").read_text()
                for pid in children.split():
                    try:
                        maps = Path(f"/proc/{pid}/maps").read_text()
                    except FileNotFoundError:
                        continue
                    if str(binary) in maps:
                        mapped_pid = int(pid)
                        break
                if mapped_pid is None:
                    time.sleep(0.001)
            with binary.open("ab") as changed:
                changed.write(b"\nchanged retained binary\n")
            (tmp_path / "binary-change.json").write_text(
                json.dumps(
                    {
                        "worker_pid": mapped_pid,
                        "original_sha256": runtime[1],
                        "changed_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    }
                )
            )
            child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=10)
    assert child.returncode != 0
    assert "binary or request changed during execution" in (output / "batch-0000.log").read_text()
    assert not (output / "report.json").exists()
    assert hashlib.sha256(Path(receipt["binary"]).read_bytes()).hexdigest() == runtime[1]


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_malformed_binary_identity_refuses_before_output(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
    digest: str,
) -> None:
    """Reject malformed caller identities before creating a run or dispatching workers."""
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="expected_binary_sha256"):
        run_rustbca_reference(
            request_case, output, python=runtime[0], expected_binary_sha256=digest
        )
    assert not output.exists()


def test_modified_completed_batch_is_refused_before_final_report(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Change an actual completed output while later seed batches are running."""
    source = tmp_path / "input.json"
    source.write_text(json.dumps(request_case))
    output = tmp_path / "run"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(source),
            str(output),
            "--python",
            str(runtime[0]),
            "--binary-sha256",
            runtime[1],
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            deadline = time.monotonic() + 30
            while not (output / "batch-0001-input.json").exists():
                assert child.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.001)
            with (output / "batch-0000-output.json").open("ab") as changed:
                changed.write(b"\n ")
            _, stderr = child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert child.returncode != 0
    assert "retained artifact changed during execution" in stderr
    assert not (output / "report.json").exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("seed", 1, "child receipt does not match requested custody"),
        ("number_reflection", 2.0, "diagnostics exceed the declared coefficient contract"),
        ("sputtering_yield", 0.000025, "aggregate count is not integral"),
        ("sputtering_yield", 1e308, "aggregate count is not integral"),
        ("energy_reflection", float("nan"), "returned nonfinite diagnostics"),
        ("number_reflection", False, "diagnostics must be numeric scalars"),
        ("sputtered_count", -1, "child counts do not match diagnostics"),
        ("reflected_count", False, "child counts do not match diagnostics"),
    ],
)
def test_changed_child_identity_refuses_before_aggregation(
    runtime: tuple[Path, str],
    request_case: dict[str, Any],
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    """Refuse changed identity or invalid diagnostics before ingesting a real worker result."""
    request_case.update(energies_eV=[1000], angles_deg=[0], seeds=[0, 1], samples_per_seed=10_000)
    source = tmp_path / "input.json"
    source.write_text(json.dumps(request_case))
    output = tmp_path / "run"
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(source),
            str(output),
            "--python",
            str(runtime[0]),
            "--binary-sha256",
            runtime[1],
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        worker_pid = None
        while worker_pid is None:
            assert child.poll() is None, "parent exited before dispatching a worker"
            assert time.monotonic() < deadline, "worker dispatch deadline expired"
            children = Path(f"/proc/{child.pid}/task/{child.pid}/children").read_text().split()
            if children:
                assert len(children) == 1
                worker_pid = int(children[0])
            else:
                time.sleep(0.001)
        os.kill(child.pid, signal.SIGSTOP)
        while True:
            stopped, status = os.waitpid(child.pid, os.WUNTRACED | os.WNOHANG)
            if stopped:
                assert os.WIFSTOPPED(status)
                break
            assert time.monotonic() < deadline, "parent stop deadline expired"
            time.sleep(0.001)
        # The actual worker continues; only its waiting parent is stopped.
        while True:
            status_text = Path(f"/proc/{worker_pid}/status").read_text()
            if any(
                line.startswith("State:") and "Z (zombie)" in line
                for line in status_text.splitlines()
            ):
                break
            assert time.monotonic() < deadline, "actual worker did not finish in time"
            time.sleep(0.001)
        result = output / "batch-0000-output.json"
        original = result.read_bytes()
        payload = json.loads(original)
        assert payload["seed"] == 0
        assert payload["binary_sha256"] == runtime[1]
        payload[field] = value
        result.write_text(json.dumps(payload, allow_nan=True))
        (tmp_path / "child-identity-change.json").write_text(
            json.dumps(
                {
                    "worker_pid": worker_pid,
                    "parent_pid": child.pid,
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "changed_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                    "original_seed": 0,
                    "changed_field": field,
                    "changed_value": repr(value),
                }
            )
        )
        os.kill(child.pid, signal.SIGCONT)
        _, stderr = child.communicate(timeout=30)
        assert child.returncode != 0
        assert "ValueError: RustBCA " + message in stderr
        assert not (output / "batch-0001-input.json").exists()
        assert not (output / "report.json").exists()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate(timeout=10)
