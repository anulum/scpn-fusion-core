# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — retained RustBCA native build integration tests
"""Exercise actual build evidence and runtime binding through public APIs."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from validation.rustbca_build_receipt import verify_rustbca_build_receipt
from validation.rustbca_reference import run_rustbca_reference


@pytest.fixture()
def build(tmp_path: Path) -> tuple[Path, str, str]:
    """Copy retained evidence from a completed actual native build, without compiling a fake."""
    original = Path(os.environ["RUSTBCA_BUILD_RECEIPT"])
    root = tmp_path / "build"
    root.mkdir()
    shutil.copytree(original.parent / "source", root / "source")
    for name in ("build.log", "build_source.py"):
        shutil.copy2(original.parent / name, root / name)
    record = json.loads(original.read_bytes())
    shutil.copy2(record["binary"], root / "native.so")
    record["binary"] = str(root / "native.so")
    receipt = root / "result.json"
    receipt.write_text(json.dumps(record))
    return receipt, hashlib.sha256(receipt.read_bytes()).hexdigest(), record["binary_sha256"]


def test_retained_build_is_not_independent_attestation(build: tuple[Path, str, str]) -> None:
    """Successful file custody cannot establish third-party provenance or reproducibility."""
    result = verify_rustbca_build_receipt(
        build[0], expected_receipt_sha256=build[1], expected_binary_sha256=build[2]
    )
    assert result["retained_build_inputs_verified"]
    assert not result["independent_attestation_verified"]
    assert not result["binary_reproducibility_verified"]


@pytest.mark.parametrize(
    "name",
    [
        "source/src/lib.rs",
        "source/Cargo.lock",
        "build.log",
        "build_source.py",
        "native.so",
        "result.json",
    ],
)
def test_changed_retained_bytes_refuse(build: tuple[Path, str, str], name: str) -> None:
    """Changing any retained build dependency invalidates the public custody check."""
    path = build[0].parent / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError):
        verify_rustbca_build_receipt(
            build[0], expected_receipt_sha256=build[1], expected_binary_sha256=build[2]
        )


def test_unrecorded_build_source_refuses(build: tuple[Path, str, str]) -> None:
    """An extra Cargo build script cannot hide outside the pinned source inventory."""
    (build[0].parent / "source/build.rs").write_text("fn main() {}")
    with pytest.raises(ValueError, match="unrecorded"):
        verify_rustbca_build_receipt(
            build[0], expected_receipt_sha256=build[1], expected_binary_sha256=build[2]
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("returncode", 1),
        ("returncode", False),
        ("source_unchanged", False),
        ("upstream", None),
        ("source_files", []),
        ("source_files", {}),
        ("binary", "native.so"),
        ("binary", None),
        ("cargo_lock_sha256", "0" * 64),
        ("binary_sha256", "0" * 64),
    ],
)
def test_rehashed_invalid_record_refuses(
    build: tuple[Path, str, str], key: str, value: Any
) -> None:
    """A caller's digest does not override pinned inputs or unsuccessful build outcomes."""
    record = json.loads(build[0].read_bytes())
    record[key] = value
    build[0].write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify_rustbca_build_receipt(
            build[0],
            expected_receipt_sha256=hashlib.sha256(build[0].read_bytes()).hexdigest(),
            expected_binary_sha256=build[2],
        )


def test_actual_cli_binds_build_to_runtime(build: tuple[Path, str, str], tmp_path: Path) -> None:
    """The user CLI records verified local build custody while retaining physical refusal."""
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "validation/reference_data/rustbca_request.json").read_bytes())
    request.update(energies_eV=[1000], angles_deg=[0], samples_per_seed=32, seeds=[0, 1])
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request))
    output = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(source),
            str(output),
            "--python",
            os.environ["RUSTBCA_PYTHON"],
            "--binary-sha256",
            build[2],
            "--build-receipt",
            str(build[0]),
            "--build-receipt-sha256",
            build[1],
        ],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    result = json.loads((output / "report.json").read_bytes())
    assert result["build_custody"]["receipt_sha256"] == build[1]
    assert result["build_custody"]["retained_build_inputs_verified"]
    assert not result["provenance_verified"] and not result["actionable"]
    assert all(row["batches"][0]["binary_sha256"] == build[2] for row in result["rows"])


def test_invalid_build_refuses_before_output(build: tuple[Path, str, str], tmp_path: Path) -> None:
    """Reject a bad build receipt before dispatching any runtime batch or creating output."""
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "validation/reference_data/rustbca_request.json").read_bytes())
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="receipt digest"):
        run_rustbca_reference(
            request,
            output,
            python=Path(os.environ["RUSTBCA_PYTHON"]),
            expected_binary_sha256=build[2],
            build_receipt=build[0],
            expected_build_receipt_sha256="0" * 64,
        )
    assert not output.exists()


def test_source_change_during_actual_run_refuses(
    build: tuple[Path, str, str], tmp_path: Path
) -> None:
    """Post-run verification catches source changes after successful batch dispatch."""
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "validation/reference_data/rustbca_request.json").read_bytes())
    output = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_rustbca_reference,
            request,
            output,
            python=Path(os.environ["RUSTBCA_PYTHON"]),
            expected_binary_sha256=build[2],
            build_receipt=build[0],
            expected_build_receipt_sha256=build[1],
        )
        deadline = time.monotonic() + 15
        while not (output / "batch-0000-output.json").exists():
            if future.done():
                future.result()
                pytest.fail("runtime completed before first batch became observable")
            if time.monotonic() >= deadline:
                pytest.fail("first real batch did not complete within the observation budget")
            time.sleep(0.002)
        source = build[0].parent / "source/src/lib.rs"
        source.write_bytes(source.read_bytes() + b"\n// changed after dispatch\n")
        with pytest.raises(ValueError, match="retained build file changed"):
            future.result(timeout=30)
    assert not (output / "report.json").exists()
    assert (output / "build-receipt.json").read_bytes() == build[0].read_bytes()


@pytest.mark.parametrize("payload", [None, [], "receipt"])
def test_rehashed_nonobject_receipt_refuses(build: tuple[Path, str, str], payload: Any) -> None:
    """A matching byte digest does not make a non-object JSON payload a build record."""
    raw = json.dumps(payload).encode()
    build[0].write_bytes(raw)
    with pytest.raises(ValueError, match="receipt must be an object"):
        verify_rustbca_build_receipt(
            build[0],
            expected_receipt_sha256=hashlib.sha256(raw).hexdigest(),
            expected_binary_sha256=build[2],
        )


@pytest.mark.parametrize("receipt_only", [True, False])
def test_partial_build_binding_refuses_before_output(
    build: tuple[Path, str, str], tmp_path: Path, receipt_only: bool
) -> None:
    """Neither an unpinned receipt nor a digest without its receipt permits dispatch."""
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "validation/reference_data/rustbca_request.json").read_bytes())
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="must be supplied together"):
        run_rustbca_reference(
            request,
            output,
            python=Path(os.environ["RUSTBCA_PYTHON"]),
            expected_binary_sha256=build[2],
            build_receipt=build[0] if receipt_only else None,
            expected_build_receipt_sha256=None if receipt_only else build[1],
        )
    assert not output.exists()


@pytest.mark.parametrize("stage", ["verification", "retention"])
def test_receipt_changed_during_verification_refuses(
    build: tuple[Path, str, str], stage: str
) -> None:
    """Reject receipt mutation during verification or before CLI receipt retention."""
    log = build[0].parent / "build.log"
    original_log = log.read_bytes()
    original_receipt = build[0].read_bytes()
    log.unlink()
    os.mkfifo(log)
    root = Path(__file__).resolve().parents[2]
    output = build[0].parent / "run"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "validation.rustbca_reference",
            str(root / "validation/reference_data/rustbca_request.json"),
            str(output),
            "--python",
            os.environ["RUSTBCA_PYTHON"],
            "--binary-sha256",
            build[2],
            "--build-receipt",
            str(build[0]),
            "--build-receipt-sha256",
            build[1],
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    descriptor = None
    deadline = time.monotonic() + 15
    phases = [(log, original_log)]
    if stage == "retention":
        phases.append((build[0], original_receipt))
    try:
        for phase, (path, raw) in enumerate(phases):
            while descriptor is None:
                assert process.poll() is None, "CLI exited before reaching the controlled read"
                if time.monotonic() >= deadline:
                    pytest.fail("CLI did not reach the controlled read within the budget")
                try:
                    descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as exc:
                    if exc.errno != errno.ENXIO:
                        raise
                    time.sleep(0.001)
            if stage == "retention" and phase == 0:
                # The log reader has already parsed the original pinned receipt.
                # Make its final verification read wait on a distinct FIFO.
                build[0].unlink()
                os.mkfifo(build[0])
            else:
                # A connected second reader owns the old FIFO inode. It receives
                # original bytes; the later retention read sees the changed file.
                if stage == "retention":
                    build[0].unlink()
                build[0].write_bytes(original_receipt + b"\n")
            offset = 0
            while offset < len(raw):
                assert process.poll() is None, "CLI exited before consuming the original bytes"
                if time.monotonic() >= deadline:
                    pytest.fail("controlled write exceeded the verification budget")
                try:
                    offset += os.write(descriptor, memoryview(raw)[offset:])
                except BlockingIOError:
                    time.sleep(0.001)
            os.close(descriptor)
            descriptor = None
        stdout, stderr = process.communicate(timeout=max(0.001, deadline - time.monotonic()))
        assert process.returncode != 0
        assert not stdout
        message = "during verification" if stage == "verification" else "before retention"
        assert "ValueError: RustBCA build receipt changed " + message in stderr
        (build[0].parent / "receipt-refusal.log").write_text(stderr)
        assert output.exists() == (stage == "retention")
        assert not (output / "build-receipt.json").exists()
        assert not (output / "batch-0000-input.json").exists()
        assert not (output / "report.json").exists()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        log.unlink()
        log.write_bytes(original_log)
        if build[0].is_fifo():
            build[0].unlink()
            build[0].write_bytes(original_receipt)
    assert hashlib.sha256(build[0].read_bytes()).hexdigest() != build[1]
