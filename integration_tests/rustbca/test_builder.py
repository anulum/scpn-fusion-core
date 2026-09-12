# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — native build and wheel installation integration tests
"""Verify actual builder failures and install its real wheel into a clean runtime."""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import time
import venv
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from tools.build_rustbca_reference import build_rustbca_reference
from validation.rustbca_reference import run_rustbca_reference


@pytest.fixture()
def pinned_checkout(tmp_path: Path) -> Path:
    """Copy exact maintained code and pinned inputs for isolated filesystem changes."""
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "rustbca-input-custody-checkout"
    paths = [
        "tools/build_rustbca_reference.py",
        "validation/rustbca_build_receipt.py",
        "validation/reference_data/rustbca_source_files.json",
        "validation/reference_data/rustbca.Cargo.lock",
    ]
    for name in paths:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        assert destination.read_bytes() == (root / name).read_bytes()
    return checkout


@pytest.fixture(params=["unexpected_name", "duplicate", "content", "missing", "symlink", "fifo"])
def changed_archive(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, str]:
    """Change one member of the actual pinned archive while retaining its other bytes."""
    destination = tmp_path / "changed-source.tar.gz"
    with (
        tarfile.open(os.environ["RUSTBCA_SOURCE_ARCHIVE"], "r:gz") as source,
        tarfile.open(destination, "w:gz") as changed,
    ):
        selected = next(member.name for member in source.getmembers() if member.isfile())
        for member in source.getmembers():
            stream = source.extractfile(member) if member.isfile() else None
            raw = stream.read() if stream is not None else b""
            if member.name == selected:
                if request.param == "missing":
                    continue
                if request.param == "duplicate":
                    changed.addfile(member, io.BytesIO(raw))
                member = copy.copy(member)
                if request.param == "unexpected_name":
                    member.name += ".changed"
                if request.param == "content":
                    raw += b"\nchanged source\n"
                    member.size = len(raw)
                if request.param in {"symlink", "fifo"}:
                    member.type = (
                        tarfile.SYMTYPE if request.param == "symlink" else tarfile.FIFOTYPE
                    )
                    member.linkname = "unexpected-link-target" if request.param == "symlink" else ""
                    member.size = 0
                    raw = b""
            changed.addfile(member, io.BytesIO(raw) if member.isfile() else None)
    message = {
        "unexpected_name": "Unexpected RustBCA source archive member",
        "duplicate": "Unexpected RustBCA source archive member",
        "content": "RustBCA archive member digest mismatch",
        "missing": "Incomplete RustBCA source archive",
        "symlink": "Unexpected RustBCA source archive member",
        "fifo": "Unexpected RustBCA source archive member",
    }[request.param]
    return destination, message


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("rustbca_source_files.json", "source inventory mismatch"),
        ("rustbca.Cargo.lock", "Cargo lock mismatch"),
    ],
)
def test_changed_pinned_build_input_refuses_through_actual_cli(
    completed_build: tuple[Path, dict[str, Any]],
    pinned_checkout: Path,
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    """Execute byte-identical maintained code against a changed inventory or lockfile."""
    _, record = completed_build
    checkout = pinned_checkout
    (checkout / "validation/reference_data" / filename).write_bytes(b"{}\n")
    output = tmp_path / "build"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.build_rustbca_reference",
            os.environ["RUSTBCA_SOURCE_ARCHIVE"],
            str(output),
            "--python",
            str(record["environment"]["PYO3_PYTHON"]),
            "--cargo",
            str(record["command"][0]),
            "--rustc",
            str(record["environment"]["RUSTC"]),
        ],
        cwd=checkout,
        env=os.environ | {"PYTHONPATH": str(checkout)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("stall_reader", [False, True], ids=["consuming", "stalled"])
def test_archive_changed_after_hash_refuses_before_compilation(
    completed_build: tuple[Path, dict[str, Any]],
    pinned_checkout: Path,
    changed_archive: tuple[Path, str],
    tmp_path: Path,
    stall_reader: bool,
) -> None:
    """Coordinate a real archive replacement after hashing through the next file read."""
    _, record = completed_build
    archive = tmp_path / "source.tar.gz"
    shutil.copyfile(os.environ["RUSTBCA_SOURCE_ARCHIVE"], archive)
    inventory = pinned_checkout / "validation/reference_data/rustbca_source_files.json"
    inventory_raw = inventory.read_bytes()
    inventory.unlink()
    os.mkfifo(inventory)
    output = tmp_path / "build"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.build_rustbca_reference",
            str(archive),
            str(output),
            "--python",
            str(record["environment"]["PYO3_PYTHON"]),
            "--cargo",
            str(record["command"][0]),
            "--rustc",
            str(record["environment"]["RUSTC"]),
        ],
        cwd=pinned_checkout,
        env=os.environ | {"PYTHONPATH": str(pinned_checkout)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            deadline = time.monotonic() + 30
            while True:
                assert child.poll() is None, "builder exited before reading its inventory"
                assert time.monotonic() < deadline, "builder did not open its inventory"
                try:
                    descriptor = os.open(inventory, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as exc:
                    if exc.errno != errno.ENXIO:
                        raise
                    time.sleep(0.001)
            try:
                if stall_reader:
                    capacity = fcntl.fcntl(descriptor, fcntl.F_SETPIPE_SZ, 4096)
                    assert capacity < len(inventory_raw)
                    os.kill(child.pid, signal.SIGSTOP)
                    while True:
                        stopped, status = os.waitpid(child.pid, os.WUNTRACED | os.WNOHANG)
                        if stopped:
                            assert os.WIFSTOPPED(status)
                            break
                        assert time.monotonic() < deadline, "inventory reader did not stop"
                        time.sleep(0.001)
                assert not output.exists()
                changed_archive[0].replace(archive)
                write_deadline = (
                    min(deadline, time.monotonic() + 0.25) if stall_reader else deadline
                )
                expectation = (
                    pytest.raises(TimeoutError, match="inventory write deadline")
                    if stall_reader
                    else nullcontext()
                )
                offset = 0
                blocked_writes = 0
                with expectation:
                    while offset < len(inventory_raw):
                        if time.monotonic() >= write_deadline:
                            raise TimeoutError("inventory write deadline expired")
                        assert child.poll() is None, "builder exited while reading inventory"
                        try:
                            offset += os.write(descriptor, memoryview(inventory_raw)[offset:])
                        except BlockingIOError:
                            blocked_writes += 1
                            time.sleep(0.001)
            finally:
                os.close(descriptor)
            if not stall_reader:
                _, stderr = child.communicate(timeout=30)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=10)
    if stall_reader:
        assert child.returncode == -signal.SIGKILL
        assert 0 < offset < len(inventory_raw) and blocked_writes > 0
        assert not output.exists()
        with pytest.raises(ProcessLookupError):
            os.killpg(child.pid, 0)
        (tmp_path / "stalled-reader.json").write_text(
            json.dumps(
                {
                    "pid": child.pid,
                    "returncode": child.returncode,
                    "bytes_written": offset,
                    "inventory_bytes": len(inventory_raw),
                    "blocked_writes": blocked_writes,
                    "process_group_reaped": True,
                }
            )
        )
        return
    assert child.returncode != 0
    assert changed_archive[1] in stderr
    failure = json.loads((output / "result.json").read_bytes())
    assert failure["status"] == "failed" and failure["error_type"] == "ValueError"
    assert "compiler_pid" not in failure
    assert not (output / "target").exists()
    assert not list(output.glob("*.whl"))


@pytest.fixture()
def completed_build() -> tuple[Path, dict[str, Any]]:
    """Require a successful CLI build made from the exact current maintained recipe."""
    receipt = Path(os.environ["RUSTBCA_BUILD_RECEIPT"])
    record = json.loads(receipt.read_bytes())
    recipe = Path(__file__).resolve().parents[2] / "tools/build_rustbca_reference.py"
    assert record["recipe_sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert record["returncode"] == 0 and record["wheel_built"]
    return receipt, record


def test_real_wheel_installs_and_runs(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Install the actual built wheel offline and run diagnostics with the same build receipt."""
    receipt, record = completed_build
    wheel = Path(record["wheel"])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == record["wheel_sha256"]
    with zipfile.ZipFile(wheel) as archive:
        assert (
            archive.read("rustbca-3.0.0.dist-info/LICENSE")
            == (receipt.parent / "source/LICENSE").read_bytes()
        )
        assert b"libRustBCA" in archive.read("rustbca-3.0.0.dist-info/RECORD")
    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(runtime)
    python = runtime / "bin/python"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "validation/reference_data/rustbca_request.json").read_bytes())
    request.update(energies_eV=[1000], angles_deg=[0], samples_per_seed=64, seeds=[0, 1])
    result = run_rustbca_reference(
        request,
        tmp_path / "run",
        python=python,
        expected_binary_sha256=record["binary_sha256"],
        build_receipt=receipt,
        expected_build_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )
    assert result["build_custody"]["retained_build_inputs_verified"]
    assert not result["provenance_verified"] and not result["actionable"]


@pytest.mark.parametrize(
    "budget", [0, -1, float("nan"), float("inf"), True, 1801, 10**1000, -(10**1000)]
)
def test_invalid_build_budget_refuses(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path, budget: float
) -> None:
    """Invalid execution budgets are rejected before creating any build directory."""
    _, record = completed_build
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_rustbca_reference(
            Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
            output,
            python=Path(sys.executable),
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
            timeout_seconds=budget,
        )
    assert not output.exists()


def test_changed_archive_refuses(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """A byte change to the real upstream archive fails before unpacking or compiling."""
    _, record = completed_build
    source = Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"])
    archive = tmp_path / "changed.tar.gz"
    archive.write_bytes(source.read_bytes() + b"changed")
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="archive digest"):
        build_rustbca_reference(
            archive,
            output,
            python=Path(sys.executable),
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
        )
    assert not output.exists()


def test_expired_build_retains_failure(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """A real runtime probe exceeding the total budget leaves an explicit failed receipt."""
    _, record = completed_build
    output = tmp_path / "build"
    with pytest.raises(subprocess.TimeoutExpired):
        build_rustbca_reference(
            Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
            output,
            python=Path(sys.executable),
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
            timeout_seconds=1e-12,
        )
    failure = json.loads((output / "result.json").read_bytes())
    assert failure["status"] == "failed" and failure["returncode"] == -1
    assert failure["error_type"] == "TimeoutExpired"
    assert not list(output.glob("*.whl"))


@pytest.mark.parametrize("interrupt_cleanup", [False, True], ids=["timeout", "cleanup-cancel"])
def test_actual_compiler_timeout_reaps_process_group(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path, interrupt_cleanup: bool
) -> None:
    """Preserve a real Cargo timeout through cancellation and verify process-group exit."""
    _, record = completed_build
    output = tmp_path / "build"
    interruptions: list[int] = []

    def interrupt(signum: int, frame: FrameType | None) -> None:
        """Cancel once when the actual compiler exits, excluding earlier tool probes."""
        if (output / "process.json").exists() and not interruptions:
            interruptions.append(signum)
            raise KeyboardInterrupt("application cancellation during compiler cleanup")

    previous = signal.getsignal(signal.SIGCHLD)
    started = time.monotonic()
    try:
        if interrupt_cleanup:
            signal.signal(signal.SIGCHLD, interrupt)
        with pytest.raises(subprocess.TimeoutExpired):
            build_rustbca_reference(
                Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
                output,
                python=Path(sys.executable),
                cargo=Path(record["command"][0]),
                rustc=Path(record["environment"]["RUSTC"]),
                timeout_seconds=5,
            )
    finally:
        signal.signal(signal.SIGCHLD, previous)
    failure = json.loads((output / "result.json").read_bytes())
    assert failure["error_type"] == "TimeoutExpired"
    assert time.monotonic() - started < 12
    if interrupt_cleanup:
        assert interruptions == [signal.SIGCHLD]
        errors = [failure.get("compiler_reap_error", {}), failure.get("compiler_signal_error", {})]
        assert any(error.get("type") == "KeyboardInterrupt" for error in errors)
    assert failure["status"] == "failed" and failure["compiler_reaped"]
    assert failure["compiler_returncode"] < 0
    assert "Compiling" in (output / "build.log").read_text()
    assert not Path(f"/proc/{failure['compiler_pid']}").exists()
    deadline = time.monotonic() + 5
    while True:
        try:
            os.killpg(failure["compiler_pgid"], 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            pytest.fail("timed-out compiler process group remains present")
        time.sleep(0.02)
    assert not list(output.glob("*.whl"))


def test_actual_dependency_fetch_retains_locked_graph(
    completed_build: tuple[Path, dict[str, Any]],
) -> None:
    """The reference build records a successful locked fetch before compilation."""
    receipt, record = completed_build
    assert record["fetch_dependencies"] is True
    assert record["dependency_fetch_command"][1:] == ["fetch", "--locked"]
    assert record["dependency_fetch_returncode"] == 0 and record["dependency_fetch_reaped"]
    assert record["lock_unchanged"]
    assert (
        hashlib.sha256((receipt.parent / "dependency-fetch.log").read_bytes()).hexdigest()
        == record["dependency_fetch_log_sha256"]
    )


def test_frozen_build_without_dependencies_retains_failure_and_reaps_cargo(
    completed_build: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual empty cache cannot cause the default build to fetch dependencies."""
    _, record = completed_build
    cache = tmp_path / "cargo-cache"
    cache.mkdir()
    monkeypatch.setenv("CARGO_HOME", str(cache))
    output = tmp_path / "build"
    with pytest.raises(subprocess.CalledProcessError):
        build_rustbca_reference(
            Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
            output,
            python=Path(record["environment"]["PYO3_PYTHON"]),
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
            timeout_seconds=30,
        )
    failure = json.loads((output / "result.json").read_bytes())
    assert failure["status"] == "failed" and failure["error_type"] == "CalledProcessError"
    assert failure["fetch_dependencies"] is False and failure["compiler_reaped"]
    assert failure["compiler_returncode"] > 0
    assert failure["environment"]["CARGO_HOME"] == str(cache)
    assert "--frozen" in failure["command"]
    log = (output / "build.log").read_text()
    assert "--frozen" in log or "offline" in log
    assert "dependency_fetch_pid" not in failure
    assert not list(output.glob("*.whl"))
    with pytest.raises(ProcessLookupError):
        os.killpg(failure["compiler_pgid"], 0)


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_fetch_mode_requires_boolean(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path, value: Any
) -> None:
    """Ambiguous network permission cannot silently enable dependency acquisition."""
    _, record = completed_build
    with pytest.raises(ValueError, match="fetch_dependencies"):
        build_rustbca_reference(
            Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
            tmp_path / "build",
            python=Path(sys.executable),
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
            fetch_dependencies=value,
        )
    assert not (tmp_path / "build").exists()


def test_actual_unsupported_wheel_refuses_before_build(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Install the pinned unsupported wheel offline and exercise the actual builder refusal."""
    _, record = completed_build
    artifact = Path(os.environ["RUSTBCA_UNSUPPORTED_WHEEL"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "177f9c9b0d45c47873b619f5b650346d632cdc35fb5e4d25058e09c9e581433d"
    )
    environment = tmp_path / "unsupported-writer"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-index",
            "--no-deps",
            str(artifact),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    if os.environ.get("COVERAGE_PROCESS_START"):
        site = subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
            timeout=10,
        ).strip()
        # Append instrumentation dependencies so the real installed old writer wins.
        hook = (
            "import os,sys; "
            f"sys.path.append({sysconfig.get_path('purelib')!r}) "
            "if os.environ.get('COVERAGE_PROCESS_START') else None; "
            "__import__('coverage').process_startup() "
            "if os.environ.get('COVERAGE_PROCESS_START') else None\n"
        )
        (Path(site) / "rustbca_coverage.pth").write_text(hook)
    identity = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import importlib.metadata as m,json; "
                "d=m.distribution('wheel'); print(json.dumps([d.version,str(d.locate_file(''))]))",
            ],
            text=True,
            timeout=10,
        )
    )
    assert identity[0] == "0.42.0"
    assert Path(identity[1]).resolve().is_relative_to(environment.resolve())
    output = tmp_path / "build"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "tools.build_rustbca_reference",
            os.environ["RUSTBCA_SOURCE_ARCHIVE"],
            str(output),
            "--python",
            str(python),
            "--cargo",
            record["command"][0],
            "--rustc",
            record["environment"]["RUSTC"],
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=15,
    )
    (tmp_path / "unsupported-wheel-refusal.log").write_text(result.stderr)
    assert result.returncode != 0
    assert "ValueError: RustBCA wheel writer requires wheel==0.47.0" in result.stderr
    assert not output.exists()


def test_actual_pypy_refuses_before_compilation(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Probe an actual pinned PyPy runtime and refuse it before starting Cargo."""
    _, record = completed_build
    python = Path(os.environ["RUSTBCA_UNSUPPORTED_PYTHON"])
    spec = json.loads(
        (
            Path(__file__).resolve().parents[2] / "requirements/rustbca.unsupported-python.json"
        ).read_text()
    )
    identity = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json,sys; "
                "print(json.dumps([sys.implementation.name,list(sys.version_info[:3])]))",
            ],
            text=True,
            timeout=15,
        )
    )
    assert identity == [spec["implementation"], spec["python_version"]]
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="requires a Linux CPython runtime"):
        build_rustbca_reference(
            Path(os.environ["RUSTBCA_SOURCE_ARCHIVE"]),
            output,
            python=python,
            cargo=Path(record["command"][0]),
            rustc=Path(record["environment"]["RUSTC"]),
            timeout_seconds=30,
        )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed" and result["returncode"] == -1
    assert result["error_type"] == "ValueError"
    assert not (output / "process.json").exists()
    assert not (output / "target").exists()
    assert not list(output.glob("*.whl"))


@pytest.mark.parametrize("boundary", ["deadline", "locked_fetch"])
def test_metadata_boundary_refuses_compiler_dispatch(
    completed_build: tuple[Path, dict[str, Any]], tmp_path: Path, boundary: str
) -> None:
    """Refuse compilation after real metadata backpressure or a changed lock at fetch."""
    _, record = completed_build
    output = tmp_path / "build"
    budget = 3 if boundary == "deadline" else 30
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.build_rustbca_reference",
            os.environ["RUSTBCA_SOURCE_ARCHIVE"],
            str(output),
            "--python",
            sys.executable,
            "--cargo",
            record["command"][0],
            "--rustc",
            record["environment"]["RUSTC"],
            "--timeout-seconds",
            str(budget),
            *(["--fetch-dependencies"] if boundary == "locked_fetch" else []),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    descriptor = None
    deadline = time.monotonic() + 30
    inputs = output / "inputs.json"
    captured = bytearray()
    try:
        while not output.exists():
            assert child.poll() is None, "builder exited before creating output"
            assert time.monotonic() < deadline, "output creation deadline expired"
            time.sleep(0.001)
        os.mkfifo(inputs)
        descriptor = os.open(inputs, os.O_RDONLY | os.O_NONBLOCK)
        fcntl.fcntl(descriptor, fcntl.F_SETPIPE_SZ, 4096)
        capacity = fcntl.fcntl(descriptor, fcntl.F_GETPIPE_SZ)
        while not captured:
            assert child.poll() is None, "builder exited before writing metadata"
            assert time.monotonic() < deadline, "metadata write deadline expired"
            try:
                captured.extend(os.read(descriptor, 1))
            except BlockingIOError:
                pass
            if not captured:
                time.sleep(0.001)
        # The real metadata payload exceeds the pipe capacity. Holding the read
        # end applies backpressure after the runtime probe and before Cargo.
        if boundary == "deadline":
            release_at = time.monotonic() + 3.1
            while time.monotonic() < release_at:
                assert child.poll() is None, "builder exited while its metadata write was held"
                time.sleep(0.005)
        else:
            lock = output / "source/Cargo.lock"
            assert hashlib.sha256(lock.read_bytes()).hexdigest() == record["cargo_lock_sha256"]
            with lock.open("ab") as changed:
                changed.write(b"\n")
        while True:
            assert time.monotonic() < deadline, "metadata drain deadline expired"
            try:
                chunk = os.read(descriptor, 65536)
            except BlockingIOError:
                time.sleep(0.001)
                continue
            if not chunk:
                break
            captured.extend(chunk)
        os.close(descriptor)
        descriptor = None
        _, stderr = child.communicate(timeout=max(0.001, deadline - time.monotonic()))
        assert child.returncode != 0
        error = "TimeoutExpired" if boundary == "deadline" else "ValueError"
        assert error in stderr
        assert len(captured) > capacity
        metadata = json.loads(captured)
        assert metadata["command"][0] == record["command"][0]
        assert metadata["runtime"]["implementation"] == "cpython"
        failure = json.loads((output / "result.json").read_text())
        assert failure["status"] == "failed" and failure["error_type"] == error
        if boundary == "locked_fetch":
            assert "dependency fetch changed the locked graph" in stderr
            assert failure["dependency_fetch_returncode"] == 0
            assert failure["dependency_fetch_reaped"]
            assert failure["dependency_fetch_command"][1:] == ["fetch", "--locked"]
        assert "compiler_pid" not in failure
        assert not (output / "process.json").exists()
        assert not (output / "build.log").exists()
        assert not (output / "target").exists()
        inputs.unlink()
        inputs.write_bytes(captured)
        (tmp_path / "metadata-backpressure.json").write_text(
            json.dumps(
                {
                    "parent_pid": child.pid,
                    "pipe_capacity": capacity,
                    "metadata_bytes": len(captured),
                    "hold_minimum_seconds": 3.1 if boundary == "deadline" else 0,
                    "boundary": boundary,
                    "build_budget_seconds": budget,
                    "compiler_started": False,
                }
            )
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate(timeout=10)
