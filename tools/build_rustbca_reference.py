# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — pinned RustBCA native reference build
"""Build an optional RustBCA wheel and retain source-to-binary observations."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, BinaryIO, cast

from validation.rustbca_build_receipt import (
    CARGO_LOCK_SHA256,
    SOURCE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    verify_rustbca_build_receipt,
)

ARCHIVE_SHA256 = "3aded40a7a3282ac41e4a1c90198511f26bf8d418ec2994914b6bf930424f4b7"


def _sha(path: Path) -> str:
    """Return the SHA-256 digest of the exact retained file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel(binary: Path, source: Path, output: Path, runtime: dict[str, str]) -> Path:
    """Package the local platform extension with upstream licence and source identity."""
    tag = f"{runtime['interpreter']}-{runtime['abi']}-{runtime['platform']}"
    destination = output / f"rustbca-3.0.0-{tag}.whl"
    info = "rustbca-3.0.0.dist-info/"
    wheel_file = importlib.import_module("wheel.wheelfile").WheelFile
    with wheel_file(str(destination), "w") as archive:
        archive.writestr("libRustBCA" + runtime["suffix"], binary.read_bytes())
        archive.writestr(
            info + "WHEEL",
            "Wheel-Version: 1.0\nGenerator: scpn-rustbca-build\nRoot-Is-Purelib: false\nTag: "
            + tag
            + "\n",
        )
        archive.writestr(
            info + "METADATA",
            "Metadata-Version: 2.1\nName: RustBCA\nVersion: 3.0.0\nLicense: GPLv3 (see included upstream LICENSE)\nHome-page: https://github.com/lcpp-org/RustBCA\n\nLocally compiled reference from upstream commit "
            + SOURCE_COMMIT
            + ".\n",
        )
        archive.writestr(info + "LICENSE", (source / "LICENSE").read_bytes())
    return destination


def _run_cargo_phase(
    command: list[str],
    source: Path,
    output: Path,
    env: dict[str, str],
    deadline: float,
    record: dict[str, Any],
    phase: str,
    log_name: str,
) -> None:
    """Run one Cargo phase within the shared deadline and account for its process group."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, 0)
    with (output / log_name).open("wb") as log:
        child = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            record[phase + "_pid"] = child.pid
            record[phase + "_pgid"] = child.pid
            name = "process.json" if phase == "compiler" else phase + "-process.json"
            (output / name).write_text(
                json.dumps({"pid": child.pid, "pgid": child.pid, "deadline_monotonic": deadline})
                + "\n"
            )
            code = child.wait(timeout=max(0.001, deadline - time.monotonic()))
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            cleanup_deadline = time.monotonic() + 5.0
            record[phase + "_cleanup_deadline_monotonic"] = cleanup_deadline
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except BaseException as cleanup_error:
                record[phase + "_signal_error"] = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            try:
                child.wait(timeout=max(0.001, cleanup_deadline - time.monotonic()))
            except BaseException as cleanup_error:
                record[phase + "_reap_error"] = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            raise
        finally:
            record[phase + "_returncode"] = child.returncode
            record[phase + "_reaped"] = child.returncode is not None
            record[phase + "_log_sha256"] = _sha(output / log_name)


def build_rustbca_reference(
    source_archive: Path,
    output: Path,
    *,
    python: Path,
    cargo: Path,
    rustc: Path,
    timeout_seconds: float = 600.0,
    fetch_dependencies: bool = False,
) -> dict[str, Any]:
    """Build a fresh frozen native library, local-platform wheel and custody receipt.

    Parameters
    ----------
    source_archive : Path
        Exact pinned upstream tar.gz, already acquired by the caller.
    output : Path
        New retained build directory. Existing directories are never overwritten.
    python : Path
        CPython interpreter whose ABI the optional extension targets.
    cargo, rustc : Path
        Explicit compiler executable paths, hashed before and after compilation.
        Required dependencies must already be available to Cargo's frozen resolver.
    timeout_seconds : float
        Finite total subprocess budget in seconds, greater than zero and at most 1800.
        Failed phases additionally allow at most five seconds to reap their leader;
        failure receipts retain the PID and any signalling or reaping error.

    fetch_dependencies : bool
        Explicitly permit Cargo fetch --locked before frozen compilation. Both phases
        share the deadline; downloaded dependencies remain pinned by the lockfile.

    Returns
    -------
    dict[str, Any]
        Local observation binding sources, compiler identities, binary and wheel.
        This does not certify hermetic builds, independent attestation or physics.

    Raises
    ------
    ValueError
        For invalid limits, source identity, unsupported runtime or changed inputs.
    FileExistsError
        If output already exists.
    subprocess.SubprocessError
        If a compiler/probe fails or exceeds the remaining deadline.
    OSError
        If required input files or executables are unavailable.

    Notes
    -----
    The builder uses two Cargo jobs and a fresh target directory. On failure it
    retains result.json with a failure status, signals its compiler process group
    and attempts bounded leader reaping. The receipt distinguishes an unreaped
    leader from successful reaping; reaping alone does not prove group absence.
    wheel 0.47.0 is required to write checked RECORD entries. The generated
    wheel uses the local platform tag, never an unaudited manylinux claim.
    """
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 1800:
        raise ValueError("timeout_seconds must be finite and in (0, 1800]")
    if type(fetch_dependencies) is not bool:
        raise ValueError("fetch_dependencies must be a boolean")
    if importlib.metadata.version("wheel") != "0.47.0":
        raise ValueError("RustBCA wheel writer requires wheel==0.47.0")
    if _sha(source_archive) != ARCHIVE_SHA256:
        raise ValueError("RustBCA archive digest mismatch")
    data = Path(__file__).resolve().parents[1] / "validation/reference_data"
    files = json.loads((data / "rustbca_source_files.json").read_bytes())
    if (
        hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("RustBCA source inventory mismatch")
    lock = data / "rustbca.Cargo.lock"
    if _sha(lock) != CARGO_LOCK_SHA256:
        raise ValueError("RustBCA Cargo lock mismatch")
    output = output.absolute()
    python, cargo, rustc = python.absolute(), cargo.absolute(), rustc.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = time.monotonic() + timeout_seconds
    record: dict[str, Any] = {
        "schema": "scpn-fusion.rustbca-native-build-observation.v1",
        "started_unix": started,
        "returncode": -1,
    }
    try:
        source = output / "source"
        source.mkdir()
        seen = set()
        with tarfile.open(source_archive, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                prefix, _, name = member.name.partition("/")
                if (
                    prefix != "RustBCA-" + SOURCE_COMMIT
                    or name not in files
                    or name in seen
                    or not member.isfile()
                ):
                    raise ValueError("Unexpected RustBCA source archive member")
                # isfile() above establishes extractfile()'s regular-file return type.
                with cast(BinaryIO, archive.extractfile(member)) as stream:
                    raw = stream.read()
                if hashlib.sha256(raw).hexdigest() != files[name]:
                    raise ValueError("RustBCA archive member digest mismatch")
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                seen.add(name)
        if seen != set(files):
            raise ValueError("Incomplete RustBCA source archive")
        (source / "Cargo.lock").write_bytes(lock.read_bytes())
        recipe = output / "build_source.py"
        recipe.write_bytes(Path(__file__).read_bytes())
        env = {
            key: os.environ[key]
            for key in ("HOME", "PATH", "CARGO_HOME", "RUSTUP_HOME")
            if key in os.environ
        }
        env.update(
            RUSTC=str(rustc),
            PYO3_PYTHON=str(python),
            CARGO_TARGET_DIR=str(output / "target"),
            CARGO_INCREMENTAL="0",
            LC_ALL="C",
        )
        query = "import sys,sysconfig,json; print(json.dumps({'implementation':sys.implementation.name,'interpreter':'cp'+str(sys.version_info.major)+str(sys.version_info.minor),'abi':'cp'+sysconfig.get_config_var('SOABI').split('-')[1],'platform':sysconfig.get_platform().replace('-','_').replace('.','_'),'suffix':sysconfig.get_config_var('EXT_SUFFIX')}))"
        runtime = json.loads(
            subprocess.check_output(
                [str(python), "-c", query],
                env=env,
                text=True,
                timeout=max(0.001, deadline - time.monotonic()),
            )
        )
        if runtime["implementation"] != "cpython" or not runtime["platform"].startswith("linux_"):
            raise ValueError("RustBCA build currently requires a Linux CPython runtime")
        command = [
            str(cargo),
            "build",
            "--frozen",
            "--release",
            "--lib",
            "--features",
            "python,parry3d,pythonize,pyo3/extension-module",
            "--jobs",
            "2",
        ]
        record.update(
            upstream={
                "repository": "https://github.com/lcpp-org/RustBCA",
                "commit": SOURCE_COMMIT,
                "archive_sha256": ARCHIVE_SHA256,
                "license_sha256": files["LICENSE"],
            },
            source_files=files,
            cargo_lock_sha256=CARGO_LOCK_SHA256,
            command=command,
            environment=env,
            cargo_sha256=_sha(cargo),
            rustc_sha256=_sha(rustc),
            python_sha256=_sha(python.resolve()),
            recipe_sha256=_sha(recipe),
            wheel_writer_version="0.47.0",
            runtime=runtime,
            fetch_dependencies=fetch_dependencies,
            independent_attestation_verified=False,
            reproducible_binary_verified=False,
        )
        (output / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")
        if fetch_dependencies:
            fetch_command = [str(cargo), "fetch", "--locked"]
            record["dependency_fetch_command"] = fetch_command
            _run_cargo_phase(
                fetch_command,
                source,
                output,
                env,
                deadline,
                record,
                "dependency_fetch",
                "dependency-fetch.log",
            )
            if _sha(source / "Cargo.lock") != CARGO_LOCK_SHA256:
                raise ValueError("RustBCA dependency fetch changed the locked graph")
        _run_cargo_phase(command, source, output, env, deadline, record, "compiler", "build.log")
        binary = output / "target/release/liblibRustBCA.so"
        record.update(
            returncode=0,
            binary=str(binary),
            binary_sha256=_sha(binary),
            log_sha256=_sha(output / "build.log"),
            source_unchanged=all(_sha(source / name) == digest for name, digest in files.items()),
            lock_unchanged=_sha(source / "Cargo.lock") == CARGO_LOCK_SHA256,
            tools_unchanged=_sha(cargo) == record["cargo_sha256"]
            and _sha(rustc) == record["rustc_sha256"]
            and _sha(python.resolve()) == record["python_sha256"],
        )
        result = output / "result.json"
        result.write_text(json.dumps(record, indent=2) + "\n")
        verify_rustbca_build_receipt(
            result,
            expected_receipt_sha256=_sha(result),
            expected_binary_sha256=record["binary_sha256"],
        )
        wheel = _wheel(binary, source, output, runtime)
        record.update(
            wheel=str(wheel),
            wheel_sha256=_sha(wheel),
            wheel_built=True,
            finished_unix=time.time(),
            status="complete_local_observation",
        )
        result.write_text(json.dumps(record, indent=2) + "\n")
        return record
    except BaseException as exc:
        record.update(
            returncode=-1,
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            finished_unix=time.time(),
        )
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        raise


def main() -> None:
    """Build the pinned optional reference from caller-selected local tools and archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--fetch-dependencies", action="store_true")
    args = parser.parse_args()
    build_rustbca_reference(
        args.archive,
        args.output,
        python=args.python,
        cargo=args.cargo,
        rustc=args.rustc,
        timeout_seconds=args.timeout_seconds,
        fetch_dependencies=args.fetch_dependencies,
    )


if __name__ == "__main__":
    main()
