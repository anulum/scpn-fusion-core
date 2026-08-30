# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Client
"""Bounded process-isolated client that never imports TORAX or JAX."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import ToraxFailureCode, ToraxRunOutcome, ToraxRunRequest
from .serialization import canonical_sha256, file_sha256, load_json_object, write_json_atomic
from .worker import failure_outcome

MAX_CAPTURE_BYTES = 64 * 1024


@dataclass(frozen=True)
class ToraxRuntimeClient:
    """Launch a supplied TORAX Python interpreter through the public CLI."""

    python_executable: Path
    working_directory: Path | None = None

    def run(
        self,
        request: ToraxRunRequest,
        *,
        request_path: Path,
        result_path: Path,
        sidecar_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> ToraxRunOutcome:
        """Execute one request with a whole-process-group wall-time bound."""
        source_root = Path(__file__).resolve().parents[3]
        base_directory = (
            source_root.parent if self.working_directory is None else self.working_directory
        ).resolve()
        effective_python = _absolute_from(base_directory, self.python_executable)
        if not effective_python.is_file():
            return failure_outcome(
                request,
                ToraxFailureCode.BACKEND_UNAVAILABLE,
                f"TORAX Python executable is unavailable: {self.python_executable}",
                sim_error="BACKEND_UNAVAILABLE",
            )
        effective_request_path = _resolve_from(base_directory, request_path)
        effective_result_path = _resolve_from(base_directory, result_path)
        effective_sidecar_path = _resolve_from(base_directory, sidecar_path)
        write_json_atomic(effective_request_path, request.to_dict())
        effective_result_path.parent.mkdir(parents=True, exist_ok=True)
        effective_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(effective_python),
            "-m",
            "scpn_fusion.integrations.torax",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--output-sidecar",
            str(sidecar_path),
        ]
        child_environment = os.environ.copy()
        child_environment["PYTHONPATH"] = str(source_root)
        if environment is not None:
            child_environment.update(environment)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=child_environment,
            cwd=base_directory,
        )
        try:
            stdout, stderr = process.communicate(timeout=request.clock.timeout_s)
        except subprocess.TimeoutExpired:
            _signal_process_group(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                _signal_process_group(process.pid, signal.SIGKILL)
                process.communicate()
            outcome = failure_outcome(
                request,
                ToraxFailureCode.TIMEOUT,
                f"TORAX process exceeded {request.clock.timeout_s:.6g} seconds",
                sim_error="TIMEOUT",
            )
            write_json_atomic(effective_result_path, outcome.to_dict())
            return outcome
        if effective_result_path.is_file():
            try:
                return self.load_verified_outcome(
                    result_path=result_path,
                    expected_sidecar_path=sidecar_path,
                )
            except ValueError as error:
                outcome = failure_outcome(
                    request,
                    ToraxFailureCode.PROVENANCE_FAILURE,
                    f"TORAX result custody is invalid: {error}",
                    sim_error="PROVENANCE_FAILURE",
                )
                write_json_atomic(effective_result_path, outcome.to_dict())
                return outcome
        diagnostic = _bounded_diagnostic(stdout, stderr)
        outcome = failure_outcome(
            request,
            ToraxFailureCode.PROCESS_FAILURE,
            f"TORAX worker exited {process.returncode} without a result record; {diagnostic}",
            sim_error="PROCESS_FAILURE",
        )
        write_json_atomic(effective_result_path, outcome.to_dict())
        return outcome

    def load_verified_outcome(
        self,
        *,
        result_path: Path,
        expected_sidecar_path: Path,
    ) -> ToraxRunOutcome:
        """Load an outcome and verify every declared sidecar custody digest."""
        source_root = Path(__file__).resolve().parents[3]
        base_directory = (
            source_root.parent if self.working_directory is None else self.working_directory
        ).resolve()
        effective_result_path = _resolve_from(base_directory, result_path)
        effective_sidecar_path = _resolve_from(base_directory, expected_sidecar_path)
        outcome = ToraxRunOutcome.from_dict(load_json_object(effective_result_path))
        if outcome.artifact is None:
            if outcome.success:
                raise ValueError("successful outcome is missing its artifact")
            return outcome
        artifact = outcome.artifact
        manifest_path = effective_sidecar_path.with_suffix(
            effective_sidecar_path.suffix + ".manifest.json"
        )
        artifact_sidecar = _resolve_from(base_directory, Path(artifact.sidecar_path))
        artifact_manifest = _resolve_from(base_directory, Path(artifact.manifest_path))
        if artifact_sidecar != effective_sidecar_path:
            raise ValueError("outcome sidecar path differs from the caller-selected path")
        if artifact_manifest != manifest_path:
            raise ValueError("outcome manifest path differs from the derived path")
        if file_sha256(effective_sidecar_path) != artifact.sidecar_sha256:
            raise ValueError("sidecar SHA-256 mismatch")
        if effective_sidecar_path.stat().st_size != artifact.sidecar_bytes:
            raise ValueError("sidecar byte count mismatch")
        if file_sha256(manifest_path) != artifact.manifest_sha256:
            raise ValueError("manifest SHA-256 mismatch")
        manifest = load_json_object(manifest_path)
        if manifest.get("schema") != "scpn-fusion-core.torax-datatree-manifest.v1":
            raise ValueError("manifest schema mismatch")
        sidecar_raw = manifest.get("sidecar")
        if not isinstance(sidecar_raw, Mapping):
            raise ValueError("manifest sidecar record must be an object")
        sidecar = sidecar_raw
        if sidecar.get("sha256") != artifact.sidecar_sha256:
            raise ValueError("manifest sidecar SHA-256 disagrees with outcome")
        if sidecar.get("bytes") != artifact.sidecar_bytes:
            raise ValueError("manifest sidecar size disagrees with outcome")
        groups = manifest.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("manifest has no DataTree groups")
        variable_count = manifest.get("variable_count")
        if (
            isinstance(variable_count, bool)
            or not isinstance(variable_count, int)
            or variable_count <= 0
        ):
            raise ValueError("manifest variable_count must be a positive integer")
        if manifest.get("content_sha256") != canonical_sha256(groups):
            raise ValueError("manifest content SHA-256 mismatch")
        inventory = {
            "schema": manifest["schema"],
            "group_count": manifest.get("group_count"),
            "variable_count": variable_count,
            "groups": groups,
        }
        if manifest.get("inventory_sha256") != canonical_sha256(inventory):
            raise ValueError("manifest inventory SHA-256 mismatch")
        return outcome


def _bounded_diagnostic(stdout: bytes, stderr: bytes) -> str:
    combined = (stdout[-MAX_CAPTURE_BYTES:] + b"\n" + stderr[-MAX_CAPTURE_BYTES:]).decode(
        "utf-8", errors="replace"
    )
    return " ".join(combined.split())[-MAX_CAPTURE_BYTES:] or "no captured output"


def _resolve_from(base_directory: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


def _absolute_from(base_directory: Path, path: Path) -> Path:
    return path.absolute() if path.is_absolute() else (base_directory / path).absolute()


def _signal_process_group(process_id: int, selected_signal: signal.Signals) -> None:
    try:
        os.killpg(process_id, selected_signal)
    except ProcessLookupError:
        pass


__all__ = ["MAX_CAPTURE_BYTES", "ToraxRuntimeClient"]
