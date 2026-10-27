# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Worker
"""Lazy TORAX execution inside the dedicated backend environment."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, cast

from .contracts import (
    TORAX_VERSION,
    ToraxFailureCode,
    ToraxProvenance,
    ToraxRunOutcome,
    ToraxRunRequest,
)
from .projection import (
    build_projection,
    publish_manifest,
    reached_time_ns,
    write_complete_sidecar,
)
from .serialization import canonical_sha256, file_sha256


def execute_request(
    request: ToraxRunRequest,
    *,
    sidecar_path: Path,
    manifest_path: Path,
) -> ToraxRunOutcome:
    """Execute real TORAX and retain complete output for success or solver failure."""
    started_at = _utc_now()
    provenance_base = _provenance_base(request, started_at)
    try:
        import jax
        import torax  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as error:
        return failure_outcome(
            request,
            ToraxFailureCode.BACKEND_UNAVAILABLE,
            f"TORAX backend import failed: {type(error).__name__}: {error}",
            sim_error="BACKEND_UNAVAILABLE",
            provenance_base=provenance_base,
        )
    installed_version = str(getattr(torax, "__version__", "unknown"))
    runtime_backend = str(jax.default_backend())
    if installed_version != request.expected_torax_version:
        return failure_outcome(
            request,
            ToraxFailureCode.BACKEND_VERSION_MISMATCH,
            f"expected TORAX {request.expected_torax_version}, found {installed_version}",
            sim_error="BACKEND_VERSION_MISMATCH",
            provenance_base={
                **provenance_base,
                "torax_version": installed_version,
                "runtime_backend": runtime_backend,
            },
        )
    try:
        torax_config = torax.ToraxConfig.from_dict(_thaw(request.torax_config))
    except Exception as error:
        return failure_outcome(
            request,
            ToraxFailureCode.CONFIGURATION_REJECTED,
            f"TORAX rejected the configuration: {type(error).__name__}: {error}",
            sim_error="CONFIGURATION_REJECTED",
            provenance_base={
                **provenance_base,
                "torax_version": installed_version,
                "runtime_backend": runtime_backend,
            },
        )
    try:
        data_tree, history = torax.run_simulation(
            torax_config,
            progress_bar=False,
            max_steps=request.clock.max_steps,
        )
    except Exception as error:
        return failure_outcome(
            request,
            ToraxFailureCode.PROCESS_FAILURE,
            f"TORAX execution failed: {type(error).__name__}: {error}",
            sim_error="PROCESS_FAILURE",
            provenance_base={
                **provenance_base,
                "torax_version": installed_version,
                "runtime_backend": runtime_backend,
            },
        )
    sim_error = str(getattr(history.sim_error, "name", history.sim_error))
    try:
        write_complete_sidecar(data_tree, sidecar_path)
        artifact = publish_manifest(data_tree, sidecar_path, manifest_path)
        projection = build_projection(data_tree, request)
        reached_ns = reached_time_ns(data_tree)
    except Exception as error:
        return failure_outcome(
            request,
            ToraxFailureCode.OUTPUT_SCHEMA_MISMATCH,
            f"TORAX output validation failed: {type(error).__name__}: {error}",
            sim_error=sim_error,
            provenance_base={
                **provenance_base,
                "torax_version": installed_version,
                "runtime_backend": runtime_backend,
            },
        )
    provenance = _make_provenance(
        {
            **provenance_base,
            "torax_version": installed_version,
            "runtime_backend": runtime_backend,
        },
        _utc_now(),
    )
    try:
        failure_code = ToraxFailureCode.from_sim_error(sim_error)
    except ValueError:
        failure_code = ToraxFailureCode.OUTPUT_SCHEMA_MISMATCH
    failure_message: str | None = None
    if sim_error == "NO_ERROR" and reached_ns != request.clock.final_ns:
        failure_code = ToraxFailureCode.DID_NOT_REACH_T_FINAL
        failure_message = (
            f"TORAX reported NO_ERROR but reached {reached_ns} ns, "
            f"not requested {request.clock.final_ns} ns"
        )
    elif sim_error != "NO_ERROR":
        if failure_code is ToraxFailureCode.OUTPUT_SCHEMA_MISMATCH:
            failure_message = f"unrecognized TORAX SimError: {sim_error}"
        else:
            failure_message = f"TORAX terminated with SimError.{sim_error}"
    return ToraxRunOutcome(
        request_id=request.request_id,
        event_id=request.event_id,
        complete=failure_code is None,
        reached_time_ns=reached_ns,
        sim_error=sim_error,
        provenance=provenance,
        projection=projection,
        artifact=artifact,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def failure_outcome(
    request: ToraxRunRequest,
    code: ToraxFailureCode,
    message: str,
    *,
    sim_error: str,
    provenance_base: Mapping[str, str] | None = None,
) -> ToraxRunOutcome:
    """Construct a complete typed failure without importing TORAX."""
    started = _utc_now()
    base = dict(_provenance_base(request, started) if provenance_base is None else provenance_base)
    return ToraxRunOutcome(
        request_id=request.request_id,
        event_id=request.event_id,
        complete=False,
        reached_time_ns=request.clock.initial_ns,
        sim_error=sim_error,
        provenance=_make_provenance(base, _utc_now()),
        projection=None,
        artifact=None,
        failure_code=code,
        failure_message=message,
    )


def invalid_request_outcome(raw: Mapping[str, object], message: str) -> ToraxRunOutcome:
    """Create the CLI failure record when a request cannot be parsed."""
    request_id = raw.get("request_id")
    event_id = raw.get("event_id")
    now = _utc_now()
    zero = "0" * 64
    return ToraxRunOutcome(
        request_id=request_id if isinstance(request_id, str) and request_id else "unavailable",
        event_id=event_id if isinstance(event_id, str) and event_id else "unavailable",
        complete=False,
        reached_time_ns=0,
        sim_error="INVALID_REQUEST",
        provenance=ToraxProvenance(
            torax_version=TORAX_VERSION,
            torax_license="Apache-2.0",
            source_repo_commit="unavailable",
            python_version=platform.python_version(),
            platform=platform.platform(),
            runtime_backend="not_started",
            precision="float64",
            request_sha256=canonical_sha256(dict(raw)),
            config_sha256=zero,
            deck_sha256=zero,
            runner_sha256=file_sha256(Path(__file__)),
            started_at_utc=now,
            finished_at_utc=now,
        ),
        projection=None,
        artifact=None,
        failure_code=ToraxFailureCode.INVALID_REQUEST,
        failure_message=message,
    )


def _provenance_base(request: ToraxRunRequest, started_at: str) -> dict[str, str]:
    custody = request.custody
    return {
        "torax_version": request.expected_torax_version,
        "torax_license": "Apache-2.0",
        "source_repo_commit": cast(str, custody["source_repo_commit"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime_backend": "not_started",
        "precision": "float64",
        "request_sha256": canonical_sha256(request.to_dict()),
        "config_sha256": canonical_sha256(request.torax_config),
        "deck_sha256": cast(str, custody["deck_sha256"]),
        "runner_sha256": file_sha256(Path(__file__)),
        "started_at_utc": started_at,
    }


def _make_provenance(base: Mapping[str, str], finished_at: str) -> ToraxProvenance:
    return ToraxProvenance(
        torax_version=base["torax_version"],
        torax_license=base["torax_license"],
        source_repo_commit=base["source_repo_commit"],
        python_version=base["python_version"],
        platform=base["platform"],
        runtime_backend=base["runtime_backend"],
        precision=base["precision"],
        request_sha256=base["request_sha256"],
        config_sha256=base["config_sha256"],
        deck_sha256=base["deck_sha256"],
        runner_sha256=base["runner_sha256"],
        started_at_utc=base["started_at_utc"],
        finished_at_utc=finished_at,
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


__all__ = ["execute_request", "failure_outcome", "invalid_request_outcome"]
