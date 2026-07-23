# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed report contract for the IDA geometry-feedback trace."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, cast

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_payload_sha256: Callable[[dict[str, Any]], str] = _same_case._payload_sha256
_walk_finite: Callable[[object], None] = _same_case._walk_finite

SCHEMA_VERSION = "scpn-fusion.ida-geometry-feedback-trace.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-GEOMETRY-FEEDBACK-TRACE"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
COLD_CHECKPOINT_INDICES = (0, 14, 29, 49, 79, 99, 100, 109, 119, 129, 149, 179)
WARM_CHECKPOINT_INDICES = (0, 1, 4, 9, 19)
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
INPUT_CONTRACT = {
    "anderson_depth": 8,
    "anderson_solver": "lstsq",
    "cold_checkpoint_indices": list(COLD_CHECKPOINT_INDICES),
    "cold_ip_ramp": 30,
    "cold_iteration_cap": 180,
    "cutoff_width": 0.03,
    "evaluation_case_id": EVALUATION_CASE_ID,
    "grid_shape": [129, 129],
    "inner_solver": "bicgstab",
    "mixing": 0.5,
    "mu0_si": 1.2566370614359173e-6,
    "tol": 1.0e-9,
    "use_mg_preconditioner": True,
    "warm_checkpoint_indices": list(WARM_CHECKPOINT_INDICES),
    "warm_ip_ramp": 1,
    "warm_iteration_cap": 20,
}
SOURCE_PATHS = {
    "checkpoint_trace": "src/scpn_fusion/core/jax_predictive_checkpoint_trace.py",
    "compiled_forward": "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
    "contract": "validation/ida_geometry_feedback_trace_contract.py",
    "metrics": "validation/ida_geometry_feedback_trace_metrics.py",
    "renderer": "validation/ida_geometry_feedback_trace_render.py",
    "same_case_benchmark": "validation/benchmark_ida_same_case.py",
    "solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "source_ablation": "validation/diagnose_ida_fixed_reference_source.py",
    "trace": "validation/trace_ida_geometry_feedback.py",
}
SAME_CASE_REPORT_PATH = "validation/reports/ida_same_case_evidence.json"
SOURCE_ABLATION_REPORT_PATH = "validation/reports/ida_fixed_reference_source_ablation.json"
FREEGS_PUBLIC_EXAMPLE_PATH = "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py"
_ENVIRONMENT_FIELDS = {
    "affinity_cpu_count",
    "backend",
    "devices",
    "freegs_version",
    "host_load_1m_5m_15m",
    "isolated_host",
    "jax_version",
    "jaxlib_version",
    "machine",
    "platform",
    "python_version",
    "x64_enabled",
}
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "blockers",
    "claim_boundary",
    "environment",
    "generated_at",
    "input_contract",
    "payload_sha256",
    "routing",
    "runs",
    "schema_version",
    "source_artifacts",
    "source_reports",
    "status",
    "terminal_parity",
}
_RUN_FIELDS = {
    "checkpoint_indices_requested",
    "checkpoints",
    "iteration_cap",
    "iteration_count",
    "terminal_psi_sha256",
    "terminated_early",
}
_CHECKPOINT_FIELDS = {
    "converged",
    "fixed_point",
    "geometry",
    "ip_fraction",
    "ip_now_a",
    "iteration_index",
    "phase",
    "physical_residual",
    "production_current",
    "psi_after_sha256",
    "psi_before_sha256",
    "reference_boundary_counterfactual",
    "separatrix_refinement",
    "terminal",
}
_CURRENT_FIELDS = {
    "centroid_delta_m",
    "cosine_similarity",
    "current_density_sha256",
    "total_variation_distance",
}
_SHA256_LENGTH = 64
_GIT_OID_LENGTH = 40
_BASE_BLOCKERS = {
    "facility_validation_not_bound",
    "pcs_and_safety_programmes_not_bound",
    "same_case_accuracy_threshold_failed",
    "statistically_held_out_case_missing",
}


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _GIT_OID_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a full lowercase Git object ID")
    return value


def _require_number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def phase_for_iteration(run_name: str, iteration_index: int) -> str:
    """Return the frozen continuation phase for one iteration."""
    if run_name == "warm":
        return "warm_polish"
    if run_name != "cold":
        raise ValueError(f"unknown run name {run_name!r}")
    if iteration_index < 30:
        return "ip_ramp"
    if iteration_index < 100:
        return "pre_separatrix"
    if iteration_index < 120:
        return "separatrix_homotopy"
    return "post_homotopy"


def _routing(
    runs: dict[str, dict[str, Any]],
    terminal_parity: dict[str, Any],
) -> dict[str, Any]:
    sequence: list[tuple[str, dict[str, Any]]] = []
    for run_name in ("cold", "warm"):
        sequence.extend((run_name, row) for row in runs[run_name]["checkpoints"])
    warm_terminal = next(row for row in runs["warm"]["checkpoints"] if row["terminal"])
    terminal_tv = float(warm_terminal["production_current"]["total_variation_distance"])
    terminal_counterfactual_tv = float(
        warm_terminal["reference_boundary_counterfactual"]["total_variation_distance"]
    )
    pairs = list(zip(sequence[:-1], sequence[1:], strict=True))
    largest_from, largest_to = max(
        pairs,
        key=lambda pair: (
            float(pair[1][1]["production_current"]["total_variation_distance"])
            - float(pair[0][1]["production_current"]["total_variation_distance"])
        ),
    )
    largest_delta = (
        float(largest_to[1]["production_current"]["total_variation_distance"])
        - float(largest_from[1]["production_current"]["total_variation_distance"])
    )
    half_threshold = 0.5 * terminal_tv
    half_run, half_row = next(
        (run_name, row)
        for run_name, row in sequence
        if float(row["production_current"]["total_variation_distance"]) >= half_threshold
    )
    parity_ok = bool(terminal_parity["matches_same_case_candidate"])
    target = (
        f"{largest_to[1]['phase']}_geometry_source_feedback"
        if parity_ok
        else "compiled_trace_parity_failure"
    )
    return {
        "first_half_terminal_tv": {
            "iteration_index": half_row["iteration_index"],
            "phase": half_row["phase"],
            "run": half_run,
            "threshold": half_threshold,
            "tv": half_row["production_current"]["total_variation_distance"],
        },
        "largest_sparse_tv_increase": {
            "delta": largest_delta,
            "from_iteration_index": largest_from[1]["iteration_index"],
            "from_run": largest_from[0],
            "phase": largest_to[1]["phase"],
            "to_iteration_index": largest_to[1]["iteration_index"],
            "to_run": largest_to[0],
        },
        "next_ratcheting_target": target,
        "terminal_boundary_counterfactual_tv_reduction": (
            terminal_tv - terminal_counterfactual_tv
        ),
        "terminal_production_tv": terminal_tv,
        "terminal_reference_boundary_tv": terminal_counterfactual_tv,
        "trace_matches_same_case_candidate": parity_ok,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, str]],
    source_reports: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    terminal_parity: dict[str, Any],
    source_commit: str,
    source_worktree_clean: bool,
) -> dict[str, Any]:
    """Build and validate one self-digested trace report."""
    if not generated_at.strip():
        raise ValueError("generated_at must not be empty")
    blockers = [
        "facility_validation_not_bound",
        "pcs_and_safety_programmes_not_bound",
        "same_case_accuracy_threshold_failed",
        "statistically_held_out_case_missing",
    ]
    if not source_worktree_clean:
        blockers.append("source_worktree_not_clean")
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "blockers": blockers,
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "environment": environment,
        "generated_at": generated_at,
        "input_contract": dict(INPUT_CONTRACT),
        "payload_sha256": "",
        "routing": _routing(runs, terminal_parity),
        "runs": runs,
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": {
            **source_artifacts,
            "repository": {"git_commit": source_commit, "path": "."},
        },
        "source_reports": source_reports,
        "status": "diagnostic_complete_claims_blocked",
        "terminal_parity": terminal_parity,
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def _validate_current(row: object, *, field: str) -> None:
    if not isinstance(row, dict) or set(row) != _CURRENT_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    _require_sha256(row["current_density_sha256"], field=f"{field}.current_density_sha256")
    centroid = row["centroid_delta_m"]
    if not isinstance(centroid, dict) or set(centroid) != {"r", "z"}:
        raise ValueError(f"{field}.centroid_delta_m fields are invalid")
    for axis in ("r", "z"):
        _require_number(centroid[axis], field=f"{field}.centroid_delta_m.{axis}")
    for name in ("cosine_similarity", "total_variation_distance"):
        value = _require_number(row[name], field=f"{field}.{name}", minimum=0.0)
        if value > 1.0:
            raise ValueError(f"{field}.{name} must be <= 1")


def _validate_checkpoint(row: object, *, run_name: str, field: str) -> None:
    if not isinstance(row, dict) or set(row) != _CHECKPOINT_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    index = row["iteration_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"{field}.iteration_index is invalid")
    if row["phase"] != phase_for_iteration(run_name, index):
        raise ValueError(f"{field}.phase is inconsistent")
    if not isinstance(row["converged"], bool) or not isinstance(row["terminal"], bool):
        raise ValueError(f"{field} flags are invalid")
    for name in ("psi_before_sha256", "psi_after_sha256"):
        _require_sha256(row[name], field=f"{field}.{name}")
    for name in ("ip_fraction", "ip_now_a", "separatrix_refinement"):
        _require_number(row[name], field=f"{field}.{name}", minimum=0.0)
    if row["ip_fraction"] > 1.0 + 1.0e-12 or row["separatrix_refinement"] > 1.0:
        raise ValueError(f"{field} continuation fractions must be <= 1")
    geometry = row["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {
        "active_boundary_wb",
        "active_span_wb",
        "axis_wb",
        "physical_boundary_wb",
        "physical_span_wb",
    }:
        raise ValueError(f"{field}.geometry fields are invalid")
    for name, value in geometry.items():
        _require_number(value, field=f"{field}.geometry.{name}")
    fixed = row["fixed_point"]
    if not isinstance(fixed, dict) or set(fixed) != {
        "accepted_update_relative_l2",
        "residual_linf_wb",
        "residual_relative_l2",
    }:
        raise ValueError(f"{field}.fixed_point fields are invalid")
    residual = row["physical_residual"]
    if not isinstance(residual, dict) or set(residual) != {
        "interior_relative_rms",
        "wall_relative_l2",
    }:
        raise ValueError(f"{field}.physical_residual fields are invalid")
    for group_name, group in (("fixed_point", fixed), ("physical_residual", residual)):
        for name, value in group.items():
            _require_number(value, field=f"{field}.{group_name}.{name}", minimum=0.0)
    _validate_current(row["production_current"], field=f"{field}.production_current")
    _validate_current(
        row["reference_boundary_counterfactual"],
        field=f"{field}.reference_boundary_counterfactual",
    )


def _validate_runs(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"cold", "warm"}:
        raise ValueError("runs fields are invalid")
    for run_name, expected_indices, iteration_cap in (
        ("cold", COLD_CHECKPOINT_INDICES, 180),
        ("warm", WARM_CHECKPOINT_INDICES, 20),
    ):
        run = runs[run_name]
        if not isinstance(run, dict) or set(run) != _RUN_FIELDS:
            raise ValueError(f"runs.{run_name} fields are invalid")
        if run["checkpoint_indices_requested"] != list(expected_indices):
            raise ValueError(f"runs.{run_name} requested indices are invalid")
        count = run["iteration_count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= iteration_cap:
            raise ValueError(f"runs.{run_name}.iteration_count is invalid")
        if run["iteration_cap"] != iteration_cap:
            raise ValueError(f"runs.{run_name}.iteration_cap is invalid")
        if run["terminated_early"] is not (count < iteration_cap):
            raise ValueError(f"runs.{run_name}.terminated_early is inconsistent")
        _require_sha256(
            run["terminal_psi_sha256"],
            field=f"runs.{run_name}.terminal_psi_sha256",
        )
        checkpoints = run["checkpoints"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"runs.{run_name}.checkpoints must be non-empty")
        for position, row in enumerate(checkpoints):
            _validate_checkpoint(row, run_name=run_name, field=f"runs.{run_name}.{position}")
        actual = [row["iteration_index"] for row in checkpoints]
        expected = sorted({*(index for index in expected_indices if index < count), count - 1})
        if actual != expected:
            raise ValueError(f"runs.{run_name} checkpoint coverage is inconsistent")
        terminals = [row for row in checkpoints if row["terminal"]]
        if len(terminals) != 1 or terminals[0]["iteration_index"] != count - 1:
            raise ValueError(f"runs.{run_name} terminal checkpoint is inconsistent")
        if terminals[0]["psi_after_sha256"] != run["terminal_psi_sha256"]:
            raise ValueError(f"runs.{run_name} terminal digest is inconsistent")
    cold_terminal = next(row for row in runs["cold"]["checkpoints"] if row["terminal"])
    warm_initial = runs["warm"]["checkpoints"][0]
    if warm_initial["psi_before_sha256"] != cold_terminal["psi_after_sha256"]:
        raise ValueError("cold-to-warm trace continuity is inconsistent")
    return cast(dict[str, dict[str, Any]], runs)


def _validate_sources(report: dict[str, Any]) -> None:
    artifacts = report.get("source_artifacts")
    expected = {*SOURCE_PATHS, "freegs_public_example", "repository"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("source_artifacts fields are invalid")
    for name in {*SOURCE_PATHS, "freegs_public_example"}:
        row = artifacts[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        expected_path = SOURCE_PATHS.get(name, FREEGS_PUBLIC_EXAMPLE_PATH)
        if row["path"] != expected_path:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        _require_sha256(row["sha256"], field=f"source_artifacts.{name}.sha256")
    repository = artifacts["repository"]
    if (
        not isinstance(repository, dict)
        or set(repository) != {"git_commit", "path"}
        or repository["path"] != "."
    ):
        raise ValueError("source_artifacts.repository fields are invalid")
    _require_git_oid(repository.get("git_commit"), field="source_artifacts.repository.git_commit")


def _validate_source_reports(report: dict[str, Any]) -> None:
    reports = report.get("source_reports")
    if not isinstance(reports, dict) or set(reports) != {"same_case", "source_ablation"}:
        raise ValueError("source_reports fields are invalid")
    for name, path in (
        ("same_case", SAME_CASE_REPORT_PATH),
        ("source_ablation", SOURCE_ABLATION_REPORT_PATH),
    ):
        row = reports[name]
        if not isinstance(row, dict) or set(row) != {
            "path",
            "payload_sha256",
            "source_commit",
        }:
            raise ValueError(f"source_reports.{name} fields are invalid")
        if row["path"] != path:
            raise ValueError(f"source_reports.{name}.path is invalid")
        _require_sha256(row["payload_sha256"], field=f"source_reports.{name}.payload_sha256")
        _require_git_oid(row["source_commit"], field=f"source_reports.{name}.source_commit")


def validate_report(report: dict[str, Any]) -> None:
    """Reject schema drift, tamper, overclaim, and inconsistent routing."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("trace report top-level fields do not match the v1 schema")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("unsupported geometry-feedback trace contract")
    _walk_finite(report)
    if report.get("payload_sha256") != _payload_sha256(report):
        raise ValueError("payload_sha256 does not match report content")
    _require_sha256(report.get("payload_sha256"), field="payload_sha256")
    if not isinstance(report.get("generated_at"), str) or not report["generated_at"].strip():
        raise ValueError("generated_at must not be empty")
    blockers = report.get("blockers")
    if (
        not isinstance(blockers, list)
        or set(blockers) not in (_BASE_BLOCKERS, _BASE_BLOCKERS | {"source_worktree_not_clean"})
        or len(blockers) != len(set(blockers))
    ):
        raise ValueError("blockers are inconsistent")
    if report.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if report.get("input_contract") != INPUT_CONTRACT:
        raise ValueError("input_contract does not match the frozen compiled trace")
    if report.get("status") != "diagnostic_complete_claims_blocked":
        raise ValueError("geometry-feedback trace cannot promote an admitted status")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields are invalid")
    if environment["x64_enabled"] is not True:
        raise ValueError("environment must bind a JAX FP64 execution")
    _validate_sources(report)
    _validate_source_reports(report)
    runs = _validate_runs(report)
    parity = report.get("terminal_parity")
    if not isinstance(parity, dict) or set(parity) != {
        "expected_same_case_candidate_sha256",
        "matches_same_case_candidate",
        "traced_candidate_sha256",
    }:
        raise ValueError("terminal_parity fields are invalid")
    for name in ("expected_same_case_candidate_sha256", "traced_candidate_sha256"):
        _require_sha256(parity[name], field=f"terminal_parity.{name}")
    expected_match = parity["expected_same_case_candidate_sha256"] == parity["traced_candidate_sha256"]
    if parity["matches_same_case_candidate"] is not expected_match:
        raise ValueError("terminal_parity match flag is inconsistent")
    if parity["traced_candidate_sha256"] != runs["warm"]["terminal_psi_sha256"]:
        raise ValueError("terminal_parity traced digest is inconsistent")
    if report.get("routing") != _routing(runs, cast(dict[str, Any], parity)):
        raise ValueError("routing is inconsistent with checkpoint measurements")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise checkpoint table and routing summary."""
    renderer = importlib.import_module("validation.ida_geometry_feedback_trace_render")
    return cast(str, renderer.render_markdown(report))
