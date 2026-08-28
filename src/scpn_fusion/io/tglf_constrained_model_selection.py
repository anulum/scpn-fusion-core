# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained TGLF Model Selection
"""Fresh-holdout TGLF selection with charge-constrained particle outputs."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import ModuleType
from typing import Any, Final, cast

import numpy as np

import scpn_fusion.core.tglf_surrogate_candidates as candidate_module
from scpn_fusion.core.tglf_surrogate_candidates import (
    CompactNeuralEnsemble,
    QuadraticPolynomialCandidate,
    RandomisedTreeEnsemble,
    TGLFRegressionCandidate,
)
import scpn_fusion.io._tglf_constrained_targets as target_module
from scpn_fusion.io._tglf_constrained_targets import (
    PARTICLE_CLOSURE_P95_MAX,
    constrained_coordinate_names,
    encode_constrained_targets,
    particle_closure_summary,
    reconstruct_constrained_prediction,
)
import scpn_fusion.io._tglf_model_selection_data as data_module
from scpn_fusion.io._tglf_model_selection_data import (
    TGLFModelStudyData,
    load_tglf_model_study_data,
    object_value,
)
import scpn_fusion.io._tglf_model_selection_metrics as metrics_module
from scpn_fusion.io._tglf_model_selection_metrics import (
    ABS_NORMALISED_BIAS_MAX,
    CHANNEL_NRMSE_MAX,
    SIGN_AGREEMENT_MIN,
    THRESHOLD_NRMSE_MAX,
    evaluate_rows,
    indices,
    latency_orientation,
)
from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_dataset_contract import sha256_file
from scpn_fusion.io.tglf_development_plan import (
    TGLF_DEVELOPMENT_GACODE_REVISION,
    TGLF_EXPANDED_SELECTION_SEED,
)

TGLF_CONSTRAINED_SELECTION_SCHEMA: Final = "scpn-fusion.tglf-constrained-selection.v1"
TGLF_EXPANDED_PLAN_SHA256: Final = (
    "9acbe7db76abf04b6a23f7b96e48faf79304e134881e7e01f0c452498b5302fb"
)
EXPECTED_SPLIT_ROWS: Final = {"train": 108, "calibration": 69, "test": 39}


def _candidate_factories() -> dict[str, tuple[dict[str, Any], TGLFRegressionCandidate]]:
    seed = TGLF_EXPANDED_SELECTION_SEED
    return {
        "quadratic_polynomial": (
            {"ridge": 1.0e-3},
            QuadraticPolynomialCandidate(ridge=1.0e-3),
        ),
        "randomised_tree_ensemble": (
            {
                "trees": 64,
                "maximum_depth": 6,
                "minimum_leaf_rows": 2,
                "split_probes": 8,
                "seed": seed,
            },
            RandomisedTreeEnsemble(
                trees=64,
                maximum_depth=6,
                minimum_leaf_rows=2,
                split_probes=8,
                seed=seed,
            ),
        ),
        "compact_neural_ensemble": (
            {
                "members": 5,
                "hidden_width": 24,
                "epochs": 1500,
                "learning_rate": 1.0e-2,
                "l2": 1.0e-4,
                "seed": seed,
            },
            CompactNeuralEnsemble(
                members=5,
                hidden_width=24,
                epochs=1500,
                learning_rate=1.0e-2,
                l2=1.0e-4,
                seed=seed,
            ),
        ),
    }


def constrained_eligibility(evaluation: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply frozen channel gates without a false exchange zero-sum rule."""
    if evaluation.get("failed_rows") != 0:
        return False, ["non-finite prediction rows"]
    reasons: list[str] = []
    channels = cast(dict[str, dict[str, float | int | None]], evaluation["channels"])
    for name, metrics in channels.items():
        nrmse = metrics["normalised_rmse"]
        bias = metrics["normalised_bias"]
        sign = metrics["sign_agreement"]
        if nrmse is not None and float(nrmse) > CHANNEL_NRMSE_MAX:
            reasons.append(f"{name} normalised RMSE exceeds {CHANNEL_NRMSE_MAX}")
        if bias is not None and abs(float(bias)) > ABS_NORMALISED_BIAS_MAX:
            reasons.append(f"{name} absolute normalised bias exceeds {ABS_NORMALISED_BIAS_MAX}")
        if sign is not None and float(sign) < SIGN_AGREEMENT_MIN:
            reasons.append(f"{name} sign agreement is below {SIGN_AGREEMENT_MIN}")
    threshold = cast(dict[str, Any], evaluation["by_stratum"]).get("threshold")
    if not isinstance(threshold, dict):
        reasons.append("threshold stratum is absent")
    else:
        threshold_channels = cast(dict[str, dict[str, float | int | None]], threshold["channels"])
        for name, metrics in threshold_channels.items():
            nrmse = metrics["normalised_rmse"]
            if nrmse is not None and float(nrmse) > THRESHOLD_NRMSE_MAX:
                reasons.append(f"threshold {name} normalised RMSE exceeds {THRESHOLD_NRMSE_MAX}")
    constrained = cast(dict[str, float], evaluation["constrained_particle_closure"])
    if constrained["p95"] > PARTICLE_CLOSURE_P95_MAX:
        reasons.append(
            f"particle closure p95 exceeds exact reconstruction bound {PARTICLE_CLOSURE_P95_MAX}"
        )
    return not reasons, reasons


def constrained_candidate_rank(
    report: dict[str, Any],
) -> tuple[float, float, float, float, int, float]:
    """Return the frozen calibration-only ranking without exchange closure."""
    calibration = cast(dict[str, Any], report["calibration"])
    summary = cast(dict[str, float], calibration["summary"])
    closure = cast(dict[str, float], calibration["constrained_particle_closure"])
    latency = cast(dict[str, float], report["latency_orientation"])
    return (
        summary["median_normalised_rmse"],
        summary["p95_normalised_rmse"],
        summary["worst_normalised_rmse"],
        closure["p95"],
        cast(int, report["state_bytes"]),
        latency["median_microseconds_per_row"],
    )


def _evaluate_prediction(
    data: TGLFModelStudyData,
    row_indices: np.ndarray[Any, np.dtype[np.int64]],
    coordinate_prediction: np.ndarray[Any, np.dtype[np.float64]],
    evaluation_scales: np.ndarray[Any, np.dtype[np.float64]],
) -> dict[str, Any]:
    prediction = reconstruct_constrained_prediction(data, row_indices, coordinate_prediction)
    evaluation = evaluate_rows(data, row_indices, prediction, evaluation_scales)
    if evaluation["failed_rows"] == 0:
        evaluation["constrained_particle_closure"] = particle_closure_summary(
            data, row_indices, prediction
        )
    else:
        evaluation["constrained_particle_closure"] = None
    evaluation["exchange_semantics"] = {
        "quantity": "species-resolved turbulent exchange power S/S_GB",
        "all_species_zero_sum_gate": False,
        "tgyro_consumer": "sum active thermal-ion channels, then apply opposite equation signs",
    }
    return evaluation


def _fit_candidates(
    data: TGLFModelStudyData,
    *,
    coordinate_targets: np.ndarray[Any, np.dtype[np.float64]],
    coordinate_scales: np.ndarray[Any, np.dtype[np.float64]],
    evaluation_scales: np.ndarray[Any, np.dtype[np.float64]],
    latency_repeats: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, TGLFRegressionCandidate], str, list[str]]:
    train = indices(data.splits, "train")
    calibration = indices(data.splits, "calibration")
    scaled_targets = coordinate_targets / coordinate_scales
    reports: dict[str, dict[str, Any]] = {}
    fitted: dict[str, TGLFRegressionCandidate] = {}
    for name, (configuration, candidate) in _candidate_factories().items():
        try:
            start = time.perf_counter_ns()
            candidate.fit(data.features[train], scaled_targets[train])
            fit_seconds = (time.perf_counter_ns() - start) / 1.0e9
            coordinate_prediction = (
                candidate.predict(data.features[calibration]) * coordinate_scales
            )
            calibration_report = _evaluate_prediction(
                data, calibration, coordinate_prediction, evaluation_scales
            )
            is_eligible, reasons = constrained_eligibility(calibration_report)
            reports[name] = {
                "configuration": configuration,
                "fit_seconds_orientation_only": fit_seconds,
                "state_bytes": candidate.state_bytes() + coordinate_scales.nbytes,
                "latency_orientation": latency_orientation(
                    candidate, data.features, repeats=latency_repeats
                ),
                "calibration": calibration_report,
                "calibration_eligible": is_eligible,
                "calibration_ineligibility_reasons": reasons,
            }
            fitted[name] = candidate
        except (ArithmeticError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            reports[name] = {
                "configuration": configuration,
                "fit_failure": f"{type(exc).__name__}: {exc}",
                "calibration_eligible": False,
                "calibration_ineligibility_reasons": ["candidate fit or evaluation failed"],
            }
    ranked = [name for name, report in reports.items() if "calibration" in report]
    if not ranked:
        raise RuntimeError("all constrained TGLF candidates failed before calibration")
    eligible_names = [name for name in ranked if reports[name]["calibration_eligible"]]
    leader = min(
        eligible_names or ranked, key=lambda name: constrained_candidate_rank(reports[name])
    )
    return reports, fitted, leader, eligible_names


def _evaluate_fresh_test(
    data: TGLFModelStudyData,
    reports: dict[str, dict[str, Any]],
    fitted: dict[str, TGLFRegressionCandidate],
    coordinate_scales: np.ndarray[Any, np.dtype[np.float64]],
    evaluation_scales: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    test = indices(data.splits, "test")
    for name, candidate in fitted.items():
        coordinate_prediction = candidate.predict(data.features[test]) * coordinate_scales
        evaluation = _evaluate_prediction(data, test, coordinate_prediction, evaluation_scales)
        passed, reasons = constrained_eligibility(evaluation)
        reports[name]["test"] = evaluation
        reports[name]["test_gate_passed"] = passed
        reports[name]["test_gate_failure_reasons"] = reasons


def _implementation_contract() -> dict[str, str]:
    modules: tuple[tuple[str, str, ModuleType], ...] = (
        ("candidate_module", "src/scpn_fusion/core/tglf_surrogate_candidates.py", candidate_module),
        ("data_module", "src/scpn_fusion/io/_tglf_model_selection_data.py", data_module),
        ("metrics_module", "src/scpn_fusion/io/_tglf_model_selection_metrics.py", metrics_module),
        ("target_module", "src/scpn_fusion/io/_tglf_constrained_targets.py", target_module),
        (
            "study_module",
            "src/scpn_fusion/io/tglf_constrained_model_selection.py",
            sys.modules[__name__],
        ),
    )
    contract: dict[str, str] = {}
    for label, logical_path, module in modules:
        source = module.__file__
        if source is None:
            raise RuntimeError(f"implementation module has no source file: {logical_path}")
        contract[label] = logical_path
        contract[f"{label}_sha256"] = sha256_file(Path(source))
    return contract


def _scientific_digest(report: dict[str, Any]) -> str:
    candidates = {
        name: {
            key: value
            for key, value in candidate_report.items()
            if key not in {"fit_seconds_orientation_only", "latency_orientation"}
        }
        for name, candidate_report in report["candidates"].items()
    }
    projection = {
        key: report[key]
        for key in (
            "schema_version",
            "status",
            "purpose",
            "seed",
            "selection_locks",
            "source",
            "implementation",
            "representation",
            "gates",
            "selection",
            "admission",
        )
    }
    projection["candidates"] = candidates
    encoded = json.dumps(
        projection, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_tglf_constrained_model_selection(
    dataset_root: str | Path,
    *,
    selection_lock_paths: tuple[str | Path, ...],
    latency_repeats: int = 31,
) -> dict[str, Any]:
    """Run the frozen constrained study and open test only after leader lock."""
    if not selection_lock_paths:
        raise ValueError("at least one selection lock path is required")
    if latency_repeats < 3:
        raise ValueError("latency repeats must be at least three")
    locks: list[dict[str, str]] = []
    for raw_path in selection_lock_paths:
        path = Path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"selection lock must be a regular non-symlink file: {path}")
        locks.append({"name": path.name, "sha256": sha256_file(path)})

    data = load_tglf_model_study_data(dataset_root)
    split_rows = dict(Counter(data.splits))
    if split_rows != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"expanded split rows differ from lock: {split_rows}")
    root = Path(dataset_root)
    plan = object_value(checked_json_load(root / "plan.json"), "plan")
    manifest = object_value(checked_json_load(root / "manifest.json"), "manifest")
    if (
        plan.get("profile") != "expanded"
        or plan.get("seed") != TGLF_EXPANDED_SELECTION_SEED
        or plan.get("plan_sha256") != TGLF_EXPANDED_PLAN_SHA256
    ):
        raise ValueError("expanded corpus plan differs from the frozen lock")
    source = object_value(manifest.get("source"), "manifest.source")
    if source.get("revision") != TGLF_DEVELOPMENT_GACODE_REVISION:
        raise ValueError("expanded corpus GACODE revision differs from the frozen lock")

    train = indices(data.splits, "train")
    coordinates = encode_constrained_targets(data)
    coordinate_scales = np.sqrt(np.mean(coordinates[train] ** 2, axis=0))
    coordinate_scales = np.where(coordinate_scales > 1.0e-12, coordinate_scales, 1.0)
    evaluation_scales = np.sqrt(np.mean(data.targets[train] ** 2, axis=0))
    evaluation_scales = np.where(evaluation_scales > 1.0e-12, evaluation_scales, 1.0)
    reports, fitted, leader, eligible = _fit_candidates(
        data,
        coordinate_targets=coordinates,
        coordinate_scales=coordinate_scales,
        evaluation_scales=evaluation_scales,
        latency_repeats=latency_repeats,
    )
    _evaluate_fresh_test(data, reports, fitted, coordinate_scales, evaluation_scales)
    leader_eligible = bool(reports[leader]["calibration_eligible"])
    leader_test_passed = bool(reports[leader]["test_gate_passed"])
    report: dict[str, Any] = {
        "schema_version": TGLF_CONSTRAINED_SELECTION_SCHEMA,
        "status": "passed",
        "purpose": "fresh-holdout constrained mean-model family selection",
        "seed": TGLF_EXPANDED_SELECTION_SEED,
        "selection_locks": locks,
        "source": {
            "dataset_id": manifest.get("dataset_id"),
            "plan_sha256": plan["plan_sha256"],
            "tree_sha256": data.verification.get("tree_sha256"),
            "gacode_revision": source["revision"],
            "rows": len(data.splits),
            "split_rows": split_rows,
            "split_groups": {
                split: len(
                    {
                        group
                        for group, role in zip(data.groups, data.splits, strict=True)
                        if role == split
                    }
                )
                for split in EXPECTED_SPLIT_ROWS
            },
            "old_model04_rows_loaded": False,
            "fresh_test_opened_after_calibration_leader_lock": True,
        },
        "implementation": _implementation_contract(),
        "representation": {
            "features": list(data.feature_names),
            "physical_targets": list(data.target_names),
            "model_coordinates": list(constrained_coordinate_names(data)),
            "particle_reconstruction": "Gamma_e=-sum_i(Z_i*Gamma_i)/Z_e",
            "ambipolarity_by_construction": True,
            "inactive_species_outputs_zeroed": True,
            "exchange_quantity": "species-resolved turbulent exchange power S/S_GB",
            "exchange_all_species_zero_sum_gate": False,
            "tgyro_exchange_consumer": "sum thermal-ion channels and apply opposite energy-equation signs",
            "coordinate_scales": coordinate_scales.tolist(),
            "evaluation_scales": evaluation_scales.tolist(),
        },
        "gates": {
            "channel_normalised_rmse_max": CHANNEL_NRMSE_MAX,
            "absolute_normalised_bias_max": ABS_NORMALISED_BIAS_MAX,
            "sign_agreement_min": SIGN_AGREEMENT_MIN,
            "threshold_normalised_rmse_max": THRESHOLD_NRMSE_MAX,
            "particle_closure_p95_max": PARTICLE_CLOSURE_P95_MAX,
            "exchange_closure_gate": None,
        },
        "candidates": reports,
        "selection": {
            "calibration_leader": leader,
            "calibration_eligible_candidates": eligible,
            "calibration_leader_eligible": leader_eligible,
            "test_gate_passed": leader_test_passed,
            "test_opened_after_leader_lock": True,
        },
        "admission": {
            "mean_model_admitted_for_uq": leader_eligible and leader_test_passed,
            "weights_persisted": False,
            "runtime_promoted": False,
            "uq_ood_admitted": False,
            "experimental_validation": False,
            "cross_solver_parity": False,
            "hardware_neutral_performance": False,
        },
        "runtime_context": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "performance_claim_admitted": False,
        },
    }
    report["scientific_projection_sha256"] = _scientific_digest(report)
    return report


def write_tglf_constrained_selection_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write one strict report atomically without mutating source custody."""
    destination = Path(path)
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise ValueError("report destination must be absent or a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


__all__ = [
    "TGLF_CONSTRAINED_SELECTION_SCHEMA",
    "constrained_candidate_rank",
    "constrained_eligibility",
    "run_tglf_constrained_model_selection",
    "write_tglf_constrained_selection_report",
]
