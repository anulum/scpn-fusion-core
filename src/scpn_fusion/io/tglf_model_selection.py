# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Model Selection
"""Frozen-split model-family evaluation for official GACODE TGLF records.

The public facade verifies corpus custody, fits deterministic candidate
families on training rows, selects on calibration rows, and opens test rows
only after leader lock. Candidate evidence never promotes a runtime model.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import ModuleType
from typing import Any, Final

import numpy as np

import scpn_fusion.core.tglf_surrogate_candidates as candidate_module
from scpn_fusion.core.tglf_surrogate_candidates import (
    CompactNeuralEnsemble,
    QuadraticPolynomialCandidate,
    RandomisedTreeEnsemble,
    TGLFRegressionCandidate,
)
import scpn_fusion.io._tglf_model_selection_data as data_module
from scpn_fusion.io._tglf_model_selection_data import (
    SPLITS,
    TGLFModelStudyData,
    load_tglf_model_study_data,
    object_value,
)
import scpn_fusion.io._tglf_model_selection_metrics as metrics_module
from scpn_fusion.io._tglf_model_selection_metrics import (
    ABS_NORMALISED_BIAS_MAX,
    CHANNEL_NRMSE_MAX,
    CLOSURE_P95_MAX,
    SIGN_AGREEMENT_MIN,
    THRESHOLD_NRMSE_MAX,
    candidate_rank,
    eligibility,
    evaluate_rows,
    indices,
    latency_orientation,
)
from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_dataset_contract import sha256_file

TGLF_MODEL_SELECTION_SCHEMA: Final = "scpn-fusion.tglf-model-selection.v1"
TGLF_MODEL_SELECTION_SEED: Final = 20260826


def _candidate_factories() -> dict[str, tuple[dict[str, Any], TGLFRegressionCandidate]]:
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
                "seed": TGLF_MODEL_SELECTION_SEED,
            },
            RandomisedTreeEnsemble(
                trees=64,
                maximum_depth=6,
                minimum_leaf_rows=2,
                split_probes=8,
                seed=TGLF_MODEL_SELECTION_SEED,
            ),
        ),
        "compact_neural_ensemble": (
            {
                "members": 5,
                "hidden_width": 24,
                "epochs": 1500,
                "learning_rate": 1.0e-2,
                "l2": 1.0e-4,
                "seed": TGLF_MODEL_SELECTION_SEED,
            },
            CompactNeuralEnsemble(
                members=5,
                hidden_width=24,
                epochs=1500,
                learning_rate=1.0e-2,
                l2=1.0e-4,
                seed=TGLF_MODEL_SELECTION_SEED,
            ),
        ),
    }


def _fit_candidates(
    data: TGLFModelStudyData,
    *,
    target_scales: np.ndarray[Any, np.dtype[np.float64]],
    latency_repeats: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, TGLFRegressionCandidate], str, list[str]]:
    train_indices = indices(data.splits, "train")
    calibration_indices = indices(data.splits, "calibration")
    scaled_targets = data.targets / target_scales
    reports: dict[str, dict[str, Any]] = {}
    fitted: dict[str, TGLFRegressionCandidate] = {}
    for name, (configuration, candidate) in _candidate_factories().items():
        try:
            start = time.perf_counter_ns()
            candidate.fit(data.features[train_indices], scaled_targets[train_indices])
            fit_seconds = (time.perf_counter_ns() - start) / 1.0e9
            calibration_prediction = (
                candidate.predict(data.features[calibration_indices]) * target_scales
            )
            calibration = evaluate_rows(
                data, calibration_indices, calibration_prediction, target_scales
            )
            eligible, reasons = eligibility(calibration)
            reports[name] = {
                "configuration": configuration,
                "fit_seconds_orientation_only": fit_seconds,
                "state_bytes": candidate.state_bytes() + target_scales.nbytes,
                "latency_orientation": latency_orientation(
                    candidate, data.features, repeats=latency_repeats
                ),
                "calibration": calibration,
                "calibration_eligible": eligible,
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
        raise RuntimeError("all TGLF model candidates failed before calibration")
    eligible_names = [name for name in ranked if reports[name]["calibration_eligible"]]
    leader = min(eligible_names or ranked, key=lambda name: candidate_rank(reports[name]))
    return reports, fitted, leader, eligible_names


def _evaluate_test(
    data: TGLFModelStudyData,
    reports: dict[str, dict[str, Any]],
    fitted: dict[str, TGLFRegressionCandidate],
    target_scales: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    test_indices = indices(data.splits, "test")
    for name, candidate in fitted.items():
        prediction = candidate.predict(data.features[test_indices]) * target_scales
        evaluation = evaluate_rows(data, test_indices, prediction, target_scales)
        passed, reasons = eligibility(evaluation)
        reports[name]["test"] = evaluation
        reports[name]["test_gate_passed"] = passed
        reports[name]["test_gate_failure_reasons"] = reasons


def _implementation_contract() -> dict[str, str]:
    modules: tuple[tuple[str, str, ModuleType], ...] = (
        ("candidate_module", "src/scpn_fusion/core/tglf_surrogate_candidates.py", candidate_module),
        ("data_module", "src/scpn_fusion/io/_tglf_model_selection_data.py", data_module),
        ("metrics_module", "src/scpn_fusion/io/_tglf_model_selection_metrics.py", metrics_module),
        ("study_module", "src/scpn_fusion/io/tglf_model_selection.py", sys.modules[__name__]),
    )
    result: dict[str, str] = {}
    for label, logical_path, module in modules:
        module_file = module.__file__
        if module_file is None:
            raise RuntimeError(f"implementation module has no source file: {logical_path}")
        result[label] = logical_path
        result[f"{label}_sha256"] = sha256_file(Path(module_file))
    return result


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
        projection,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_tglf_model_selection(
    dataset_root: str | Path,
    *,
    selection_lock_paths: tuple[str | Path, ...],
    latency_repeats: int = 31,
) -> dict[str, Any]:
    """Run the frozen three-family TGLF candidate study."""
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
    train_indices = indices(data.splits, "train")
    target_scales = np.sqrt(np.mean(data.targets[train_indices] ** 2, axis=0))
    target_scales = np.where(target_scales > 1.0e-12, target_scales, 1.0)
    reports, fitted, leader, eligible_names = _fit_candidates(
        data,
        target_scales=target_scales,
        latency_repeats=latency_repeats,
    )
    _evaluate_test(data, reports, fitted, target_scales)
    leader_is_eligible = bool(reports[leader]["calibration_eligible"])
    leader_test_passed = bool(reports[leader]["test_gate_passed"])

    root = Path(dataset_root)
    plan = object_value(checked_json_load(root / "plan.json"), "plan")
    manifest = object_value(checked_json_load(root / "manifest.json"), "manifest")
    development = object_value(manifest.get("development"), "manifest.development")
    source = object_value(manifest.get("source"), "manifest.source")
    plan_design_sha256 = plan.get("plan_sha256")
    gacode_revision = plan.get("gacode_revision")
    if (
        not isinstance(plan_design_sha256, str)
        or development.get("plan_sha256") != plan_design_sha256
    ):
        raise ValueError("plan and manifest development digests differ")
    if not isinstance(gacode_revision, str) or source.get("revision") != gacode_revision:
        raise ValueError("plan and manifest GACODE revisions differ")

    report: dict[str, Any] = {
        "schema_version": TGLF_MODEL_SELECTION_SCHEMA,
        "status": "passed" if leader_is_eligible and leader_test_passed else "failed",
        "purpose": "candidate-family-selection-not-runtime-promotion",
        "seed": TGLF_MODEL_SELECTION_SEED,
        "selection_locks": locks,
        "source": {
            "dataset_id": data.verification["dataset_id"],
            "gacode_revision": gacode_revision,
            "plan_design_sha256": plan_design_sha256,
            "plan_file_sha256": sha256_file(root / "plan.json"),
            "records_sha256": sha256_file(root / "dataset.json"),
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "tree_sha256": data.verification["tree_sha256"],
            "samples": int(data.features.shape[0]),
            "split_counts": {split: int(indices(data.splits, split).size) for split in SPLITS},
            "group_counts": {
                split: len(
                    {
                        data.groups[index]
                        for index in range(len(data.groups))
                        if data.splits[index] == split
                    }
                )
                for split in SPLITS
            },
        },
        "implementation": _implementation_contract(),
        "representation": {
            "dtype": "float64",
            "feature_names": list(data.feature_names),
            "target_names": list(data.target_names),
            "feature_count": len(data.feature_names),
            "target_count": len(data.target_names),
            "target_scales_training_rms": {
                name: float(target_scales[index]) for index, name in enumerate(data.target_names)
            },
            "missing_species_policy": "presence-zero-and-masked-zero-flux-targets",
        },
        "gates": {
            "channel_normalised_rmse_max": CHANNEL_NRMSE_MAX,
            "absolute_normalised_bias_max": ABS_NORMALISED_BIAS_MAX,
            "sign_agreement_min": SIGN_AGREEMENT_MIN,
            "threshold_channel_normalised_rmse_max": THRESHOLD_NRMSE_MAX,
            "closure_p95_max": CLOSURE_P95_MAX,
        },
        "selection": {
            "surface": "calibration-only",
            "calibration_leader": leader,
            "calibration_leader_eligible": leader_is_eligible,
            "eligible_candidates": eligible_names,
            "test_gate_passed": leader_test_passed,
            "test_does_not_reselect": True,
        },
        "candidates": reports,
        "runtime_context": {
            "classification": "loaded-host-orientation-only",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_affinity_cpus": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            ),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        },
        "admission": {
            "weights_promoted": False,
            "runtime_integrated": False,
            "uncertainty_calibrated": False,
            "ood_gate": False,
            "independent_gyrokinetic_parity": False,
            "experimental_accuracy": False,
            "torax_comparison": False,
            "hardware_neutral_performance": False,
        },
    }
    report["scientific_projection_sha256"] = _scientific_digest(report)
    json.dumps(report, allow_nan=False)
    return report


def write_tglf_model_selection_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically persist a strict finite MODEL-04 JSON report."""
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError("output_path must be absent or a regular non-symlink file")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() and (temporary.is_symlink() or not temporary.is_file()):
        raise ValueError("temporary output path is unsafe")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


__all__ = [
    "TGLF_MODEL_SELECTION_SCHEMA",
    "TGLF_MODEL_SELECTION_SEED",
    "TGLFModelStudyData",
    "load_tglf_model_study_data",
    "run_tglf_model_selection",
    "write_tglf_model_selection_report",
]
