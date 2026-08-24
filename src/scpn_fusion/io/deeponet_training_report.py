# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Training Reports
"""Running evidence, artifact payload, and completed-report composition."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from scpn_fusion.core.deeponet_training_contracts import (
    PreparedTraining,
    RuntimeBackendParity,
    TrainingConfig,
)
from scpn_fusion.io.deeponet_training_recovery import (
    OptimizerRecovery,
    OptimizerState,
    serialize_network,
)
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import sha256_file
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedSplit,
    MachineConditionedTrainingData,
    array_sha256,
)


def running_report(
    data: MachineConditionedTrainingData,
    split: MachineConditionedSplit,
    split_hashes: dict[str, str],
    config: TrainingConfig,
    statistics_sha256: str,
    *,
    training_schema: str,
) -> dict[str, Any]:
    """Compose fail-closed evidence before optimisation starts.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated cohort and provenance.
    split : MachineConditionedSplit
        Four disjoint role assignments.
    split_hashes : dict[str, str]
        SHA-256 digest for each role's ordered indices.
    config : TrainingConfig
        Immutable run configuration.
    statistics_sha256 : str
        Digest of the training-only statistics stage.
    training_schema : str
        Versioned report schema identifier.

    Returns
    -------
    dict[str, Any]
        JSON-compatible running report with claims closed by default.
    """
    return {
        "schema_version": training_schema,
        "status": "running",
        "claims": {
            "class": "synthetic_fixed_machine_coordinate_operator_candidate",
            "facility_validated": False,
            "cross_machine_validated": False,
            "experimental_shot_data": False,
            "free_boundary_prediction": False,
            "ida_or_efit_replacement": False,
        },
        "dataset": {
            "dataset_id": data.manifest["dataset_id"],
            "manifest_sha256": data.manifest_sha256,
            "inputs_sha256": data.inputs_sha256,
            "fields_sha256": data.fields_sha256,
            "samples": len(data.inputs),
            "grid_shape": list(data.grid_shape),
            "feature_names": list(data.feature_names),
            "machine_name": data.manifest["machine"]["name"],
        },
        "split": {
            "seed": config.seed,
            "fractions": {
                "validation": config.validation_fraction,
                "calibration": config.calibration_fraction,
                "test": config.test_fraction,
            },
            "samples": {
                "training": len(split.training),
                "validation": len(split.validation),
                "calibration": len(split.calibration),
                "test": len(split.test),
            },
            "indices_sha256": split_hashes,
            "transforms_fit_on": "training_indices_only",
            "model_selection_on": "fixed_validation_probe_only",
            "calibration_and_test_opened": "after_best_step_frozen",
        },
        "recovery": {
            "checkpoint_dir": str(config.checkpoint_dir),
            "resume_requested": config.resume,
            "statistics_sha256": statistics_sha256,
        },
    }


def artifact_payload(
    prepared: PreparedTraining,
    config: TrainingConfig,
    state: OptimizerState,
    *,
    artifact_schema: str,
    training_schema: str,
) -> dict[str, Any]:
    """Compose the pickle-free manifest-bound runtime artifact.

    Parameters
    ----------
    prepared : PreparedTraining
        Authenticated data, transforms, split hashes, and source identity.
    config : TrainingConfig
        Immutable run configuration.
    state : OptimizerState
        Completed optimiser state with validation-selected parameters.
    artifact_schema, training_schema : str
        Versioned runtime and training schema identifiers.

    Returns
    -------
    dict[str, Any]
        NumPy-compatible arrays for the production runtime NPZ.
    """
    payload: dict[str, Any] = {
        "artifact_schema": np.asarray([artifact_schema]),
        "input_mean": prepared.input_mean,
        "input_std": prepared.input_std,
        "coordinates_rz_m": prepared.coordinates,
        "coordinate_mean": prepared.coordinate_mean,
        "coordinate_std": prepared.coordinate_std,
        "field_mean": prepared.field_mean,
        "field_scale": np.asarray([prepared.field_scale]),
        "basis_width": np.asarray([config.basis_width], dtype=np.int64),
        "grid_nh": np.asarray([prepared.data.grid_shape[0]], dtype=np.int64),
        "grid_nw": np.asarray([prepared.data.grid_shape[1]], dtype=np.int64),
        "feature_names": np.asarray(prepared.data.feature_names),
        "dataset_manifest_sha256": np.asarray([prepared.data.manifest_sha256]),
        "selected_step": np.asarray([state.best_step], dtype=np.int64),
        "training_schema": np.asarray([training_schema]),
        "source_sha256_names": prepared.identity["source_sha256_names"],
        "source_sha256_values": prepared.identity["source_sha256_values"],
    }
    for role, digest in prepared.split_hashes.items():
        payload[f"{role}_indices_sha256"] = np.asarray([digest])
    serialize_network(payload, "branch", state.best_params["branch"])
    serialize_network(payload, "trunk", state.best_params["trunk"])
    return payload


def completed_report_sections(
    prepared: PreparedTraining,
    config: TrainingConfig,
    state: OptimizerState,
    *,
    stopped_early: bool,
    elapsed_seconds: float,
    validation_metrics: dict[str, float],
    calibration_metrics: dict[str, float],
    test_metrics: dict[str, float],
    conformal_alpha: float,
    conformal_rank: int,
    conformal_bound: float,
    test_coverage: float,
    recovery: OptimizerRecovery,
    runtime_prediction: np.ndarray[Any, np.dtype[np.float64]],
    runtime_parity: float,
    runtime_backend: str,
    backend_parity: RuntimeBackendParity,
) -> dict[str, Any]:
    """Compose final evidence after held-out evaluation has completed.

    Parameters
    ----------
    prepared : PreparedTraining
        Frozen data, transforms, split, and recovery identity.
    config : TrainingConfig
        Immutable run configuration.
    state : OptimizerState
        Final optimiser and validation-selection state.
    stopped_early : bool
        Whether validation patience ended optimisation before ``steps``.
    elapsed_seconds : float
        Local wall-clock training duration.
    validation_metrics, calibration_metrics, test_metrics : dict[str, float]
        Full-field metrics measured on each held-out role.
    conformal_alpha : float
        Miscoverage target used for split-conformal calibration.
    conformal_rank : int
        One-based finite-sample order-statistic rank.
    conformal_bound, test_coverage : float
        Calibrated relative-L2 bound and untouched-test empirical coverage.
    recovery : OptimizerRecovery
        Authenticated optimiser recovery pointer.
    runtime_prediction : ndarray[float64]
        Production-runtime parity probe.
    runtime_parity : float
        Maximum absolute Wb/rad difference from the JAX training path.
    runtime_backend : str
        Selected production inference tier, ``rust`` or ``numpy``.
    backend_parity : RuntimeBackendParity
        Rust-versus-NumPy evidence over every untouched-test row.

    Returns
    -------
    dict[str, Any]
        JSON-compatible final report sections.
    """
    return {
        "status": "completed_local_candidate_not_promoted",
        "architecture": {
            "operator": "DeepONet_branch_trunk_inner_product",
            "activation": "SiLU",
            "branch_inputs": "17_causal_pre_solve_controls",
            "trunk_inputs": "normalised_R_Z_coordinates",
            "branch_hidden": list(config.branch_hidden),
            "trunk_hidden": list(config.trunk_hidden),
            "basis_width": config.basis_width,
            "machine_conditioning": "manifest_bound_single_machine_only",
            "cross_machine_claim": False,
        },
        "training": {
            "precision": "float32_parameters_and_updates_float64_artifact_and_metrics",
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "requested_steps": config.steps,
            "completed_steps": state.completed_steps,
            "selected_step": state.best_step,
            "stopped_early": stopped_early,
            "final_training_loss": state.final_training_loss,
            "best_validation_probe_loss": state.best_validation_loss,
            "shot_batch_size": config.shot_batch_size,
            "coordinate_batch_size": config.coordinate_batch_size,
            "validation_probe_shots": len(prepared.probe_indices),
            "validation_probe_coordinates": len(prepared.probe_coordinate_indices),
            "validation_probe_samples_sha256": array_sha256(prepared.probe_indices),
            "validation_probe_coordinates_sha256": array_sha256(prepared.probe_coordinate_indices),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip": config.gradient_clip,
            "evaluation_every": config.evaluation_every,
            "evaluation_steps": state.evaluation_steps,
            "training_losses": state.training_losses,
            "validation_losses": state.validation_losses,
            "field_mean_and_scale_fit_on": "training_indices_only",
            "relative_sample_weight_reference": prepared.field_norm_reference,
            "elapsed_seconds": elapsed_seconds,
        },
        "held_out_validation": validation_metrics,
        "post_selection_calibration": calibration_metrics,
        "untouched_final_test": test_metrics,
        "conformal_relative_l2": {
            "alpha": conformal_alpha,
            "finite_sample_rank_one_based": conformal_rank,
            "bound": conformal_bound,
            "test_empirical_coverage": test_coverage,
        },
        "recovery": {
            **prepared.report["recovery"],
            "optimizer_stage_file": recovery["stage_file"],
            "optimizer_stage_sha256": recovery["stage_sha256"],
            "optimizer_completed_steps": recovery["completed_steps"],
        },
        "artifact": {
            "path": str(config.output_path),
            "sha256": sha256_file(config.output_path),
            "promotion_status": "local_candidate_not_promoted",
            "runtime_load_predict_finite": bool(np.all(np.isfinite(runtime_prediction))),
            "runtime_backend": runtime_backend,
            "runtime_prediction_shape": list(runtime_prediction.shape),
            "runtime_training_path_parity_max_abs": runtime_parity,
            "rust_numpy_untouched_test_parity": backend_parity,
        },
    }


__all__ = ["artifact_payload", "completed_report_sections", "running_report"]
