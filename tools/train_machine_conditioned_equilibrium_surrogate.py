# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned Equilibrium Surrogate Trainer
"""Train a recovery-safe v2 equilibrium surrogate from authenticated local data."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad, vmap
from numpy.typing import NDArray

_jax_config_update = cast(Callable[[str, bool], None], jax.config.update)
_jax_config_update("jax_enable_x64", True)

from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumAccelerator
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import sha256_file
from scpn_fusion.io.machine_conditioned_surrogate_cli import run_training_cli
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedTrainingData,
    StreamingPCAState,
    array_sha256,
    atomic_json,
    atomic_savez,
    deterministic_four_way_split,
    fit_streaming_randomized_pca,
    iter_field_rows,
    load_machine_conditioned_training_data,
)
from scpn_fusion.io.safe_loaders import checked_json_load

logger = logging.getLogger(__name__)
FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
Layer: TypeAlias = dict[str, jax.Array]
Params: TypeAlias = list[Layer]

TRAINING_SCHEMA = "scpn-fusion.machine-conditioned-equilibrium-surrogate-training.v2"
OPTIMIZER_RECOVERY_SCHEMA = "scpn-fusion.machine-conditioned-adam-recovery.v2"
OPTIMIZER_CHECKPOINT_VERSION = 2
MAX_OPTIMIZER_RECOVERY_BYTES = 2 * 1024 * 1024
DEFAULT_HIDDEN_SIZES = (512, 256, 128)
DEFAULT_PCA_COMPONENTS = 64
DEFAULT_PCA_OVERSAMPLING = 16
DEFAULT_PCA_POWER_ITERATIONS = 1
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_GRADIENT_CLIP = 0.5
REPO_ROOT = Path(__file__).resolve().parents[1]


def init_mlp_params(
    key: jax.Array,
    *,
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
) -> Params:
    """Create deterministic He-initialized dense ReLU parameters."""
    dimensions = (input_dim, *hidden_sizes, output_dim)
    params: Params = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True):
        key, subkey = random.split(key)
        params.append(
            {
                "W": random.normal(subkey, (fan_in, fan_out), dtype=jnp.float64)
                * jnp.sqrt(2.0 / fan_in),
                "b": jnp.zeros(fan_out, dtype=jnp.float64),
            }
        )
    return params


def model_forward(params: Params, features: jax.Array) -> jax.Array:
    """Evaluate one dense ReLU network row."""
    activation = features
    for index, layer in enumerate(params):
        activation = activation @ layer["W"] + layer["b"]
        if index + 1 < len(params):
            activation = jax.nn.relu(activation)
    return activation


def _mse(params: Params, features: jax.Array, targets: jax.Array) -> jax.Array:
    predictions = vmap(lambda row: model_forward(params, row))(features)
    return jnp.mean(jnp.square(predictions - targets))


def hybrid_latent_loss(
    predictions: jax.Array,
    targets: jax.Array,
    component_weights: jax.Array,
    sample_weights: jax.Array,
    field_loss_weight: float,
) -> jax.Array:
    """Blend standardized latent MSE with a relative field-aligned objective.

    Orthonormal PCA makes squared raw-latent error equal to squared error in
    the reconstructible field subspace. ``component_weights`` undo latent
    standardization, while ``sample_weights`` divide by each field's squared
    norm. Both weight vectors are scaled by training-only constants so this
    objective remains numerically comparable to standardized latent MSE.
    """
    squared_error = jnp.square(predictions - targets)
    standardized_mse = jnp.mean(squared_error)
    field_aligned_mse = jnp.mean(
        sample_weights * jnp.mean(squared_error * component_weights, axis=1)
    )
    return (1.0 - field_loss_weight) * standardized_mse + (field_loss_weight * field_aligned_mse)


def _hybrid_objective(
    params: Params,
    features: jax.Array,
    targets: jax.Array,
    component_weights: jax.Array,
    sample_weights: jax.Array,
    field_loss_weight: float,
) -> jax.Array:
    predictions = vmap(lambda row: model_forward(params, row))(features)
    return hybrid_latent_loss(
        predictions,
        targets,
        component_weights,
        sample_weights,
        field_loss_weight,
    )


@jax.jit
def adam_step(
    params: Params,
    first_moment: Params,
    second_moment: Params,
    features: jax.Array,
    targets: jax.Array,
    learning_rate: float,
    gradient_clip: float,
    step: int,
) -> tuple[Params, Params, Params, jax.Array]:
    """Run one deterministic full-batch Adam update."""
    loss, gradients = value_and_grad(_mse)(params, features, targets)
    gradients = cast(
        Params,
        jax.tree_util.tree_map(
            lambda gradient: jnp.clip(gradient, -gradient_clip, gradient_clip), gradients
        ),
    )
    beta_1, beta_2, epsilon = 0.9, 0.999, 1.0e-8
    first_moment = cast(
        Params,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_1 * moment + (1.0 - beta_1) * gradient,
            first_moment,
            gradients,
        ),
    )
    second_moment = cast(
        Params,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_2 * moment + (1.0 - beta_2) * gradient * gradient,
            second_moment,
            gradients,
        ),
    )
    corrected_first = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_1**step), first_moment
    )
    corrected_second = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_2**step), second_moment
    )
    params = cast(
        Params,
        jax.tree_util.tree_map(
            lambda parameter, moment_1, moment_2: (
                parameter - learning_rate * moment_1 / (jnp.sqrt(moment_2) + epsilon)
            ),
            params,
            corrected_first,
            corrected_second,
        ),
    )
    return params, first_moment, second_moment, loss


@jax.jit
def hybrid_adam_step(
    params: Params,
    first_moment: Params,
    second_moment: Params,
    features: jax.Array,
    targets: jax.Array,
    component_weights: jax.Array,
    sample_weights: jax.Array,
    field_loss_weight: float,
    learning_rate: float,
    gradient_clip: float,
    step: int,
) -> tuple[Params, Params, Params, jax.Array]:
    """Run one deterministic full-batch Adam update on the hybrid objective."""
    loss, gradients = value_and_grad(_hybrid_objective)(
        params,
        features,
        targets,
        component_weights,
        sample_weights,
        field_loss_weight,
    )
    gradients = cast(
        Params,
        jax.tree_util.tree_map(
            lambda gradient: jnp.clip(gradient, -gradient_clip, gradient_clip), gradients
        ),
    )
    beta_1, beta_2, epsilon = 0.9, 0.999, 1.0e-8
    first_moment = cast(
        Params,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_1 * moment + (1.0 - beta_1) * gradient,
            first_moment,
            gradients,
        ),
    )
    second_moment = cast(
        Params,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_2 * moment + (1.0 - beta_2) * gradient * gradient,
            second_moment,
            gradients,
        ),
    )
    corrected_first = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_1**step), first_moment
    )
    corrected_second = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_2**step), second_moment
    )
    params = cast(
        Params,
        jax.tree_util.tree_map(
            lambda parameter, moment_1, moment_2: (
                parameter - learning_rate * moment_1 / (jnp.sqrt(moment_2) + epsilon)
            ),
            params,
            corrected_first,
            corrected_second,
        ),
    )
    return params, first_moment, second_moment, loss


@jax.jit
def validation_mse(params: Params, features: jax.Array, targets: jax.Array) -> jax.Array:
    """Evaluate normalized latent validation MSE."""
    return _mse(params, features, targets)


@jax.jit
def validation_hybrid_loss(
    params: Params,
    features: jax.Array,
    targets: jax.Array,
    component_weights: jax.Array,
    sample_weights: jax.Array,
    field_loss_weight: float,
) -> jax.Array:
    """Evaluate the selection objective on validation rows only."""
    return _hybrid_objective(
        params,
        features,
        targets,
        component_weights,
        sample_weights,
        field_loss_weight,
    )


def _serialize_params(payload: dict[str, Any], prefix: str, params: Params) -> None:
    payload[f"{prefix}_n_layers"] = np.asarray([len(params)], dtype=np.int64)
    for index, layer in enumerate(params):
        for name in ("W", "b"):
            payload[f"{prefix}_{index}_{name}"] = np.asarray(layer[name], dtype=np.float64)


def _deserialize_params(archive: Any, prefix: str) -> Params:
    count = int(archive[f"{prefix}_n_layers"][0])
    return [
        {
            name: jnp.asarray(archive[f"{prefix}_{index}_{name}"], dtype=jnp.float64)
            for name in ("W", "b")
        }
        for index in range(count)
    ]


def _optimizer_identity(
    *,
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    validation_indices: IndexArray,
    calibration_indices: IndexArray,
    test_indices: IndexArray,
    pca_checkpoint_sha256: str,
    seed: int,
    validation_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
    field_loss_weight: float,
    hidden_sizes: tuple[int, ...],
    learning_rate: float,
    gradient_clip: float,
    evaluation_every: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_cli.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_training.py",
        REPO_ROOT / "src/scpn_fusion/core/neural_equilibrium.py",
    )
    return {
        "checkpoint_version": np.asarray([OPTIMIZER_CHECKPOINT_VERSION], dtype=np.int64),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "inputs_sha256": np.asarray([data.inputs_sha256]),
        "fields_sha256": np.asarray([data.fields_sha256]),
        "train_indices_sha256": np.asarray([array_sha256(train_indices)]),
        "validation_indices_sha256": np.asarray([array_sha256(validation_indices)]),
        "calibration_indices_sha256": np.asarray([array_sha256(calibration_indices)]),
        "test_indices_sha256": np.asarray([array_sha256(test_indices)]),
        "pca_checkpoint_sha256": np.asarray([pca_checkpoint_sha256]),
        "seed": np.asarray([seed], dtype=np.int64),
        "validation_fraction": np.asarray([validation_fraction], dtype=np.float64),
        "calibration_fraction": np.asarray([calibration_fraction], dtype=np.float64),
        "test_fraction": np.asarray([test_fraction], dtype=np.float64),
        "field_loss_weight": np.asarray([field_loss_weight], dtype=np.float64),
        "hidden_sizes": np.asarray(hidden_sizes, dtype=np.int64),
        "learning_rate": np.asarray([learning_rate], dtype=np.float64),
        "gradient_clip": np.asarray([gradient_clip], dtype=np.float64),
        "evaluation_every": np.asarray([evaluation_every], dtype=np.int64),
        "early_stopping_patience": np.asarray([early_stopping_patience], dtype=np.int64),
        "source_sha256_names": np.asarray(
            [str(path.relative_to(REPO_ROOT)) for path in source_paths]
        ),
        "source_sha256_values": np.asarray([sha256_file(path) for path in source_paths]),
    }


def _validate_identity(archive: Any, identity: Mapping[str, Any]) -> None:
    for name, value in identity.items():
        if name not in archive or not np.array_equal(archive[name], value):
            raise ValueError(f"optimizer checkpoint identity mismatch for {name}")


def _save_optimizer_checkpoint(
    checkpoint_dir: Path,
    *,
    identity: Mapping[str, Any],
    params: Params,
    first_moment: Params,
    second_moment: Params,
    best_params: Params,
    completed_epochs: int,
    final_training_loss: float,
    best_validation_loss: float,
    best_epoch: int,
    evaluations_without_improvement: int,
    evaluation_epochs: list[int],
    training_losses: list[float],
    validation_losses: list[float],
) -> None:
    payload: dict[str, Any] = dict(identity)
    payload.update(
        {
            "completed_epochs": np.asarray([completed_epochs], dtype=np.int64),
            "final_training_loss": np.asarray([final_training_loss]),
            "best_validation_loss": np.asarray([best_validation_loss]),
            "best_epoch": np.asarray([best_epoch], dtype=np.int64),
            "evaluations_without_improvement": np.asarray(
                [evaluations_without_improvement], dtype=np.int64
            ),
            "evaluation_epochs": np.asarray(evaluation_epochs, dtype=np.int64),
            "training_losses": np.asarray(training_losses, dtype=np.float64),
            "validation_losses": np.asarray(validation_losses, dtype=np.float64),
        }
    )
    _serialize_params(payload, "params", params)
    _serialize_params(payload, "first", first_moment)
    _serialize_params(payload, "second", second_moment)
    _serialize_params(payload, "best", best_params)
    stage_path = checkpoint_dir / f"optimizer_epoch_{completed_epochs:08d}.npz"
    atomic_savez(stage_path, payload)
    atomic_json(
        checkpoint_dir / "optimizer_recovery.json",
        {
            "schema_version": OPTIMIZER_RECOVERY_SCHEMA,
            "completed_epochs": completed_epochs,
            "stage_file": stage_path.name,
            "stage_sha256": sha256_file(stage_path),
        },
    )


def _load_optimizer_checkpoint(
    checkpoint_dir: Path,
    *,
    identity: Mapping[str, Any],
) -> tuple[
    Params, Params, Params, Params, int, float, float, int, int, list[int], list[float], list[float]
]:
    raw = checked_json_load(
        checkpoint_dir / "optimizer_recovery.json",
        max_bytes=MAX_OPTIMIZER_RECOVERY_BYTES,
    )
    if not isinstance(raw, dict):
        raise ValueError("optimizer recovery state must be an object")
    recovery = cast(dict[str, Any], raw)
    if recovery.get("schema_version") != OPTIMIZER_RECOVERY_SCHEMA:
        raise ValueError(f"optimizer recovery schema must be {OPTIMIZER_RECOVERY_SCHEMA}")
    filename = recovery.get("stage_file")
    digest = recovery.get("stage_sha256")
    expected_epochs = recovery.get("completed_epochs")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(digest, str)
        or not isinstance(expected_epochs, int)
    ):
        raise ValueError("optimizer recovery metadata is invalid")
    stage_path = checkpoint_dir / filename
    if not stage_path.is_file() or stage_path.is_symlink() or sha256_file(stage_path) != digest:
        raise ValueError("optimizer recovery stage SHA-256 mismatch")
    with np.load(stage_path, allow_pickle=False) as archive:
        _validate_identity(archive, identity)
        result = (
            _deserialize_params(archive, "params"),
            _deserialize_params(archive, "first"),
            _deserialize_params(archive, "second"),
            _deserialize_params(archive, "best"),
            int(archive["completed_epochs"][0]),
            float(archive["final_training_loss"][0]),
            float(archive["best_validation_loss"][0]),
            int(archive["best_epoch"][0]),
            int(archive["evaluations_without_improvement"][0]),
            np.asarray(archive["evaluation_epochs"], dtype=np.int64).tolist(),
            np.asarray(archive["training_losses"], dtype=np.float64).tolist(),
            np.asarray(archive["validation_losses"], dtype=np.float64).tolist(),
        )
    if result[4] != expected_epochs:
        raise ValueError("optimizer recovery epoch mismatch")
    return result


def _encode_fields(
    pca: StreamingPCAState,
    fields: FloatArray,
    indices: IndexArray,
    *,
    chunk_rows: int,
) -> FloatArray:
    latent: FloatArray = np.empty((len(indices), len(pca.components)), dtype=np.float64)
    for start, stop, rows in iter_field_rows(fields, indices, chunk_rows=chunk_rows):
        latent[start:stop] = pca.transform(rows)
    return latent


def _field_norm_squared(
    fields: FloatArray,
    indices: IndexArray,
    *,
    chunk_rows: int,
) -> FloatArray:
    """Compute per-row field norms without materialising the indexed cohort."""
    norms = np.empty(len(indices), dtype=np.float64)
    for start, stop, rows in iter_field_rows(fields, indices, chunk_rows=chunk_rows):
        norms[start:stop] = np.einsum("ij,ij->i", rows, rows)
    return np.maximum(norms, 1.0e-30)


def _field_metrics(
    *,
    pca: StreamingPCAState,
    params: Params | None,
    normalized_inputs: FloatArray | None,
    latent_mean: FloatArray,
    latent_std: FloatArray,
    fields: FloatArray,
    indices: IndexArray,
    true_latent: FloatArray,
    chunk_rows: int,
) -> tuple[dict[str, float], FloatArray]:
    squared_error = 0.0
    elements = 0
    relative_l2: list[float] = []
    squared_latent_error = 0.0
    latent_elements = 0
    for start, stop, rows in iter_field_rows(fields, indices, chunk_rows=chunk_rows):
        if params is None:
            predicted_latent = true_latent[start:stop]
        else:
            if normalized_inputs is None:
                raise ValueError("normalized inputs are required for model field metrics")
            predicted_norm = np.asarray(
                vmap(lambda row: model_forward(params, row))(
                    jnp.asarray(normalized_inputs[start:stop])
                )
            )
            predicted_latent = predicted_norm * latent_std + latent_mean
            latent_delta = predicted_latent - true_latent[start:stop]
            squared_latent_error += float(np.sum(latent_delta * latent_delta))
            latent_elements += latent_delta.size
        delta = pca.inverse_transform(predicted_latent) - rows
        squared_error += float(np.sum(delta * delta))
        elements += delta.size
        numerator = np.linalg.norm(delta, axis=1)
        denominator = np.maximum(np.linalg.norm(rows, axis=1), 1.0e-15)
        relative_l2.extend(np.asarray(numerator / denominator).tolist())
    metrics = {
        "field_rmse": float(np.sqrt(squared_error / max(elements, 1))),
        "mean_relative_l2": float(np.mean(relative_l2)),
        "p95_relative_l2": float(np.percentile(relative_l2, 95.0)),
        "max_relative_l2": float(np.max(relative_l2)),
    }
    if params is not None:
        metrics["latent_rmse"] = float(np.sqrt(squared_latent_error / max(latent_elements, 1)))
    return metrics, np.asarray(relative_l2, dtype=np.float64)


def run_training(
    *,
    dataset_dir: Path,
    output_path: Path,
    report_path: Path,
    checkpoint_dir: Path,
    epochs: int,
    seed: int = 42,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.05,
    field_loss_weight: float = 0.9,
    n_components: int = DEFAULT_PCA_COMPONENTS,
    pca_oversampling: int = DEFAULT_PCA_OVERSAMPLING,
    pca_power_iterations: int = DEFAULT_PCA_POWER_ITERATIONS,
    pca_chunk_rows: int = 256,
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
    evaluation_every: int = 100,
    checkpoint_every: int = 500,
    early_stopping_patience: int = 50,
    resume: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one local machine-conditioned v2 surrogate candidate."""
    if (
        epochs < 1
        or evaluation_every < 1
        or checkpoint_every < 1
        or early_stopping_patience < 1
        or learning_rate <= 0.0
        or gradient_clip <= 0.0
        or n_components < 1
        or any(size < 1 for size in hidden_sizes)
        or not 0.0 <= field_loss_weight <= 1.0
    ):
        raise ValueError("training hyperparameters must be positive")
    started = time.perf_counter()
    data = load_machine_conditioned_training_data(dataset_dir, full_field_scan=True)
    if len(data.inputs) < 5:
        raise ValueError("successor training requires at least five authenticated samples")
    split = deterministic_four_way_split(
        len(data.inputs),
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_indices = split.training
    validation_indices = split.validation
    calibration_indices = split.calibration
    test_indices = split.test
    actual_components = min(n_components, len(train_indices) - 1, int(np.prod(data.grid_shape)))
    pca_source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_cli.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_training.py",
    )
    pca_identity = {
        "training_schema": TRAINING_SCHEMA,
        "dataset_manifest_sha256": data.manifest_sha256,
        "fields_sha256": data.fields_sha256,
        "train_indices_sha256": array_sha256(train_indices),
        "validation_indices_sha256": array_sha256(validation_indices),
        "calibration_indices_sha256": array_sha256(calibration_indices),
        "test_indices_sha256": array_sha256(test_indices),
        "n_components": actual_components,
        "oversampling": pca_oversampling,
        "power_iterations": pca_power_iterations,
        "seed": seed,
        "chunk_rows": pca_chunk_rows,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in pca_source_paths
        },
    }
    report: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA,
        "status": "running",
        "claims": {
            "class": "synthetic_machine_conditioned_fixed_support_surrogate_candidate",
            "facility_validated": False,
            "experimental_shot_data": False,
            "free_boundary_prediction": False,
            "ida_or_efit_replacement": False,
        },
        "dataset": {
            "dataset_id": data.manifest["dataset_id"],
            "path": str(data.root),
            "manifest_sha256": data.manifest_sha256,
            "inputs_sha256": data.inputs_sha256,
            "fields_sha256": data.fields_sha256,
            "samples": len(data.inputs),
            "feature_names": list(data.feature_names),
            "grid_shape": list(data.grid_shape),
        },
        "split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "calibration_fraction": calibration_fraction,
            "test_fraction": test_fraction,
            "training_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "calibration_samples": len(calibration_indices),
            "test_samples": len(test_indices),
            "train_indices_sha256": array_sha256(train_indices),
            "validation_indices_sha256": array_sha256(validation_indices),
            "calibration_indices_sha256": array_sha256(calibration_indices),
            "test_indices_sha256": array_sha256(test_indices),
            "transforms_fit_on": "training_indices_only",
            "model_selection_on": "validation_indices_only",
            "uncertainty_calibration_on": "calibration_indices_after_selection_only",
            "final_evaluation_on": "test_indices_after_selection_only",
        },
        "recovery": {
            "checkpoint_dir": str(checkpoint_dir),
            "resume_requested": resume,
        },
    }
    atomic_json(report_path, report)
    pca, train_latent = fit_streaming_randomized_pca(
        data.fields,
        train_indices,
        n_components=actual_components,
        oversampling=pca_oversampling,
        power_iterations=pca_power_iterations,
        seed=seed,
        chunk_rows=pca_chunk_rows,
        checkpoint_dir=checkpoint_dir / "pca",
        identity=pca_identity,
        resume=resume,
    )
    pca_checkpoint = checkpoint_dir / "pca" / "pca_complete.npz"
    pca_checkpoint_sha256 = sha256_file(pca_checkpoint)
    validation_latent = _encode_fields(
        pca, data.fields, validation_indices, chunk_rows=pca_chunk_rows
    )
    x_train = np.asarray(data.inputs[train_indices], dtype=np.float64)
    x_validation = np.asarray(data.inputs[validation_indices], dtype=np.float64)
    input_mean = np.mean(x_train, axis=0)
    input_raw_std = np.std(x_train, axis=0)
    input_std = np.where(input_raw_std < 1.0e-12, 1.0, input_raw_std)
    latent_mean = np.mean(train_latent, axis=0)
    latent_raw_std = np.std(train_latent, axis=0)
    latent_std = np.where(latent_raw_std < 1.0e-12, 1.0, latent_raw_std)
    x_train_norm = np.asarray((x_train - input_mean) / input_std, dtype=np.float64)
    x_validation_norm = np.asarray((x_validation - input_mean) / input_std, dtype=np.float64)
    train_latent_norm = np.asarray((train_latent - latent_mean) / latent_std, dtype=np.float64)
    validation_latent_norm = np.asarray(
        (validation_latent - latent_mean) / latent_std, dtype=np.float64
    )
    train_field_norm_squared = _field_norm_squared(
        data.fields, train_indices, chunk_rows=pca_chunk_rows
    )
    validation_field_norm_squared = _field_norm_squared(
        data.fields, validation_indices, chunk_rows=pca_chunk_rows
    )
    field_norm_reference = float(np.mean(train_field_norm_squared))
    component_weight_scale = float(np.mean(latent_std * latent_std))
    component_weights = np.asarray(
        (latent_std * latent_std) / component_weight_scale, dtype=np.float64
    )
    train_sample_weights = np.asarray(
        field_norm_reference / train_field_norm_squared, dtype=np.float64
    )
    validation_sample_weights = np.asarray(
        field_norm_reference / validation_field_norm_squared, dtype=np.float64
    )
    identity = _optimizer_identity(
        data=data,
        train_indices=train_indices,
        validation_indices=validation_indices,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        pca_checkpoint_sha256=pca_checkpoint_sha256,
        seed=seed,
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        field_loss_weight=field_loss_weight,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        evaluation_every=evaluation_every,
        early_stopping_patience=early_stopping_patience,
    )
    optimizer_recovery_path = checkpoint_dir / "optimizer_recovery.json"
    if resume and optimizer_recovery_path.exists():
        (
            params,
            first_moment,
            second_moment,
            best_params,
            completed_epochs,
            final_training_loss,
            best_validation_loss,
            best_epoch,
            without_improvement,
            evaluation_epochs,
            training_losses,
            validation_losses,
        ) = _load_optimizer_checkpoint(checkpoint_dir, identity=identity)
        if completed_epochs > epochs:
            raise ValueError("optimizer checkpoint exceeds requested epoch target")
    else:
        params = init_mlp_params(
            random.PRNGKey(seed),
            input_dim=17,
            hidden_sizes=hidden_sizes,
            output_dim=actual_components,
        )
        first_moment = cast(Params, jax.tree_util.tree_map(jnp.zeros_like, params))
        second_moment = cast(Params, jax.tree_util.tree_map(jnp.zeros_like, params))
        best_params = cast(Params, jax.tree_util.tree_map(lambda value: value, params))
        completed_epochs = 0
        final_training_loss = float("nan")
        best_validation_loss = float("inf")
        best_epoch = 0
        without_improvement = 0
        evaluation_epochs, training_losses, validation_losses = [], [], []

    x_train_jax = jnp.asarray(x_train_norm)
    y_train_jax = jnp.asarray(train_latent_norm)
    x_validation_jax = jnp.asarray(x_validation_norm)
    y_validation_jax = jnp.asarray(validation_latent_norm)
    component_weights_jax = jnp.asarray(component_weights)
    train_sample_weights_jax = jnp.asarray(train_sample_weights)
    validation_sample_weights_jax = jnp.asarray(validation_sample_weights)
    stopped_early = without_improvement >= early_stopping_patience
    first_epoch = epochs if stopped_early else completed_epochs
    for epoch in range(first_epoch, epochs):
        step = epoch + 1
        params, first_moment, second_moment, loss = hybrid_adam_step(
            params,
            first_moment,
            second_moment,
            x_train_jax,
            y_train_jax,
            component_weights_jax,
            train_sample_weights_jax,
            field_loss_weight,
            learning_rate,
            gradient_clip,
            step,
        )
        final_training_loss = float(loss)
        completed_epochs = step
        evaluate = step == 1 or step % evaluation_every == 0 or step == epochs
        if evaluate:
            current_validation = float(
                validation_hybrid_loss(
                    params,
                    x_validation_jax,
                    y_validation_jax,
                    component_weights_jax,
                    validation_sample_weights_jax,
                    field_loss_weight,
                )
            )
            evaluation_epochs.append(step)
            training_losses.append(final_training_loss)
            validation_losses.append(current_validation)
            if current_validation < best_validation_loss:
                best_validation_loss = current_validation
                best_epoch = step
                best_params = cast(Params, jax.tree_util.tree_map(lambda value: value, params))
                without_improvement = 0
            else:
                without_improvement += 1
            logger.info(
                "epoch=%d train_loss=%.8g validation_loss=%.8g best_epoch=%d",
                step,
                final_training_loss,
                current_validation,
                best_epoch,
            )
        checkpoint = step % checkpoint_every == 0 or step == epochs
        if checkpoint or (evaluate and without_improvement >= early_stopping_patience):
            _save_optimizer_checkpoint(
                checkpoint_dir,
                identity=identity,
                params=params,
                first_moment=first_moment,
                second_moment=second_moment,
                best_params=best_params,
                completed_epochs=completed_epochs,
                final_training_loss=final_training_loss,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                evaluations_without_improvement=without_improvement,
                evaluation_epochs=evaluation_epochs,
                training_losses=training_losses,
                validation_losses=validation_losses,
            )
        if evaluate and without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    if best_epoch < 1 or not np.isfinite(best_validation_loss):
        raise RuntimeError("training produced no finite validation selection")

    # These partitions are first transformed and inspected only after the
    # validation-selected parameter state is frozen.
    calibration_latent = _encode_fields(
        pca, data.fields, calibration_indices, chunk_rows=pca_chunk_rows
    )
    test_latent = _encode_fields(pca, data.fields, test_indices, chunk_rows=pca_chunk_rows)
    x_calibration = np.asarray(data.inputs[calibration_indices], dtype=np.float64)
    x_test = np.asarray(data.inputs[test_indices], dtype=np.float64)
    x_calibration_norm = np.asarray((x_calibration - input_mean) / input_std, dtype=np.float64)
    x_test_norm = np.asarray((x_test - input_mean) / input_std, dtype=np.float64)

    pca_validation_metrics, _ = _field_metrics(
        pca=pca,
        params=None,
        normalized_inputs=None,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fields=data.fields,
        indices=validation_indices,
        true_latent=validation_latent,
        chunk_rows=pca_chunk_rows,
    )
    validation_metrics, _ = _field_metrics(
        pca=pca,
        params=best_params,
        normalized_inputs=x_validation_norm,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fields=data.fields,
        indices=validation_indices,
        true_latent=validation_latent,
        chunk_rows=pca_chunk_rows,
    )
    calibration_metrics, calibration_scores = _field_metrics(
        pca=pca,
        params=best_params,
        normalized_inputs=x_calibration_norm,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fields=data.fields,
        indices=calibration_indices,
        true_latent=calibration_latent,
        chunk_rows=pca_chunk_rows,
    )
    pca_test_metrics, _ = _field_metrics(
        pca=pca,
        params=None,
        normalized_inputs=None,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fields=data.fields,
        indices=test_indices,
        true_latent=test_latent,
        chunk_rows=pca_chunk_rows,
    )
    test_metrics, test_scores = _field_metrics(
        pca=pca,
        params=best_params,
        normalized_inputs=x_test_norm,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fields=data.fields,
        indices=test_indices,
        true_latent=test_latent,
        chunk_rows=pca_chunk_rows,
    )
    conformal_alpha = 0.05
    conformal_rank = min(
        len(calibration_scores),
        int(np.ceil((len(calibration_scores) + 1) * (1.0 - conformal_alpha))),
    )
    conformal_bound = float(np.sort(calibration_scores)[conformal_rank - 1])
    conformal_test_coverage = float(np.mean(test_scores <= conformal_bound))
    weights_payload: dict[str, Any] = {
        "n_components": np.asarray([actual_components], dtype=np.int64),
        "grid_nh": np.asarray([data.grid_shape[0]], dtype=np.int64),
        "grid_nw": np.asarray([data.grid_shape[1]], dtype=np.int64),
        "n_input_features": np.asarray([17], dtype=np.int64),
        "pca_mean": pca.mean,
        "pca_components": pca.components,
        "pca_evr": pca.explained_variance_ratio,
        "input_mean": input_mean,
        "input_std": input_std,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "n_layers": np.asarray([len(best_params)], dtype=np.int64),
        "feature_names": np.asarray(data.feature_names),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "training_schema": np.asarray([TRAINING_SCHEMA]),
        "selected_epoch": np.asarray([best_epoch], dtype=np.int64),
        "field_loss_weight": np.asarray([field_loss_weight], dtype=np.float64),
        "train_indices_sha256": np.asarray([array_sha256(train_indices)]),
        "validation_indices_sha256": np.asarray([array_sha256(validation_indices)]),
        "calibration_indices_sha256": np.asarray([array_sha256(calibration_indices)]),
        "test_indices_sha256": np.asarray([array_sha256(test_indices)]),
        "source_sha256_names": identity["source_sha256_names"],
        "source_sha256_values": identity["source_sha256_values"],
    }
    for index, layer in enumerate(best_params):
        weights_payload[f"w{index}"] = np.asarray(layer["W"], dtype=np.float64)
        weights_payload[f"b{index}"] = np.asarray(layer["b"], dtype=np.float64)
    atomic_savez(output_path, weights_payload)
    accelerator = NeuralEquilibriumAccelerator()
    accelerator.load_weights(output_path)
    runtime_prediction = accelerator.predict(input_mean)
    reference_latent_norm = np.asarray(model_forward(best_params, jnp.zeros(17, dtype=jnp.float64)))
    reference_prediction = pca.inverse_transform(
        (reference_latent_norm * latent_std + latent_mean)[np.newaxis, :]
    ).reshape(data.grid_shape)
    runtime_parity = float(np.max(np.abs(runtime_prediction - reference_prediction)))
    if runtime_prediction.shape != data.grid_shape or not np.all(np.isfinite(runtime_prediction)):
        raise RuntimeError("candidate failed production runtime load/predict")
    if runtime_parity > 1.0e-8:
        raise RuntimeError(f"runtime/training path mismatch: {runtime_parity}")
    optimizer_recovery = checked_json_load(
        optimizer_recovery_path, max_bytes=MAX_OPTIMIZER_RECOVERY_BYTES
    )
    if not isinstance(optimizer_recovery, dict):
        raise RuntimeError("completed optimizer recovery state is invalid")
    report["recovery"].update(
        {
            "pca_checkpoint_sha256": pca_checkpoint_sha256,
            "optimizer_stage_file": optimizer_recovery["stage_file"],
            "optimizer_stage_sha256": optimizer_recovery["stage_sha256"],
            "optimizer_completed_epochs": optimizer_recovery["completed_epochs"],
        }
    )
    elapsed = time.perf_counter() - started
    report.update(
        {
            "status": "completed_local_candidate_not_promoted",
            "pca": {
                "method": "streaming_randomized_svd_train_only",
                "components": actual_components,
                "oversampling": pca_oversampling,
                "power_iterations": pca_power_iterations,
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio)),
                "checkpoint_sha256": pca_checkpoint_sha256,
                "validation_reconstruction": pca_validation_metrics,
                "test_reconstruction": pca_test_metrics,
            },
            "training": {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "requested_epochs": epochs,
                "completed_epochs": completed_epochs,
                "selected_epoch": best_epoch,
                "stopped_early": stopped_early,
                "final_training_loss": final_training_loss,
                "selection_objective": "hybrid_relative_field_aligned_and_standardized_latent_mse",
                "field_loss_weight": field_loss_weight,
                "standardized_latent_loss_weight": 1.0 - field_loss_weight,
                "component_weight_contract": "latent_variance_over_mean_training_latent_variance",
                "sample_weight_contract": "mean_training_field_norm_squared_over_row_field_norm_squared",
                "field_norm_reference": field_norm_reference,
                "component_weight_scale": component_weight_scale,
                "best_validation_objective": best_validation_loss,
                "evaluation_epochs": evaluation_epochs,
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "hidden_sizes": list(hidden_sizes),
                "learning_rate": learning_rate,
                "gradient_clip": gradient_clip,
                "evaluation_every": evaluation_every,
                "early_stopping_patience": early_stopping_patience,
                "source_sha256": {
                    str(name): str(value)
                    for name, value in zip(
                        identity["source_sha256_names"],
                        identity["source_sha256_values"],
                        strict=True,
                    )
                },
                "elapsed_seconds": elapsed,
            },
            "held_out_validation": validation_metrics,
            "post_selection_calibration": calibration_metrics,
            "untouched_final_test": test_metrics,
            "conformal_relative_l2": {
                "alpha": conformal_alpha,
                "calibration_samples": len(calibration_scores),
                "finite_sample_rank_one_based": conformal_rank,
                "bound": conformal_bound,
                "test_samples": len(test_scores),
                "test_empirical_coverage": conformal_test_coverage,
                "contract": "calibration relative-L2 nonconformity after validation selection",
            },
            "artifact": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "promotion_status": "local_candidate_not_promoted",
                "runtime_load_predict_finite": True,
                "runtime_prediction_shape": list(runtime_prediction.shape),
                "runtime_training_path_parity_max_abs": runtime_parity,
            },
        }
    )
    atomic_json(report_path, report)
    return report


def main() -> None:
    run_training_cli(
        run_training,
        default_epochs=20_000,
        default_pca_components=DEFAULT_PCA_COMPONENTS,
        default_pca_oversampling=DEFAULT_PCA_OVERSAMPLING,
        default_pca_power_iterations=DEFAULT_PCA_POWER_ITERATIONS,
    )


if __name__ == "__main__":
    main()
