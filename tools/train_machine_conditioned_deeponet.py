# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned DeepONet Trainer
"""Train a recovery-safe fixed-machine equilibrium branch-trunk operator."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad
from numpy.typing import NDArray

from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import sha256_file
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedTrainingData,
    array_sha256,
    atomic_json,
    atomic_savez,
    deterministic_four_way_split,
    iter_field_rows,
    load_machine_conditioned_training_data,
)
from scpn_fusion.io.safe_loaders import checked_json_load

logger = logging.getLogger(__name__)
FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
Layer: TypeAlias = dict[str, jax.Array]
Params: TypeAlias = list[Layer]
OperatorParams: TypeAlias = dict[str, Params]

TRAINING_SCHEMA = "scpn-fusion.machine-conditioned-equilibrium-deeponet-training.v1"
ARTIFACT_SCHEMA = "scpn-fusion.equilibrium-deeponet.v1"
STATISTICS_SCHEMA = "scpn-fusion.equilibrium-deeponet-statistics.v1"
OPTIMIZER_SCHEMA = "scpn-fusion.equilibrium-deeponet-adam.v1"
MAX_RECOVERY_BYTES = 2 * 1024 * 1024
DEFAULT_BRANCH_HIDDEN = (256, 256)
DEFAULT_TRUNK_HIDDEN = (128, 128)
DEFAULT_BASIS_WIDTH = 64
REPO_ROOT = Path(__file__).resolve().parents[1]


def init_network(
    key: jax.Array,
    *,
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
) -> Params:
    """Create deterministic float32 He-initialized SiLU network parameters."""
    dimensions = (input_dim, *hidden_sizes, output_dim)
    params: Params = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True):
        key, subkey = random.split(key)
        params.append(
            {
                "W": random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32)
                * jnp.sqrt(2.0 / fan_in),
                "b": jnp.zeros(fan_out, dtype=jnp.float32),
            }
        )
    return params


def network_forward(params: Params, values: jax.Array) -> jax.Array:
    """Evaluate a dense SiLU network over a row batch."""
    activation = values
    for index, layer in enumerate(params):
        activation = activation @ layer["W"] + layer["b"]
        if index + 1 < len(params):
            activation = jax.nn.silu(activation)
    return activation


def operator_forward(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
) -> jax.Array:
    """Evaluate branch coefficients against learned coordinate basis values."""
    branch = network_forward(params["branch"], features)
    trunk = network_forward(params["trunk"], coordinates)
    return branch @ trunk.T / jnp.sqrt(branch.shape[1])


def relative_field_objective(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
) -> jax.Array:
    """Return field-normalized coordinate-sampled physical error."""
    squared_error = jnp.square(operator_forward(params, features, coordinates) - targets)
    return jnp.mean(sample_weights * jnp.mean(squared_error, axis=1))


@jax.jit
def adamw_step(
    params: OperatorParams,
    first_moment: OperatorParams,
    second_moment: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    step: int,
) -> tuple[OperatorParams, OperatorParams, OperatorParams, jax.Array]:
    """Run one deterministic AdamW step with global gradient clipping."""
    loss, gradients = value_and_grad(relative_field_objective)(
        params, features, coordinates, targets, sample_weights
    )
    gradient_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(gradients))
    )
    gradient_scale = jnp.minimum(1.0, gradient_clip / jnp.maximum(gradient_norm, 1.0e-12))
    gradients = cast(
        OperatorParams,
        jax.tree_util.tree_map(lambda gradient: gradient * gradient_scale, gradients),
    )
    beta_1, beta_2, epsilon = 0.9, 0.999, 1.0e-8
    first_moment = cast(
        OperatorParams,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_1 * moment + (1.0 - beta_1) * gradient,
            first_moment,
            gradients,
        ),
    )
    second_moment = cast(
        OperatorParams,
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
        OperatorParams,
        jax.tree_util.tree_map(
            lambda parameter, moment_1, moment_2: (
                parameter
                - learning_rate
                * (moment_1 / (jnp.sqrt(moment_2) + epsilon) + weight_decay * parameter)
            ),
            params,
            corrected_first,
            corrected_second,
        ),
    )
    return params, first_moment, second_moment, loss


@jax.jit
def validation_objective(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
) -> jax.Array:
    """Evaluate a frozen validation probe used only for model selection."""
    return relative_field_objective(params, features, coordinates, targets, sample_weights)


def _serialize_network(payload: dict[str, Any], prefix: str, params: Params) -> None:
    payload[f"{prefix}_n_layers"] = np.asarray([len(params)], dtype=np.int64)
    for index, layer in enumerate(params):
        payload[f"{prefix}_{index}_W"] = np.asarray(layer["W"], dtype=np.float64)
        payload[f"{prefix}_{index}_b"] = np.asarray(layer["b"], dtype=np.float64)


def _deserialize_network(archive: Any, prefix: str) -> Params:
    return [
        {
            "W": jnp.asarray(archive[f"{prefix}_{index}_W"], dtype=jnp.float32),
            "b": jnp.asarray(archive[f"{prefix}_{index}_b"], dtype=jnp.float32),
        }
        for index in range(int(archive[f"{prefix}_n_layers"][0]))
    ]


def _serialize_operator(payload: dict[str, Any], prefix: str, params: OperatorParams) -> None:
    _serialize_network(payload, f"{prefix}_branch", params["branch"])
    _serialize_network(payload, f"{prefix}_trunk", params["trunk"])


def _deserialize_operator(archive: Any, prefix: str) -> OperatorParams:
    return {
        "branch": _deserialize_network(archive, f"{prefix}_branch"),
        "trunk": _deserialize_network(archive, f"{prefix}_trunk"),
    }


def _validate_identity(archive: Any, identity: Mapping[str, Any]) -> None:
    for name, value in identity.items():
        if name not in archive or not np.array_equal(archive[name], value):
            raise ValueError(f"DeepONet recovery identity mismatch for {name}")


def _load_coordinates(data: MachineConditionedTrainingData) -> FloatArray:
    arrays = cast(dict[str, Any], data.manifest["arrays"])
    r_spec = cast(dict[str, Any], arrays["grid_r_m"])
    z_spec = cast(dict[str, Any], arrays["grid_z_m"])
    r = np.load(data.root / str(r_spec["file"]), allow_pickle=False)
    z = np.load(data.root / str(z_spec["file"]), allow_pickle=False)
    if r.shape != (data.grid_shape[1],) or z.shape != (data.grid_shape[0],):
        raise ValueError("DeepONet coordinate arrays do not match the field grid")
    grid_r, grid_z = np.meshgrid(r, z, indexing="xy")
    return np.asarray(np.column_stack((grid_r.ravel(), grid_z.ravel())), dtype=np.float64)


def _training_statistics(
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    *,
    chunk_rows: int,
) -> tuple[FloatArray, float, FloatArray]:
    width = int(np.prod(data.grid_shape))
    field_sum = np.zeros(width, dtype=np.float64)
    field_sum_squared = 0.0
    field_norm_squared = np.empty(len(train_indices), dtype=np.float64)
    for start, stop, rows in iter_field_rows(data.fields, train_indices, chunk_rows=chunk_rows):
        field_sum += np.sum(rows, axis=0)
        row_norms = np.einsum("ij,ij->i", rows, rows)
        field_norm_squared[start:stop] = row_norms
        field_sum_squared += float(np.sum(row_norms))
    field_mean = field_sum / len(train_indices)
    centered_sum_squared = max(
        field_sum_squared - len(train_indices) * float(field_mean @ field_mean), 1.0e-30
    )
    field_scale = float(np.sqrt(centered_sum_squared / (len(train_indices) * width)))
    return field_mean, field_scale, np.maximum(field_norm_squared, 1.0e-30)


def _statistics_identity(
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    source_paths: tuple[Path, ...],
) -> dict[str, Any]:
    return {
        "statistics_schema": np.asarray([STATISTICS_SCHEMA]),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "fields_sha256": np.asarray([data.fields_sha256]),
        "train_indices_sha256": np.asarray([array_sha256(train_indices)]),
        "source_sha256_names": np.asarray(
            [str(path.relative_to(REPO_ROOT)) for path in source_paths]
        ),
        "source_sha256_values": np.asarray([sha256_file(path) for path in source_paths]),
    }


def _load_or_compute_statistics(
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    *,
    chunk_rows: int,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    resume: bool,
) -> tuple[FloatArray, float, FloatArray, str]:
    state_path = checkpoint_dir / "statistics_recovery.json"
    statistics_path = checkpoint_dir / "statistics.npz"
    if resume:
        raw = checked_json_load(state_path, max_bytes=MAX_RECOVERY_BYTES)
        if not isinstance(raw, dict):
            raise ValueError("DeepONet statistics recovery must be an object")
        recovery = cast(dict[str, Any], raw)
        digest = recovery.get("sha256")
        if recovery.get("schema_version") != STATISTICS_SCHEMA or not isinstance(digest, str):
            raise ValueError("DeepONet statistics recovery metadata is invalid")
        if sha256_file(statistics_path) != digest:
            raise ValueError("DeepONet statistics recovery SHA-256 mismatch")
        with np.load(statistics_path, allow_pickle=False) as archive:
            _validate_identity(archive, identity)
            return (
                np.asarray(archive["field_mean"], dtype=np.float64),
                float(archive["field_scale"][0]),
                np.asarray(archive["field_norm_squared"], dtype=np.float64),
                digest,
            )
    field_mean, field_scale, field_norm_squared = _training_statistics(
        data, train_indices, chunk_rows=chunk_rows
    )
    payload = dict(identity)
    payload.update(
        {
            "field_mean": field_mean,
            "field_scale": np.asarray([field_scale]),
            "field_norm_squared": field_norm_squared,
        }
    )
    atomic_savez(statistics_path, payload)
    digest = sha256_file(statistics_path)
    atomic_json(
        state_path,
        {"schema_version": STATISTICS_SCHEMA, "file": statistics_path.name, "sha256": digest},
    )
    return field_mean, field_scale, field_norm_squared, digest


def _extract_targets(
    data: MachineConditionedTrainingData,
    sample_indices: IndexArray,
    coordinate_indices: IndexArray,
    field_mean: FloatArray,
    field_scale: float,
) -> FloatArray:
    n_r = data.grid_shape[1]
    z_indices = coordinate_indices // n_r
    r_indices = coordinate_indices % n_r
    values = data.fields[
        sample_indices[:, np.newaxis], z_indices[np.newaxis, :], r_indices[np.newaxis, :]
    ]
    return np.asarray(
        (values - field_mean[coordinate_indices][np.newaxis, :]) / field_scale,
        dtype=np.float64,
    )


def _deterministic_probe(
    *,
    seed: int,
    sample_count: int,
    coordinate_count: int,
    available_samples: int,
    available_coordinates: int,
) -> tuple[IndexArray, IndexArray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD33F]))
    samples = np.sort(
        rng.choice(available_samples, size=min(sample_count, available_samples), replace=False)
    )
    coordinates = np.sort(
        rng.choice(
            available_coordinates,
            size=min(coordinate_count, available_coordinates),
            replace=False,
        )
    )
    return np.asarray(samples, dtype=np.int64), np.asarray(coordinates, dtype=np.int64)


def _training_batch(
    *,
    step: int,
    seed: int,
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    normalized_inputs: FloatArray,
    normalized_coordinates: FloatArray,
    field_mean: FloatArray,
    field_scale: float,
    sample_weights: FloatArray,
    shot_batch_size: int,
    coordinate_batch_size: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    shot_positions = np.asarray(
        rng.choice(
            len(train_indices), size=min(shot_batch_size, len(train_indices)), replace=False
        ),
        dtype=np.int64,
    )
    coordinate_indices = np.asarray(
        rng.choice(
            len(normalized_coordinates),
            size=min(coordinate_batch_size, len(normalized_coordinates)),
            replace=False,
        ),
        dtype=np.int64,
    )
    targets = _extract_targets(
        data,
        train_indices[shot_positions],
        coordinate_indices,
        field_mean,
        field_scale,
    )
    return (
        normalized_inputs[shot_positions],
        normalized_coordinates[coordinate_indices],
        targets,
        sample_weights[shot_positions],
    )


def _optimizer_identity(
    *,
    data: MachineConditionedTrainingData,
    split_hashes: Mapping[str, str],
    statistics_sha256: str,
    seed: int,
    branch_hidden: tuple[int, ...],
    trunk_hidden: tuple[int, ...],
    basis_width: int,
    shot_batch_size: int,
    coordinate_batch_size: int,
    validation_probe_samples: IndexArray,
    validation_probe_coordinates: IndexArray,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    evaluation_every: int,
    early_stopping_patience: int,
    source_paths: tuple[Path, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "optimizer_schema": np.asarray([OPTIMIZER_SCHEMA]),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "statistics_sha256": np.asarray([statistics_sha256]),
        "seed": np.asarray([seed], dtype=np.int64),
        "branch_hidden": np.asarray(branch_hidden, dtype=np.int64),
        "trunk_hidden": np.asarray(trunk_hidden, dtype=np.int64),
        "basis_width": np.asarray([basis_width], dtype=np.int64),
        "shot_batch_size": np.asarray([shot_batch_size], dtype=np.int64),
        "coordinate_batch_size": np.asarray([coordinate_batch_size], dtype=np.int64),
        "validation_probe_samples_sha256": np.asarray([array_sha256(validation_probe_samples)]),
        "validation_probe_coordinates_sha256": np.asarray(
            [array_sha256(validation_probe_coordinates)]
        ),
        "learning_rate": np.asarray([learning_rate]),
        "weight_decay": np.asarray([weight_decay]),
        "gradient_clip": np.asarray([gradient_clip]),
        "evaluation_every": np.asarray([evaluation_every], dtype=np.int64),
        "early_stopping_patience": np.asarray([early_stopping_patience], dtype=np.int64),
        "source_sha256_names": np.asarray(
            [str(path.relative_to(REPO_ROOT)) for path in source_paths]
        ),
        "source_sha256_values": np.asarray([sha256_file(path) for path in source_paths]),
    }
    for role, digest in split_hashes.items():
        payload[f"{role}_indices_sha256"] = np.asarray([digest])
    return payload


def _save_optimizer(
    checkpoint_dir: Path,
    *,
    identity: Mapping[str, Any],
    params: OperatorParams,
    first_moment: OperatorParams,
    second_moment: OperatorParams,
    best_params: OperatorParams,
    completed_steps: int,
    final_training_loss: float,
    best_validation_loss: float,
    best_step: int,
    evaluations_without_improvement: int,
    evaluation_steps: list[int],
    training_losses: list[float],
    validation_losses: list[float],
) -> None:
    payload: dict[str, Any] = dict(identity)
    payload.update(
        {
            "completed_steps": np.asarray([completed_steps], dtype=np.int64),
            "final_training_loss": np.asarray([final_training_loss]),
            "best_validation_loss": np.asarray([best_validation_loss]),
            "best_step": np.asarray([best_step], dtype=np.int64),
            "evaluations_without_improvement": np.asarray(
                [evaluations_without_improvement], dtype=np.int64
            ),
            "evaluation_steps": np.asarray(evaluation_steps, dtype=np.int64),
            "training_losses": np.asarray(training_losses),
            "validation_losses": np.asarray(validation_losses),
        }
    )
    _serialize_operator(payload, "current", params)
    _serialize_operator(payload, "first", first_moment)
    _serialize_operator(payload, "second", second_moment)
    _serialize_operator(payload, "best", best_params)
    stage = checkpoint_dir / f"optimizer_step_{completed_steps:08d}.npz"
    atomic_savez(stage, payload)
    atomic_json(
        checkpoint_dir / "optimizer_recovery.json",
        {
            "schema_version": OPTIMIZER_SCHEMA,
            "completed_steps": completed_steps,
            "stage_file": stage.name,
            "stage_sha256": sha256_file(stage),
        },
    )


def _load_optimizer(
    checkpoint_dir: Path, *, identity: Mapping[str, Any]
) -> tuple[
    OperatorParams,
    OperatorParams,
    OperatorParams,
    OperatorParams,
    int,
    float,
    float,
    int,
    int,
    list[int],
    list[float],
    list[float],
]:
    raw = checked_json_load(
        checkpoint_dir / "optimizer_recovery.json", max_bytes=MAX_RECOVERY_BYTES
    )
    if not isinstance(raw, dict):
        raise ValueError("DeepONet optimizer recovery must be an object")
    recovery = cast(dict[str, Any], raw)
    filename = recovery.get("stage_file")
    digest = recovery.get("stage_sha256")
    expected_steps = recovery.get("completed_steps")
    if (
        recovery.get("schema_version") != OPTIMIZER_SCHEMA
        or not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(digest, str)
        or not isinstance(expected_steps, int)
    ):
        raise ValueError("DeepONet optimizer recovery metadata is invalid")
    stage = checkpoint_dir / filename
    if not stage.is_file() or stage.is_symlink() or sha256_file(stage) != digest:
        raise ValueError("DeepONet optimizer recovery SHA-256 mismatch")
    with np.load(stage, allow_pickle=False) as archive:
        _validate_identity(archive, identity)
        result = (
            _deserialize_operator(archive, "current"),
            _deserialize_operator(archive, "first"),
            _deserialize_operator(archive, "second"),
            _deserialize_operator(archive, "best"),
            int(archive["completed_steps"][0]),
            float(archive["final_training_loss"][0]),
            float(archive["best_validation_loss"][0]),
            int(archive["best_step"][0]),
            int(archive["evaluations_without_improvement"][0]),
            np.asarray(archive["evaluation_steps"], dtype=np.int64).tolist(),
            np.asarray(archive["training_losses"], dtype=np.float64).tolist(),
            np.asarray(archive["validation_losses"], dtype=np.float64).tolist(),
        )
    if result[4] != expected_steps:
        raise ValueError("DeepONet optimizer recovery step mismatch")
    return result


def _field_metrics(
    runtime: DeepONetEquilibriumAccelerator,
    data: MachineConditionedTrainingData,
    indices: IndexArray,
    *,
    chunk_rows: int,
) -> tuple[dict[str, float], FloatArray]:
    squared_error = 0.0
    elements = 0
    scores: list[float] = []
    for start in range(0, len(indices), chunk_rows):
        stop = min(start + chunk_rows, len(indices))
        rows = indices[start:stop]
        predicted = runtime.predict_batch(np.asarray(data.inputs[rows], dtype=np.float64))
        truth = np.asarray(data.fields[rows], dtype=np.float64)
        delta = predicted - truth
        squared_error += float(np.sum(delta * delta))
        elements += delta.size
        numerator = np.linalg.norm(delta.reshape(len(rows), -1), axis=1)
        denominator = np.maximum(np.linalg.norm(truth.reshape(len(rows), -1), axis=1), 1.0e-15)
        scores.extend(np.asarray(numerator / denominator).tolist())
    values = np.asarray(scores, dtype=np.float64)
    return (
        {
            "field_rmse": float(np.sqrt(squared_error / max(elements, 1))),
            "mean_relative_l2": float(np.mean(values)),
            "p95_relative_l2": float(np.percentile(values, 95.0)),
            "max_relative_l2": float(np.max(values)),
        },
        values,
    )


def run_training(
    *,
    dataset_dir: Path,
    output_path: Path,
    report_path: Path,
    checkpoint_dir: Path,
    steps: int,
    seed: int = 42,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.05,
    branch_hidden: tuple[int, ...] = DEFAULT_BRANCH_HIDDEN,
    trunk_hidden: tuple[int, ...] = DEFAULT_TRUNK_HIDDEN,
    basis_width: int = DEFAULT_BASIS_WIDTH,
    shot_batch_size: int = 256,
    coordinate_batch_size: int = 512,
    validation_probe_shots: int = 1024,
    validation_probe_coordinates: int = 2048,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-6,
    gradient_clip: float = 1.0,
    statistics_chunk_rows: int = 256,
    evaluation_every: int = 250,
    checkpoint_every: int = 500,
    early_stopping_patience: int = 40,
    resume: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one local, manifest-bound fixed-machine DeepONet."""
    positive = (
        steps,
        basis_width,
        shot_batch_size,
        coordinate_batch_size,
        validation_probe_shots,
        validation_probe_coordinates,
        statistics_chunk_rows,
        evaluation_every,
        checkpoint_every,
        early_stopping_patience,
    )
    if any(value < 1 for value in positive) or any(value < 1 for value in branch_hidden):
        raise ValueError("DeepONet integer hyperparameters must be positive")
    if any(value < 1 for value in trunk_hidden):
        raise ValueError("DeepONet trunk widths must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or gradient_clip <= 0.0:
        raise ValueError("DeepONet optimizer hyperparameters are invalid")

    started = time.perf_counter()
    data = load_machine_conditioned_training_data(dataset_dir, full_field_scan=True)
    split = deterministic_four_way_split(
        len(data.inputs),
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_hashes = {
        role: array_sha256(indices)
        for role, indices in {
            "training": split.training,
            "validation": split.validation,
            "calibration": split.calibration,
            "test": split.test,
        }.items()
    }
    coordinates = _load_coordinates(data)
    coordinate_mean = np.mean(coordinates, axis=0)
    coordinate_std = np.std(coordinates, axis=0)
    input_rows = np.asarray(data.inputs[split.training], dtype=np.float64)
    input_mean = np.mean(input_rows, axis=0)
    raw_input_std = np.std(input_rows, axis=0)
    input_std = np.where(raw_input_std < 1.0e-12, 1.0, raw_input_std)
    normalized_inputs = np.asarray((input_rows - input_mean) / input_std, dtype=np.float64)
    normalized_coordinates = np.asarray(
        (coordinates - coordinate_mean) / coordinate_std, dtype=np.float64
    )
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/scpn_fusion/core/deeponet_equilibrium.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_training.py",
    )
    statistics_identity = _statistics_identity(data, split.training, source_paths)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    field_mean, field_scale, train_field_norm_squared, statistics_sha256 = (
        _load_or_compute_statistics(
            data,
            split.training,
            chunk_rows=statistics_chunk_rows,
            checkpoint_dir=checkpoint_dir,
            identity=statistics_identity,
            resume=resume,
        )
    )
    field_norm_reference = float(np.mean(train_field_norm_squared))
    train_sample_weights = np.asarray(
        field_norm_reference / train_field_norm_squared, dtype=np.float64
    )
    probe_positions, probe_coordinate_indices = _deterministic_probe(
        seed=seed,
        sample_count=validation_probe_shots,
        coordinate_count=validation_probe_coordinates,
        available_samples=len(split.validation),
        available_coordinates=len(coordinates),
    )
    probe_indices = split.validation[probe_positions]
    probe_features = np.asarray(
        (np.asarray(data.inputs[probe_indices]) - input_mean) / input_std, dtype=np.float64
    )
    probe_coordinates = normalized_coordinates[probe_coordinate_indices]
    probe_targets = _extract_targets(
        data, probe_indices, probe_coordinate_indices, field_mean, field_scale
    )
    probe_norms = np.empty(len(probe_indices), dtype=np.float64)
    for start, stop, rows in iter_field_rows(
        data.fields,
        np.sort(probe_indices),
        chunk_rows=statistics_chunk_rows,
    ):
        probe_norms[start:stop] = np.einsum("ij,ij->i", rows, rows)
    # iter_field_rows requires sorted indices, while probe feature/target order
    # follows the sorted deterministic positions and is therefore already sorted.
    probe_sample_weights = np.asarray(
        field_norm_reference / np.maximum(probe_norms, 1.0e-30), dtype=np.float64
    )
    identity = _optimizer_identity(
        data=data,
        split_hashes=split_hashes,
        statistics_sha256=statistics_sha256,
        seed=seed,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        basis_width=basis_width,
        shot_batch_size=shot_batch_size,
        coordinate_batch_size=coordinate_batch_size,
        validation_probe_samples=probe_indices,
        validation_probe_coordinates=probe_coordinate_indices,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        evaluation_every=evaluation_every,
        early_stopping_patience=early_stopping_patience,
        source_paths=source_paths,
    )
    report: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA,
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
            "seed": seed,
            "fractions": {
                "validation": validation_fraction,
                "calibration": calibration_fraction,
                "test": test_fraction,
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
            "checkpoint_dir": str(checkpoint_dir),
            "resume_requested": resume,
            "statistics_sha256": statistics_sha256,
        },
    }
    atomic_json(report_path, report)

    if resume and (checkpoint_dir / "optimizer_recovery.json").exists():
        (
            params,
            first_moment,
            second_moment,
            best_params,
            completed_steps,
            final_training_loss,
            best_validation_loss,
            best_step,
            without_improvement,
            evaluation_steps,
            training_losses,
            validation_losses,
        ) = _load_optimizer(checkpoint_dir, identity=identity)
        if completed_steps > steps:
            raise ValueError("DeepONet recovery exceeds the requested step target")
    else:
        branch_key, trunk_key = random.split(random.PRNGKey(seed))
        params = {
            "branch": init_network(
                branch_key,
                input_dim=len(data.feature_names),
                hidden_sizes=branch_hidden,
                output_dim=basis_width,
            ),
            "trunk": init_network(
                trunk_key,
                input_dim=2,
                hidden_sizes=trunk_hidden,
                output_dim=basis_width,
            ),
        }
        first_moment = cast(OperatorParams, jax.tree_util.tree_map(jnp.zeros_like, params))
        second_moment = cast(OperatorParams, jax.tree_util.tree_map(jnp.zeros_like, params))
        best_params = cast(OperatorParams, jax.tree_util.tree_map(lambda value: value, params))
        completed_steps = 0
        final_training_loss = float("nan")
        best_validation_loss = float("inf")
        best_step = 0
        without_improvement = 0
        evaluation_steps, training_losses, validation_losses = [], [], []

    probe_jax = tuple(
        jnp.asarray(values, dtype=jnp.float32)
        for values in (
            probe_features,
            probe_coordinates,
            probe_targets,
            probe_sample_weights,
        )
    )
    stopped_early = without_improvement >= early_stopping_patience
    first_step = steps if stopped_early else completed_steps
    for zero_based_step in range(first_step, steps):
        step = zero_based_step + 1
        batch = _training_batch(
            step=step,
            seed=seed,
            data=data,
            train_indices=split.training,
            normalized_inputs=normalized_inputs,
            normalized_coordinates=normalized_coordinates,
            field_mean=field_mean,
            field_scale=field_scale,
            sample_weights=train_sample_weights,
            shot_batch_size=shot_batch_size,
            coordinate_batch_size=coordinate_batch_size,
        )
        batch_jax = tuple(jnp.asarray(values, dtype=jnp.float32) for values in batch)
        params, first_moment, second_moment, loss = adamw_step(
            params,
            first_moment,
            second_moment,
            *batch_jax,
            learning_rate,
            weight_decay,
            gradient_clip,
            step,
        )
        final_training_loss = float(loss)
        completed_steps = step
        evaluate = step == 1 or step % evaluation_every == 0 or step == steps
        if evaluate:
            current_validation = float(validation_objective(params, *probe_jax))
            evaluation_steps.append(step)
            training_losses.append(final_training_loss)
            validation_losses.append(current_validation)
            if current_validation < best_validation_loss:
                best_validation_loss = current_validation
                best_step = step
                best_params = cast(
                    OperatorParams, jax.tree_util.tree_map(lambda value: value, params)
                )
                without_improvement = 0
            else:
                without_improvement += 1
            logger.info(
                "step=%d train_loss=%.8g validation_probe_loss=%.8g best_step=%d",
                step,
                final_training_loss,
                current_validation,
                best_step,
            )
        checkpoint = step % checkpoint_every == 0 or step == steps
        if checkpoint or (evaluate and without_improvement >= early_stopping_patience):
            _save_optimizer(
                checkpoint_dir,
                identity=identity,
                params=params,
                first_moment=first_moment,
                second_moment=second_moment,
                best_params=best_params,
                completed_steps=completed_steps,
                final_training_loss=final_training_loss,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
                evaluations_without_improvement=without_improvement,
                evaluation_steps=evaluation_steps,
                training_losses=training_losses,
                validation_losses=validation_losses,
            )
        if evaluate and without_improvement >= early_stopping_patience:
            stopped_early = True
            break
    if best_step < 1 or not np.isfinite(best_validation_loss):
        raise RuntimeError("DeepONet training produced no finite validation selection")

    artifact: dict[str, Any] = {
        "artifact_schema": np.asarray([ARTIFACT_SCHEMA]),
        "input_mean": input_mean,
        "input_std": input_std,
        "coordinates_rz_m": coordinates,
        "coordinate_mean": coordinate_mean,
        "coordinate_std": coordinate_std,
        "field_mean": field_mean,
        "field_scale": np.asarray([field_scale]),
        "basis_width": np.asarray([basis_width], dtype=np.int64),
        "grid_nh": np.asarray([data.grid_shape[0]], dtype=np.int64),
        "grid_nw": np.asarray([data.grid_shape[1]], dtype=np.int64),
        "feature_names": np.asarray(data.feature_names),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "selected_step": np.asarray([best_step], dtype=np.int64),
        "training_schema": np.asarray([TRAINING_SCHEMA]),
        "source_sha256_names": identity["source_sha256_names"],
        "source_sha256_values": identity["source_sha256_values"],
    }
    for role, digest in split_hashes.items():
        artifact[f"{role}_indices_sha256"] = np.asarray([digest])
    _serialize_network(artifact, "branch", best_params["branch"])
    _serialize_network(artifact, "trunk", best_params["trunk"])
    atomic_savez(output_path, artifact)
    runtime = DeepONetEquilibriumAccelerator()
    runtime.load_weights(output_path)
    runtime_prediction = runtime.predict(input_mean)
    reference_normalized = np.asarray(
        operator_forward(
            best_params,
            jnp.zeros((1, len(input_mean)), dtype=jnp.float32),
            jnp.asarray(normalized_coordinates, dtype=jnp.float32),
        )[0],
        dtype=np.float64,
    )
    reference_prediction = (field_mean + field_scale * reference_normalized).reshape(
        data.grid_shape
    )
    runtime_parity = float(np.max(np.abs(runtime_prediction - reference_prediction)))
    if runtime_parity > 1.0e-4:
        raise RuntimeError(f"DeepONet runtime/training parity failed: {runtime_parity}")

    validation_metrics, _ = _field_metrics(runtime, data, split.validation, chunk_rows=64)
    calibration_metrics, calibration_scores = _field_metrics(
        runtime, data, split.calibration, chunk_rows=64
    )
    test_metrics, test_scores = _field_metrics(runtime, data, split.test, chunk_rows=64)
    conformal_alpha = 0.05
    conformal_rank = min(
        len(calibration_scores),
        int(np.ceil((len(calibration_scores) + 1) * (1.0 - conformal_alpha))),
    )
    conformal_bound = float(np.sort(calibration_scores)[conformal_rank - 1])
    elapsed = time.perf_counter() - started
    optimizer_recovery = checked_json_load(
        checkpoint_dir / "optimizer_recovery.json", max_bytes=MAX_RECOVERY_BYTES
    )
    if not isinstance(optimizer_recovery, dict):
        raise RuntimeError("DeepONet optimizer recovery is invalid after completion")
    report.update(
        {
            "status": "completed_local_candidate_not_promoted",
            "architecture": {
                "operator": "DeepONet_branch_trunk_inner_product",
                "activation": "SiLU",
                "branch_inputs": "17_causal_pre_solve_controls",
                "trunk_inputs": "normalized_R_Z_coordinates",
                "branch_hidden": list(branch_hidden),
                "trunk_hidden": list(trunk_hidden),
                "basis_width": basis_width,
                "machine_conditioning": "manifest_bound_single_machine_only",
                "cross_machine_claim": False,
            },
            "training": {
                "precision": "float32_parameters_and_updates_float64_artifact_and_metrics",
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "requested_steps": steps,
                "completed_steps": completed_steps,
                "selected_step": best_step,
                "stopped_early": stopped_early,
                "final_training_loss": final_training_loss,
                "best_validation_probe_loss": best_validation_loss,
                "shot_batch_size": shot_batch_size,
                "coordinate_batch_size": coordinate_batch_size,
                "validation_probe_shots": len(probe_indices),
                "validation_probe_coordinates": len(probe_coordinate_indices),
                "validation_probe_samples_sha256": array_sha256(probe_indices),
                "validation_probe_coordinates_sha256": array_sha256(probe_coordinate_indices),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_clip": gradient_clip,
                "evaluation_every": evaluation_every,
                "evaluation_steps": evaluation_steps,
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "field_mean_and_scale_fit_on": "training_indices_only",
                "relative_sample_weight_reference": field_norm_reference,
                "elapsed_seconds": elapsed,
            },
            "held_out_validation": validation_metrics,
            "post_selection_calibration": calibration_metrics,
            "untouched_final_test": test_metrics,
            "conformal_relative_l2": {
                "alpha": conformal_alpha,
                "finite_sample_rank_one_based": conformal_rank,
                "bound": conformal_bound,
                "test_empirical_coverage": float(np.mean(test_scores <= conformal_bound)),
            },
            "recovery": {
                **report["recovery"],
                "optimizer_stage_file": optimizer_recovery["stage_file"],
                "optimizer_stage_sha256": optimizer_recovery["stage_sha256"],
                "optimizer_completed_steps": optimizer_recovery["completed_steps"],
            },
            "artifact": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "promotion_status": "local_candidate_not_promoted",
                "runtime_load_predict_finite": bool(np.all(np.isfinite(runtime_prediction))),
                "runtime_prediction_shape": list(runtime_prediction.shape),
                "runtime_training_path_parity_max_abs": runtime_parity,
            },
        }
    )
    atomic_json(report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a recovery-safe fixed-machine equilibrium DeepONet."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--basis-width", type=int, default=DEFAULT_BASIS_WIDTH)
    parser.add_argument("--shot-batch-size", type=int, default=256)
    parser.add_argument("--coordinate-batch-size", type=int, default=512)
    parser.add_argument("--validation-probe-shots", type=int, default=1024)
    parser.add_argument("--validation-probe-coordinates", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--statistics-chunk-rows", type=int, default=256)
    parser.add_argument("--evaluation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_training(
        dataset_dir=args.dataset_dir,
        output_path=args.out,
        report_path=args.report,
        checkpoint_dir=args.checkpoint_dir,
        steps=args.steps,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        test_fraction=args.test_fraction,
        basis_width=args.basis_width,
        shot_batch_size=args.shot_batch_size,
        coordinate_batch_size=args.coordinate_batch_size,
        validation_probe_shots=args.validation_probe_shots,
        validation_probe_coordinates=args.validation_probe_coordinates,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        statistics_chunk_rows=args.statistics_chunk_rows,
        evaluation_every=args.evaluation_every,
        checkpoint_every=args.checkpoint_every,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume,
    )
    print(report["status"])


if __name__ == "__main__":
    main()
