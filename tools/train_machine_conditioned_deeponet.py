# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned DeepONet Trainer
"""Orchestrate one recovery-safe fixed-machine equilibrium DeepONet run."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator
from scpn_fusion.core.deeponet_training import (
    OperatorParams,
    adamw_step,
    init_network,
    operator_forward,
    validation_objective,
)
from scpn_fusion.core.deeponet_training_contracts import (
    FloatArray,
    IndexArray,
    PreparedTraining,
    TrainingConfig,
)
from scpn_fusion.io.deeponet_training_data import (
    deterministic_probe,
    extract_targets,
    field_metrics,
    load_coordinates,
    runtime_backend_parity,
    training_batch,
)
from scpn_fusion.io.deeponet_training_recovery import (
    MAX_RECOVERY_BYTES,
    OptimizerState,
    load_optimizer,
    load_or_compute_statistics,
    optimizer_identity,
    save_optimizer,
    statistics_identity,
)
from scpn_fusion.io.deeponet_training_report import (
    artifact_payload,
    completed_report_sections,
    running_report,
)
from scpn_fusion.io.machine_conditioned_deeponet_cli import run_deeponet_cli
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedSplit,
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
TRAINING_SCHEMA = "scpn-fusion.machine-conditioned-equilibrium-deeponet-training.v1"
ARTIFACT_SCHEMA = "scpn-fusion.equilibrium-deeponet.v1"
DEFAULT_BRANCH_HIDDEN = (256, 256)
DEFAULT_TRUNK_HIDDEN = (128, 128)
DEFAULT_BASIS_WIDTH = 64
REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate_config(config: TrainingConfig) -> None:
    integers = (
        config.steps,
        config.basis_width,
        config.shot_batch_size,
        config.coordinate_batch_size,
        config.validation_probe_shots,
        config.validation_probe_coordinates,
        config.statistics_chunk_rows,
        config.evaluation_every,
        config.checkpoint_every,
        config.early_stopping_patience,
        *config.branch_hidden,
        *config.trunk_hidden,
    )
    if any(value < 1 for value in integers):
        raise ValueError("DeepONet integer hyperparameters must be positive")
    continuous = (
        config.validation_fraction,
        config.calibration_fraction,
        config.test_fraction,
        config.learning_rate,
        config.weight_decay,
        config.gradient_clip,
    )
    if not all(np.isfinite(value) for value in continuous):
        raise ValueError("DeepONet continuous hyperparameters must be finite")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("DeepONet learning rate or weight decay is invalid")
    if config.gradient_clip <= 0.0:
        raise ValueError("DeepONet gradient clip must be positive")


def _split_hashes(split: MachineConditionedSplit) -> dict[str, str]:
    roles = {
        "training": split.training,
        "validation": split.validation,
        "calibration": split.calibration,
        "test": split.test,
    }
    return {role: array_sha256(indices) for role, indices in roles.items()}


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        REPO_ROOT / "src/scpn_fusion/core/deeponet_equilibrium.py",
        REPO_ROOT / "src/scpn_fusion/core/deeponet_training.py",
        REPO_ROOT / "src/scpn_fusion/core/deeponet_training_contracts.py",
        REPO_ROOT / "src/scpn_fusion/core/_multi_compat.py",
        REPO_ROOT / "src/scpn_fusion/core/_multi_compat_providers.py",
        REPO_ROOT / "src/scpn_fusion/io/deeponet_training_data.py",
        REPO_ROOT / "src/scpn_fusion/io/deeponet_training_recovery.py",
        REPO_ROOT / "src/scpn_fusion/io/deeponet_training_report.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_deeponet_cli.py",
        REPO_ROOT / "src/scpn_fusion/io/machine_conditioned_surrogate_training.py",
        REPO_ROOT / "src/scpn_fusion/io/safe_loaders.py",
        REPO_ROOT / "scpn-fusion-rs/Cargo.toml",
        REPO_ROOT / "scpn-fusion-rs/Cargo.lock",
        REPO_ROOT / "scpn-fusion-rs/crates/fusion-ml/Cargo.toml",
        REPO_ROOT / "scpn-fusion-rs/crates/fusion-ml/src/deeponet_equilibrium.rs",
        REPO_ROOT / "scpn-fusion-rs/crates/fusion-ml/src/lib.rs",
        REPO_ROOT / "scpn-fusion-rs/crates/fusion-python/src/bindings/ml.rs",
        REPO_ROOT / "scpn-fusion-rs/crates/fusion-python/src/lib.rs",
    )


def _probe_arrays(
    data: MachineConditionedTrainingData,
    split: MachineConditionedSplit,
    config: TrainingConfig,
    *,
    input_mean: FloatArray,
    input_std: FloatArray,
    normalised_coordinates: FloatArray,
    field_mean: FloatArray,
    field_scale: float,
    field_norm_reference: float,
) -> tuple[IndexArray, IndexArray, tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    probe_positions, coordinate_indices = deterministic_probe(
        seed=config.seed,
        sample_count=config.validation_probe_shots,
        coordinate_count=config.validation_probe_coordinates,
        available_samples=len(split.validation),
        available_coordinates=len(normalised_coordinates),
    )
    probe_indices = split.validation[probe_positions]
    features = np.asarray(
        (np.asarray(data.inputs[probe_indices]) - input_mean) / input_std, dtype=np.float64
    )
    targets = extract_targets(data, probe_indices, coordinate_indices, field_mean, field_scale)
    norms = np.empty(len(probe_indices), dtype=np.float64)
    for start, stop, rows in iter_field_rows(
        data.fields, probe_indices, chunk_rows=config.statistics_chunk_rows
    ):
        norms[start:stop] = np.einsum("ij,ij->i", rows, rows)
    weights = np.asarray(field_norm_reference / np.maximum(norms, 1.0e-30), dtype=np.float64)
    return (
        probe_indices,
        coordinate_indices,
        (features, normalised_coordinates[coordinate_indices], targets, weights),
    )


def _prepare_training(config: TrainingConfig) -> PreparedTraining:
    data = load_machine_conditioned_training_data(config.dataset_dir, full_field_scan=True)
    split = deterministic_four_way_split(
        len(data.inputs),
        validation_fraction=config.validation_fraction,
        calibration_fraction=config.calibration_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    split_hashes = _split_hashes(split)
    coordinates = load_coordinates(data)
    coordinate_mean = np.mean(coordinates, axis=0)
    coordinate_std = np.std(coordinates, axis=0)
    if not np.all(np.isfinite(coordinate_std)) or np.any(coordinate_std <= 0.0):
        raise ValueError("DeepONet coordinate scales must be finite and positive")
    normalised_coordinates = np.asarray(
        (coordinates - coordinate_mean) / coordinate_std, dtype=np.float64
    )
    input_rows = np.asarray(data.inputs[split.training], dtype=np.float64)
    input_mean = np.mean(input_rows, axis=0)
    raw_input_std = np.std(input_rows, axis=0)
    input_std = np.where(raw_input_std < 1.0e-12, 1.0, raw_input_std)
    normalised_inputs = np.asarray((input_rows - input_mean) / input_std, dtype=np.float64)
    sources = _source_paths()
    stats_identity = statistics_identity(data, split.training, sources, repo_root=REPO_ROOT)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    field_mean, field_scale, field_norm_squared, statistics_sha256 = load_or_compute_statistics(
        data,
        split.training,
        chunk_rows=config.statistics_chunk_rows,
        checkpoint_dir=config.checkpoint_dir,
        identity=stats_identity,
        resume=config.resume,
    )
    field_norm_reference = float(np.mean(field_norm_squared))
    train_sample_weights = np.asarray(field_norm_reference / field_norm_squared, dtype=np.float64)
    probe_indices, probe_coordinate_indices, probe_arrays = _probe_arrays(
        data,
        split,
        config,
        input_mean=input_mean,
        input_std=input_std,
        normalised_coordinates=normalised_coordinates,
        field_mean=field_mean,
        field_scale=field_scale,
        field_norm_reference=field_norm_reference,
    )
    identity = optimizer_identity(
        data=data,
        split_hashes=split_hashes,
        statistics_sha256=statistics_sha256,
        seed=config.seed,
        branch_hidden=config.branch_hidden,
        trunk_hidden=config.trunk_hidden,
        basis_width=config.basis_width,
        shot_batch_size=config.shot_batch_size,
        coordinate_batch_size=config.coordinate_batch_size,
        validation_probe_samples=probe_indices,
        validation_probe_coordinates=probe_coordinate_indices,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        evaluation_every=config.evaluation_every,
        early_stopping_patience=config.early_stopping_patience,
        source_paths=sources,
        repo_root=REPO_ROOT,
    )
    report = running_report(
        data,
        split,
        split_hashes,
        config,
        statistics_sha256,
        training_schema=TRAINING_SCHEMA,
    )
    atomic_json(config.report_path, report)
    return PreparedTraining(
        data=data,
        split=split,
        split_hashes=split_hashes,
        coordinates=coordinates,
        coordinate_mean=coordinate_mean,
        coordinate_std=coordinate_std,
        normalised_coordinates=normalised_coordinates,
        input_mean=input_mean,
        input_std=input_std,
        normalised_inputs=normalised_inputs,
        field_mean=field_mean,
        field_scale=field_scale,
        field_norm_reference=field_norm_reference,
        train_sample_weights=train_sample_weights,
        probe_indices=probe_indices,
        probe_coordinate_indices=probe_coordinate_indices,
        probe_arrays=probe_arrays,
        identity=identity,
        report=report,
    )


def _new_optimizer_state(config: TrainingConfig, n_features: int) -> OptimizerState:
    branch_key, trunk_key = random.split(random.PRNGKey(config.seed))
    params: OperatorParams = {
        "branch": init_network(
            branch_key,
            input_dim=n_features,
            hidden_sizes=config.branch_hidden,
            output_dim=config.basis_width,
        ),
        "trunk": init_network(
            trunk_key,
            input_dim=2,
            hidden_sizes=config.trunk_hidden,
            output_dim=config.basis_width,
        ),
    }
    zeros = cast(OperatorParams, jax.tree_util.tree_map(jnp.zeros_like, params))
    return OptimizerState(
        params=params,
        first_moment=zeros,
        second_moment=cast(OperatorParams, jax.tree_util.tree_map(jnp.zeros_like, params)),
        best_params=cast(OperatorParams, jax.tree_util.tree_map(lambda value: value, params)),
        completed_steps=0,
        final_training_loss=float("nan"),
        best_validation_loss=float("inf"),
        best_step=0,
        evaluations_without_improvement=0,
        evaluation_steps=[],
        training_losses=[],
        validation_losses=[],
    )


def _optimise(prepared: PreparedTraining, config: TrainingConfig, state: OptimizerState) -> bool:
    probe = tuple(jnp.asarray(values, dtype=jnp.float32) for values in prepared.probe_arrays)
    stopped_early = state.evaluations_without_improvement >= config.early_stopping_patience
    first_step = config.steps if stopped_early else state.completed_steps
    for zero_based_step in range(first_step, config.steps):
        step = zero_based_step + 1
        batch = training_batch(
            step=step,
            seed=config.seed,
            data=prepared.data,
            train_indices=prepared.split.training,
            normalised_inputs=prepared.normalised_inputs,
            normalised_coordinates=prepared.normalised_coordinates,
            field_mean=prepared.field_mean,
            field_scale=prepared.field_scale,
            sample_weights=prepared.train_sample_weights,
            shot_batch_size=config.shot_batch_size,
            coordinate_batch_size=config.coordinate_batch_size,
        )
        jax_batch = tuple(jnp.asarray(values, dtype=jnp.float32) for values in batch)
        state.params, state.first_moment, state.second_moment, loss = adamw_step(
            state.params,
            state.first_moment,
            state.second_moment,
            *jax_batch,
            config.learning_rate,
            config.weight_decay,
            config.gradient_clip,
            step,
        )
        state.final_training_loss = float(loss)
        state.completed_steps = step
        evaluate = step == 1 or step % config.evaluation_every == 0 or step == config.steps
        if evaluate:
            _update_selection(state, float(validation_objective(state.params, *probe)), step)
        checkpoint = step % config.checkpoint_every == 0 or step == config.steps
        if checkpoint or (
            evaluate and state.evaluations_without_improvement >= config.early_stopping_patience
        ):
            save_optimizer(config.checkpoint_dir, identity=prepared.identity, state=state)
        if evaluate and state.evaluations_without_improvement >= config.early_stopping_patience:
            return True
    return stopped_early


def _update_selection(state: OptimizerState, validation_loss: float, step: int) -> None:
    state.evaluation_steps.append(step)
    state.training_losses.append(state.final_training_loss)
    state.validation_losses.append(validation_loss)
    if validation_loss < state.best_validation_loss:
        state.best_validation_loss = validation_loss
        state.best_step = step
        state.best_params = cast(
            OperatorParams, jax.tree_util.tree_map(lambda value: value, state.params)
        )
        state.evaluations_without_improvement = 0
    else:
        state.evaluations_without_improvement += 1
    logger.info(
        "step=%d train_loss=%.8g validation_probe_loss=%.8g best_step=%d",
        step,
        state.final_training_loss,
        validation_loss,
        state.best_step,
    )


def _finalise(
    prepared: PreparedTraining,
    config: TrainingConfig,
    state: OptimizerState,
    *,
    stopped_early: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if state.best_step < 1 or not np.isfinite(state.best_validation_loss):
        raise RuntimeError("DeepONet training produced no finite validation selection")
    atomic_savez(
        config.output_path,
        artifact_payload(
            prepared,
            config,
            state,
            artifact_schema=ARTIFACT_SCHEMA,
            training_schema=TRAINING_SCHEMA,
        ),
    )
    runtime = DeepONetEquilibriumAccelerator()
    runtime.load_weights(config.output_path)
    runtime_prediction = runtime.predict(prepared.input_mean)
    reference_normalised = np.asarray(
        operator_forward(
            state.best_params,
            jnp.zeros((1, len(prepared.input_mean)), dtype=jnp.float32),
            jnp.asarray(prepared.normalised_coordinates, dtype=jnp.float32),
        )[0],
        dtype=np.float64,
    )
    reference = (prepared.field_mean + prepared.field_scale * reference_normalised).reshape(
        prepared.data.grid_shape
    )
    runtime_parity = float(np.max(np.abs(runtime_prediction - reference)))
    if runtime_parity > 1.0e-4:
        raise RuntimeError(f"DeepONet runtime/training parity failed: {runtime_parity}")
    validation_metrics, _ = field_metrics(
        runtime, prepared.data, prepared.split.validation, chunk_rows=64
    )
    calibration_metrics, calibration_scores = field_metrics(
        runtime, prepared.data, prepared.split.calibration, chunk_rows=64
    )
    test_metrics, test_scores = field_metrics(
        runtime, prepared.data, prepared.split.test, chunk_rows=64
    )
    numpy_reference = DeepONetEquilibriumAccelerator(prefer_rust=False)
    numpy_reference.load_weights(config.output_path)
    backend_parity = runtime_backend_parity(
        runtime,
        numpy_reference,
        prepared.data,
        prepared.split.test,
        chunk_rows=64,
    )
    if backend_parity["within_tolerance"] is False:
        raise RuntimeError(
            "DeepONet Rust/NumPy untouched-test parity failed: "
            f"ratio={backend_parity['max_tolerance_ratio']}"
        )
    alpha = 0.05
    rank = min(
        len(calibration_scores),
        int(np.ceil((len(calibration_scores) + 1) * (1.0 - alpha))),
    )
    bound = float(np.sort(calibration_scores)[rank - 1])
    recovery = checked_json_load(
        config.checkpoint_dir / "optimizer_recovery.json", max_bytes=MAX_RECOVERY_BYTES
    )
    if not isinstance(recovery, dict):
        raise RuntimeError("DeepONet optimizer recovery is invalid after completion")
    report = prepared.report
    report.update(
        completed_report_sections(
            prepared,
            config,
            state,
            stopped_early=stopped_early,
            elapsed_seconds=elapsed_seconds,
            validation_metrics=validation_metrics,
            calibration_metrics=calibration_metrics,
            test_metrics=test_metrics,
            conformal_alpha=alpha,
            conformal_rank=rank,
            conformal_bound=bound,
            test_coverage=float(np.mean(test_scores <= bound)),
            recovery=cast(dict[str, Any], recovery),
            runtime_prediction=runtime_prediction,
            runtime_parity=runtime_parity,
            runtime_backend=runtime.backend,
            backend_parity=backend_parity,
        )
    )
    atomic_json(config.report_path, report)
    return report


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
    """Train and evaluate one local, manifest-bound fixed-machine DeepONet.

    Parameters
    ----------
    dataset_dir : Path
        Authenticated machine-conditioned v2 cohort.
    output_path, report_path, checkpoint_dir : Path
        Local candidate NPZ, evidence JSON, and recovery directory.
    steps, seed : int
        Positive optimiser-step target and deterministic run seed.
    validation_fraction, calibration_fraction, test_fraction : float
        Positive held-out fractions whose sum is less than one.
    branch_hidden, trunk_hidden : tuple[int, ...]
        Hidden widths of the control and coordinate networks.
    basis_width : int
        Shared branch/trunk output width.
    shot_batch_size, coordinate_batch_size : int
        Maximum training minibatch sizes.
    validation_probe_shots, validation_probe_coordinates : int
        Maximum frozen validation-probe sizes.
    learning_rate, weight_decay, gradient_clip : float
        AdamW step size, decoupled decay, and global gradient ceiling.
    statistics_chunk_rows : int
        Maximum fields materialised per training-statistics chunk.
    evaluation_every, checkpoint_every, early_stopping_patience : int
        Validation, recovery, and early-selection cadence.
    resume : bool
        Continue only from an identity-matching recovery point.

    Returns
    -------
    dict[str, Any]
        Final local-candidate report with split, metric, recovery, and runtime
        parity evidence.

    Raises
    ------
    OSError
        If authenticated inputs or local custody files cannot be accessed.
    RuntimeError
        If selection or runtime parity fails.
    ValueError
        If configuration, data, or recovery state violates its contract.
    """
    config = TrainingConfig(
        dataset_dir=dataset_dir,
        output_path=output_path,
        report_path=report_path,
        checkpoint_dir=checkpoint_dir,
        steps=steps,
        seed=seed,
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        basis_width=basis_width,
        shot_batch_size=shot_batch_size,
        coordinate_batch_size=coordinate_batch_size,
        validation_probe_shots=validation_probe_shots,
        validation_probe_coordinates=validation_probe_coordinates,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        statistics_chunk_rows=statistics_chunk_rows,
        evaluation_every=evaluation_every,
        checkpoint_every=checkpoint_every,
        early_stopping_patience=early_stopping_patience,
        resume=resume,
    )
    _validate_config(config)
    started = time.perf_counter()
    prepared = _prepare_training(config)
    if config.resume and (config.checkpoint_dir / "optimizer_recovery.json").exists():
        state = load_optimizer(config.checkpoint_dir, identity=prepared.identity)
        if state.completed_steps > config.steps:
            raise ValueError("DeepONet recovery exceeds the requested step target")
    else:
        state = _new_optimizer_state(config, len(prepared.data.feature_names))
    stopped_early = _optimise(prepared, config, state)
    return _finalise(
        prepared,
        config,
        state,
        stopped_early=stopped_early,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> None:
    """Run the DeepONet command-line adapter."""
    run_deeponet_cli(run_training, default_basis_width=DEFAULT_BASIS_WIDTH)


if __name__ == "__main__":
    main()
