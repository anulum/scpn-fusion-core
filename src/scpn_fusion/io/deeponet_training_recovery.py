# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Training Recovery
"""Identity-bound statistics, optimiser recovery, and artifact serialisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import jax.numpy as jnp
import numpy as np

from scpn_fusion.core.deeponet_training import OperatorParams, Params
from scpn_fusion.io.deeponet_training_data import training_statistics
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import sha256_file
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedTrainingData,
    array_sha256,
    atomic_json,
    atomic_savez,
)
from scpn_fusion.io.safe_loaders import checked_json_load

STATISTICS_SCHEMA = "scpn-fusion.equilibrium-deeponet-statistics.v1"
OPTIMIZER_SCHEMA = "scpn-fusion.equilibrium-deeponet-adam.v1"
MAX_RECOVERY_BYTES = 2 * 1024 * 1024


@dataclass
class OptimizerState:
    """Mutable state required for exact AdamW continuation.

    The state retains current parameters and both optimiser-moment trees,
    validation-selected parameters, absolute completed/selected steps, latest
    training and best validation objectives, validation-patience state, and
    loss histories aligned with the recorded evaluation steps.
    """

    params: OperatorParams
    first_moment: OperatorParams
    second_moment: OperatorParams
    best_params: OperatorParams
    completed_steps: int
    final_training_loss: float
    best_validation_loss: float
    best_step: int
    evaluations_without_improvement: int
    evaluation_steps: list[int]
    training_losses: list[float]
    validation_losses: list[float]


def serialize_network(payload: dict[str, Any], prefix: str, params: Params) -> None:
    """Append one dense network to a pickle-free NPZ payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Mutable array mapping that will be written with ``numpy.savez``.
    prefix : str
        Stable namespace for layer-count, weight, and bias members.
    params : Params
        Ordered dense layers to serialise as float64 arrays.
    """
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
    serialize_network(payload, f"{prefix}_branch", params["branch"])
    serialize_network(payload, f"{prefix}_trunk", params["trunk"])


def _deserialize_operator(archive: Any, prefix: str) -> OperatorParams:
    return {
        "branch": _deserialize_network(archive, f"{prefix}_branch"),
        "trunk": _deserialize_network(archive, f"{prefix}_trunk"),
    }


def _validate_identity(archive: Any, identity: Mapping[str, Any]) -> None:
    for name, value in identity.items():
        if name not in archive or not np.array_equal(archive[name], value):
            raise ValueError(f"DeepONet recovery identity mismatch for {name}")


def statistics_identity(
    data: MachineConditionedTrainingData,
    train_indices: np.ndarray[Any, np.dtype[np.int64]],
    source_paths: tuple[Path, ...],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the immutable identity of training-only field statistics.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated cohort and manifest digests.
    train_indices : ndarray[int64]
        Rows assigned exclusively to training.
    source_paths : tuple[Path, ...]
        Exact implementation files whose bytes affect the result.
    repo_root : Path
        Root used to store portable relative source names.

    Returns
    -------
    dict[str, Any]
        Pickle-free arrays binding dataset, split, and source SHA-256 values.
    """
    return {
        "statistics_schema": np.asarray([STATISTICS_SCHEMA]),
        "dataset_manifest_sha256": np.asarray([data.manifest_sha256]),
        "fields_sha256": np.asarray([data.fields_sha256]),
        "train_indices_sha256": np.asarray([array_sha256(train_indices)]),
        "source_sha256_names": np.asarray(
            [str(path.relative_to(repo_root)) for path in source_paths]
        ),
        "source_sha256_values": np.asarray([sha256_file(path) for path in source_paths]),
    }


def load_or_compute_statistics(
    data: MachineConditionedTrainingData,
    train_indices: np.ndarray[Any, np.dtype[np.int64]],
    *,
    chunk_rows: int,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    resume: bool,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], float, np.ndarray[Any, np.dtype[np.float64]], str
]:
    """Load authenticated statistics or fit and checkpoint them atomically.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated cohort used for training-only statistics.
    train_indices : ndarray[int64]
        Rows assigned exclusively to training.
    chunk_rows : int
        Maximum fields materialised per statistics chunk.
    checkpoint_dir : Path
        Local recovery directory.
    identity : Mapping[str, Any]
        Expected dataset, split, and source identity arrays.
    resume : bool
        Load existing state when true; compute and save otherwise.

    Returns
    -------
    tuple[ndarray[float64], float, ndarray[float64], str]
        Field mean, residual scale, training-row norms, and statistics digest.

    Raises
    ------
    ValueError
        If recovery metadata, bytes, or identity do not authenticate.
    """
    state_path = checkpoint_dir / "statistics_recovery.json"
    statistics_path = checkpoint_dir / "statistics.npz"
    if resume:
        raw = checked_json_load(state_path, max_bytes=MAX_RECOVERY_BYTES)
        if not isinstance(raw, dict):
            raise ValueError("DeepONet statistics recovery must be an object")
        recovery = cast(dict[str, Any], raw)
        filename = recovery.get("file")
        digest = recovery.get("sha256")
        if (
            recovery.get("schema_version") != STATISTICS_SCHEMA
            or filename != statistics_path.name
            or not isinstance(digest, str)
        ):
            raise ValueError("DeepONet statistics recovery metadata is invalid")
        if (
            not statistics_path.is_file()
            or statistics_path.is_symlink()
            or sha256_file(statistics_path) != digest
        ):
            raise ValueError("DeepONet statistics recovery SHA-256 mismatch")
        with np.load(statistics_path, allow_pickle=False) as archive:
            _validate_identity(archive, identity)
            return (
                np.asarray(archive["field_mean"], dtype=np.float64),
                float(archive["field_scale"][0]),
                np.asarray(archive["field_norm_squared"], dtype=np.float64),
                digest,
            )
    field_mean, field_scale, field_norm_squared = training_statistics(
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


def optimizer_identity(
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
    validation_probe_samples: np.ndarray[Any, np.dtype[np.int64]],
    validation_probe_coordinates: np.ndarray[Any, np.dtype[np.int64]],
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    evaluation_every: int,
    early_stopping_patience: int,
    source_paths: tuple[Path, ...],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind every trajectory-affecting input into optimiser recovery.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated cohort and manifest digest.
    split_hashes : Mapping[str, str]
        SHA-256 digest for every split role.
    statistics_sha256 : str
        Digest of the authenticated training-only statistics stage.
    seed : int
        Run seed.
    branch_hidden, trunk_hidden : tuple[int, ...]
        Network hidden widths.
    basis_width, shot_batch_size, coordinate_batch_size : int
        Operator width and minibatch sizes.
    validation_probe_samples, validation_probe_coordinates : ndarray[int64]
        Frozen validation probe identities.
    learning_rate, weight_decay, gradient_clip : float
        AdamW and gradient-clipping parameters.
    evaluation_every, early_stopping_patience : int
        Validation schedule and selection patience.
    source_paths : tuple[Path, ...]
        Exact implementation files whose bytes affect the trajectory.
    repo_root : Path
        Root used for portable relative source names.

    Returns
    -------
    dict[str, Any]
        Pickle-free identity arrays stored in every optimiser stage.
    """
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
            [str(path.relative_to(repo_root)) for path in source_paths]
        ),
        "source_sha256_values": np.asarray([sha256_file(path) for path in source_paths]),
    }
    for role, digest in split_hashes.items():
        payload[f"{role}_indices_sha256"] = np.asarray([digest])
    return payload


def save_optimizer(
    checkpoint_dir: Path,
    *,
    identity: Mapping[str, Any],
    state: OptimizerState,
) -> None:
    """Atomically save one complete optimiser continuation point.

    Parameters
    ----------
    checkpoint_dir : Path
        Local directory receiving the NPZ stage and JSON recovery pointer.
    identity : Mapping[str, Any]
        Expected trajectory identity embedded in the stage.
    state : OptimizerState
        Parameters, moments, selection state, and loss history to persist.
    """
    payload: dict[str, Any] = dict(identity)
    payload.update(
        {
            "completed_steps": np.asarray([state.completed_steps], dtype=np.int64),
            "final_training_loss": np.asarray([state.final_training_loss]),
            "best_validation_loss": np.asarray([state.best_validation_loss]),
            "best_step": np.asarray([state.best_step], dtype=np.int64),
            "evaluations_without_improvement": np.asarray(
                [state.evaluations_without_improvement], dtype=np.int64
            ),
            "evaluation_steps": np.asarray(state.evaluation_steps, dtype=np.int64),
            "training_losses": np.asarray(state.training_losses),
            "validation_losses": np.asarray(state.validation_losses),
        }
    )
    _serialize_operator(payload, "current", state.params)
    _serialize_operator(payload, "first", state.first_moment)
    _serialize_operator(payload, "second", state.second_moment)
    _serialize_operator(payload, "best", state.best_params)
    stage = checkpoint_dir / f"optimizer_step_{state.completed_steps:08d}.npz"
    atomic_savez(stage, payload)
    atomic_json(
        checkpoint_dir / "optimizer_recovery.json",
        {
            "schema_version": OPTIMIZER_SCHEMA,
            "completed_steps": state.completed_steps,
            "stage_file": stage.name,
            "stage_sha256": sha256_file(stage),
        },
    )


def load_optimizer(checkpoint_dir: Path, *, identity: Mapping[str, Any]) -> OptimizerState:
    """Authenticate and restore one exact optimiser continuation point.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing the JSON pointer and declared NPZ stage.
    identity : Mapping[str, Any]
        Expected trajectory identity for every embedded member.

    Returns
    -------
    OptimizerState
        Exact parameters, moments, selection state, and loss history.

    Raises
    ------
    ValueError
        If metadata, stage bytes, identity, or completed-step values disagree.
    """
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
        state = OptimizerState(
            params=_deserialize_operator(archive, "current"),
            first_moment=_deserialize_operator(archive, "first"),
            second_moment=_deserialize_operator(archive, "second"),
            best_params=_deserialize_operator(archive, "best"),
            completed_steps=int(archive["completed_steps"][0]),
            final_training_loss=float(archive["final_training_loss"][0]),
            best_validation_loss=float(archive["best_validation_loss"][0]),
            best_step=int(archive["best_step"][0]),
            evaluations_without_improvement=int(archive["evaluations_without_improvement"][0]),
            evaluation_steps=np.asarray(archive["evaluation_steps"], dtype=np.int64).tolist(),
            training_losses=np.asarray(archive["training_losses"], dtype=np.float64).tolist(),
            validation_losses=np.asarray(archive["validation_losses"], dtype=np.float64).tolist(),
        )
    if state.completed_steps != expected_steps:
        raise ValueError("DeepONet optimizer recovery step mismatch")
    return state


__all__ = [
    "MAX_RECOVERY_BYTES",
    "OPTIMIZER_SCHEMA",
    "OptimizerState",
    "STATISTICS_SCHEMA",
    "load_optimizer",
    "load_or_compute_statistics",
    "optimizer_identity",
    "save_optimizer",
    "serialize_network",
    "statistics_identity",
]
