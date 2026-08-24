# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Training Contracts
"""Typed configuration and prepared-data contracts for DeepONet training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedSplit,
    MachineConditionedTrainingData,
)

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]


class RuntimeBackendParity(TypedDict):
    """JSON-safe Rust-versus-NumPy evidence over an authenticated split."""

    evaluated: bool
    native_backend: str
    reference_backend: str
    sample_count: int
    relative_tolerance: float
    absolute_tolerance: float
    max_absolute_difference: float | None
    max_tolerance_ratio: float | None
    max_ulp_difference: int | None
    within_tolerance: bool | None


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for one local DeepONet run.

    Paths identify authenticated input, local candidate/report outputs, and
    recovery custody. Split fractions assign held-out roles. Network widths,
    minibatch sizes, AdamW parameters, evaluation cadence, and the seed fully
    define the optimisation trajectory. ``resume`` permits only an exact
    identity match.
    """

    dataset_dir: Path
    output_path: Path
    report_path: Path
    checkpoint_dir: Path
    steps: int
    seed: int
    validation_fraction: float
    calibration_fraction: float
    test_fraction: float
    branch_hidden: tuple[int, ...]
    trunk_hidden: tuple[int, ...]
    basis_width: int
    shot_batch_size: int
    coordinate_batch_size: int
    validation_probe_shots: int
    validation_probe_coordinates: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    statistics_chunk_rows: int
    evaluation_every: int
    checkpoint_every: int
    early_stopping_patience: int
    resume: bool


@dataclass(frozen=True)
class PreparedTraining:
    """Training-only transforms, validation probe, and recovery identity.

    This value binds the authenticated cohort, four disjoint split roles,
    metre-valued coordinate grid, training-only control/field scaling,
    relative-field weights, fixed validation probe, source hashes, and the
    running evidence report passed between orchestration stages.
    """

    data: MachineConditionedTrainingData
    split: MachineConditionedSplit
    split_hashes: dict[str, str]
    coordinates: FloatArray
    coordinate_mean: FloatArray
    coordinate_std: FloatArray
    normalised_coordinates: FloatArray
    input_mean: FloatArray
    input_std: FloatArray
    normalised_inputs: FloatArray
    field_mean: FloatArray
    field_scale: float
    field_norm_reference: float
    train_sample_weights: FloatArray
    probe_indices: IndexArray
    probe_coordinate_indices: IndexArray
    probe_arrays: tuple[FloatArray, FloatArray, FloatArray, FloatArray]
    identity: dict[str, Any]
    report: dict[str, Any]


__all__ = [
    "FloatArray",
    "IndexArray",
    "PreparedTraining",
    "RuntimeBackendParity",
    "TrainingConfig",
]
