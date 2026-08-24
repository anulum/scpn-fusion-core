# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Training Data
"""Coordinate, statistics, minibatch, and held-out metric preparation."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator
from scpn_fusion.core.deeponet_training_contracts import RuntimeBackendParity
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedTrainingData,
    iter_field_rows,
)

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
TrainingBatch: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def load_coordinates(data: MachineConditionedTrainingData) -> FloatArray:
    """Load the authenticated R/Z grid in flattened field order.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Verified dataset whose manifest declares the coordinate arrays.

    Returns
    -------
    FloatArray
        Metre-valued coordinates with shape ``(n_z * n_r, 2)``.

    Raises
    ------
    ValueError
        If the coordinate vectors disagree with the authenticated field grid.
    """
    arrays = cast(dict[str, Any], data.manifest["arrays"])
    r_spec = cast(dict[str, Any], arrays["grid_r_m"])
    z_spec = cast(dict[str, Any], arrays["grid_z_m"])
    r = np.load(data.root / str(r_spec["file"]), allow_pickle=False)
    z = np.load(data.root / str(z_spec["file"]), allow_pickle=False)
    if r.shape != (data.grid_shape[1],) or z.shape != (data.grid_shape[0],):
        raise ValueError("DeepONet coordinate arrays do not match the field grid")
    grid_r, grid_z = np.meshgrid(r, z, indexing="xy")
    return np.asarray(np.column_stack((grid_r.ravel(), grid_z.ravel())), dtype=np.float64)


def training_statistics(
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    *,
    chunk_rows: int,
) -> tuple[FloatArray, float, FloatArray]:
    """Fit field mean, residual scale, and norms on training rows.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated field cohort in Wb/rad.
    train_indices : IndexArray
        Rows assigned exclusively to training.
    chunk_rows : int
        Maximum number of fields materialised per streaming chunk.

    Returns
    -------
    tuple[FloatArray, float, FloatArray]
        Flattened spatial mean in Wb/rad, positive global residual scale in
        Wb/rad, and one squared field norm per training row.
    """
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
    centred_sum_squared = max(
        field_sum_squared - len(train_indices) * float(field_mean @ field_mean), 1.0e-30
    )
    field_scale = float(np.sqrt(centred_sum_squared / (len(train_indices) * width)))
    return field_mean, field_scale, np.maximum(field_norm_squared, 1.0e-30)


def extract_targets(
    data: MachineConditionedTrainingData,
    sample_indices: IndexArray,
    coordinate_indices: IndexArray,
    field_mean: FloatArray,
    field_scale: float,
) -> FloatArray:
    """Read and normalise a rectangular shot-coordinate target selection.

    Parameters
    ----------
    data : MachineConditionedTrainingData
        Authenticated field cohort.
    sample_indices, coordinate_indices : IndexArray
        Shot rows and flattened Z-R positions to select.
    field_mean : FloatArray
        Training-only flattened spatial mean in Wb/rad.
    field_scale : float
        Positive training-only global scale in Wb/rad.

    Returns
    -------
    FloatArray
        Dimensionless targets with shape ``(shots, coordinates)``.
    """
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


def deterministic_probe(
    *,
    seed: int,
    sample_count: int,
    coordinate_count: int,
    available_samples: int,
    available_coordinates: int,
) -> tuple[IndexArray, IndexArray]:
    """Select a reproducible validation-only shot and coordinate probe.

    Parameters
    ----------
    seed : int
        Run seed used to derive the probe generator.
    sample_count, coordinate_count : int
        Requested probe sizes.
    available_samples, available_coordinates : int
        Bounds of the validation and coordinate populations.

    Returns
    -------
    tuple[IndexArray, IndexArray]
        Sorted unique sample positions and coordinate indices.
    """
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


def training_batch(
    *,
    step: int,
    seed: int,
    data: MachineConditionedTrainingData,
    train_indices: IndexArray,
    normalised_inputs: FloatArray,
    normalised_coordinates: FloatArray,
    field_mean: FloatArray,
    field_scale: float,
    sample_weights: FloatArray,
    shot_batch_size: int,
    coordinate_batch_size: int,
) -> TrainingBatch:
    """Build a deterministic physical minibatch from an absolute step.

    Parameters
    ----------
    step, seed : int
        One-based optimiser step and fixed run seed.
    data : MachineConditionedTrainingData
        Authenticated cohort used to read physical targets.
    train_indices : IndexArray
        Rows assigned exclusively to training.
    normalised_inputs, normalised_coordinates : FloatArray
        Training controls and coordinate grid after training-only scaling.
    field_mean : FloatArray
        Training-only flattened spatial mean in Wb/rad.
    field_scale : float
        Positive training-only residual scale in Wb/rad.
    sample_weights : FloatArray
        One relative-field weight per training row.
    shot_batch_size, coordinate_batch_size : int
        Maximum sampled shots and coordinates.

    Returns
    -------
    TrainingBatch
        Input rows, coordinate rows, normalised targets, and shot weights.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    shot_positions = np.asarray(
        rng.choice(
            len(train_indices), size=min(shot_batch_size, len(train_indices)), replace=False
        ),
        dtype=np.int64,
    )
    coordinate_indices = np.asarray(
        rng.choice(
            len(normalised_coordinates),
            size=min(coordinate_batch_size, len(normalised_coordinates)),
            replace=False,
        ),
        dtype=np.int64,
    )
    targets = extract_targets(
        data,
        train_indices[shot_positions],
        coordinate_indices,
        field_mean,
        field_scale,
    )
    return (
        normalised_inputs[shot_positions],
        normalised_coordinates[coordinate_indices],
        targets,
        sample_weights[shot_positions],
    )


def field_metrics(
    runtime: DeepONetEquilibriumAccelerator,
    data: MachineConditionedTrainingData,
    indices: IndexArray,
    *,
    chunk_rows: int,
) -> tuple[dict[str, float], FloatArray]:
    """Measure full-field error and retain row-wise relative-L2 scores.

    Parameters
    ----------
    runtime : DeepONetEquilibriumAccelerator
        Loaded runtime used for production-path inference.
    data : MachineConditionedTrainingData
        Authenticated truth fields in Wb/rad.
    indices : IndexArray
        Held-out shot rows to evaluate.
    chunk_rows : int
        Maximum inference rows per chunk.

    Returns
    -------
    tuple[dict[str, float], FloatArray]
        Field RMSE and relative-L2 summary, plus one relative-L2 score per row.

    Raises
    ------
    ValueError
        If no held-out rows are supplied or ``chunk_rows`` is not positive.
    """
    if len(indices) == 0:
        raise ValueError("DeepONet field metrics require at least one sample")
    if chunk_rows < 1:
        raise ValueError("DeepONet field metric chunk size must be positive")
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


def runtime_backend_parity(
    native: DeepONetEquilibriumAccelerator,
    reference: DeepONetEquilibriumAccelerator,
    data: MachineConditionedTrainingData,
    indices: IndexArray,
    *,
    chunk_rows: int,
    relative_tolerance: float = 1.0e-14,
    absolute_tolerance: float = 1.0e-14,
) -> RuntimeBackendParity:
    """Compare native and NumPy inference over an authenticated held-out split.

    Parameters
    ----------
    native, reference : DeepONetEquilibriumAccelerator
        Loaded runtimes for the Rust-first and NumPy-only execution paths.
    data : MachineConditionedTrainingData
        Authenticated causal inputs associated with the evaluated split.
    indices : IndexArray
        Untouched held-out rows; every row is evaluated exactly once.
    chunk_rows : int
        Maximum number of shot predictions materialised per runtime call.
    relative_tolerance, absolute_tolerance : float
        Element-wise parity bounds in Wb/rad, applied as
        ``abs(delta) <= atol + rtol * abs(reference)``.

    Returns
    -------
    RuntimeBackendParity
        Maximum absolute, normalised-tolerance, and IEEE-754 ULP differences.
        Metrics are unavailable when the compiled Rust backend is not loaded.

    Raises
    ------
    ValueError
        If the split is empty, the chunk size or a tolerance is not positive,
        or the reference runtime is not NumPy.
    """
    if len(indices) == 0:
        raise ValueError("DeepONet parity requires at least one sample")
    if chunk_rows < 1:
        raise ValueError("DeepONet parity chunk size must be positive")
    if reference.backend != "numpy":
        raise ValueError("DeepONet parity reference must use the NumPy backend")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("DeepONet parity tolerances must be positive")
    evidence = RuntimeBackendParity(
        evaluated=False,
        native_backend=native.backend,
        reference_backend=reference.backend,
        sample_count=len(indices),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_absolute_difference=None,
        max_tolerance_ratio=None,
        max_ulp_difference=None,
        within_tolerance=None,
    )
    if native.backend != "rust":
        return evidence

    max_absolute = 0.0
    max_ratio = 0.0
    max_ulp = 0
    sign_bit = np.uint64(1 << 63)
    for start in range(0, len(indices), chunk_rows):
        rows = indices[start : min(start + chunk_rows, len(indices))]
        features = np.asarray(data.inputs[rows], dtype=np.float64)
        native_fields = native.predict_batch(features)
        reference_fields = reference.predict_batch(features)
        delta = np.abs(native_fields - reference_fields)
        tolerance = absolute_tolerance + relative_tolerance * np.abs(reference_fields)
        max_absolute = max(max_absolute, float(np.max(delta)))
        max_ratio = max(max_ratio, float(np.max(delta / tolerance)))

        native_values = np.where(native_fields == 0.0, 0.0, native_fields)
        reference_values = np.where(reference_fields == 0.0, 0.0, reference_fields)
        native_bits = native_values.view(np.uint64)
        reference_bits = reference_values.view(np.uint64)
        native_ordered = np.where(native_bits & sign_bit, ~native_bits, native_bits | sign_bit)
        reference_ordered = np.where(
            reference_bits & sign_bit,
            ~reference_bits,
            reference_bits | sign_bit,
        )
        ulp_delta = np.maximum(native_ordered, reference_ordered) - np.minimum(
            native_ordered, reference_ordered
        )
        max_ulp = max(max_ulp, int(np.max(ulp_delta)))

    evidence["evaluated"] = True
    evidence["max_absolute_difference"] = max_absolute
    evidence["max_tolerance_ratio"] = max_ratio
    evidence["max_ulp_difference"] = max_ulp
    evidence["within_tolerance"] = max_ratio <= 1.0
    return evidence


__all__ = [
    "TrainingBatch",
    "deterministic_probe",
    "extract_targets",
    "field_metrics",
    "load_coordinates",
    "runtime_backend_parity",
    "training_batch",
    "training_statistics",
]
