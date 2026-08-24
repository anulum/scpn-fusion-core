# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned Surrogate Training Data
"""Authenticated v2 data loading and train-only streaming randomized PCA."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.machine_conditioned_equilibrium_dataset import (
    SCHEMA_VERSION,
    sha256_file,
    verify_machine_conditioned_dataset,
)
from scpn_fusion.io.safe_loaders import checked_json_load

PCA_RECOVERY_SCHEMA = "scpn-fusion.machine-conditioned-streaming-pca.v1"
MAX_PCA_RECOVERY_BYTES = 2 * 1024 * 1024
FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class MachineConditionedTrainingData:
    """Authenticated arrays and metadata required by successor training."""

    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    inputs: FloatArray
    fields: FloatArray
    feature_names: tuple[str, ...]
    grid_shape: tuple[int, int]
    inputs_sha256: str
    fields_sha256: str


@dataclass(frozen=True)
class StreamingPCAState:
    """A fitted PCA basis learned exclusively from declared training rows."""

    mean: FloatArray
    components: FloatArray
    explained_variance_ratio: FloatArray
    singular_values: FloatArray
    total_centered_sum_squares: float

    def transform(self, rows: FloatArray) -> FloatArray:
        """Project flattened field rows into the fitted basis."""
        matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)
        return np.asarray((matrix - self.mean) @ self.components.T, dtype=np.float64)

    def inverse_transform(self, latent: FloatArray) -> FloatArray:
        """Reconstruct flattened field rows from latent coordinates."""
        return np.asarray(latent @ self.components + self.mean, dtype=np.float64)


def array_sha256(array: NDArray[np.generic]) -> str:
    """Hash one contiguous numerical array without a Python bytes copy."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def deterministic_split(
    n_samples: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[IndexArray, IndexArray]:
    """Return sorted, disjoint and reproducible training/validation indices."""
    if n_samples < 3:
        raise ValueError("at least three samples are required for a held-out split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    n_validation = max(1, int(round(n_samples * validation_fraction)))
    if n_validation >= n_samples:
        raise ValueError("validation_fraction leaves no training samples")
    permutation = np.random.default_rng(seed).permutation(n_samples)
    validation = np.sort(permutation[:n_validation]).astype(np.int64)
    training = np.sort(permutation[n_validation:]).astype(np.int64)
    return training, validation


def load_machine_conditioned_training_data(
    dataset_dir: str | Path,
    *,
    full_field_scan: bool = True,
) -> MachineConditionedTrainingData:
    """Authenticate and mmap a machine-conditioned v2 cohort for training."""
    root = Path(dataset_dir).resolve()
    verification = verify_machine_conditioned_dataset(root, full_field_scan=full_field_scan)
    if verification["status"] != "passed":
        raise ValueError(
            f"machine-conditioned dataset failed verification: {verification['failures']}"
        )
    raw_manifest = checked_json_load(root / "manifest.json")
    if not isinstance(raw_manifest, dict):
        raise ValueError("dataset manifest must be an object")
    manifest = cast(dict[str, Any], raw_manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"dataset schema must be {SCHEMA_VERSION}")
    features = manifest.get("features")
    arrays = manifest.get("arrays")
    if not isinstance(features, list) or not isinstance(arrays, dict):
        raise ValueError("dataset manifest lacks features or arrays")
    feature_names = tuple(str(item["name"]) for item in features)
    if len(feature_names) != 17 or len(set(feature_names)) != 17:
        raise ValueError("successor training requires exactly 17 unique v2 features")
    if any(item.get("role") != "pre_solve_control" for item in features):
        raise ValueError("every successor input must be a declared pre-solve control")
    input_spec = cast(dict[str, Any], arrays["inputs"])
    field_spec = cast(dict[str, Any], arrays["psi_total"])
    inputs = np.load(root / str(input_spec["file"]), mmap_mode="r", allow_pickle=False)
    fields = np.load(root / str(field_spec["file"]), mmap_mode="r", allow_pickle=False)
    if inputs.ndim != 2 or inputs.shape[1] != 17:
        raise ValueError("inputs must have shape (samples, 17)")
    if fields.ndim != 3 or fields.shape[0] != inputs.shape[0]:
        raise ValueError("psi_total must have shape (samples, z, r)")
    grid_shape = (int(fields.shape[1]), int(fields.shape[2]))
    return MachineConditionedTrainingData(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(root / "manifest.json"),
        inputs=cast(FloatArray, inputs),
        fields=cast(FloatArray, fields),
        feature_names=feature_names,
        grid_shape=grid_shape,
        inputs_sha256=str(input_spec["sha256"]),
        fields_sha256=str(field_spec["sha256"]),
    )


def atomic_savez(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace an NPZ after syncing its data and parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **payload)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace JSON after syncing its data and parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _field_rows(fields: FloatArray, indices: IndexArray, start: int, stop: int) -> FloatArray:
    rows = np.asarray(fields[indices[start:stop]], dtype=np.float64)
    return np.asarray(rows.reshape(stop - start, -1), dtype=np.float64)


def iter_field_rows(
    fields: FloatArray,
    indices: IndexArray,
    *,
    chunk_rows: int,
) -> Iterator[tuple[int, int, FloatArray]]:
    """Yield flattened indexed field rows in bounded deterministic chunks."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    for start in range(0, len(indices), chunk_rows):
        stop = min(start + chunk_rows, len(indices))
        yield start, stop, _field_rows(fields, indices, start, stop)


def _canonical_qr(matrix: FloatArray) -> FloatArray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return np.asarray(q * signs[np.newaxis, :], dtype=np.float64)


def _canonical_components(components: FloatArray) -> FloatArray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in range(len(result)):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0.0:
            result[row] *= -1.0
    return result


def _write_pca_stage(
    checkpoint_dir: Path,
    *,
    stage: str,
    identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Path:
    stage_path = checkpoint_dir / f"pca_{stage}.npz"
    atomic_savez(stage_path, payload)
    atomic_json(
        checkpoint_dir / "pca_recovery.json",
        {
            "schema_version": PCA_RECOVERY_SCHEMA,
            "identity": dict(identity),
            "stage": stage,
            "stage_file": stage_path.name,
            "stage_sha256": sha256_file(stage_path),
        },
    )
    return stage_path


def _load_pca_recovery(
    checkpoint_dir: Path,
    *,
    identity: Mapping[str, Any],
) -> tuple[str, dict[str, FloatArray]]:
    state_path = checkpoint_dir / "pca_recovery.json"
    raw = checked_json_load(state_path, max_bytes=MAX_PCA_RECOVERY_BYTES)
    if not isinstance(raw, dict):
        raise ValueError("PCA recovery state must be an object")
    state = cast(dict[str, Any], raw)
    if state.get("schema_version") != PCA_RECOVERY_SCHEMA:
        raise ValueError(f"PCA recovery schema must be {PCA_RECOVERY_SCHEMA}")
    if state.get("identity") != dict(identity):
        raise ValueError("PCA recovery identity mismatch")
    stage = state.get("stage")
    filename = state.get("stage_file")
    digest = state.get("stage_sha256")
    if (
        not isinstance(stage, str)
        or not isinstance(filename, str)
        or Path(filename).name != filename
    ):
        raise ValueError("PCA recovery stage metadata is invalid")
    stage_path = checkpoint_dir / filename
    if not stage_path.is_file() or stage_path.is_symlink() or sha256_file(stage_path) != digest:
        raise ValueError("PCA recovery stage SHA-256 mismatch")
    with np.load(stage_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    return stage, cast(dict[str, FloatArray], payload)


def fit_streaming_randomized_pca(
    fields: FloatArray,
    train_indices: IndexArray,
    *,
    n_components: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
    chunk_rows: int,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    resume: bool = False,
) -> tuple[StreamingPCAState, FloatArray]:
    """Fit deterministic randomized PCA without materialising all training fields."""
    if fields.ndim < 2:
        raise ValueError("fields must have a leading sample axis and at least one feature axis")
    if train_indices.ndim != 1 or len(train_indices) < 2:
        raise ValueError("train_indices must contain at least two rows")
    if (
        np.any(np.diff(train_indices) <= 0)
        or train_indices[0] < 0
        or train_indices[-1] >= len(fields)
    ):
        raise ValueError("train_indices must be sorted, unique and in bounds")
    width = int(np.prod(fields.shape[1:], dtype=np.int64))
    if not 1 <= n_components < min(len(train_indices), width):
        raise ValueError("n_components must be positive and below train rank bounds")
    if oversampling < 0 or power_iterations < 0 or chunk_rows < 1:
        raise ValueError("oversampling, power_iterations and chunk_rows are invalid")
    sketch_width = min(n_components + oversampling, len(train_indices), width)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recovery_path = checkpoint_dir / "pca_recovery.json"
    stage = "none"
    payload: dict[str, FloatArray] = {}
    if resume:
        if not recovery_path.exists():
            raise FileNotFoundError("PCA resume requested without recovery state")
        stage, payload = _load_pca_recovery(checkpoint_dir, identity=identity)
    elif recovery_path.exists():
        raise FileExistsError("PCA recovery state already exists; use resume")

    if stage == "complete":
        state = StreamingPCAState(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            components=np.asarray(payload["components"], dtype=np.float64),
            explained_variance_ratio=np.asarray(
                payload["explained_variance_ratio"], dtype=np.float64
            ),
            singular_values=np.asarray(payload["singular_values"], dtype=np.float64),
            total_centered_sum_squares=float(payload["total_centered_sum_squares"][0]),
        )
        return state, np.asarray(payload["train_latent"], dtype=np.float64)

    if stage == "none":
        mean: FloatArray = np.zeros(width, dtype=np.float64)
        for _, _, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
            mean += np.sum(rows, axis=0, dtype=np.float64)
        mean /= float(len(train_indices))
        _write_pca_stage(
            checkpoint_dir,
            stage="mean",
            identity=identity,
            payload={"mean": mean},
        )
        stage, payload = "mean", {"mean": mean}
    mean = np.asarray(payload["mean"], dtype=np.float64)

    if stage == "mean":
        omega = np.random.default_rng(seed).standard_normal((width, sketch_width))
        sketch: FloatArray = np.empty((len(train_indices), sketch_width), dtype=np.float64)
        total_ss = 0.0
        for start, stop, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
            centered = rows - mean
            total_ss += float(np.sum(centered * centered, dtype=np.float64))
            sketch[start:stop] = centered @ omega
        if not np.isfinite(total_ss) or total_ss <= 0.0:
            raise ValueError("training fields have no finite centered variance")
        range_basis = _canonical_qr(sketch)
        _write_pca_stage(
            checkpoint_dir,
            stage="range",
            identity=identity,
            payload={
                "mean": mean,
                "basis": range_basis,
                "total_centered_sum_squares": np.asarray([total_ss]),
            },
        )
        stage = "range"
        payload = {
            "mean": mean,
            "basis": range_basis,
            "total_centered_sum_squares": np.asarray([total_ss]),
        }

    basis: FloatArray | None = (
        np.asarray(payload["basis"], dtype=np.float64) if "basis" in payload else None
    )
    total_ss = (
        float(payload["total_centered_sum_squares"][0])
        if "total_centered_sum_squares" in payload
        else 0.0
    )
    completed_power = int(stage.removeprefix("power_")) if stage.startswith("power_") else 0
    if stage in {"range"} or stage.startswith("power_"):
        if basis is None:
            raise ValueError("PCA recovery stage lacks its range basis")
        for iteration in range(completed_power, power_iterations):
            feature_basis: FloatArray = np.zeros((width, sketch_width), dtype=np.float64)
            for start, stop, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
                feature_basis += (rows - mean).T @ basis[start:stop]
            feature_basis = _canonical_qr(feature_basis)
            next_range = np.empty_like(basis)
            for start, stop, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
                next_range[start:stop] = (rows - mean) @ feature_basis
            basis = _canonical_qr(next_range)
            stage = f"power_{iteration + 1}"
            _write_pca_stage(
                checkpoint_dir,
                stage=stage,
                identity=identity,
                payload={
                    "mean": mean,
                    "basis": basis,
                    "total_centered_sum_squares": np.asarray([total_ss]),
                },
            )

        low_rank: FloatArray = np.zeros((sketch_width, width), dtype=np.float64)
        for start, stop, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
            low_rank += basis[start:stop].T @ (rows - mean)
        _, singular_values, right_vectors = np.linalg.svd(low_rank, full_matrices=False)
        components = _canonical_components(right_vectors[:n_components])
        singular_values = np.asarray(singular_values[:n_components], dtype=np.float64)
        explained = np.asarray(singular_values * singular_values / total_ss, dtype=np.float64)
        _write_pca_stage(
            checkpoint_dir,
            stage="components",
            identity=identity,
            payload={
                "mean": mean,
                "components": components,
                "explained_variance_ratio": explained,
                "singular_values": singular_values,
                "total_centered_sum_squares": np.asarray([total_ss]),
            },
        )
        stage = "components"
        payload = {
            "mean": mean,
            "components": components,
            "explained_variance_ratio": explained,
            "singular_values": singular_values,
            "total_centered_sum_squares": np.asarray([total_ss]),
        }

    if stage != "components":
        raise ValueError(f"unsupported PCA recovery stage: {stage}")
    state = StreamingPCAState(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        components=np.asarray(payload["components"], dtype=np.float64),
        explained_variance_ratio=np.asarray(payload["explained_variance_ratio"], dtype=np.float64),
        singular_values=np.asarray(payload["singular_values"], dtype=np.float64),
        total_centered_sum_squares=float(payload["total_centered_sum_squares"][0]),
    )
    train_latent: FloatArray = np.empty((len(train_indices), n_components), dtype=np.float64)
    for start, stop, rows in iter_field_rows(fields, train_indices, chunk_rows=chunk_rows):
        train_latent[start:stop] = state.transform(rows)
    complete_payload = {
        "mean": state.mean,
        "components": state.components,
        "explained_variance_ratio": state.explained_variance_ratio,
        "singular_values": state.singular_values,
        "total_centered_sum_squares": np.asarray([state.total_centered_sum_squares]),
        "train_latent": train_latent,
    }
    _write_pca_stage(
        checkpoint_dir,
        stage="complete",
        identity=identity,
        payload=complete_payload,
    )
    return state, train_latent


__all__ = [
    "MachineConditionedTrainingData",
    "PCA_RECOVERY_SCHEMA",
    "StreamingPCAState",
    "atomic_json",
    "atomic_savez",
    "array_sha256",
    "deterministic_split",
    "fit_streaming_randomized_pca",
    "iter_field_rows",
    "load_machine_conditioned_training_data",
]
