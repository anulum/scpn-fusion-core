# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Recoverable NPY Dataset Store
"""Crash-safe fixed-shape NPY storage for long deterministic generation jobs."""

from __future__ import annotations

import json
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.safe_loaders import checked_json_load

RECOVERY_SCHEMA = "scpn-fusion.recoverable-npy-dataset.v1"
MAX_RECOVERY_BYTES = 2 * 1024 * 1024
FloatArray: TypeAlias = NDArray[np.float64]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _normalise_shapes(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    if not shapes:
        raise ValueError("at least one array shape is required")
    result: dict[str, tuple[int, ...]] = {}
    for name, shape in shapes.items():
        if not name or Path(name).name != name or not shape or any(value < 1 for value in shape):
            raise ValueError(f"invalid array contract: {name}={shape}")
        result[name] = tuple(int(value) for value in shape)
    return result


def _array_digest(array: FloatArray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


class RecoverableNpyDataset:
    """A fixed-shape float64 NPY store with an external atomic checkpoint."""

    def __init__(
        self,
        *,
        partial_dir: Path,
        recovery_path: Path,
        arrays: dict[str, FloatArray],
        sample_array_names: frozenset[str],
        run_contract: dict[str, Any],
        accepted_samples: int,
        next_candidate_index: int,
        rejection_counts: Counter[str],
        committed_chunks: list[dict[str, Any]],
        static_sha256: dict[str, str],
    ) -> None:
        self.partial_dir = partial_dir
        self.recovery_path = recovery_path
        self.arrays = arrays
        self.sample_array_names = sample_array_names
        self.run_contract = run_contract
        self.accepted_samples = accepted_samples
        self.written_samples = accepted_samples
        self.next_candidate_index = next_candidate_index
        self.rejection_counts = rejection_counts
        self.committed_chunks = committed_chunks
        self.static_sha256 = static_sha256

    @classmethod
    def create(
        cls,
        *,
        partial_dir: Path,
        recovery_path: Path,
        shapes: Mapping[str, tuple[int, ...]],
        sample_array_names: frozenset[str],
        run_contract: Mapping[str, Any],
        initial_arrays: Mapping[str, FloatArray] | None = None,
    ) -> RecoverableNpyDataset:
        """Create a new store and its zero-progress durable checkpoint."""
        normalised = _normalise_shapes(shapes)
        if not sample_array_names or not sample_array_names.issubset(normalised):
            raise ValueError("sample array names must be a non-empty subset of array shapes")
        static_names = set(normalised) - sample_array_names
        if set(initial_arrays or {}) != static_names:
            raise ValueError("initial arrays must provide exactly every static array")
        if partial_dir.exists() or recovery_path.exists():
            raise FileExistsError("partial dataset or recovery checkpoint already exists")
        partial_dir.mkdir(parents=False)
        arrays: dict[str, FloatArray] = {}
        try:
            for name, shape in normalised.items():
                array = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
                    partial_dir / f"{name}.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=shape,
                )
                arrays[name] = cast(FloatArray, array)
            store = cls(
                partial_dir=partial_dir,
                recovery_path=recovery_path,
                arrays=arrays,
                sample_array_names=sample_array_names,
                run_contract=dict(run_contract),
                accepted_samples=0,
                next_candidate_index=0,
                rejection_counts=Counter(),
                committed_chunks=[],
                static_sha256={},
            )
            for name, value in (initial_arrays or {}).items():
                store.write_array(name, value)
            store.static_sha256 = {
                name: _array_digest(store.arrays[name]) for name in sorted(static_names)
            }
            store.checkpoint(accepted_samples=0, next_candidate_index=0)
            return store
        except Exception:
            for array in arrays.values():
                if isinstance(array, np.memmap):
                    array.flush()
            raise

    @classmethod
    def resume(
        cls,
        *,
        partial_dir: Path,
        recovery_path: Path,
        shapes: Mapping[str, tuple[int, ...]],
        sample_array_names: frozenset[str],
        run_contract: Mapping[str, Any],
    ) -> RecoverableNpyDataset:
        """Open a checkpoint only when its immutable run contract matches exactly."""
        normalised = _normalise_shapes(shapes)
        if not sample_array_names or not sample_array_names.issubset(normalised):
            raise ValueError("sample array names must be a non-empty subset of array shapes")
        if not partial_dir.is_dir() or partial_dir.is_symlink():
            raise FileNotFoundError(f"partial dataset is missing: {partial_dir}")
        if not recovery_path.is_file() or recovery_path.is_symlink():
            raise FileNotFoundError(f"recovery checkpoint is missing: {recovery_path}")
        raw = checked_json_load(recovery_path, max_bytes=MAX_RECOVERY_BYTES)
        if not isinstance(raw, dict):
            raise ValueError("recovery checkpoint must be a JSON object")
        state = cast(dict[str, Any], raw)
        if state.get("schema_version") != RECOVERY_SCHEMA:
            raise ValueError(f"recovery schema must be {RECOVERY_SCHEMA}")
        if state.get("run_contract") != dict(run_contract):
            raise ValueError("recovery run contract does not match this invocation")
        declared_shapes = {name: list(shape) for name, shape in normalised.items()}
        if state.get("array_shapes") != declared_shapes:
            raise ValueError("recovery array shapes do not match this invocation")
        if state.get("sample_array_names") != sorted(sample_array_names):
            raise ValueError("recovery sample array names do not match this invocation")
        accepted = state.get("accepted_samples")
        next_candidate = state.get("next_candidate_index")
        raw_rejections = state.get("rejection_counts")
        raw_chunks = state.get("committed_chunks")
        raw_static_sha256 = state.get("static_sha256")
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted < 0
            or isinstance(next_candidate, bool)
            or not isinstance(next_candidate, int)
            or next_candidate < accepted
            or not isinstance(raw_rejections, dict)
            or not isinstance(raw_chunks, list)
            or not isinstance(raw_static_sha256, dict)
        ):
            raise ValueError("recovery progress counters are invalid")
        arrays: dict[str, FloatArray] = {}
        expected_files = {f"{name}.npy" for name in normalised}
        actual_files = {
            path.name for path in partial_dir.iterdir() if path.is_file() or path.is_symlink()
        }
        allowed_files = expected_files | {"manifest.json"}
        if not expected_files.issubset(actual_files) or not actual_files.issubset(allowed_files):
            raise ValueError("partial dataset file inventory does not match the recovery contract")
        for name, shape in normalised.items():
            path = partial_dir / f"{name}.npy"
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"recovery array is missing or unsafe: {path.name}")
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
            if array.dtype != np.dtype(np.float64) or array.shape != shape:
                raise ValueError(f"recovery array contract mismatch: {path.name}")
            arrays[name] = cast(FloatArray, array)
        rejections: Counter[str] = Counter()
        for reason, count in raw_rejections.items():
            if (
                not isinstance(reason, str)
                or not reason
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("recovery rejection counts are invalid")
            rejections[reason] = count
        if next_candidate != accepted + sum(rejections.values()):
            raise ValueError("recovery candidate cursor does not match accepted and rejected counts")
        static_names = set(normalised) - sample_array_names
        expected_static_sha256 = {
            name: _array_digest(arrays[name]) for name in sorted(static_names)
        }
        if raw_static_sha256 != expected_static_sha256:
            raise ValueError("recovered static array SHA-256 does not match the checkpoint")
        chunks: list[dict[str, Any]] = []
        cursor = 0
        for index, raw_chunk in enumerate(raw_chunks):
            if not isinstance(raw_chunk, dict):
                raise ValueError(f"recovery committed_chunks[{index}] must be an object")
            chunk = cast(dict[str, Any], raw_chunk)
            start, stop, digests = chunk.get("start"), chunk.get("stop"), chunk.get("sha256")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start != cursor
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or stop <= cursor
                or stop > accepted
                or not isinstance(digests, dict)
                or set(digests) != sample_array_names
            ):
                raise ValueError("recovery committed chunk contract is invalid")
            expected_digests = {
                name: _array_digest(arrays[name][cursor:stop])
                for name in sorted(sample_array_names)
            }
            if digests != expected_digests:
                raise ValueError("recovered sample chunk SHA-256 does not match the checkpoint")
            chunks.append(chunk)
            cursor = stop
        if cursor != accepted:
            raise ValueError("recovery committed chunks do not cover the durable sample prefix")
        return cls(
            partial_dir=partial_dir,
            recovery_path=recovery_path,
            arrays=arrays,
            sample_array_names=sample_array_names,
            run_contract=dict(run_contract),
            accepted_samples=accepted,
            next_candidate_index=next_candidate,
            rejection_counts=rejections,
            committed_chunks=chunks,
            static_sha256=expected_static_sha256,
        )

    def write_array(self, name: str, value: FloatArray) -> None:
        """Write one complete static array after an exact shape and finiteness check."""
        target = self.arrays[name]
        source = np.asarray(value, dtype=np.float64)
        if source.shape != target.shape or not np.all(np.isfinite(source)):
            raise ValueError(f"invalid static array value for {name}")
        target[...] = source

    def require_array_equal(self, name: str, expected: FloatArray) -> None:
        """Authenticate regenerated static metadata against its recovered bytes."""
        value = np.asarray(expected, dtype=np.float64)
        if value.shape != self.arrays[name].shape or not np.array_equal(self.arrays[name], value):
            raise ValueError(f"recovered static array differs from regenerated {name}")

    def write_sample(self, row: int, values: Mapping[str, FloatArray]) -> None:
        """Write a complete accepted sample into its deterministic output row."""
        if row != self.written_samples:
            raise ValueError(f"sample row {row} does not equal the next unwritten position")
        if not values:
            raise ValueError("an accepted sample must contain at least one array")
        if set(values) != self.sample_array_names:
            raise ValueError("accepted sample arrays do not match the recovery contract")
        for name, value in values.items():
            target = self.arrays[name]
            source = np.asarray(value, dtype=np.float64)
            if target.ndim < 1 or row >= target.shape[0] or source.shape != target.shape[1:]:
                raise ValueError(f"invalid sample shape for {name}")
            if not np.all(np.isfinite(source)):
                raise ValueError(f"non-finite sample for {name}")
            target[row] = source
        self.written_samples += 1

    def checkpoint(
        self,
        *,
        accepted_samples: int,
        next_candidate_index: int,
        rejection_counts: Mapping[str, int] | None = None,
    ) -> None:
        """Flush array writes before atomically advancing the durable progress cursor."""
        if (
            accepted_samples < self.accepted_samples
            or accepted_samples > self.written_samples
            or next_candidate_index < accepted_samples
        ):
            raise ValueError("checkpoint progress cannot move backwards")
        next_rejection_counts = (
            Counter(rejection_counts) if rejection_counts is not None else self.rejection_counts
        )
        if next_candidate_index != accepted_samples + sum(next_rejection_counts.values()):
            raise ValueError("candidate cursor must equal accepted plus rejected candidates")
        next_chunks = list(self.committed_chunks)
        if accepted_samples > self.accepted_samples:
            chunk = {
                "start": self.accepted_samples,
                "stop": accepted_samples,
                "sha256": {
                    name: _array_digest(
                        self.arrays[name][self.accepted_samples : accepted_samples]
                    )
                    for name in sorted(self.sample_array_names)
                },
            }
            next_chunks.append(chunk)
        for name, array in self.arrays.items():
            if isinstance(array, np.memmap):
                array.flush()
            descriptor = os.open(self.partial_dir / f"{name}.npy", os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        state = {
            "schema_version": RECOVERY_SCHEMA,
            "run_contract": self.run_contract,
            "array_shapes": {name: list(array.shape) for name, array in self.arrays.items()},
            "sample_array_names": sorted(self.sample_array_names),
            "accepted_samples": accepted_samples,
            "next_candidate_index": next_candidate_index,
            "rejection_counts": dict(sorted(next_rejection_counts.items())),
            "committed_chunks": next_chunks,
            "static_sha256": self.static_sha256,
        }
        _atomic_json_write(self.recovery_path, state)
        self.accepted_samples = accepted_samples
        self.next_candidate_index = next_candidate_index
        self.rejection_counts = next_rejection_counts
        self.committed_chunks = next_chunks

    def remove_recovery_checkpoint(self) -> None:
        """Remove the external checkpoint after atomic dataset installation."""
        self.recovery_path.unlink()
        _sync_directory(self.recovery_path.parent)


__all__ = ["RECOVERY_SCHEMA", "RecoverableNpyDataset"]
