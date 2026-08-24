# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Equilibrium Runtime
"""Bounded branch-trunk operator runtime for fixed-machine equilibrium fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.safe_loaders import checked_np_load

FloatArray: TypeAlias = NDArray[np.float64]


def _silu(values: FloatArray) -> FloatArray:
    sigmoid = np.empty_like(values)
    nonnegative = values >= 0.0
    sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    negative_exponential = np.exp(values[~nonnegative])
    sigmoid[~nonnegative] = negative_exponential / (1.0 + negative_exponential)
    return np.asarray(values * sigmoid, dtype=np.float64)


def _load_layers(archive: Any, prefix: str) -> list[tuple[FloatArray, FloatArray]]:
    count = int(archive[f"{prefix}_n_layers"][0])
    if count < 1 or count > 16:
        raise ValueError(f"{prefix} layer count is outside the supported range")
    layers: list[tuple[FloatArray, FloatArray]] = []
    for index in range(count):
        weight = np.asarray(archive[f"{prefix}_{index}_W"], dtype=np.float64)
        bias = np.asarray(archive[f"{prefix}_{index}_b"], dtype=np.float64)
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[1] != len(bias):
            raise ValueError(f"{prefix} layer {index} has inconsistent dimensions")
        if layers and layers[-1][0].shape[1] != weight.shape[0]:
            raise ValueError(f"{prefix} layer {index} input width is inconsistent")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError(f"{prefix} layer {index} contains non-finite values")
        layers.append((weight, bias))
    return layers


def _forward(layers: list[tuple[FloatArray, FloatArray]], values: FloatArray) -> FloatArray:
    activation = values
    for index, (weight, bias) in enumerate(layers):
        activation = activation @ weight + bias
        if index + 1 < len(layers):
            activation = _silu(activation)
    return np.asarray(activation, dtype=np.float64)


class DeepONetEquilibriumAccelerator:
    """Evaluate one manifest-bound equilibrium DeepONet candidate.

    The branch network encodes 17 causal pre-solve controls. The trunk network
    encodes the artifact's fixed physical ``(R, Z)`` grid in metres. Their
    scaled inner product reconstructs ``psi_total`` in Wb/rad around the
    training-only spatial mean. This follows the branch-trunk construction of
    Lu et al. (2021), DOI: 10.1038/s42256-021-00302-5, while binding the branch
    to a finite control vector rather than claiming a cross-machine operator.

    Notes
    -----
    Call :meth:`load_weights` before inference. One instance represents only
    the machine manifest and coordinate grid stored in its NPZ artifact.
    """

    def __init__(self, *, prefer_rust: bool = True) -> None:
        """Create an unloaded runtime with optional native-first dispatch.

        Parameters
        ----------
        prefer_rust : bool, optional
            Use the validated Rust/PyO3 inference kernel when installed. Set
            false to retain the NumPy reference path for parity measurement.
        """
        self._branch: list[tuple[FloatArray, FloatArray]] = []
        self._trunk: list[tuple[FloatArray, FloatArray]] = []
        self._input_mean: FloatArray | None = None
        self._input_std: FloatArray | None = None
        self._coordinates: FloatArray | None = None
        self._coordinate_mean: FloatArray | None = None
        self._coordinate_std: FloatArray | None = None
        self._field_mean: FloatArray | None = None
        self._field_scale = 0.0
        self._basis_width = 0
        self._grid_shape = (0, 0)
        self._trunk_cache: FloatArray | None = None
        self.feature_names: tuple[str, ...] = ()
        self.machine_manifest_sha256 = ""
        self.backend = "unloaded"
        self.is_loaded = False
        self._prefer_rust = prefer_rust
        self._native: Any | None = None

    def load_weights(self, path: str | Path) -> None:
        """Authenticate and load a pickle-free DeepONet NPZ artifact.

        Parameters
        ----------
        path : str | Path
            Artifact produced by the machine-conditioned DeepONet trainer.

        Raises
        ------
        OSError
            If the artifact cannot be opened.
        ValueError
            If the schema, network dimensions, feature order, grid, digest,
            scales, or numerical values violate the runtime contract.
        """
        with checked_np_load(path, allow_pickle=False) as archive:
            required = {
                "artifact_schema",
                "branch_n_layers",
                "trunk_n_layers",
                "input_mean",
                "input_std",
                "coordinates_rz_m",
                "coordinate_mean",
                "coordinate_std",
                "field_mean",
                "field_scale",
                "basis_width",
                "grid_nh",
                "grid_nw",
                "feature_names",
                "dataset_manifest_sha256",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"DeepONet artifact is missing keys: {missing}")
            schema = str(archive["artifact_schema"][0])
            if schema != "scpn-fusion.equilibrium-deeponet.v1":
                raise ValueError("unsupported DeepONet artifact schema")
            branch = _load_layers(archive, "branch")
            trunk = _load_layers(archive, "trunk")
            input_mean = np.asarray(archive["input_mean"], dtype=np.float64)
            input_std = np.asarray(archive["input_std"], dtype=np.float64)
            coordinates = np.asarray(archive["coordinates_rz_m"], dtype=np.float64)
            coordinate_mean = np.asarray(archive["coordinate_mean"], dtype=np.float64)
            coordinate_std = np.asarray(archive["coordinate_std"], dtype=np.float64)
            field_mean = np.asarray(archive["field_mean"], dtype=np.float64)
            field_scale = float(archive["field_scale"][0])
            basis_width = int(archive["basis_width"][0])
            grid_shape = (int(archive["grid_nh"][0]), int(archive["grid_nw"][0]))
            feature_names = tuple(str(value) for value in archive["feature_names"])
            manifest_sha256 = str(archive["dataset_manifest_sha256"][0])

        if input_mean.shape != input_std.shape or input_mean.ndim != 1:
            raise ValueError("DeepONet input normalisation has inconsistent dimensions")
        if len(feature_names) != len(input_mean) or len(set(feature_names)) != len(feature_names):
            raise ValueError("DeepONet feature contract is inconsistent")
        if coordinates.shape != (grid_shape[0] * grid_shape[1], 2):
            raise ValueError("DeepONet coordinate grid is inconsistent")
        if coordinate_mean.shape != (2,) or coordinate_std.shape != (2,):
            raise ValueError("DeepONet coordinate normalisation must contain R and Z")
        if field_mean.shape != (len(coordinates),):
            raise ValueError("DeepONet field mean does not match the coordinate grid")
        if branch[0][0].shape[0] != len(input_mean) or trunk[0][0].shape[0] != 2:
            raise ValueError("DeepONet network inputs do not match the declared contract")
        if branch[-1][0].shape[1] != basis_width or trunk[-1][0].shape[1] != basis_width:
            raise ValueError("DeepONet branch and trunk basis widths differ")
        arrays = (input_mean, input_std, coordinates, coordinate_mean, coordinate_std, field_mean)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("DeepONet artifact contains non-finite arrays")
        if np.any(input_std <= 0.0) or np.any(coordinate_std <= 0.0) or field_scale <= 0.0:
            raise ValueError("DeepONet normalisation scales must be positive")
        if len(manifest_sha256) != 64:
            raise ValueError("DeepONet machine manifest digest is invalid")

        self._branch = branch
        self._trunk = trunk
        self._input_mean = input_mean
        self._input_std = input_std
        self._coordinates = coordinates
        self._coordinate_mean = coordinate_mean
        self._coordinate_std = coordinate_std
        self._field_mean = field_mean
        self._field_scale = field_scale
        self._basis_width = basis_width
        self._grid_shape = grid_shape
        self._trunk_cache = None
        self.feature_names = feature_names
        self.machine_manifest_sha256 = manifest_sha256
        self._native = None
        if self._prefer_rust:
            try:
                from scpn_fusion.core._multi_compat import dispatch_kernel_class

                native_class = dispatch_kernel_class("deeponet_equilibrium")
            except (AttributeError, ImportError, RuntimeError, TypeError):
                native_class = None
            if native_class is not None:
                self._native = native_class(
                    (
                        [weight.tolist() for weight, _ in branch],
                        [bias.tolist() for _, bias in branch],
                    ),
                    (
                        [weight.tolist() for weight, _ in trunk],
                        [bias.tolist() for _, bias in trunk],
                    ),
                    (input_mean.tolist(), input_std.tolist()),
                    (
                        coordinates.tolist(),
                        coordinate_mean.tolist(),
                        coordinate_std.tolist(),
                    ),
                    (
                        field_mean.tolist(),
                        field_scale,
                        basis_width,
                        grid_shape[0],
                        grid_shape[1],
                    ),
                )
        self.backend = "rust" if self._native is not None else "numpy"
        self.is_loaded = True

    def _require_arrays(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        if not self.is_loaded:
            raise RuntimeError("DeepONet weights have not been loaded")
        arrays = (
            self._input_mean,
            self._input_std,
            self._coordinates,
            self._coordinate_mean,
            self._coordinate_std,
            self._field_mean,
        )
        if any(array is None for array in arrays):
            raise RuntimeError("DeepONet runtime state is incomplete")
        return cast(
            tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray],
            arrays,
        )

    def predict_batch(self, features: FloatArray) -> FloatArray:
        """Predict poloidal-flux fields for causal pre-solve feature rows.

        Parameters
        ----------
        features : FloatArray
            Array with shape ``(batch, 17)`` in the artifact-declared feature
            order and units.

        Returns
        -------
        FloatArray
            ``psi_total`` with shape ``(batch, n_z, n_r)`` in Wb/rad.

        Raises
        ------
        RuntimeError
            If weights are absent or inference produces non-finite output.
        ValueError
            If the input shape or values violate the feature contract.
        """
        input_mean, input_std, coordinates, coordinate_mean, coordinate_std, field_mean = (
            self._require_arrays()
        )
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(input_mean):
            raise ValueError(f"DeepONet inputs must have shape (batch, {len(input_mean)})")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("DeepONet inputs must be finite")
        if self._native is not None:
            try:
                flat = np.asarray(self._native.predict_batch(matrix), dtype=np.float64)
            except ValueError as exc:
                raise RuntimeError(f"DeepONet native inference failed: {exc}") from exc
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                branch = _forward(self._branch, (matrix - input_mean) / input_std)
                if self._trunk_cache is None:
                    normalised_coordinates = (coordinates - coordinate_mean) / coordinate_std
                    self._trunk_cache = _forward(self._trunk, normalised_coordinates)
                normalised_field = branch @ self._trunk_cache.T / np.sqrt(self._basis_width)
                flat = field_mean[np.newaxis, :] + self._field_scale * normalised_field
        result = np.asarray(flat.reshape(len(matrix), *self._grid_shape), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("DeepONet inference produced non-finite output")
        return result

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict one poloidal-flux field on the bound coordinate grid.

        Parameters
        ----------
        features : FloatArray
            One 17-element causal pre-solve control vector in the
            artifact-declared order and units.

        Returns
        -------
        FloatArray
            ``psi_total`` with shape ``(n_z, n_r)`` in Wb/rad.

        Raises
        ------
        RuntimeError
            If weights are absent or inference produces non-finite output.
        ValueError
            If ``features`` is not a finite one-dimensional control vector.
        """
        row = np.asarray(features, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("DeepONet single-row input must be one-dimensional")
        return np.asarray(self.predict_batch(row[np.newaxis, :])[0], dtype=np.float64)


__all__ = ["DeepONetEquilibriumAccelerator"]
