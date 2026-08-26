# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Candidate Families
"""Deterministic regression families for TGLF surrogate model selection.

The models in this module are candidate-study surfaces. They do not confer a
promoted runtime identity. All families accept the same finite float64 feature
and target matrices and expose exact in-memory state sizes for comparative
reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.tglf_surrogate_bridge import TGLFSurrogate

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TGLFRegressionCandidate(Protocol):
    """Common fitted-candidate contract used by the selection study."""

    def fit(self, features: FloatArray, targets: FloatArray) -> Self:
        """Fit one candidate from aligned finite matrices."""

    def predict(self, features: FloatArray) -> FloatArray:
        """Return one finite prediction row per input row."""

    def state_bytes(self) -> int:
        """Return bytes occupied by fitted numerical state."""


def _validated_matrices(
    features: FloatArray,
    targets: FloatArray,
    *,
    minimum_rows: int,
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must be aligned two-dimensional matrices")
    if x.shape[0] < minimum_rows or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError(f"candidate fit requires at least {minimum_rows} non-empty rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("candidate fit matrices must contain only finite values")
    return x, y


class QuadraticPolynomialCandidate:
    """Adapter exposing the existing quadratic TGLF surrogate as a candidate."""

    def __init__(self, *, ridge: float = 1.0e-3) -> None:
        """Configure the existing per-feature quadratic ridge model."""
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        self.ridge = float(ridge)
        self._model: TGLFSurrogate | None = None

    def fit(self, features: FloatArray, targets: FloatArray) -> Self:
        """Fit the existing quadratic ridge implementation."""
        x, y = _validated_matrices(features, targets, minimum_rows=2)
        feature_names = tuple(f"feature_{index}" for index in range(x.shape[1]))
        target_names = tuple(f"target_{index}" for index in range(y.shape[1]))
        self._model = TGLFSurrogate(
            features=feature_names,
            targets=target_names,
            ridge=self.ridge,
        ).fit(x, y)
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict with the fitted quadratic baseline."""
        if self._model is None:
            raise RuntimeError("quadratic candidate is not fit")
        return self._model.predict(np.asarray(features, dtype=np.float64))

    def state_bytes(self) -> int:
        """Return bytes held by fitted baseline arrays."""
        if self._model is None:
            raise RuntimeError("quadratic candidate is not fit")
        arrays = (
            self._model._mean,
            self._model._std,
            self._model._weights,
        )
        if any(array is None for array in arrays):
            raise RuntimeError("quadratic candidate fitted state is incomplete")
        return int(sum(array.nbytes for array in arrays if array is not None))


@dataclass(frozen=True)
class _TreeState:
    features: IntArray
    thresholds: FloatArray
    left_children: IntArray
    right_children: IntArray
    values: FloatArray

    @property
    def state_bytes(self) -> int:
        return int(
            self.features.nbytes
            + self.thresholds.nbytes
            + self.left_children.nbytes
            + self.right_children.nbytes
            + self.values.nbytes
        )


class RandomisedTreeEnsemble:
    """Extremely randomised multi-output regression-tree ensemble.

    Each node samples a square-root subset of features and uniformly probes a
    fixed number of thresholds. The split minimises summed target squared error
    on the supplied target scale. The caller therefore supplies training-only
    normalised targets when channels have different physical units.
    """

    def __init__(
        self,
        *,
        trees: int = 64,
        maximum_depth: int = 6,
        minimum_leaf_rows: int = 2,
        split_probes: int = 8,
        seed: int = 20260826,
    ) -> None:
        """Configure deterministic tree count, depth and split sampling."""
        if trees < 1:
            raise ValueError("trees must be positive")
        if maximum_depth < 1:
            raise ValueError("maximum_depth must be positive")
        if minimum_leaf_rows < 1:
            raise ValueError("minimum_leaf_rows must be positive")
        if split_probes < 1:
            raise ValueError("split_probes must be positive")
        self.trees = int(trees)
        self.maximum_depth = int(maximum_depth)
        self.minimum_leaf_rows = int(minimum_leaf_rows)
        self.split_probes = int(split_probes)
        self.seed = int(seed)
        self._states: tuple[_TreeState, ...] = ()
        self._feature_width: int | None = None
        self._target_width: int | None = None

    def _build_tree(
        self,
        x: FloatArray,
        y: FloatArray,
        rng: np.random.Generator,
    ) -> _TreeState:
        features: list[int] = []
        thresholds: list[float] = []
        left_children: list[int] = []
        right_children: list[int] = []
        values: list[FloatArray] = []
        candidate_width = max(1, int(math.ceil(math.sqrt(x.shape[1]))))

        def add_node(indices: IntArray, depth: int) -> int:
            node_index = len(features)
            features.append(-1)
            thresholds.append(math.nan)
            left_children.append(-1)
            right_children.append(-1)
            values.append(np.mean(y[indices], axis=0, dtype=np.float64))
            if depth >= self.maximum_depth or indices.size < 2 * self.minimum_leaf_rows:
                return node_index

            selected = rng.choice(x.shape[1], size=candidate_width, replace=False)
            best: tuple[float, int, float, IntArray, IntArray] | None = None
            for feature_index_raw in selected:
                feature_index = int(feature_index_raw)
                column = x[indices, feature_index]
                lower = float(np.min(column))
                upper = float(np.max(column))
                if not upper > lower:
                    continue
                for threshold_raw in rng.uniform(lower, upper, size=self.split_probes):
                    threshold = float(threshold_raw)
                    left_mask = column <= threshold
                    left = indices[left_mask]
                    right = indices[~left_mask]
                    if left.size < self.minimum_leaf_rows or right.size < self.minimum_leaf_rows:
                        continue
                    left_error = y[left] - np.mean(y[left], axis=0, dtype=np.float64)
                    right_error = y[right] - np.mean(y[right], axis=0, dtype=np.float64)
                    score = float(
                        np.sum(left_error * left_error) + np.sum(right_error * right_error)
                    )
                    if best is None or score < best[0]:
                        best = (score, feature_index, threshold, left, right)
            if best is None:
                return node_index

            _, feature_index, threshold, left, right = best
            features[node_index] = feature_index
            thresholds[node_index] = threshold
            left_children[node_index] = add_node(left, depth + 1)
            right_children[node_index] = add_node(right, depth + 1)
            return node_index

        add_node(np.arange(x.shape[0], dtype=np.int64), 0)
        return _TreeState(
            features=np.asarray(features, dtype=np.int64),
            thresholds=np.asarray(thresholds, dtype=np.float64),
            left_children=np.asarray(left_children, dtype=np.int64),
            right_children=np.asarray(right_children, dtype=np.int64),
            values=np.asarray(values, dtype=np.float64),
        )

    def fit(self, features: FloatArray, targets: FloatArray) -> Self:
        """Fit all randomised trees from the same aligned training rows."""
        x, y = _validated_matrices(features, targets, minimum_rows=4)
        rng = np.random.default_rng(self.seed)
        self._states = tuple(self._build_tree(x, y, rng) for _ in range(self.trees))
        self._feature_width = x.shape[1]
        self._target_width = y.shape[1]
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        """Average deterministic predictions from every fitted tree."""
        if not self._states or self._feature_width is None or self._target_width is None:
            raise RuntimeError("tree ensemble is not fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self._feature_width or not np.all(np.isfinite(x)):
            raise ValueError(f"features must be finite with width {self._feature_width}")
        result = np.zeros((x.shape[0], self._target_width), dtype=np.float64)
        for tree in self._states:
            for row_index, row in enumerate(x):
                node = 0
                while tree.features[node] >= 0:
                    feature_index = int(tree.features[node])
                    if row[feature_index] <= tree.thresholds[node]:
                        node = int(tree.left_children[node])
                    else:
                        node = int(tree.right_children[node])
                result[row_index] += tree.values[node]
        result /= float(len(self._states))
        return result

    def state_bytes(self) -> int:
        """Return bytes occupied by all fitted tree arrays."""
        if not self._states:
            raise RuntimeError("tree ensemble is not fit")
        return int(sum(tree.state_bytes for tree in self._states))


@dataclass(frozen=True)
class _NetworkState:
    first_weights: FloatArray
    first_bias: FloatArray
    second_weights: FloatArray
    second_bias: FloatArray

    @property
    def state_bytes(self) -> int:
        return int(
            self.first_weights.nbytes
            + self.first_bias.nbytes
            + self.second_weights.nbytes
            + self.second_bias.nbytes
        )


class CompactNeuralEnsemble:
    """Deterministic float64 ensemble of one-hidden-layer tanh regressors."""

    def __init__(
        self,
        *,
        members: int = 5,
        hidden_width: int = 24,
        epochs: int = 1500,
        learning_rate: float = 1.0e-2,
        l2: float = 1.0e-4,
        seed: int = 20260826,
    ) -> None:
        """Configure ensemble size and fixed full-batch Adam training."""
        if members < 1 or hidden_width < 1 or epochs < 1:
            raise ValueError("members, hidden_width and epochs must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(l2) or l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        self.members = int(members)
        self.hidden_width = int(hidden_width)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)
        self._feature_mean: FloatArray | None = None
        self._feature_std: FloatArray | None = None
        self._states: tuple[_NetworkState, ...] = ()

    def _fit_member(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        seed: int,
    ) -> _NetworkState:
        rng = np.random.default_rng(seed)
        input_width = x.shape[1]
        target_width = y.shape[1]
        w1 = rng.normal(
            0.0,
            math.sqrt(1.0 / input_width),
            size=(input_width, self.hidden_width),
        )
        b1 = np.zeros(self.hidden_width, dtype=np.float64)
        w2 = rng.normal(
            0.0,
            math.sqrt(1.0 / self.hidden_width),
            size=(self.hidden_width, target_width),
        )
        b2 = np.zeros(target_width, dtype=np.float64)
        parameters = [w1, b1, w2, b2]
        first_moments = [np.zeros_like(value) for value in parameters]
        second_moments = [np.zeros_like(value) for value in parameters]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1.0e-8
        scale = 2.0 / float(x.shape[0] * target_width)

        for step in range(1, self.epochs + 1):
            hidden = np.tanh(x @ w1 + b1)
            prediction = hidden @ w2 + b2
            output_gradient = scale * (prediction - y)
            gradient_w2 = hidden.T @ output_gradient + 2.0 * self.l2 * w2
            gradient_b2 = np.sum(output_gradient, axis=0)
            hidden_gradient = (output_gradient @ w2.T) * (1.0 - hidden * hidden)
            gradient_w1 = x.T @ hidden_gradient + 2.0 * self.l2 * w1
            gradient_b1 = np.sum(hidden_gradient, axis=0)
            gradients = [gradient_w1, gradient_b1, gradient_w2, gradient_b2]

            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                first_moments[index] *= beta1
                first_moments[index] += (1.0 - beta1) * gradient
                second_moments[index] *= beta2
                second_moments[index] += (1.0 - beta2) * gradient * gradient
                corrected_first = first_moments[index] / (1.0 - beta1**step)
                corrected_second = second_moments[index] / (1.0 - beta2**step)
                parameter -= (
                    self.learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
                )

        return _NetworkState(
            first_weights=np.asarray(w1, dtype=np.float64),
            first_bias=np.asarray(b1, dtype=np.float64),
            second_weights=np.asarray(w2, dtype=np.float64),
            second_bias=np.asarray(b2, dtype=np.float64),
        )

    def fit(self, features: FloatArray, targets: FloatArray) -> Self:
        """Fit every member with training-only input standardisation."""
        x, y = _validated_matrices(features, targets, minimum_rows=4)
        self._feature_mean = np.mean(x, axis=0, dtype=np.float64)
        feature_std = np.std(x, axis=0, dtype=np.float64)
        self._feature_std = np.where(feature_std > 1.0e-12, feature_std, 1.0)
        normalised = (x - self._feature_mean) / self._feature_std
        self._states = tuple(
            self._fit_member(normalised, y, seed=self.seed + 1009 * index)
            for index in range(self.members)
        )
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        """Average predictions from all fitted neural members."""
        if self._feature_mean is None or self._feature_std is None or not self._states:
            raise RuntimeError("neural ensemble is not fit")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self._feature_mean.size or not np.all(np.isfinite(x)):
            raise ValueError(f"features must be finite with width {self._feature_mean.size}")
        normalised = (x - self._feature_mean) / self._feature_std
        prediction = np.zeros(
            (x.shape[0], self._states[0].second_bias.size),
            dtype=np.float64,
        )
        for state in self._states:
            hidden = np.tanh(normalised @ state.first_weights + state.first_bias)
            prediction += hidden @ state.second_weights + state.second_bias
        prediction /= float(len(self._states))
        return prediction

    def state_bytes(self) -> int:
        """Return bytes occupied by input scaling and all member parameters."""
        if self._feature_mean is None or self._feature_std is None or not self._states:
            raise RuntimeError("neural ensemble is not fit")
        return int(
            self._feature_mean.nbytes
            + self._feature_std.nbytes
            + sum(state.state_bytes for state in self._states)
        )


__all__ = [
    "CompactNeuralEnsemble",
    "QuadraticPolynomialCandidate",
    "RandomisedTreeEnsemble",
    "TGLFRegressionCandidate",
]
