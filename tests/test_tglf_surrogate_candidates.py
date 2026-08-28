# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Candidate Tests
"""Determinism, numerical learning and fail-closed candidate tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core.tglf_surrogate_bridge import TGLFSurrogate
from scpn_fusion.core.tglf_surrogate_candidates import (
    CompactNeuralEnsemble,
    QuadraticPolynomialCandidate,
    RandomisedTreeEnsemble,
    TGLFRegressionCandidate,
)


def _regression_problem() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    generator = np.random.default_rng(20260826)
    features = generator.uniform(-1.0, 1.0, size=(48, 6))
    targets = np.column_stack(
        (
            0.4 + 0.8 * features[:, 0] - 0.2 * features[:, 1] ** 2,
            features[:, 2] * features[:, 3] + 0.1 * features[:, 4],
            -0.3 * features[:, 0] + 0.7 * features[:, 5],
        )
    )
    return features, targets


def test_quadratic_adapter_exactly_matches_existing_surrogate() -> None:
    """The polynomial candidate is an adapter, not a second implementation."""
    features, targets = _regression_problem()
    candidate = QuadraticPolynomialCandidate(ridge=1.0e-3).fit(features, targets)
    baseline = TGLFSurrogate(
        features=tuple(f"feature_{index}" for index in range(features.shape[1])),
        targets=tuple(f"target_{index}" for index in range(targets.shape[1])),
        ridge=1.0e-3,
    ).fit(features, targets)
    np.testing.assert_array_equal(candidate.predict(features), baseline.predict(features))
    assert candidate.state_bytes() > 0


def test_tree_ensemble_is_seeded_finite_and_learns_training_structure() -> None:
    """Two independent fixed-seed ensembles have identical finite predictions."""
    features, targets = _regression_problem()
    first = RandomisedTreeEnsemble(
        trees=12,
        maximum_depth=5,
        minimum_leaf_rows=2,
        split_probes=6,
        seed=17,
    ).fit(features, targets)
    second = RandomisedTreeEnsemble(
        trees=12,
        maximum_depth=5,
        minimum_leaf_rows=2,
        split_probes=6,
        seed=17,
    ).fit(features, targets)
    prediction = first.predict(features)
    np.testing.assert_array_equal(prediction, second.predict(features))
    assert np.all(np.isfinite(prediction))
    assert np.sqrt(np.mean((prediction - targets) ** 2)) < np.std(targets)
    assert first.state_bytes() == second.state_bytes() > 0


def test_compact_neural_ensemble_is_seeded_and_improves_over_target_mean() -> None:
    """Fixed full-batch training is reproducible and resolves nonlinear signal."""
    features, targets = _regression_problem()
    first = CompactNeuralEnsemble(
        members=2,
        hidden_width=12,
        epochs=400,
        learning_rate=1.0e-2,
        l2=1.0e-4,
        seed=29,
    ).fit(features, targets)
    second = CompactNeuralEnsemble(
        members=2,
        hidden_width=12,
        epochs=400,
        learning_rate=1.0e-2,
        l2=1.0e-4,
        seed=29,
    ).fit(features, targets)
    prediction = first.predict(features)
    np.testing.assert_array_equal(prediction, second.predict(features))
    mean_error = np.sqrt(np.mean((targets - np.mean(targets, axis=0)) ** 2))
    fitted_error = np.sqrt(np.mean((targets - prediction) ** 2))
    assert fitted_error < 0.35 * mean_error
    assert first.state_bytes() == second.state_bytes() > 0


@pytest.mark.parametrize(
    "candidate",
    [
        QuadraticPolynomialCandidate(),
        RandomisedTreeEnsemble(trees=2),
        CompactNeuralEnsemble(members=1, epochs=1),
    ],
)
def test_candidates_reject_unfitted_prediction_and_nonfinite_fit(
    candidate: TGLFRegressionCandidate,
) -> None:
    """Every family fails closed before fit and on non-finite training matrices."""
    features, targets = _regression_problem()
    with pytest.raises(RuntimeError, match="not fit"):
        candidate.predict(features)
    with pytest.raises(RuntimeError, match="not fit"):
        candidate.state_bytes()
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        candidate.fit(features, targets)


def test_candidate_configuration_and_prediction_width_are_validated() -> None:
    """Invalid hyperparameters and incompatible prediction widths are rejected."""
    with pytest.raises(ValueError):
        QuadraticPolynomialCandidate(ridge=0.0)
    with pytest.raises(ValueError):
        RandomisedTreeEnsemble(trees=0)
    with pytest.raises(ValueError):
        CompactNeuralEnsemble(epochs=0)
    features, targets = _regression_problem()
    tree = RandomisedTreeEnsemble(trees=2).fit(features, targets)
    neural = CompactNeuralEnsemble(members=1, epochs=2).fit(features, targets)
    with pytest.raises(ValueError, match="width"):
        tree.predict(features[:, :-1])
    with pytest.raises(ValueError, match="width"):
        neural.predict(features[:, :-1])


@pytest.mark.parametrize(
    "factory",
    [
        QuadraticPolynomialCandidate,
        lambda: RandomisedTreeEnsemble(trees=1),
        lambda: CompactNeuralEnsemble(members=1, epochs=1),
    ],
)
def test_every_candidate_fit_surface_rejects_misaligned_or_empty_matrices(
    factory: Callable[[], TGLFRegressionCandidate],
) -> None:
    """All three public fit surfaces enforce the shared finite matrix contract."""
    features, targets = _regression_problem()
    invalid_pairs = (
        (features[0], targets),
        (features, targets[0]),
        (features[:-1], targets),
        (features[:1], targets[:1]),
        (np.empty((4, 0), dtype=np.float64), targets[:4]),
        (features[:4], np.empty((4, 0), dtype=np.float64)),
    )
    for invalid_features, invalid_targets in invalid_pairs:
        with pytest.raises(ValueError, match="aligned|non-empty rows"):
            factory().fit(invalid_features, invalid_targets)

    nonfinite_targets = targets.copy()
    nonfinite_targets[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        factory().fit(features, nonfinite_targets)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: QuadraticPolynomialCandidate(ridge=float("nan")),
        lambda: QuadraticPolynomialCandidate(ridge=-1.0),
        lambda: RandomisedTreeEnsemble(trees=0),
        lambda: RandomisedTreeEnsemble(maximum_depth=0),
        lambda: RandomisedTreeEnsemble(minimum_leaf_rows=0),
        lambda: RandomisedTreeEnsemble(split_probes=0),
        lambda: CompactNeuralEnsemble(members=0),
        lambda: CompactNeuralEnsemble(hidden_width=0),
        lambda: CompactNeuralEnsemble(epochs=0),
        lambda: CompactNeuralEnsemble(learning_rate=0.0),
        lambda: CompactNeuralEnsemble(learning_rate=float("nan")),
        lambda: CompactNeuralEnsemble(l2=-1.0),
        lambda: CompactNeuralEnsemble(l2=float("nan")),
    ],
)
def test_each_candidate_hyperparameter_fails_closed(factory: Callable[[], object]) -> None:
    """Every independent constructor bound rejects its invalid edge."""
    with pytest.raises(ValueError):
        factory()


def test_fitted_candidates_accept_one_row_and_reject_nonfinite_or_rank_three_inputs() -> None:
    """Tree and neural prediction surfaces accept vectors but reject malformed batches."""
    features, targets = _regression_problem()
    candidates: tuple[TGLFRegressionCandidate, ...] = (
        RandomisedTreeEnsemble(trees=2, maximum_depth=3).fit(features, targets),
        CompactNeuralEnsemble(members=1, hidden_width=4, epochs=2).fit(features, targets),
    )
    for candidate in candidates:
        assert candidate.predict(features[0]).shape == (1, targets.shape[1])
        with pytest.raises(ValueError, match="width"):
            candidate.predict(features.reshape(2, 4, -1))
        nonfinite = features[:1].copy()
        nonfinite[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            candidate.predict(nonfinite)


def test_candidate_state_byte_counts_match_exact_owned_arrays() -> None:
    """Reported state sizes equal the fitted NumPy arrays owned by each family."""
    features, targets = _regression_problem()
    polynomial = QuadraticPolynomialCandidate().fit(features, targets)
    assert polynomial._model is not None
    assert polynomial._model._mean is not None
    assert polynomial._model._std is not None
    assert polynomial._model._weights is not None
    assert polynomial.state_bytes() == sum(
        array.nbytes
        for array in (
            polynomial._model._mean,
            polynomial._model._std,
            polynomial._model._weights,
        )
    )

    tree = RandomisedTreeEnsemble(trees=2, maximum_depth=3).fit(features, targets)
    assert tree.state_bytes() == sum(state.state_bytes for state in tree._states)

    neural = CompactNeuralEnsemble(members=2, hidden_width=4, epochs=2).fit(features, targets)
    assert neural._feature_mean is not None
    assert neural._feature_std is not None
    assert neural.state_bytes() == (
        neural._feature_mean.nbytes
        + neural._feature_std.nbytes
        + sum(state.state_bytes for state in neural._states)
    )


def test_polynomial_incomplete_state_guard_follows_a_real_fit() -> None:
    """A corrupted fitted baseline cannot report a misleading partial state size."""
    features, targets = _regression_problem()
    candidate = QuadraticPolynomialCandidate().fit(features, targets)
    assert candidate._model is not None
    candidate._model._weights = None
    with pytest.raises(RuntimeError, match="incomplete"):
        candidate.state_bytes()


def test_constant_feature_tree_returns_its_real_mean_leaf() -> None:
    """An unsplittable public fit remains a valid deterministic mean predictor."""
    features = np.ones((8, 3), dtype=np.float64)
    targets = np.arange(16, dtype=np.float64).reshape(8, 2)
    candidate = RandomisedTreeEnsemble(trees=1, maximum_depth=3).fit(features, targets)
    prediction = candidate.predict(features[0])
    np.testing.assert_array_equal(prediction[0], np.mean(targets, axis=0))
