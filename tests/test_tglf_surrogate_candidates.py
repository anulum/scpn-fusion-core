# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Candidate Tests
"""Determinism, numerical learning and fail-closed candidate tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core.tglf_surrogate_bridge import TGLFSurrogate
from scpn_fusion.core.tglf_surrogate_candidates import (
    CompactNeuralEnsemble,
    QuadraticPolynomialCandidate,
    RandomisedTreeEnsemble,
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
def test_candidates_reject_unfitted_prediction_and_nonfinite_fit(candidate: object) -> None:
    """Every family fails closed before fit and on non-finite training matrices."""
    features, targets = _regression_problem()
    with pytest.raises(RuntimeError, match="not fit"):
        candidate.predict(features)  # type: ignore[attr-defined]
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        candidate.fit(features, targets)  # type: ignore[attr-defined]


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
