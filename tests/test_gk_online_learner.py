# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Online Learner Tests
"""Tests for validation-guarded gyrokinetic online learning."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.gk_online_learner import LearnerConfig, OnlineLearner

FloatArray = NDArray[np.float64]
Weights = dict[str, FloatArray]


def _random_samples(n: int, rng: np.random.Generator) -> list[tuple[FloatArray, FloatArray]]:
    """Return deterministic finite 10D-to-3D training samples."""
    return [
        (
            np.asarray(rng.random(10), dtype=np.float64),
            np.asarray(rng.random(3), dtype=np.float64),
        )
        for _ in range(n)
    ]


def _weights(rng: np.random.Generator) -> Weights:
    """Return a complete finite three-layer MLP weight dictionary."""
    return {
        "w1": rng.normal(0.0, 0.05, (10, 64)),
        "b1": np.zeros(64, dtype=np.float64),
        "w2": rng.normal(0.0, 0.05, (64, 32)),
        "b2": np.zeros(32, dtype=np.float64),
        "w3": rng.normal(0.0, 0.05, (32, 3)),
        "b3": np.zeros(3, dtype=np.float64),
    }


def test_buffer_not_full_initially() -> None:
    """New learners do not report a full retraining buffer."""
    learner = OnlineLearner()
    assert not learner.buffer_full


def test_buffer_fills() -> None:
    """Adding enough samples trips the retraining trigger."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert learner.buffer_full


def test_try_retrain_skips_empty() -> None:
    """Retraining is skipped while the sample buffer is empty."""
    learner = OnlineLearner()
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_skips_below_threshold() -> None:
    """Retraining is skipped until the configured buffer size is reached."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert not learner.buffer_full
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_runs_when_full() -> None:
    """A full buffer trains a three-layer surrogate from scratch."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=20, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(20, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    assert result is not None
    assert "w1" in result
    assert "w3" in result
    assert learner.generation == 1


def test_retrain_clears_buffer() -> None:
    """Accepted retraining clears the consumed sample buffer."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.buffer) == 0


def test_max_generations() -> None:
    """Retraining stops after the configured generation cap."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, max_generations=1, n_epochs=2))
    rng = np.random.default_rng(42)

    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()

    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    assert result is None


def test_retrain_history_tracked() -> None:
    """Accepted retraining records generation and validation loss."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.retrain_history) == 1
    assert "val_loss" in learner.retrain_history[0]


def test_reset() -> None:
    """Reset clears buffered samples, generations, and history."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    learner.reset()
    assert learner.generation == 0
    assert len(learner.buffer) == 0
    assert len(learner.retrain_history) == 0


def test_weights_have_correct_shapes() -> None:
    """Generated model weights use the expected dense-network shapes."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    assert result is not None
    assert result["w1"].shape == (10, 64)
    assert result["b1"].shape == (64,)
    assert result["w2"].shape == (64, 32)
    assert result["w3"].shape == (32, 3)


def test_retrain_updates_hidden_layers() -> None:
    """Retraining updates all hidden-layer weights and biases."""
    learner = OnlineLearner(
        config=LearnerConfig(
            buffer_size=16,
            validation_fraction=0.25,
            n_epochs=8,
            learning_rate=5e-3,
        )
    )
    rng = np.random.default_rng(7)
    current_weights = _weights(rng)

    for sample_input, _ in _random_samples(16, rng):
        target = np.array(
            [
                0.3 * sample_input[0] - 0.2 * sample_input[1],
                sample_input[2] * sample_input[3],
                np.sin(sample_input[4]),
            ],
            dtype=np.float64,
        )
        learner.add_sample(sample_input, target)

    updated = learner.try_retrain(current_weights=current_weights)

    assert updated is not None
    assert not np.allclose(updated["w1"], current_weights["w1"])
    assert not np.allclose(updated["b1"], current_weights["b1"])
    assert not np.allclose(updated["w2"], current_weights["w2"])
    assert not np.allclose(updated["b2"], current_weights["b2"])


def test_try_retrain_rolls_back_when_validation_does_not_improve() -> None:
    """Retraining returns None and records rejection when validation loss regresses."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=8, n_epochs=2))
    learner._best_val_loss = -1.0
    rng = np.random.default_rng(19)
    for inp, tgt in _random_samples(8, rng):
        learner.add_sample(inp, tgt)

    result = learner.try_retrain(current_weights=_weights(rng))

    assert result is None
    assert learner.generation == 0
    assert len(learner.retrain_history) == 1
    assert learner.retrain_history[0]["generation"] == 0
    assert learner.retrain_history[0]["accepted"] is False
    assert float(learner.retrain_history[0]["val_loss"]) >= 0.0


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (LearnerConfig(buffer_size=1), "buffer_size"),
        (LearnerConfig(validation_fraction=0.0), "validation_fraction"),
        (LearnerConfig(validation_fraction=1.0), "validation_fraction"),
        (LearnerConfig(n_epochs=0), "n_epochs"),
        (LearnerConfig(learning_rate=0.0), "learning_rate"),
        (LearnerConfig(learning_rate=float("nan")), "learning_rate"),
        (LearnerConfig(max_generations=0), "max_generations"),
    ],
)
def test_constructor_rejects_invalid_config(config: LearnerConfig, match: str) -> None:
    """Constructor rejects invalid online-learning controls."""
    with pytest.raises(ValueError, match=match):
        OnlineLearner(config=config)


def test_add_sample_rejects_invalid_shape_or_nonfinite_values() -> None:
    """Sample validation rejects wrong shapes and non-finite arrays."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=2))

    with pytest.raises(ValueError, match="input_10d"):
        learner.add_sample(np.ones(9), np.ones(3))

    with pytest.raises(ValueError, match="target_3d"):
        learner.add_sample(np.ones(10), np.ones(4))

    bad_input = np.ones(10)
    bad_input[2] = np.nan
    with pytest.raises(ValueError, match="input_10d"):
        learner.add_sample(bad_input, np.ones(3))

    bad_target = np.ones(3)
    bad_target[1] = np.inf
    with pytest.raises(ValueError, match="target_3d"):
        learner.add_sample(np.ones(10), bad_target)
