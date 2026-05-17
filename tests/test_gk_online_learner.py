# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Online Learner Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.gk_online_learner import LearnerConfig, OnlineLearner


def _random_samples(n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(rng.random(10), rng.random(3)) for _ in range(n)]


def test_buffer_not_full_initially():
    learner = OnlineLearner()
    assert not learner.buffer_full


def test_buffer_fills():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert learner.buffer_full


def test_try_retrain_skips_empty():
    learner = OnlineLearner()
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_skips_below_threshold():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert not learner.buffer_full
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_runs_when_full():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=20, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(20, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    # May return weights or None (if validation loss doesn't improve over inf)
    # First retrain should succeed since best_val starts at inf
    assert result is not None
    assert "w1" in result
    assert "w3" in result
    assert learner.generation == 1


def test_retrain_clears_buffer():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.buffer) == 0


def test_max_generations():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, max_generations=1, n_epochs=2))
    rng = np.random.default_rng(42)

    # First retrain
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()

    # Second attempt should be blocked
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    assert result is None  # max_generations reached


def test_retrain_history_tracked():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.retrain_history) == 1
    assert "val_loss" in learner.retrain_history[0]


def test_reset():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    learner.reset()
    assert learner.generation == 0
    assert len(learner.buffer) == 0
    assert len(learner.retrain_history) == 0


def test_weights_have_correct_shapes():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    if result is not None:
        assert result["w1"].shape == (10, 64)
        assert result["b1"].shape == (64,)
        assert result["w2"].shape == (64, 32)
        assert result["w3"].shape == (32, 3)


def test_retrain_updates_hidden_layers():
    learner = OnlineLearner(
        config=LearnerConfig(
            buffer_size=16,
            validation_fraction=0.25,
            n_epochs=8,
            learning_rate=5e-3,
        )
    )
    rng = np.random.default_rng(7)
    current_weights = {
        "w1": rng.normal(0.0, 0.05, (10, 64)),
        "b1": np.zeros(64),
        "w2": rng.normal(0.0, 0.05, (64, 32)),
        "b2": np.zeros(32),
        "w3": rng.normal(0.0, 0.05, (32, 3)),
        "b3": np.zeros(3),
    }

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


def test_add_sample_rejects_invalid_shape_or_nonfinite_values():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=2))

    with np.testing.assert_raises(ValueError):
        learner.add_sample(np.ones(9), np.ones(3))

    with np.testing.assert_raises(ValueError):
        learner.add_sample(np.ones(10), np.ones(4))

    bad_input = np.ones(10)
    bad_input[2] = np.nan
    with np.testing.assert_raises(ValueError):
        learner.add_sample(bad_input, np.ones(3))

    bad_target = np.ones(3)
    bad_target[1] = np.inf
    with np.testing.assert_raises(ValueError):
        learner.add_sample(np.ones(10), bad_target)
