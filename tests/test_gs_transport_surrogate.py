# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GS-Transport Surrogate Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for the GS-transport surrogate training pipeline (P1.3).

Verifies that:
  - _generate_gs_transport_pairs() produces correctly shaped data
  - MLPSurrogate forward pass returns the right shape
  - MLPSurrogate save/load round-trips weights exactly
  - train_gs_transport_surrogate() runs end-to-end and returns expected keys
  - _relative_l2 returns 0 for identical arrays and positive for different ones
"""

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fno_training import (
    MLPSurrogate,
    _generate_gs_transport_pairs,
    _relative_l2,
    train_gs_transport_surrogate,
)


# ── Data generation ──────────────────────────────────────────────────


def test_generate_gs_transport_pairs_small():
    """Generate 5 samples and verify shapes and value constraints."""
    x, y, meta = _generate_gs_transport_pairs(n_samples=5, grid_size=50, seed=123)

    assert x.ndim == 2, f"Expected 2D, got {x.ndim}D"
    assert y.ndim == 2, f"Expected 2D, got {y.ndim}D"
    assert x.shape[1] == 50, f"Expected grid_size=50, got {x.shape[1]}"
    assert y.shape[1] == 50, f"Expected grid_size=50, got {y.shape[1]}"
    assert x.shape[0] == y.shape[0], "x and y must have equal sample count"
    assert x.shape[0] == len(meta), "metadata length must match sample count"
    assert x.shape[0] > 0, "Must produce at least 1 valid sample from 5 attempts"

    # All profiles should be finite
    assert np.all(np.isfinite(x)), "x contains NaN/Inf"
    assert np.all(np.isfinite(y)), "y contains NaN/Inf"

    # Metadata should have the expected keys
    for m in meta:
        for key in ("Ip", "BT", "kappa", "n_e20", "P_aux", "T0"):
            assert key in m, f"Missing metadata key: {key}"


# ── MLPSurrogate ─────────────────────────────────────────────────────


def test_mlp_surrogate_forward():
    """Create MLPSurrogate with default dims and verify forward pass shape."""
    model = MLPSurrogate(input_dim=50, hidden_dim=128, seed=42)

    # Single sample
    x_single = np.random.default_rng(1).standard_normal(50)
    y_single = model.forward(x_single)
    assert y_single.shape == (50,), f"Expected (50,), got {y_single.shape}"
    assert np.all(np.isfinite(y_single)), "Output contains NaN/Inf"

    # Batch of 8
    x_batch = np.random.default_rng(2).standard_normal((8, 50))
    y_batch = model.forward(x_batch)
    assert y_batch.shape == (8, 50), f"Expected (8, 50), got {y_batch.shape}"
    assert np.all(np.isfinite(y_batch)), "Batch output contains NaN/Inf"


def test_mlp_surrogate_save_load(tmp_path: Path):
    """Save and load weights, then verify output is bit-identical."""
    model = MLPSurrogate(input_dim=50, hidden_dim=64, seed=99)
    x_test = np.random.default_rng(3).standard_normal((4, 50))
    y_before = model.forward(x_test).copy()

    save_file = tmp_path / "test_mlp.npz"
    model.save_weights(save_file)
    assert save_file.exists(), "Weights file was not created"

    # Load into a fresh model (different seed to prove loading overrides init)
    model2 = MLPSurrogate(input_dim=50, hidden_dim=64, seed=12345)
    model2.load_weights(save_file)
    y_after = model2.forward(x_test)

    np.testing.assert_array_equal(y_before, y_after)


# ── End-to-end training ──────────────────────────────────────────────


def test_train_gs_transport_surrogate_smoke(tmp_path: Path):
    """Train with minimal samples and epochs, verify history keys."""
    save_file = tmp_path / "gs_surrogate_smoke.npz"
    history = train_gs_transport_surrogate(
        n_samples=10,
        epochs=3,
        lr=1e-3,
        save_path=save_file,
        seed=42,
        patience=5,
    )

    assert save_file.exists(), "Weights file was not saved"

    # Required history keys
    for key in (
        "train_loss",
        "val_loss",
        "best_epoch",
        "best_val_loss",
        "epochs_completed",
        "epochs_requested",
        "saved_path",
        "test_mse",
        "test_rel_l2",
        "n_samples_generated",
        "n_train",
        "n_val",
        "n_test",
        "data_mode",
        "machine_class_counts",
    ):
        assert key in history, f"Missing history key: {key}"

    assert history["epochs_completed"] >= 1
    assert history["data_mode"] == "gs_transport_oracle"
    assert np.isfinite(float(history["best_val_loss"]))

    # Train loss should be a list of floats
    assert isinstance(history["train_loss"], list)
    assert len(history["train_loss"]) == history["epochs_completed"]


# ── _relative_l2 unit tests ─────────────────────────────────────────


def test_relative_l2_perfect():
    """_relative_l2 with identical arrays must return 0.0."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = _relative_l2(a, a)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_relative_l2_nonzero():
    """_relative_l2 with different arrays must return a positive value."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.1, 2.2, 3.3, 4.4])
    result = _relative_l2(a, b)
    assert result > 0.0
    assert np.isfinite(result)
