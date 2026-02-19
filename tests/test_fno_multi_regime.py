# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests for multi-regime FNO training
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the multi-regime SPARC-parameterized data generator and the
corresponding training loop in fno_training.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.fno_training import (
    DEFAULT_SPARC_WEIGHTS_PATH,
    MultiLayerFNO,
    SPARC_REGIMES,
    _generate_multi_regime_pairs,
    _sample_regime_params,
    train_fno_multi_regime,
)

pytestmark = pytest.mark.experimental


# ── Regime parameter sampling ────────────────────────────────────────


class TestRegimeParams:
    """Verify parameter sampling from SPARC regimes."""

    def test_all_regimes_defined(self):
        assert "itg" in SPARC_REGIMES
        assert "tem" in SPARC_REGIMES
        assert "etg" in SPARC_REGIMES

    @pytest.mark.parametrize("regime", ["itg", "tem", "etg"])
    def test_sample_in_bounds(self, regime):
        rng = np.random.default_rng(42)
        for _ in range(50):
            params = _sample_regime_params(rng, regime)
            bounds = SPARC_REGIMES[regime]
            for key, val in params.items():
                lo, hi = bounds[key]
                assert lo <= val <= hi, f"{regime}.{key}={val} outside [{lo}, {hi}]"

    def test_different_seeds_different_params(self):
        p1 = _sample_regime_params(np.random.default_rng(1), "itg")
        p2 = _sample_regime_params(np.random.default_rng(2), "itg")
        # At least one parameter should differ
        assert any(p1[k] != p2[k] for k in p1)

    def test_regime_has_required_keys(self):
        required = {"alpha", "kappa", "nu", "damp", "k_cut"}
        for regime in SPARC_REGIMES:
            assert set(SPARC_REGIMES[regime].keys()) == required


# ── Multi-regime data generation ─────────────────────────────────────


class TestMultiRegimeGenerator:
    """Verify the multi-regime data generator."""

    def test_output_shapes(self):
        x, y, meta = _generate_multi_regime_pairs(
            n_samples=12, grid_size=32, seed=0
        )
        assert x.shape == (12, 32, 32)
        assert y.shape == (12, 32, 32)
        assert len(meta) == 12

    def test_all_regimes_sampled(self):
        """With enough samples, all three regimes should appear."""
        _, _, meta = _generate_multi_regime_pairs(
            n_samples=300, grid_size=16, seed=7
        )
        regimes_seen = {m["regime"] for m in meta}
        assert regimes_seen == {"itg", "tem", "etg"}

    def test_metadata_has_params(self):
        _, _, meta = _generate_multi_regime_pairs(
            n_samples=3, grid_size=16, seed=1
        )
        for m in meta:
            assert "regime" in m
            assert "alpha" in m
            assert "kappa" in m
            assert "nu" in m

    def test_outputs_finite(self):
        x, y, _ = _generate_multi_regime_pairs(
            n_samples=10, grid_size=32, seed=2
        )
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

    def test_deterministic(self):
        x1, y1, m1 = _generate_multi_regime_pairs(
            n_samples=5, grid_size=16, seed=99
        )
        x2, y2, m2 = _generate_multi_regime_pairs(
            n_samples=5, grid_size=16, seed=99
        )
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        assert [m["regime"] for m in m1] == [m["regime"] for m in m2]

    def test_regime_weights_bias(self):
        """Custom regime weights should bias sampling."""
        _, _, meta = _generate_multi_regime_pairs(
            n_samples=200,
            grid_size=16,
            seed=42,
            regime_weights={"itg": 10.0, "tem": 0.0, "etg": 0.0},
        )
        regimes = [m["regime"] for m in meta]
        # All should be ITG (or nearly all with uniform probability → 0)
        assert regimes.count("itg") == len(regimes)

    def test_x_and_y_differ(self):
        """x (input field) and y (evolved field) should not be identical."""
        x, y, _ = _generate_multi_regime_pairs(
            n_samples=5, grid_size=32, seed=3
        )
        for i in range(len(x)):
            assert not np.allclose(x[i], y[i])

    def test_itg_has_more_low_k_energy(self):
        """ITG fields should have relatively more low-k energy than ETG."""
        x_itg, _, _ = _generate_multi_regime_pairs(
            n_samples=50, grid_size=64, seed=10,
            regime_weights={"itg": 1.0, "tem": 0.0, "etg": 0.0},
        )
        x_etg, _, _ = _generate_multi_regime_pairs(
            n_samples=50, grid_size=64, seed=10,
            regime_weights={"itg": 0.0, "tem": 0.0, "etg": 1.0},
        )

        def low_k_fraction(fields):
            fracs = []
            for f in fields:
                fk = np.abs(np.fft.fft2(f))
                k2 = np.fft.fftfreq(64)[:, None]**2 + np.fft.fftfreq(64)[None, :]**2
                low_k_power = np.sum(fk[k2 < 0.01])
                total = np.sum(fk) + 1e-15
                fracs.append(low_k_power / total)
            return np.mean(fracs)

        # ITG should have more low-k content
        itg_frac = low_k_fraction(x_itg)
        etg_frac = low_k_fraction(x_etg)
        assert itg_frac > etg_frac, (
            f"ITG low-k fraction {itg_frac:.4f} <= ETG {etg_frac:.4f}"
        )


# ── Multi-regime training ────────────────────────────────────────────


class TestMultiRegimeTraining:
    """Smoke-test the multi-regime training loop."""

    def test_training_smoke(self, tmp_path):
        out = tmp_path / "fno_mr.npz"
        hist = train_fno_multi_regime(
            n_samples=16,
            epochs=2,
            lr=1e-3,
            modes=4,
            width=4,
            save_path=out,
            batch_size=4,
            seed=42,
            patience=2,
        )
        assert out.exists()
        assert hist["epochs_completed"] >= 1
        assert np.isfinite(float(hist["best_val_loss"]))
        assert hist["data_mode"] == "multi_regime_sparc"

    def test_regime_counts_in_history(self, tmp_path):
        out = tmp_path / "fno_mr2.npz"
        hist = train_fno_multi_regime(
            n_samples=30,
            epochs=1,
            lr=1e-3,
            modes=4,
            width=4,
            save_path=out,
            batch_size=4,
            seed=7,
            patience=1,
        )
        counts = hist["regime_counts"]
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 30

    def test_regime_val_losses_present(self, tmp_path):
        out = tmp_path / "fno_mr3.npz"
        hist = train_fno_multi_regime(
            n_samples=30,
            epochs=2,
            lr=1e-3,
            modes=4,
            width=4,
            save_path=out,
            batch_size=4,
            seed=13,
            patience=2,
        )
        rvl = hist.get("regime_val_losses")
        assert rvl is not None
        assert isinstance(rvl, dict)
        for regime, stats in rvl.items():
            assert "mean" in stats
            assert "n" in stats
            assert np.isfinite(stats["mean"])

    def test_saved_weights_loadable(self, tmp_path):
        out = tmp_path / "fno_mr4.npz"
        train_fno_multi_regime(
            n_samples=16,
            epochs=1,
            lr=1e-3,
            modes=4,
            width=4,
            save_path=out,
            batch_size=4,
            seed=21,
            patience=1,
        )
        model = MultiLayerFNO(modes=4, width=4)
        model.load_weights(out)
        pred = model.forward(np.zeros((64, 64)))
        assert pred.shape == (64, 64)
        assert np.all(np.isfinite(pred))
