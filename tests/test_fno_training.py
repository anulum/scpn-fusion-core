# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Training Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from pathlib import Path

import numpy as np

from scpn_fusion.core.fno_training import train_fno
from scpn_fusion.core.fno_turbulence_suppressor import (
    FNO_Controller,
    SpectralTurbulenceGenerator,
)


def test_fno_training_smoke(tmp_path):
    output = tmp_path / "fno_smoke.npz"
    history = train_fno(
        n_samples=8,
        epochs=1,
        lr=1e-3,
        modes=4,
        width=4,
        save_path=output,
        batch_size=2,
        seed=7,
        patience=1,
    )
    assert output.exists()
    assert history["epochs_completed"] >= 1
    assert np.isfinite(float(history["best_val_loss"]))


def test_fno_controller_loads_saved_weights(tmp_path):
    output = tmp_path / "fno_controller.npz"
    train_fno(
        n_samples=8,
        epochs=1,
        lr=1e-3,
        modes=4,
        width=4,
        save_path=output,
        batch_size=2,
        seed=13,
        patience=1,
    )

    controller = FNO_Controller(modes=4, width=4, weights_path=str(output))
    suppression, prediction = controller.predict_and_suppress(np.zeros((64, 64), dtype=np.float64))

    assert controller.loaded_weights
    assert prediction.shape == (64, 64)
    assert np.isfinite(suppression)
    assert 0.0 <= suppression <= 1.0


def test_spectral_generator_is_deterministic_for_seed() -> None:
    g1 = SpectralTurbulenceGenerator(size=24, seed=77)
    g2 = SpectralTurbulenceGenerator(size=24, seed=77)
    np.testing.assert_allclose(g1.field, g2.field, rtol=0.0, atol=0.0)
    s1 = g1.step()
    s2 = g2.step()
    np.testing.assert_allclose(s1, s2, rtol=0.0, atol=0.0)


def test_spectral_generator_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(4242)
    state = np.random.get_state()

    g = SpectralTurbulenceGenerator(size=20, seed=12)
    _ = g.step()

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
