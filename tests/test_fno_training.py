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
from scpn_fusion.core.fno_turbulence_suppressor import FNO_Controller


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
