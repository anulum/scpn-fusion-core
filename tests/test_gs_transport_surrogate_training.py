# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS-Transport Training Tests

import numpy as np
from scpn_fusion.core.gs_transport_surrogate_training import (
    train_gs_transport_surrogate,
    MLPSurrogate,
    _generate_gs_transport_pairs,
)


def test_gs_transport_training_smoke(tmp_path):
    """Verify that the training loop runs end-to-end with minimal samples."""
    save_path = tmp_path / "test_weights.npz"
    history = train_gs_transport_surrogate(
        n_samples=5, epochs=2, save_path=save_path, seed=42, patience=5
    )
    assert history["epochs_completed"] == 2
    assert save_path.exists()
    assert "train_loss" in history
    assert "val_loss" in history


def test_gs_transport_data_gen_consistency():
    """Verify that generated pairs have expected shapes and metadata consistency."""
    x, y, meta = _generate_gs_transport_pairs(n_samples=2, grid_size=50, seed=123)
    assert x.shape == (2, 50)
    assert y.shape == (2, 50)
    assert len(meta) == 2
    for m in meta:
        assert "Ip" in m
        assert "kappa" in m
        assert "T0" in m


def test_mlp_surrogate_io(tmp_path):
    """Verify MLP weight save/load round-trip."""
    model = MLPSurrogate(input_dim=10, hidden_dim=20, seed=7)
    x = np.random.rand(5, 10).astype(np.float64)
    y1 = model.forward(x)

    save_path = tmp_path / "mlp_io.npz"
    model.save_weights(save_path)

    model2 = MLPSurrogate(input_dim=10, hidden_dim=20)
    model2.load_weights(save_path)
    y2 = model2.forward(x)

    np.testing.assert_allclose(y1, y2, atol=1e-12)
