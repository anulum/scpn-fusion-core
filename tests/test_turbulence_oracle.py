from __future__ import annotations

import numpy as np

from scpn_fusion.core.turbulence_oracle import DriftWavePhysics, OracleESN


def test_drift_wave_step_outputs_finite_fields() -> None:
    sim = DriftWavePhysics(N=8)
    phi, dens = sim.step()
    assert phi.shape == (8, 8)
    assert dens.shape == (8, 8)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(dens))


def test_oracle_esn_train_predict_shapes() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(24, 4))
    model = OracleESN(input_dim=4, reservoir_size=20, spectral_radius=0.8)
    model.train(data[:-1], data[1:])
    pred = model.predict(data[-1], steps=5)
    assert pred.shape == (5, 4)
    assert np.all(np.isfinite(pred))
