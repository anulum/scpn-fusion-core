# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Turbulence Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.core.neural_turbulence import (
    NeuralTransportTrainer,
    QLKNNSurrogate,
    QLKNNTransportModel,
    TrainingDataGenerator,
    TransportInputNormalizer,
)


def test_surrogate_forward_pass():
    model = QLKNNSurrogate(hidden_layers=[32, 16])
    x = np.random.randn(10, 10)
    out = model.forward(x)

    assert out.shape == (10, 3)
    assert np.all(np.isfinite(out))


def test_input_normalization():
    norm = TransportInputNormalizer()
    r = np.linspace(0.1, 2.0, 50)
    Te = 10.0 * (1.0 - (r / 2.0) ** 2)
    Ti = 10.0 * (1.0 - (r / 2.0) ** 2)
    ne = 5.0 * (1.0 - (r / 2.0) ** 2)
    q = 1.0 + 2.0 * (r / 2.0) ** 2

    inputs = norm.from_profiles(Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r)

    assert inputs.shape == (50, 10)
    # R/L_Te should be positive
    assert np.all(inputs[:, 1] >= 0.0)


def test_analytic_targets_critical_gradient():
    gen = TrainingDataGenerator()

    # inputs = [R_L_Ti, R_L_Te, R_L_ne, q, s_hat, alpha_MHD, Ti_Te, nu_star, Z_eff, eps]
    X_sub = np.array([[1.0, 5.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.01, 1.5, 0.1]])  # Sub-critical R/L_Ti
    X_super = np.array([[10.0, 5.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.01, 1.5, 0.1]])  # Super-critical

    y_sub = gen.generate_analytic_targets(X_sub)
    y_super = gen.generate_analytic_targets(X_super)

    assert y_sub[0, 0] == 0.0  # Q_i should be 0 below threshold
    assert y_super[0, 0] > 0.0  # Q_i > 0 above threshold


def test_neural_transport_trainer():
    trainer = NeuralTransportTrainer()

    gen = TrainingDataGenerator()
    X = gen.generate_parameter_scan(200)
    y = gen.generate_analytic_targets(X)

    hist = trainer.train(X, y, epochs=50, lr=1e-3)

    assert len(hist["train_loss"]) == 50
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_surrogate_save_load(tmp_path):
    model = QLKNNSurrogate(hidden_layers=[16, 8])
    x = np.random.randn(5, 10)
    out_before = model.forward(x)

    path = str(tmp_path / "weights.npz")
    model.save_weights(path)

    model2 = QLKNNSurrogate(hidden_layers=[16, 8])
    model2.load_weights(path)
    out_after = model2.forward(x)

    np.testing.assert_array_almost_equal(out_before, out_after)


def test_qlknn_transport_model_denormalization():
    model = QLKNNSurrogate()
    t_model = QLKNNTransportModel(model)

    r = np.linspace(0.1, 2.0, 50)
    Te = 10.0 * (1.0 - (r / 2.0) ** 2)
    Ti = 10.0 * (1.0 - (r / 2.0) ** 2)
    ne = 5.0 * (1.0 - (r / 2.0) ** 2)
    q = 1.0 + 2.0 * (r / 2.0) ** 2

    fluxes = t_model.compute_fluxes(Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r)

    assert fluxes.Q_i_W_m2.shape == (50,)
    assert fluxes.Q_e_W_m2.shape == (50,)
    assert fluxes.Gamma_e_inv_m2_s.shape == (50,)
    assert np.all(np.isfinite(fluxes.Q_i_W_m2))
