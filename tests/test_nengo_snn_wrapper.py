from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.nengo_snn_wrapper import (
    NengoSNNConfig,
    NengoSNNController,
    NengoSNNControllerStub,
    _LIFPopulation,
    _Lowpass,
    _nef_decoder,
    nengo_available,
)

# ── nengo_available / config ─────────────────────────────────────────


def test_nengo_available_always_true() -> None:
    assert nengo_available() is True


def test_config_defaults() -> None:
    cfg = NengoSNNConfig()
    assert cfg.n_neurons == 200
    assert cfg.n_channels == 2
    assert cfg.dt == 0.001
    assert cfg.tau_synapse == 0.015
    assert cfg.tau_mem == 0.020


def test_config_custom() -> None:
    cfg = NengoSNNConfig(n_neurons=100, n_channels=4, gain=10.0)
    assert cfg.n_neurons == 100
    assert cfg.n_channels == 4
    assert cfg.gain == 10.0


# ── Lowpass ──────────────────────────────────────────────────────────


def test_lowpass_converges() -> None:
    lp = _Lowpass(0.01, 0.001, 1)
    for _ in range(200):
        lp.step(np.array([1.0]))
    assert abs(lp._val[0] - 1.0) < 0.01


def test_lowpass_reset() -> None:
    lp = _Lowpass(0.01, 0.001, 1)
    lp.step(np.array([5.0]))
    lp.reset()
    assert lp._val[0] == 0.0


# ── LIF Population ──────────────────────────────────────────────────


def _make_pop(n: int = 50, seed: int = 42) -> _LIFPopulation:
    rng = np.random.default_rng(seed)
    return _LIFPopulation(
        n=n,
        tau_rc=0.02,
        tau_ref=0.002,
        max_rates=rng.uniform(100, 200, n),
        intercepts=rng.uniform(-0.8, 0.8, n),
        encoders=rng.choice([-1.0, 1.0], n),
        dt=0.001,
    )


def test_lif_steady_rates_nonnegative() -> None:
    pop = _make_pop()
    rates = pop.steady_rates(np.array([0.5]))
    assert rates.shape == (50, 1)
    assert np.all(rates >= 0)


def test_lif_spikes_for_strong_input() -> None:
    pop = _make_pop(n=100, seed=0)
    pop.encoders = np.ones(100)
    total = sum(np.sum(pop.step(1.0) > 0) for _ in range(100))
    assert total > 0


def test_lif_no_spikes_below_threshold() -> None:
    pop = _LIFPopulation(
        n=50,
        tau_rc=0.02,
        tau_ref=0.002,
        max_rates=np.full(50, 100.0),
        intercepts=np.full(50, 0.9),
        encoders=np.ones(50),
        dt=0.001,
    )
    total = sum(np.sum(pop.step(0.1) > 0) for _ in range(50))
    assert total == 0


def test_lif_reset_clears_voltage() -> None:
    pop = _make_pop()
    for _ in range(10):
        pop.step(1.0)
    pop.reset()
    assert np.all(pop.voltage == 0.0)
    assert np.all(pop.ref_time == 0.0)


# ── NEF Decoder ─────────────────────────────────────────────────────


def test_nef_identity_decode() -> None:
    pop = _make_pop(n=200)
    D = _nef_decoder(pop, lambda x: x)
    x_test = np.array([-0.5, 0.0, 0.5])
    decoded = D @ pop.steady_rates(x_test)
    np.testing.assert_allclose(decoded, x_test, atol=0.15)


def test_nef_gain_decode() -> None:
    pop = _make_pop(n=200)
    gain = 5.0
    D = _nef_decoder(pop, lambda x: gain * x)
    x_test = np.array([-0.5, 0.0, 0.5])
    decoded = D @ pop.steady_rates(x_test)
    np.testing.assert_allclose(decoded, gain * x_test, atol=0.5)


# ── NengoSNNController ─────────────────────────────────────────────


def test_controller_builds() -> None:
    ctrl = NengoSNNController()
    assert ctrl._built is True
    assert ctrl._step_count == 0


def test_step_shape() -> None:
    ctrl = NengoSNNController()
    out = ctrl.step(np.array([0.1, -0.2]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


def test_step_increments_count() -> None:
    ctrl = NengoSNNController()
    ctrl.step(np.zeros(2))
    ctrl.step(np.zeros(2))
    assert ctrl._step_count == 2


def test_reset_clears_state() -> None:
    ctrl = NengoSNNController()
    ctrl.step(np.ones(2))
    ctrl.reset()
    assert ctrl._step_count == 0
    assert np.all(ctrl._last_output == 0.0)


def test_responds_to_input() -> None:
    ctrl = NengoSNNController(NengoSNNConfig(n_neurons=200, seed=42))
    for _ in range(500):
        out = ctrl.step(np.array([0.5, -0.3]))
    assert np.any(np.abs(out) > 0.01)


def test_deterministic() -> None:
    cfg = NengoSNNConfig(seed=99)
    ctrl1 = NengoSNNController(cfg)
    ctrl2 = NengoSNNController(cfg)
    err = np.array([0.3, -0.1])
    for _ in range(100):
        o1 = ctrl1.step(err)
        o2 = ctrl2.step(err)
    np.testing.assert_array_equal(o1, o2)


def test_get_spike_data_keys() -> None:
    ctrl = NengoSNNController()
    ctrl.step(np.zeros(2))
    data = ctrl.get_spike_data()
    assert "output" in data
    assert "error_ch0" in data
    assert "error_ch1" in data


def test_export_weights_contents() -> None:
    ctrl = NengoSNNController()
    weights = ctrl.export_weights()
    assert isinstance(weights, dict)
    assert "ch0_D_gain" in weights
    assert weights["ch0_D_gain"].shape == (200,)


def test_export_fpga_weights(tmp_path) -> None:
    ctrl = NengoSNNController()
    out = tmp_path / "fpga_weights.npz"
    ctrl.export_fpga_weights(out)
    assert out.exists()
    loaded = np.load(str(out))
    assert "n_neurons" in loaded
    assert "n_channels" in loaded
    assert "ch0_D_gain" in loaded


def test_export_loihi_raises() -> None:
    ctrl = NengoSNNController()
    with pytest.raises(NotImplementedError, match="Loihi export"):
        ctrl.export_loihi("out.npz")


def test_benchmark_stats() -> None:
    ctrl = NengoSNNController()
    stats = ctrl.benchmark(n_steps=10)
    assert "mean_us" in stats
    assert "p95_us" in stats
    assert stats["mean_us"] > 0.0


def test_custom_channels() -> None:
    cfg = NengoSNNConfig(n_channels=3, n_neurons=50)
    ctrl = NengoSNNController(cfg)
    out = ctrl.step(np.array([0.1, 0.2, 0.3]))
    assert out.shape == (3,)


def test_stub_raises() -> None:
    with pytest.raises(ImportError, match="deprecated"):
        NengoSNNControllerStub()
