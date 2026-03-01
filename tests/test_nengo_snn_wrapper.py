from __future__ import annotations

import pytest

from scpn_fusion.control import nengo_snn_wrapper


def test_nengo_config_defaults_are_sane() -> None:
    cfg = nengo_snn_wrapper.NengoSNNConfig()
    assert cfg.n_neurons > 0
    assert cfg.n_channels > 0
    assert cfg.dt > 0.0
    assert cfg.tau_synapse > 0.0
    assert cfg.tau_mem > 0.0


def test_nengo_availability_probe_returns_bool() -> None:
    assert isinstance(nengo_snn_wrapper.nengo_available(), bool)


def test_nengo_stub_raises_clear_error() -> None:
    with pytest.raises(ImportError, match="Nengo is required"):
        nengo_snn_wrapper.NengoSNNControllerStub()


def test_controller_raises_when_nengo_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nengo_snn_wrapper, "_nengo_available", False)
    with pytest.raises(ImportError, match="Nengo is required"):
        nengo_snn_wrapper.NengoSNNController()
