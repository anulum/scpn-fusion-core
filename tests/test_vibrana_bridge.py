from __future__ import annotations

import types

import numpy as np
import pytest

from scpn_fusion.core import vibrana_bridge


def test_init_raises_when_vibrana_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vibrana_bridge, "VIBRANA_AVAILABLE", False)
    with pytest.raises(ImportError, match="VIBRANA Engine not available"):
        vibrana_bridge.VibranaFusionBridge("dummy.json")


def test_map_physics_to_audio_updates_engine_config() -> None:
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    cfg = types.SimpleNamespace(
        carrier_frequency=0.0,
        chaos_intensity=0.0,
        binaural_beat_frequency=0.0,
        golden_ratio_harmonics=False,
    )
    bridge.engine = types.SimpleNamespace(config=cfg)

    result = bridge.map_physics_to_audio(
        t=0,
        Ip=10.0,
        err_R=0.1,
        err_Z=0.1,
        psi_matrix=np.ones((4, 4), dtype=float),
    )
    assert result["Carrier"] == pytest.approx(400.0)
    assert 0.0 <= result["Chaos"] <= 1.0
    assert result["Beat"] == 10.0
    assert bridge.engine.config.golden_ratio_harmonics is False


def test_map_physics_to_audio_sets_gamma_for_large_error() -> None:
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    cfg = types.SimpleNamespace(
        carrier_frequency=0.0,
        chaos_intensity=0.0,
        binaural_beat_frequency=0.0,
        golden_ratio_harmonics=False,
    )
    bridge.engine = types.SimpleNamespace(config=cfg)

    result = bridge.map_physics_to_audio(
        t=0,
        Ip=50.0,  # clipped by np.clip
        err_R=1.0,
        err_Z=1.0,
        psi_matrix=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float),
    )
    assert result["Carrier"] == pytest.approx(1200.0)
    assert bridge.engine.config.carrier_frequency == pytest.approx(800.0)
    assert result["Beat"] == 40.0
    assert bridge.engine.config.golden_ratio_harmonics is True
