# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vibrana Bridge Tests
"""Tests for the Vibrana plasma-to-audio sonification bridge."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import vibrana_bridge


def _engine() -> types.SimpleNamespace:
    """Build an audio-engine stub with a mutable config namespace."""
    cfg = types.SimpleNamespace(
        carrier_frequency=0.0,
        chaos_intensity=0.0,
        binaural_beat_frequency=0.0,
        golden_ratio_harmonics=False,
    )
    return types.SimpleNamespace(config=cfg)


class _DummyBrain:
    """Controller-brain stub returning a fixed actuation per step."""

    def step(self, _err: float) -> float:
        """Return a constant control output."""
        return 0.01


class _DummyKernel:
    """Fusion-kernel stub exposing the fields the sonification loop reads."""

    def __init__(self) -> None:
        self.Psi = np.ones((5, 5), dtype=float)
        self.R = np.linspace(5.0, 7.0, 5)
        self.Z = np.linspace(-1.0, 1.0, 5)
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }

    def solve_equilibrium(self) -> None:
        """No-op equilibrium solve for the stub."""
        return None


class _DummyNC:
    """Neuro-cybernetic-controller stub with a kernel and step-able brains."""

    def __init__(self) -> None:
        self.kernel = _DummyKernel()
        self.brain_R: _DummyBrain | None = None
        self.brain_Z: _DummyBrain | None = None

    def initialize_brains(self, use_quantum: bool = True) -> None:
        """Allocate the radial/vertical control brains."""
        self.brain_R = _DummyBrain()
        self.brain_Z = _DummyBrain()


def test_init_raises_when_vibrana_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constructor refuses to build when the VIBRANA engine is absent."""
    monkeypatch.setattr(vibrana_bridge, "VIBRANA_AVAILABLE", False)
    with pytest.raises(ImportError, match="VIBRANA Engine not available"):
        vibrana_bridge.VibranaFusionBridge("dummy.json")


def test_init_builds_engine_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the engine available, the constructor wires the controller and audio engine."""
    engine = _engine()
    monkeypatch.setattr(vibrana_bridge, "VIBRANA_AVAILABLE", True)
    monkeypatch.setattr(vibrana_bridge, "NeuroCyberneticController", lambda _p: _DummyNC())
    monkeypatch.setattr(
        vibrana_bridge, "CCWConfiguration", lambda **kwargs: object(), raising=False
    )
    monkeypatch.setattr(
        vibrana_bridge, "CCWStateOfTheArtEngine", lambda _cfg: engine, raising=False
    )
    monkeypatch.setattr(
        vibrana_bridge, "AttractorType", types.SimpleNamespace(LORENZ=0), raising=False
    )
    bridge = vibrana_bridge.VibranaFusionBridge("dummy.json")
    assert isinstance(bridge.nc, _DummyNC)
    assert bridge.engine is engine


def test_map_physics_to_audio_updates_engine_config() -> None:
    """Mapping physics to audio writes the carrier/beat/chaos engine config."""
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    bridge.engine = _engine()

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
    """A large control error drives the gamma beat band and harmonic flag."""
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    bridge.engine = _engine()

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


def test_map_physics_to_audio_sets_theta_for_small_error() -> None:
    """A near-zero control error selects the theta (flow-state) beat band."""
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    bridge.engine = _engine()

    result = bridge.map_physics_to_audio(
        t=0,
        Ip=10.0,
        err_R=0.0,
        err_Z=0.0,
        psi_matrix=np.ones((4, 4), dtype=float),
    )
    assert result["Beat"] == 5.0
    assert result["Chaos"] == pytest.approx(0.0)


def test_run_sonification_session_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sonification session steps the controller and renders the soundscape."""
    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    bridge.nc = _DummyNC()  # type: ignore[assignment]
    bridge.engine = _engine()

    captured: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        vibrana_bridge.VibranaFusionBridge,
        "visualize_soundscape",
        lambda _self, log: captured.append(log),
    )

    bridge.run_sonification_session(duration_steps=60)

    assert len(captured) == 1
    assert len(captured[0]) == 60
    assert {"t", "error", "carrier", "chaos"}.issubset(captured[0][0])


def test_visualize_soundscape_saves_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The soundscape visualisation renders and saves the result PNG."""
    import matplotlib.pyplot as plt

    bridge = vibrana_bridge.VibranaFusionBridge.__new__(vibrana_bridge.VibranaFusionBridge)
    saved: list[str] = []

    class _Axes:
        def plot(self, *args: object, **kwargs: object) -> None:
            return None

        def fill_between(self, *args: object, **kwargs: object) -> None:
            return None

        def set_title(self, *args: object, **kwargs: object) -> None:
            return None

        def set_ylabel(self, *args: object, **kwargs: object) -> None:
            return None

        def set_xlabel(self, *args: object, **kwargs: object) -> None:
            return None

        def set_ylim(self, *args: object, **kwargs: object) -> None:
            return None

        def grid(self, *args: object, **kwargs: object) -> None:
            return None

        def legend(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(plt, "subplots", lambda *a, **k: (object(), (_Axes(), _Axes())))
    monkeypatch.setattr(plt, "tight_layout", lambda *a, **k: None)
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))

    log = [
        {"t": 0, "error": 0.1, "carrier": 400.0, "chaos": 0.2},
        {"t": 1, "error": 0.2, "carrier": 420.0, "chaos": 0.4},
    ]
    bridge.visualize_soundscape(log)
    assert saved == ["Vibrana_Sonification_Result.png"]
