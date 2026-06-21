# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Lazarus Bridge Tests
"""Tests for the Lazarus plasma-to-bio resonance bridge and protocol generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import lazarus_bridge


class _DummyKernel:
    """Minimal fusion-kernel stub exposing the geometry the bridge reads."""

    def __init__(self) -> None:
        self.Psi = np.ones((4, 4), dtype=float)
        self.RR = np.linspace(5.5, 6.5, 4)[None, :].repeat(4, axis=0)
        self.ZZ = np.linspace(-0.5, 0.5, 4)[:, None].repeat(4, axis=1)

    def find_x_point(self, _psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], None]:
        """Return a fixed X-point location for the stub geometry."""
        return (6.2, 0.0), None

    def solve_equilibrium(self) -> None:
        """No-op equilibrium solve for the stub."""
        return None


def test_init_builds_kernel_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The constructor builds the fusion kernel and an empty regeneration log."""
    monkeypatch.setattr(lazarus_bridge, "FusionKernel", lambda _path: _DummyKernel())
    bridge = lazarus_bridge.LazarusBridge(str(tmp_path / "cfg.json"))
    assert isinstance(bridge.kernel, _DummyKernel)
    assert bridge.regeneration_log == []


def test_calculate_bio_resonance_returns_positive_value() -> None:
    """The bio-resonance metric is a positive scalar for valid geometry."""
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    val = bridge.calculate_bio_resonance()
    assert val > 0.0


def test_generate_protocol_embeds_resonance_value() -> None:
    """The generated Opentrons protocol embeds the resonance-scaled volumes."""
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    script = bridge.generate_protocol(1.25)
    assert "Resonance Score: 1.2500" in script
    assert "tert_vol = 6.25" in script


def test_run_bridge_simulation_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end bridge run solves, scores, writes the protocol, and renders."""
    monkeypatch.chdir(tmp_path)
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    bridge.regeneration_log = []

    saved: list[str] = []
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "subplots",
        lambda **kwargs: (object(), _DummyAxes()),
    )
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "savefig",
        lambda path, *a, **k: saved.append(str(path)),
    )

    # Exercise the opentrons-available branch with a stubbed simulator.
    monkeypatch.setattr(lazarus_bridge, "OPENTRONS_AVAILABLE", True)
    monkeypatch.setattr(
        lazarus_bridge,
        "simulate",
        type("S", (), {"get_protocol_api": staticmethod(lambda _v: object())}),
        raising=False,
    )

    bridge.run_bridge_simulation()

    assert (tmp_path / "lazarus_generated_protocol.py").exists()
    assert saved and "Lazarus_Bridge_Result.png" in saved[-1]


def test_run_bridge_simulation_reports_golden_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A near-unity resonance triggers the golden-ratio convergence message."""
    monkeypatch.chdir(tmp_path)
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    bridge.regeneration_log = []

    monkeypatch.setattr(type(bridge), "calculate_bio_resonance", lambda _self: 1.0)
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "subplots",
        lambda **kwargs: (object(), _DummyAxes()),
    )
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "savefig",
        lambda path, *a, **k: None,
    )
    monkeypatch.setattr(lazarus_bridge, "OPENTRONS_AVAILABLE", False)

    bridge.run_bridge_simulation()
    assert "GOLDEN RATIO CONVERGENCE" in capsys.readouterr().out


def test_run_bridge_simulation_handles_opentrons_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing opentrons simulator is caught and reported as a warning."""
    monkeypatch.chdir(tmp_path)
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    bridge.regeneration_log = []

    def _raise(_v: str) -> object:
        raise RuntimeError("robot offline")

    monkeypatch.setattr(lazarus_bridge, "OPENTRONS_AVAILABLE", True)
    monkeypatch.setattr(
        lazarus_bridge,
        "simulate",
        type("S", (), {"get_protocol_api": staticmethod(_raise)}),
        raising=False,
    )
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "subplots",
        lambda **kwargs: (object(), _DummyAxes()),
    )
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "savefig",
        lambda path, *a, **k: None,
    )

    bridge.run_bridge_simulation()
    assert "Simulation Warning" in capsys.readouterr().out


class _DummyAxes:
    """Matplotlib-axes stub that swallows the bridge's drawing calls."""

    def contour(self, *args: object, **kwargs: object) -> None:
        """Ignore contour drawing."""
        return None

    def plot(self, *args: object, **kwargs: object) -> None:
        """Ignore line drawing."""
        return None

    def set_title(self, *args: object, **kwargs: object) -> None:
        """Ignore title setting."""
        return None

    def legend(self, *args: object, **kwargs: object) -> None:
        """Ignore legend rendering."""
        return None


def test_visualize_bridge_saves_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge visualisation writes a result PNG."""
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    saved: list[str] = []

    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "subplots",
        lambda **kwargs: (object(), _DummyAxes()),
    )
    monkeypatch.setattr(
        lazarus_bridge.plt,  # type: ignore[attr-defined]
        "savefig",
        lambda path: saved.append(str(path)),
    )
    bridge.visualize_bridge(1.0)
    assert saved
    assert "Lazarus_Bridge_Result.png" in saved[-1]
