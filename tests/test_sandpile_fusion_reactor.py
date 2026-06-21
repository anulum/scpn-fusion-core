# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Sandpile Fusion Reactor Tests
"""Tests for the self-organised-criticality sandpile reactor and HJB avalanche control."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.core.sandpile_fusion_reactor as sandpile_fusion_reactor
from scpn_fusion.core.sandpile_fusion_reactor import (
    HJB_Avalanche_Controller,
    TokamakSandpile,
    run_sandpile_simulation,
)


class TestTokamakSandpileInit:
    """Sandpile construction and input validation."""

    def test_default_init(self) -> None:
        """Construction allocates a zeroed gradient lattice of the requested size."""
        sp = TokamakSandpile(size=20)
        assert sp.size == 20
        assert sp.Z.shape == (20,)
        assert np.all(sp.Z == 0)
        assert sp.edge_loss_events == 0

    def test_rejects_small_size(self) -> None:
        """A lattice smaller than the minimum is rejected."""
        with pytest.raises(ValueError, match="size must be"):
            TokamakSandpile(size=2)

    def test_rejects_bool(self) -> None:
        """A boolean size is rejected by the integer guard."""
        with pytest.raises(ValueError, match="size must be"):
            TokamakSandpile(size=True)


class TestDrive:
    """Core-fuelling drive step behaviour."""

    def test_adds_to_core(self) -> None:
        """Driving deposits one gradient unit at the core each call."""
        sp = TokamakSandpile(size=10)
        sp.drive()
        assert sp.Z[0] == 1
        sp.drive()
        assert sp.Z[0] == 2

    def test_only_core_affected(self) -> None:
        """Driving only changes the core cell, not the outer lattice."""
        sp = TokamakSandpile(size=10)
        sp.drive()
        assert np.all(sp.Z[1:] == 0)


class TestRelax:
    """Avalanche relaxation, edge-loss accounting, and suppression."""

    def test_no_avalanche_below_threshold(self) -> None:
        """A subcritical gradient triggers no avalanche."""
        sp = TokamakSandpile(size=10)
        sp.Z[0] = 1
        assert sp.relax() == 0

    def test_avalanche_above_threshold(self) -> None:
        """A supercritical gradient triggers a non-empty avalanche."""
        sp = TokamakSandpile(size=10)
        sp.Z[0] = 5
        assert sp.relax() > 0

    def test_tracks_edge_loss_events(self) -> None:
        """Edge toppling increments the per-step and cumulative edge-loss counters."""
        reactor = TokamakSandpile(size=6)
        reactor.Z[-1] = 5
        avalanche = reactor.relax(suppression_strength=0.0)
        assert avalanche > 0
        assert reactor.last_edge_loss_events > 0
        assert reactor.edge_loss_events >= reactor.last_edge_loss_events

    def test_resets_last_edge_loss_per_step(self) -> None:
        """The per-step edge-loss counter resets when no edge toppling occurs."""
        reactor = TokamakSandpile(size=6)
        reactor.Z[:] = 0
        reactor.relax(suppression_strength=0.0)
        assert reactor.last_edge_loss_events == 0

    def test_suppression_raises_threshold(self) -> None:
        """Higher suppression strength reduces the avalanche size."""
        sp1 = TokamakSandpile(size=10)
        sp1.Z[0] = 5
        av1 = sp1.relax(suppression_strength=0.0)
        sp2 = TokamakSandpile(size=10)
        sp2.Z[0] = 5
        av2 = sp2.relax(suppression_strength=1.0)
        assert av2 <= av1

    def test_rejects_invalid_suppression(self) -> None:
        """Out-of-range or non-finite suppression strength is rejected."""
        sp = TokamakSandpile(size=8)
        with pytest.raises(ValueError, match="suppression_strength"):
            sp.relax(suppression_strength=1.5)
        with pytest.raises(ValueError, match="suppression_strength"):
            sp.relax(suppression_strength=-0.1)
        with pytest.raises(ValueError, match="finite"):
            sp.relax(suppression_strength=float("nan"))


class TestCalculateProfile:
    """Cumulative gradient-to-height profile reconstruction."""

    def test_profile_shape(self) -> None:
        """The reconstructed profile has one height per lattice cell."""
        sp = TokamakSandpile(size=10)
        sp.Z = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        h = sp.calculate_profile()
        assert len(h) == 10

    def test_core_higher_than_edge(self) -> None:
        """A peaked gradient yields a core height above the edge."""
        sp = TokamakSandpile(size=10)
        sp.Z = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        h = sp.calculate_profile()
        assert h[0] > h[-1]

    def test_flat_gradient_flat_profile(self) -> None:
        """A flat unit gradient yields a linear profile from edge to core."""
        sp = TokamakSandpile(size=5)
        sp.Z = np.array([1, 1, 1, 1, 1])
        h = sp.calculate_profile()
        assert h[0] == 5
        assert h[-1] == 1


class TestHJBController:
    """Hamilton-Jacobi-Bellman avalanche-suppression controller."""

    def test_init(self) -> None:
        """Construction starts with zero shear and a positive learning rate."""
        ctrl = HJB_Avalanche_Controller()
        assert ctrl.shear == 0.0
        assert ctrl.alpha > 0

    def test_large_avalanche_increases_shear(self) -> None:
        """A large avalanche drives the controller to raise magnetic shear."""
        ctrl = HJB_Avalanche_Controller()
        ctrl.act(last_avalanche_size=10, current_core_temp=50)
        assert ctrl.shear > 0.0

    def test_low_temp_moderate_shear(self) -> None:
        """A quiet low-temperature step keeps shear within bounds."""
        ctrl = HJB_Avalanche_Controller()
        ctrl.act(last_avalanche_size=0, current_core_temp=50)
        assert 0.0 <= ctrl.shear <= 1.0

    def test_stable_high_temp_relaxes(self) -> None:
        """Sustained stable high-temperature operation relaxes the shear."""
        ctrl = HJB_Avalanche_Controller()
        ctrl.shear = 0.5
        for _ in range(50):
            ctrl.act(last_avalanche_size=0, current_core_temp=200)
        assert ctrl.shear < 0.5

    def test_shear_always_bounded(self) -> None:
        """The controller keeps shear within [0, 1] under extreme input."""
        ctrl = HJB_Avalanche_Controller()
        for _ in range(200):
            ctrl.act(last_avalanche_size=100, current_core_temp=0)
        assert 0.0 <= ctrl.shear <= 1.0


class TestSimulationLoop:
    """Coupled drive/relax/control/profile stepping."""

    def test_drive_relax_cycle(self) -> None:
        """A manual drive/control/relax/profile cycle stays non-negative."""
        sp = TokamakSandpile(size=20)
        ctrl = HJB_Avalanche_Controller()
        for _ in range(100):
            sp.drive()
            action = ctrl.act(0, sp.h[0] if sp.h[0] > 0 else 0)
            sp.relax(suppression_strength=action)
            sp.calculate_profile()
        assert sp.h[0] >= 0


def test_run_sandpile_simulation_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end demo drives the sandpile, controls it, and renders safely."""
    import matplotlib.pyplot as plt

    saved: list[str] = []

    def _capture_savefig(path: object, *args: object, **kwargs: object) -> None:
        saved.append(str(path))

    monkeypatch.setattr(sandpile_fusion_reactor, "TIME_STEPS", 200)
    monkeypatch.setattr(plt, "savefig", _capture_savefig)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    run_sandpile_simulation()

    assert saved == ["Sandpile_Fusion_Report.png"]
