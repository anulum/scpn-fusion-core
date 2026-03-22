# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Sandpile Fusion Reactor Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.sandpile_fusion_reactor import (
    HJB_Avalanche_Controller,
    TokamakSandpile,
)


class TestTokamakSandpileInit:
    def test_default_init(self):
        sp = TokamakSandpile(size=20)
        assert sp.size == 20
        assert sp.Z.shape == (20,)
        assert np.all(sp.Z == 0)
        assert sp.edge_loss_events == 0

    def test_rejects_small_size(self):
        with pytest.raises(ValueError, match="size must be"):
            TokamakSandpile(size=2)

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="size must be"):
            TokamakSandpile(size=True)


class TestDrive:
    def test_adds_to_core(self):
        sp = TokamakSandpile(size=10)
        sp.drive()
        assert sp.Z[0] == 1
        sp.drive()
        assert sp.Z[0] == 2

    def test_only_core_affected(self):
        sp = TokamakSandpile(size=10)
        sp.drive()
        assert np.all(sp.Z[1:] == 0)


class TestRelax:
    def test_no_avalanche_below_threshold(self):
        sp = TokamakSandpile(size=10)
        sp.Z[0] = 1
        assert sp.relax() == 0

    def test_avalanche_above_threshold(self):
        sp = TokamakSandpile(size=10)
        sp.Z[0] = 5
        assert sp.relax() > 0

    def test_tracks_edge_loss_events(self):
        reactor = TokamakSandpile(size=6)
        reactor.Z[-1] = 5
        avalanche = reactor.relax(suppression_strength=0.0)
        assert avalanche > 0
        assert reactor.last_edge_loss_events > 0
        assert reactor.edge_loss_events >= reactor.last_edge_loss_events

    def test_resets_last_edge_loss_per_step(self):
        reactor = TokamakSandpile(size=6)
        reactor.Z[:] = 0
        reactor.relax(suppression_strength=0.0)
        assert reactor.last_edge_loss_events == 0

    def test_suppression_raises_threshold(self):
        sp1 = TokamakSandpile(size=10)
        sp1.Z[0] = 5
        av1 = sp1.relax(suppression_strength=0.0)
        sp2 = TokamakSandpile(size=10)
        sp2.Z[0] = 5
        av2 = sp2.relax(suppression_strength=1.0)
        assert av2 <= av1

    def test_rejects_invalid_suppression(self):
        sp = TokamakSandpile(size=8)
        with pytest.raises(ValueError, match="suppression_strength"):
            sp.relax(suppression_strength=1.5)
        with pytest.raises(ValueError, match="suppression_strength"):
            sp.relax(suppression_strength=-0.1)
        with pytest.raises(ValueError, match="finite"):
            sp.relax(suppression_strength=float("nan"))


class TestCalculateProfile:
    def test_profile_shape(self):
        sp = TokamakSandpile(size=10)
        sp.Z = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        h = sp.calculate_profile()
        assert len(h) == 10

    def test_core_higher_than_edge(self):
        sp = TokamakSandpile(size=10)
        sp.Z = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        h = sp.calculate_profile()
        assert h[0] > h[-1]

    def test_flat_gradient_flat_profile(self):
        sp = TokamakSandpile(size=5)
        sp.Z = np.array([1, 1, 1, 1, 1])
        h = sp.calculate_profile()
        assert h[0] == 5
        assert h[-1] == 1


class TestHJBController:
    def test_init(self):
        ctrl = HJB_Avalanche_Controller()
        assert ctrl.shear == 0.0
        assert ctrl.alpha > 0

    def test_large_avalanche_increases_shear(self):
        ctrl = HJB_Avalanche_Controller()
        ctrl.act(last_avalanche_size=10, current_core_temp=50)
        assert ctrl.shear > 0.0

    def test_low_temp_moderate_shear(self):
        ctrl = HJB_Avalanche_Controller()
        ctrl.act(last_avalanche_size=0, current_core_temp=50)
        assert 0.0 <= ctrl.shear <= 1.0

    def test_stable_high_temp_relaxes(self):
        ctrl = HJB_Avalanche_Controller()
        ctrl.shear = 0.5
        for _ in range(50):
            ctrl.act(last_avalanche_size=0, current_core_temp=200)
        assert ctrl.shear < 0.5

    def test_shear_always_bounded(self):
        ctrl = HJB_Avalanche_Controller()
        for _ in range(200):
            ctrl.act(last_avalanche_size=100, current_core_temp=0)
        assert 0.0 <= ctrl.shear <= 1.0


class TestSimulationLoop:
    def test_drive_relax_cycle(self):
        sp = TokamakSandpile(size=20)
        ctrl = HJB_Avalanche_Controller()
        for _ in range(100):
            sp.drive()
            action = ctrl.act(0, sp.h[0] if sp.h[0] > 0 else 0)
            sp.relax(suppression_strength=action)
            sp.calculate_profile()
        assert sp.h[0] >= 0
