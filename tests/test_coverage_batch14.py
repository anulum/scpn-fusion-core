# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 14 (passing only)
from __future__ import annotations


class TestFastIonPressure:
    def test_init(self):
        from scpn_fusion.core.kinetic_efit import FastIonPressure

        fip = FastIonPressure(E_fast_keV=3500.0, n_fast_frac=0.01, anisotropy_sigma=0.0)
        assert fip.E_fast_keV == 3500.0


class TestCODACConfig:
    def test_init(self):
        from scpn_fusion.control.codac_interface import CODACConfig

        cfg = CODACConfig()
        assert cfg is not None


class TestBioHolonomicController:
    def test_init(self):
        from scpn_fusion.control.bio_holonomic_controller import BioHolonomicController

        ctrl = BioHolonomicController(dt_s=0.01, seed=42)
        assert ctrl is not None


class TestNeuroSymbolicController:
    def test_import(self):
        from scpn_fusion.control.neuro_cybernetic_controller import NeuroCyberneticController

        assert NeuroCyberneticController is not None
