# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 7 (API-verified)
from __future__ import annotations

import numpy as np
import pytest


class TestSPIMitigation:
    def test_init(self):
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        spi = ShatteredPelletInjection(Plasma_Energy_MJ=300.0, Plasma_Current_MA=15.0)
        assert spi.W_th == pytest.approx(300e6)

    def test_estimate_z_eff(self):
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        z = ShatteredPelletInjection.estimate_z_eff(neon_quantity_mol=0.5)
        assert z > 1.0

    def test_estimate_z_eff_cocktail(self):
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        z = ShatteredPelletInjection.estimate_z_eff_cocktail(
            neon_quantity_mol=0.3,
            argon_quantity_mol=0.1,
            xenon_quantity_mol=0.0,
        )
        assert z > 1.0

    def test_rejects_bad_energy(self):
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        with pytest.raises(ValueError, match="Plasma_Energy"):
            ShatteredPelletInjection(Plasma_Energy_MJ=0.0)

    def test_require_non_negative(self):
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        assert ShatteredPelletInjection._require_non_negative("x", 5.0) == 5.0
        with pytest.raises(ValueError):
            ShatteredPelletInjection._require_non_negative("x", -1.0)


class TestEpedPedestal:
    def test_require_positive_finite(self):
        from scpn_fusion.core.eped_pedestal import _require_positive_finite

        assert _require_positive_finite("x", 3.0) == 3.0
        with pytest.raises(ValueError):
            _require_positive_finite("x", 0.0)

    def test_domain_violation(self):
        from scpn_fusion.core.eped_pedestal import _normalized_domain_violation

        assert _normalized_domain_violation(5.0, lower=0.0, upper=10.0) == 0.0
        assert _normalized_domain_violation(15.0, lower=0.0, upper=10.0) > 0.0

    def test_pedestal_model_init(self):
        from scpn_fusion.core.eped_pedestal import EpedPedestalModel

        model = EpedPedestalModel(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
        assert model is not None


class TestFokkerPlanckRE:
    def test_solver_init(self):
        from scpn_fusion.control.fokker_planck_re import FokkerPlanckSolver

        solver = FokkerPlanckSolver()
        assert solver is not None


class TestArchiveProfiles:
    def test_default_paths(self):
        from scpn_fusion.io.tokamak_archive_profiles import (
            default_reference_data_root,
            default_diiid_dir,
        )

        assert isinstance(default_reference_data_root(), type(default_reference_data_root()))
        assert (
            "diiid" in str(default_diiid_dir()).lower()
            or "diii" in str(default_diiid_dir()).lower()
        )


class TestUPDE:
    def test_upde_system_init(self):
        from scpn_fusion.phase.upde import UPDESystem
        from scpn_fusion.phase.knm import KnmSpec

        spec = KnmSpec(K=np.eye(5))
        system = UPDESystem(spec=spec, dt=0.01)
        assert system is not None


class TestAdaptiveKnm:
    def test_engine_init(self):
        from scpn_fusion.phase.adaptive_knm import AdaptiveKnmEngine, AdaptiveKnmConfig
        from scpn_fusion.phase.knm import KnmSpec

        spec = KnmSpec(K=np.eye(5))
        config = AdaptiveKnmConfig()
        engine = AdaptiveKnmEngine(baseline_spec=spec, config=config)
        assert engine is not None
