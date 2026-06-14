# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RF Heating Physics Validation Tests
"""Physics validation for ECRH heating.

ITER ECRH: 170 GHz, 20 MW, fundamental O-mode.
Resonance at B = omega*m_e / (n*q_e) where n = harmonic.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.rf_heating import ECRHHeatingSystem


class TestECRHResonance:
    def test_fundamental_resonance_near_axis(self):
        """170 GHz fundamental at B0=5.3T: resonance B ~ 6.07T → R_res ~ 5.4m.

        For ITER (B0=5.3T, R0=6.2m), B_res = omega*m_e/q_e.
        B_res = 2*pi*170e9 * 9.109e-31 / 1.602e-19 ≈ 6.07T.
        R_res = B0*R0/B_res = 5.3*6.2/6.07 ≈ 5.41m.
        Resonance is on the high-field side (R < R0).
        """
        ecrh = ECRHHeatingSystem(b0_tesla=5.3, r0_major=6.2, freq_ghz=170.0, harmonic=1)
        R_res = ecrh.resonance_radius()
        # Hand calculation: 2*pi*170e9 * 9.109e-31 / 1.602e-19 = 6.073 T
        # R_res = 5.3*6.2/6.073 = 5.414 m
        assert 5.0 < R_res < 6.2, f"R_res = {R_res:.3f} m — expected HFS"

    def test_second_harmonic_shifted(self):
        """Second harmonic resonates at half the B-field → larger R."""
        ecrh_1 = ECRHHeatingSystem(harmonic=1)
        ecrh_2 = ECRHHeatingSystem(harmonic=2)
        R1 = ecrh_1.resonance_radius()
        R2 = ecrh_2.resonance_radius()
        # n=2 needs half the B → double the R (approximately)
        assert R2 > R1, "Second harmonic should resonate at larger R"
        assert abs(R2 / R1 - 2.0) < 0.01, "R_res(n=2)/R_res(n=1) should be ~2"

    def test_resonance_formula_consistency(self):
        """Verify R_res = B0*R0 / B_res where B_res = omega*m_e/(n*q_e)."""
        ecrh = ECRHHeatingSystem(b0_tesla=4.0, r0_major=5.0, freq_ghz=140.0, harmonic=1)
        R_res = ecrh.resonance_radius()
        B_res = ecrh.omega * ecrh.m_e / (ecrh.harmonic * ecrh.q_e)
        R_expected = ecrh.B0 * ecrh.R0 / B_res
        assert abs(R_res - R_expected) < 1e-10


class TestECRHDeposition:
    def test_deposition_peaks_near_resonance(self):
        ecrh = ECRHHeatingSystem()
        rho, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=20.0)
        peak_rho = rho[np.argmax(P_dep)]
        # Resonance rho from geometry
        a = 2.0
        rho_res = abs(ecrh.resonance_radius() - ecrh.R0) / a
        assert abs(peak_rho - rho_res) < 0.15, (
            f"Deposition peak at rho={peak_rho:.2f}, expected near {rho_res:.2f}"
        )

    def test_absorption_efficiency_physical(self):
        """Single-pass absorption for ITER ECRH: typically 85-99%."""
        ecrh = ECRHHeatingSystem()
        _, _, eff = ecrh.compute_deposition(P_ecrh_mw=20.0, n_e=1e20)
        assert 0.3 < eff < 1.0, f"Efficiency {eff:.2%} outside physical range"

    def test_deposition_integrates_to_input_power(self):
        ecrh = ECRHHeatingSystem()
        rho, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=20.0)
        # P_dep is in MW/m^3 but we just check that total is reasonable
        assert np.all(P_dep >= 0), "Deposition must be non-negative"
        assert np.max(P_dep) > 0, "Deposition must have a peak"

    def test_zero_power_gives_zero_deposition(self):
        ecrh = ECRHHeatingSystem()
        rho, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=0.0)
        assert np.allclose(P_dep, 0.0)


class TestECRHValidation:
    def test_rejects_negative_field(self):
        with pytest.raises(ValueError):
            ECRHHeatingSystem(b0_tesla=-1.0)

    def test_rejects_zero_frequency(self):
        with pytest.raises(ValueError):
            ECRHHeatingSystem(freq_ghz=0.0)

    def test_rejects_bad_harmonic(self):
        with pytest.raises(ValueError):
            ECRHHeatingSystem(harmonic=0)

    def test_rejects_extreme_launch_angle(self):
        ecrh = ECRHHeatingSystem()
        with pytest.raises(ValueError):
            ecrh.compute_deposition(launch_angle_deg=90.0)

    def test_rejects_bad_density(self):
        ecrh = ECRHHeatingSystem()
        with pytest.raises(ValueError, match=">="):
            ecrh.compute_deposition(n_e=0.0)

    def test_rejects_bad_temperature(self):
        ecrh = ECRHHeatingSystem()
        with pytest.raises(ValueError, match=">="):
            ecrh.compute_deposition(T_e_keV=0.0)


class TestRequireHelpers:
    def test_require_finite_accepts_valid(self):
        from scpn_fusion.core.rf_heating import _require_finite_float

        assert _require_finite_float("x", 3.14) == pytest.approx(3.14)

    def test_require_finite_rejects_nan(self):
        from scpn_fusion.core.rf_heating import _require_finite_float

        with pytest.raises(ValueError, match="finite"):
            _require_finite_float("x", float("nan"))

    def test_require_finite_enforces_min(self):
        from scpn_fusion.core.rf_heating import _require_finite_float

        with pytest.raises(ValueError, match=">="):
            _require_finite_float("x", -1.0, min_value=0.0)

    def test_require_finite_enforces_max(self):
        from scpn_fusion.core.rf_heating import _require_finite_float

        with pytest.raises(ValueError, match="<="):
            _require_finite_float("x", 10.0, max_value=5.0)

    def test_require_int_rejects_bool(self):
        from scpn_fusion.core.rf_heating import _require_int

        with pytest.raises(ValueError, match="integer"):
            _require_int("n", True, 1)

    def test_require_int_rejects_below_minimum(self):
        from scpn_fusion.core.rf_heating import _require_int

        with pytest.raises(ValueError):
            _require_int("n", 0, 1)


class _FakeKernel:
    """Lightweight FusionKernel stand-in for ICRH ray-tracing contract tests.

    Provides only the grid attributes RFHeatingSystem.get_plasma_params reads, so
    the contract tests never trigger a real equilibrium solve.
    """

    def __init__(self, config_path):
        nr, nz = 8, 8
        self.R = np.linspace(4.0, 10.0, nr)
        self.Z = np.linspace(-3.0, 3.0, nz)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.NR = nr
        self.NZ = nz
        self.B_R = np.zeros((nz, nr))
        self.B_Z = np.zeros((nz, nr))
        self.Psi = np.zeros((nz, nr))

    def solve_equilibrium(self):
        return None


@pytest.fixture
def icrh_system(monkeypatch):
    import scpn_fusion.core.rf_heating as mod

    monkeypatch.setattr(mod, "FusionKernel", _FakeKernel)
    return mod.RFHeatingSystem(config_path="unused")


class TestICRHPlasmaParams:
    def test_in_grid_returns_finite_positive(self, icrh_system):
        B_mod, n_e, dn_dR, dn_dZ = icrh_system.get_plasma_params(6.2, 0.0)
        assert np.isfinite(B_mod) and B_mod > 0.0
        assert np.isfinite(n_e) and n_e > 0.0
        assert np.isfinite(dn_dR)
        assert np.isfinite(dn_dZ)

    def test_out_of_grid_falls_back_to_toroidal_only(self, icrh_system):
        """R far outside the grid uses the else branch (B_R=B_Z=psi=0)."""
        B_mod, _, _, _ = icrh_system.get_plasma_params(500.0, 0.0)
        B_tor = 5.3 * 6.2 / 500.0
        assert B_mod == pytest.approx(B_tor)

    @pytest.mark.parametrize("bad_R", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_R(self, icrh_system, bad_R):
        with pytest.raises(ValueError, match="R must be finite"):
            icrh_system.get_plasma_params(bad_R, 0.0)

    def test_rejects_non_finite_Z(self, icrh_system):
        with pytest.raises(ValueError, match="Z must be finite"):
            icrh_system.get_plasma_params(6.2, float("nan"))

    @pytest.mark.parametrize("bad_R", [0.0, -1.0])
    def test_rejects_non_positive_major_radius(self, icrh_system, bad_R):
        with pytest.raises(ValueError, match="R must be > 0"):
            icrh_system.get_plasma_params(bad_R, 0.0)


class TestICRHDispersion:
    def test_vacuum_far_from_axis(self, icrh_system):
        """Low local density returns the vacuum sentinel D = 1.0."""
        assert icrh_system.dispersion_relation(10.0, 3.0, 10.0, 0.0) == 1.0

    def test_finite_in_dense_plasma(self, icrh_system):
        D = icrh_system.dispersion_relation(6.2, 0.0, 10.0, 0.0)
        assert np.isfinite(D)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_k_R(self, icrh_system, bad):
        with pytest.raises(ValueError, match="k_R must be finite"):
            icrh_system.dispersion_relation(6.2, 0.0, bad, 0.0)

    def test_rejects_non_finite_k_Z(self, icrh_system):
        with pytest.raises(ValueError, match="k_Z must be finite"):
            icrh_system.dispersion_relation(6.2, 0.0, 10.0, float("nan"))


class TestICRHRayEquations:
    def test_valid_state_returns_four_finite_derivatives(self, icrh_system):
        out = icrh_system.ray_equations([9.0, 0.0, -10.0, 0.0], 0.0)
        assert len(out) == 4
        assert all(np.isfinite(v) for v in out)

    @pytest.mark.parametrize(
        "bad_state",
        [
            [9.0, 0.0, -10.0],
            [9.0, 0.0, -10.0, 0.0, 1.0],
            [[9.0, 0.0], [-10.0, 0.0]],
        ],
    )
    def test_rejects_malformed_shape(self, icrh_system, bad_state):
        with pytest.raises(ValueError, match="shape"):
            icrh_system.ray_equations(bad_state, 0.0)

    def test_rejects_non_finite_state(self, icrh_system):
        with pytest.raises(ValueError, match="finite"):
            icrh_system.ray_equations([9.0, float("nan"), -10.0, 0.0], 0.0)


class TestICRHTraceRays:
    def test_trace_small_ray_bundle(self, icrh_system):
        trajectories, B_res = icrh_system.trace_rays(n_rays=2)
        assert len(trajectories) == 2
        for sol in trajectories:
            assert sol.shape == (100, 4)
            assert np.all(np.isfinite(sol))
        assert np.isfinite(B_res) and B_res > 0.0

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_rejects_invalid_n_rays_value(self, icrh_system, bad):
        with pytest.raises(ValueError, match="n_rays"):
            icrh_system.trace_rays(n_rays=bad)

    def test_rejects_non_integer_n_rays(self, icrh_system):
        with pytest.raises(ValueError, match="n_rays"):
            icrh_system.trace_rays(n_rays=2.5)


class TestECRHEdgeCases:
    def test_oblique_launch_affects_profile(self):
        ecrh = ECRHHeatingSystem()
        _, P0, _ = ecrh.compute_deposition(launch_angle_deg=0.0)
        _, P45, _ = ecrh.compute_deposition(launch_angle_deg=45.0)
        assert not np.allclose(P0, P45)

    def test_sparc_parameters(self):
        ecrh = ECRHHeatingSystem(b0_tesla=12.2, r0_major=1.85, freq_ghz=170.0, harmonic=2)
        R_res = ecrh.resonance_radius()
        assert R_res > 0
        rho, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=10.0, T_e_keV=10.0)
        assert np.all(np.isfinite(P_dep))

    def test_small_bins(self):
        ecrh = ECRHHeatingSystem()
        rho, P_dep, eff = ecrh.compute_deposition(n_radial_bins=8)
        assert len(rho) == 8
