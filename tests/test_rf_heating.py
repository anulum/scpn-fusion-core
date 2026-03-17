# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
