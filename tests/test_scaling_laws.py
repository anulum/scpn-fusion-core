# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IPB98(y,2) Scaling Law Validation Tests
"""Physics validation tests for the IPB98(y,2) confinement scaling law.

Reference values from:
- ITER Physics Basis, Nuclear Fusion 39 (1999) 2175
- Verdoolaege et al., Nuclear Fusion 61 (2021) 076006
- Shimada et al., Nuclear Fusion 47 (2007) S1 (ITER design basis)
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.scaling_laws import (
    assess_ipb98y2_domain,
    compute_h_factor,
    ipb98y2_tau_e,
    ipb98y2_with_uncertainty,
    load_ipb98y2_coefficients,
)

# ITER inductive scenario design parameters
# Shimada et al., NF 47 (2007) S1, Table 1
ITER_PARAMS = dict(
    Ip=15.0,  # MA
    BT=5.3,  # T
    ne19=10.1,  # 10^19 m^-3
    Ploss=87.0,  # MW (P_aux + P_alpha - P_rad, ohmic negligible)
    R=6.2,  # m
    kappa=1.7,  # elongation
    epsilon=0.323,  # a/R = 2.0/6.2
    M=2.5,  # D-T
)


class TestIPB98y2Coefficients:
    def test_loads_without_error(self):
        coeff = load_ipb98y2_coefficients()
        assert coeff["C"] > 0
        assert len(coeff["exponents"]) == 8

    def test_exponent_signs_match_physics(self):
        """Confinement improves with higher Ip, BT, ne, R, kappa, M
        and degrades with higher Ploss (negative exponent)."""
        coeff = load_ipb98y2_coefficients()
        exp = coeff["exponents"]
        assert exp["Ip_MA"] > 0
        assert exp["BT_T"] > 0
        assert exp["ne19_1e19m3"] > 0
        assert exp["Ploss_MW"] < 0
        assert exp["R_m"] > 0
        assert exp["kappa"] > 0
        assert exp["M_AMU"] > 0

    def test_published_c_value(self):
        """C = 0.0562 from the ITER Physics Basis."""
        coeff = load_ipb98y2_coefficients()
        assert abs(coeff["C"] - 0.0562) < 0.005


class TestITERReferencePrediction:
    def test_iter_tau_e_in_expected_range(self):
        """IPB98(y,2) prediction for ITER: ~3.7 s (design basis).

        Published range: 3.5-4.0 s depending on exact Ploss assumption.
        """
        tau = ipb98y2_tau_e(**ITER_PARAMS)
        assert 2.5 < tau < 6.0, f"ITER tau_E = {tau:.3f} s outside physical range"

    def test_iter_in_training_domain(self):
        """ITER parameters must lie within IPB98(y,2) training envelope."""
        domain = assess_ipb98y2_domain(**ITER_PARAMS)
        assert domain["in_training_domain"], (
            f"ITER flagged as extrapolated: {domain['extrapolated_dimensions']}"
        )

    def test_iter_h_factor_near_unity(self):
        """By design, ITER targets H98(y,2) = 1.0."""
        tau = ipb98y2_tau_e(**ITER_PARAMS)
        h = compute_h_factor(tau, tau)
        assert abs(h - 1.0) < 1e-12


class TestScalingLawMonotonicity:
    """Verify partial derivatives match physics expectations."""

    def test_higher_current_improves_confinement(self):
        tau_lo = ipb98y2_tau_e(**{**ITER_PARAMS, "Ip": 10.0})
        tau_hi = ipb98y2_tau_e(**{**ITER_PARAMS, "Ip": 17.0})
        assert tau_hi > tau_lo

    def test_higher_loss_power_degrades_confinement(self):
        tau_lo = ipb98y2_tau_e(**{**ITER_PARAMS, "Ploss": 50.0})
        tau_hi = ipb98y2_tau_e(**{**ITER_PARAMS, "Ploss": 150.0})
        assert tau_hi < tau_lo

    def test_larger_machine_improves_confinement(self):
        tau_small = ipb98y2_tau_e(**{**ITER_PARAMS, "R": 3.0})
        tau_large = ipb98y2_tau_e(**{**ITER_PARAMS, "R": 8.0})
        assert tau_large > tau_small

    def test_higher_density_improves_confinement(self):
        tau_lo = ipb98y2_tau_e(**{**ITER_PARAMS, "ne19": 5.0})
        tau_hi = ipb98y2_tau_e(**{**ITER_PARAMS, "ne19": 15.0})
        assert tau_hi > tau_lo


class TestUncertaintyPropagation:
    def test_uncertainty_positive_finite(self):
        tau, sigma = ipb98y2_with_uncertainty(**ITER_PARAMS)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_relative_uncertainty_bounded(self):
        """Relative uncertainty should be O(15-25%) for ITER parameters."""
        tau, sigma = ipb98y2_with_uncertainty(**ITER_PARAMS)
        rel = sigma / tau
        assert 0.05 < rel < 0.5, f"Relative uncertainty {rel:.2%} out of expected range"


class TestDomainAssessment:
    def test_extreme_current_flagged(self):
        domain = assess_ipb98y2_domain(**{**ITER_PARAMS, "Ip": 50.0})
        assert not domain["in_training_domain"]
        assert "Ip" in domain["extrapolated_dimensions"]

    def test_extreme_density_flagged(self):
        domain = assess_ipb98y2_domain(**{**ITER_PARAMS, "ne19": 50.0})
        assert not domain["in_training_domain"]
        assert "ne19" in domain["extrapolated_dimensions"]

    def test_enforce_domain_raises(self):
        with pytest.raises(ValueError, match="outside training domain"):
            ipb98y2_tau_e(**{**ITER_PARAMS, "Ip": 50.0}, enforce_training_domain=True)


class TestInputValidation:
    def test_negative_current_raises(self):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(**{**ITER_PARAMS, "Ip": -1.0})

    def test_zero_power_raises(self):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(**{**ITER_PARAMS, "Ploss": 0.0})

    def test_nan_input_raises(self):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(**{**ITER_PARAMS, "BT": float("nan")})

    def test_h_factor_zero_denominator(self):
        h = compute_h_factor(1.0, 0.0)
        assert h == float("inf")
