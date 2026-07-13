# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Transport Data Contract Tests
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport_contracts import (
    AdasChargeStateCoefficients,
    AuroraParityCase,
    ImpuritySpecies,
    _strict_axis,
)


def _make_case_with(**overrides: Any) -> AuroraParityCase:
    base: dict[str, Any] = {
        "element": "C",
        "charge_states": np.array([0.0, 1.0, 2.0]),
        "radius_m": np.array([0.0, 0.5, 1.0]),
        "time_s": np.array([0.0, 0.01, 0.02]),
        "ne_t_r": np.full((3, 3), 1e19),
        "Te_t_r": np.full((3, 3), 100.0),
        "initial_charge_state_density_rz": np.full((3, 3), 1e15),
        "diffusion_m2_s_r_z": np.full((3, 3), 0.5),
        "convection_m_s_r_z": np.full((3, 3), -1.0),
        "major_radius_m": 1.65,
    }
    base.update(overrides)
    return AuroraParityCase(**base)


class TestImpuritySpeciesValidation:
    @pytest.mark.parametrize(
        "override, match",
        [
            ({"Z_nucleus": 0}, "Z_nucleus must be positive"),
            ({"mass_amu": 0.0}, "mass_amu must be finite and positive"),
            ({"source_rate": -1.0}, "source_rate must be finite and non-negative"),
            ({"source_decay_width_rho": 0.0}, "source_decay_width_rho must be finite and positive"),
        ],
    )
    def test_rejects_invalid_fields(self, override: dict[str, Any], match: str) -> None:
        base: dict[str, Any] = {"element": "W", "Z_nucleus": 74, "mass_amu": 183.84}
        base.update(override)
        with pytest.raises(ValueError, match=match):
            ImpuritySpecies(**base)


class TestAdasChargeStateCoefficientValidation:
    def test_rejects_short_charge_axis(self) -> None:
        with pytest.raises(ValueError, match="at least two states"):
            AdasChargeStateCoefficients(
                np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
            )

    def test_rejects_non_increasing_charge(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            AdasChargeStateCoefficients(np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), np.zeros(2))

    def test_rejects_non_integer_charge(self) -> None:
        with pytest.raises(ValueError, match="integer charge"):
            AdasChargeStateCoefficients(np.array([0.0, 1.5]), np.zeros(2), np.zeros(2), np.zeros(2))

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match charge_states shape"):
            AdasChargeStateCoefficients(
                np.array([0.0, 1.0, 2.0]), np.zeros(2), np.zeros(3), np.zeros(3)
            )

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            AdasChargeStateCoefficients(
                np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)
            )


class TestAuroraParityCaseValidation:
    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(ValueError, match="radius_m must be non-negative"):
            _make_case_with(radius_m=np.array([-0.1, 0.5, 1.0]))

    def test_rejects_bad_major_radius(self) -> None:
        with pytest.raises(ValueError, match="major_radius_m must be finite and positive"):
            _make_case_with(major_radius_m=0.0)

    def test_rejects_wrong_field_shape(self) -> None:
        with pytest.raises(ValueError, match="ne_t_r must have shape"):
            _make_case_with(ne_t_r=np.full((2, 3), 1e19))

    def test_rejects_nonpositive_ne(self) -> None:
        with pytest.raises(ValueError, match="ne_t_r must be positive"):
            _make_case_with(ne_t_r=np.zeros((3, 3)))

    def test_rejects_negative_diffusion(self) -> None:
        with pytest.raises(ValueError, match="diffusion_m2_s_r_z must be non-negative"):
            _make_case_with(diffusion_m2_s_r_z=np.full((3, 3), -1.0))

    def test_rejects_optional_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="ionisation_m3_s_t_r_z must have shape"):
            _make_case_with(ionisation_m3_s_t_r_z=np.zeros((2, 3, 3)))

    def test_rejects_optional_negative(self) -> None:
        with pytest.raises(ValueError, match="recombination_m3_s_t_r_z must be finite and non"):
            _make_case_with(recombination_m3_s_t_r_z=np.full((3, 3, 3), -1.0))

    def test_rejects_effective_source_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="effective_source_m3_s_t_r_z must have shape"):
            _make_case_with(effective_source_m3_s_t_r_z=np.zeros((2, 3, 3)))

    def test_rejects_effective_source_non_finite(self) -> None:
        with pytest.raises(ValueError, match="effective_source_m3_s_t_r_z must be finite"):
            _make_case_with(effective_source_m3_s_t_r_z=np.full((3, 3, 3), np.inf))


def test_aurora_parity_case_rejects_non_finite_required_field() -> None:
    with pytest.raises(ValueError, match="convection_m_s_r_z must be finite"):
        _make_case_with(convection_m_s_r_z=np.full((3, 3), np.nan))


def test_strict_axis_rejects_short_and_non_monotonic() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        _strict_axis("x", np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        _strict_axis("x", np.array([1.0, 0.0]))
