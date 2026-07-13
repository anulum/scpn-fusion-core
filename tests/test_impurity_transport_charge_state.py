# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Charge-State Collisional-Radiative Math Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport_charge_state import (
    _source_sink_transfer_matrix,
    adas_style_charge_state_coefficients,
    advance_charge_state_collisional_radiative,
    collisional_radiative_source_sink_matrices,
)
from scpn_fusion.core.impurity_transport_contracts import AdasChargeStateCoefficients


def test_charge_state_collisional_radiative_step_conserves_density() -> None:
    radius = np.linspace(0.0, 1.0, 8)
    charge = np.array([0, 1, 2, 3], dtype=float)
    ne = np.ones_like(radius) * 1.0e20
    coeffs = adas_style_charge_state_coefficients("Ar", charge, np.linspace(100.0, 600.0, 4))
    density = np.zeros((radius.size, charge.size))
    density[:, 1] = 1.0e15 * (1.0 - 0.2 * radius)

    updated, ion, rec = advance_charge_state_collisional_radiative(density, ne, coeffs, dt=1.0e-5)

    assert ion.shape == density.shape
    assert rec.shape == density.shape
    assert np.all(updated >= 0.0)
    np.testing.assert_allclose(np.sum(updated, axis=1), np.sum(density, axis=1), rtol=1e-13)


def test_collisional_radiative_source_sink_rejects_invalid_shapes() -> None:
    charge = np.array([0, 1, 2], dtype=float)
    coeffs = adas_style_charge_state_coefficients("Ne", charge, np.array([50.0, 100.0, 200.0]))
    with pytest.raises(ValueError, match="radius"):
        collisional_radiative_source_sink_matrices(
            np.ones((4, 3)),
            np.ones(3) * 1.0e20,
            coeffs,
        )


class TestAdasStyleCoefficientValidation:
    def test_rejects_te_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Te_eV must match charge_states shape"):
            adas_style_charge_state_coefficients(
                "C", np.array([0.0, 1.0, 2.0]), np.array([100.0, 100.0])
            )

    def test_rejects_nonpositive_te(self) -> None:
        with pytest.raises(ValueError, match="Te_eV must be finite and positive"):
            adas_style_charge_state_coefficients(
                "C", np.array([0.0, 1.0, 2.0]), np.array([100.0, 0.0, 100.0])
            )

    def test_rejects_short_charge_axis(self) -> None:
        with pytest.raises(ValueError, match="at least two states"):
            adas_style_charge_state_coefficients("C", np.array([0.0]), np.array([100.0]))

    def test_rejects_non_increasing_charge(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            adas_style_charge_state_coefficients(
                "C", np.array([1.0, 1.0]), np.array([100.0, 100.0])
            )


def _cr_coeffs() -> AdasChargeStateCoefficients:
    return adas_style_charge_state_coefficients(
        "C", np.array([0.0, 1.0, 2.0]), np.array([50.0, 100.0, 200.0])
    )


class TestCollisionalRadiativeValidation:
    def test_rejects_non_2d_density(self) -> None:
        with pytest.raises(ValueError, match="radius x charge_state"):
            collisional_radiative_source_sink_matrices(np.zeros(3), np.full(3, 1e19), _cr_coeffs())

    def test_rejects_ne_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ne must match the radius dimension"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 3)), np.full(2, 1e19), _cr_coeffs()
            )

    def test_rejects_coeff_axis_mismatch(self) -> None:
        with pytest.raises(ValueError, match="coefficient charge axis"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 2)), np.full(3, 1e19), _cr_coeffs()
            )

    def test_rejects_negative_density(self) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            collisional_radiative_source_sink_matrices(
                -np.ones((3, 3)), np.full(3, 1e19), _cr_coeffs()
            )

    def test_rejects_nonpositive_ne(self) -> None:
        with pytest.raises(ValueError, match="ne must be finite and positive"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 3)), np.array([1e19, 0.0, 1e19]), _cr_coeffs()
            )

    def test_advance_rejects_bad_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            advance_charge_state_collisional_radiative(
                np.ones((3, 3)), np.full(3, 1e19), _cr_coeffs(), 0.0
            )

    def test_source_sink_transfer_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="matching radius x charge"):
            _source_sink_transfer_matrix(np.ones((3, 2)), np.ones((3, 3)))

    def test_source_sink_transfer_matrix_is_charge_conservative(self) -> None:
        """The transfer matrix rows must sum to zero (charge conservation)."""
        coeffs = _cr_coeffs()
        density = np.full((4, 3), 1.0e15)
        ionisation, recombination = collisional_radiative_source_sink_matrices(
            density, np.full(4, 1.0e19), coeffs
        )
        matrix = _source_sink_transfer_matrix(ionisation, recombination)
        assert matrix.shape == (4, 3, 3)
        np.testing.assert_allclose(np.sum(matrix, axis=2), 0.0, atol=1.0e-6)
