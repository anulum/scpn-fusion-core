# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Flux Operator and Current Diagnostic Tests
"""Exercise raw diagnostic refusals and the signed dyadic operator oracle."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.jax_gs_solver import (
    gs_delta_star_np,
    gs_equation_residual_np,
    gs_toroidal_current_density_np,
    gs_total_toroidal_current_np,
)

FloatArray = NDArray[np.float64]
Diagnostic = Callable[[FloatArray, float, float, float, float], object]
CurrentDiagnostic = Callable[[FloatArray, float, float, float, float, float], object]


@pytest.mark.parametrize(
    "diagnostic", [gs_toroidal_current_density_np, gs_total_toroidal_current_np]
)
def test_public_current_diagnostics_refuse_overflowed_permeability_radius(
    diagnostic: CurrentDiagnostic,
) -> None:
    """Refuse an infinite denominator instead of reporting zero current.

    Parameters
    ----------
    diagnostic : callable
        Actual public current-density or integrated-current diagnostic.

    """
    z = np.linspace(-1.0, 1.0, 3)
    psi = np.broadcast_to(-(z[:, None] ** 2), (3, 3)).copy()
    before = psi.tobytes()
    with pytest.raises(RuntimeError, match="nonfinite|denominator"):
        diagnostic(psi, 1.0, 3.0, -1.0, 1.0, float(np.finfo(np.float64).max))
    assert psi.tobytes() == before


@pytest.mark.parametrize("span", [1e-170, 1e200])
@pytest.mark.parametrize(
    "diagnostic", [gs_delta_star_np, gs_toroidal_current_density_np, gs_total_toroidal_current_np]
)
def test_public_diagnostics_refuse_unrepresentable_grid_coefficients(
    diagnostic: Diagnostic, span: float
) -> None:
    """Refuse finite grids whose actual stencil coefficients are not finite.

    Parameters
    ----------
    diagnostic : callable
        Actual public flux or current diagnostic.
    span : float
        Positive spacing whose square underflows or overflows.

    """
    psi = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="nonfinite|spacing|coefficient"):
        diagnostic(psi, 1.0, 3.0, 0.0, span)
    assert not np.any(psi)


@pytest.mark.parametrize(
    "diagnostic", [gs_delta_star_np, gs_toroidal_current_density_np, gs_total_toroidal_current_np]
)
def test_public_diagnostics_refuse_nonfinite_stencil_evaluation(diagnostic: Diagnostic) -> None:
    """Keep overflow of finite flux operands from becoming a diagnostic array.

    Parameters
    ----------
    diagnostic : callable
        Actual public flux or current diagnostic.

    """
    psi = np.full((3, 3), np.finfo(np.float64).max)
    before = psi.tobytes()
    with pytest.raises(RuntimeError, match="nonfinite|operator"):
        diagnostic(psi, 1.0, 3.0, -1.0, 1.0)
    assert psi.tobytes() == before


def test_public_current_integral_refuses_overflow_of_finite_current_density() -> None:
    """Refuse an unrepresentable integral after finite density reconstruction."""
    z = np.linspace(-4.0, 4.0, 3)
    psi = np.broadcast_to(-(z[:, None] ** 2), (3, 3)).copy()
    density = gs_toroidal_current_density_np(psi, 1.0, 9.0, -4.0, 4.0, mu0=5e-309)
    assert np.isfinite(density).all()
    with pytest.raises(RuntimeError, match="nonfinite|integrated"):
        gs_total_toroidal_current_np(psi, 1.0, 9.0, -4.0, 4.0, mu0=5e-309)


@pytest.mark.parametrize("h", [1 / 16, 1 / 32, 1 / 64])
def test_public_dyadic_stencil_matches_signed_decimal_current_oracle(h: float) -> None:
    """Retain the exact declared stencil and signed current on dyadic meshes.

    Parameters
    ----------
    h : float
        Original dyadic mesh spacing from the signed software fixture family.

    """
    z = np.array([-h, 0.0, h])
    psi = np.broadcast_to(-0.25 * z[:, None] ** 2, (3, 3)).copy()
    bounds = (1.0, 1.0 + 2 * h, -h, h)
    delta = gs_delta_star_np(psi, *bounds)
    density = gs_toroidal_current_density_np(psi, *bounds, mu0=1.0)
    total = gs_total_toroidal_current_np(psi, *bounds, mu0=1.0)
    assert delta[1, 1] == -0.5
    with localcontext() as context:
        context.prec = 100
        expected_density = Decimal("0.5") / Decimal.from_float(1.0 + h)
        expected_total = expected_density * Decimal.from_float(h) ** 2
    u = 2.0**-53
    gamma16 = 16 * u / (1 - 16 * u)
    assert density[1, 1] == pytest.approx(float(expected_density), rel=gamma16, abs=0.0)
    assert total == pytest.approx(float(expected_total), rel=gamma16, abs=0.0)
    assert not np.any(delta[[0, -1], :]) and not np.any(delta[:, [0, -1]])


@pytest.mark.parametrize(
    "fault", ["dimensions", "empty_interior", "radius", "height", "flux", "span", "collapsed_axis"]
)
def test_public_operator_refuses_invalid_support_before_arithmetic(fault: str) -> None:
    """Reject malformed support and actual geometry rather than inventing zeros.

    Parameters
    ----------
    fault : str
        Invalid dimensionality, support, bounds, values or actual mesh axis.

    """
    psi: FloatArray = np.zeros((3, 3))
    bounds = (1.0, 3.0, -1.0, 1.0)
    if fault == "dimensions":
        psi = psi.ravel()
    elif fault == "empty_interior":
        psi = psi[:2]
    elif fault == "radius":
        bounds = (0.0, 3.0, -1.0, 1.0)
    elif fault == "height":
        bounds = (1.0, 3.0, 1.0, 1.0)
    elif fault == "flux":
        psi[1, 1] = np.nan
    elif fault == "span":
        bounds = (1.0, 3.0, -1e308, 1e308)
    else:
        bounds = (1.0, float(np.nextafter(1.0, 2.0)), -1.0, 1.0)
    with pytest.raises(ValueError):
        gs_delta_star_np(psi, *bounds)


@pytest.mark.parametrize("mu0", [0.0, -1.0, np.nan, np.inf])
def test_public_current_refuses_invalid_model_permeability(mu0: float) -> None:
    """Refuse invalid declared model permeability before reconstructing current.

    Parameters
    ----------
    mu0 : float
        Nonpositive or nonfinite supplied model permeability.

    """
    with pytest.raises(ValueError, match="mu0"):
        gs_toroidal_current_density_np(np.zeros((3, 3)), 1.0, 3.0, -1.0, 1.0, mu0)


def test_public_current_refuses_underflowed_positive_denominator() -> None:
    """Keep a rounded-to-zero permeability-radius product from becoming current."""
    with pytest.raises(RuntimeError, match="denominator"):
        gs_toroidal_current_density_np(np.zeros((3, 3)), 1e-100, 3e-100, -1.0, 1.0, 1e-300)


def test_public_operator_refuses_overflowed_centre_coefficient() -> None:
    """Refuse an unrepresentable centre coefficient even with finite spacing squares."""
    with pytest.raises(RuntimeError, match="coefficient"):
        gs_delta_star_np(np.zeros((3, 3)), 1.0, 3.0, -1e-154, 1e-154)


@pytest.mark.parametrize("fault", ["coefficient", "source_norm"])
def test_public_equation_residual_refuses_nonfinite_arithmetic(fault: str) -> None:
    """Keep the fixed-work residual API from reporting an invalid relative metric.

    Parameters
    ----------
    fault : str
        Unrepresentable operator coefficient or norm of a finite final source.

    """
    with pytest.raises(RuntimeError, match="nonfinite"):
        gs_equation_residual_np(
            np.zeros((3, 3)),
            1.0,
            3.0,
            0.0 if fault == "coefficient" else -1.0,
            1e-170 if fault == "coefficient" else 1.0,
            3,
            3,
            1e6 if fault == "coefficient" else 1e170,
        )


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2)])
def test_public_equation_residual_requires_an_interior(shape: tuple[int, int]) -> None:
    """Refuse either short axis before measuring an empty centred stencil.

    Parameters
    ----------
    shape : tuple of int
        Actual flux grid with fewer than three radial or vertical nodes.
    """
    psi = np.zeros(shape)
    nz, nr = shape
    with pytest.raises(ValueError, match="3 x 3"):
        gs_equation_residual_np(psi, 1.0, 3.0, -1.0, 1.0, nr, nz, 0.0)
    np.testing.assert_array_equal(psi, np.zeros(shape))
