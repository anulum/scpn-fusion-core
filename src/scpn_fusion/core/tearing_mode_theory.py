# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Analytic FKR and Rutherford contracts for a Harris current sheet."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Constant-psi inner-layer matching coefficient in the FKR normalisation used
# below.  Writing it from gamma functions keeps the published asymptotic
# contract inspectable instead of fitting a benchmark-specific prefactor.
FKR_MATCHING_COEFFICIENT = (math.gamma(0.25) / (2.0 * math.pi * math.gamma(0.75))) ** 0.8


def _positive_finite(name: str, value: float) -> float:
    """Return a positive finite float or reject the physical input."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def fkr_constant_psi_growth_rate_per_s(
    *,
    delta_prime_per_m: float,
    sheet_half_width_m: float,
    wave_number_per_m: float,
    alfven_time_s: float,
    lundquist_number: float,
) -> float:
    r"""Return the FKR matched constant-psi growth rate for explicit inputs.

    This low-level form keeps :math:`S^{-3/5}`, :math:`(ka)^{2/5}`, and
    :math:`(\Delta'a)^{4/5}` independently testable. Non-positive
    :math:`\Delta'` is linearly stable and returns zero.
    """
    delta_prime = float(delta_prime_per_m)
    if not math.isfinite(delta_prime):
        raise ValueError("delta_prime_per_m must be finite")
    sheet_width = _positive_finite("sheet_half_width_m", sheet_half_width_m)
    wave_number = _positive_finite("wave_number_per_m", wave_number_per_m)
    tau_a = _positive_finite("alfven_time_s", alfven_time_s)
    lundquist = _positive_finite("lundquist_number", lundquist_number)
    if delta_prime <= 0.0:
        return 0.0
    ka = wave_number * sheet_width
    delta_a = delta_prime * sheet_width
    gamma_tau = (
        FKR_MATCHING_COEFFICIENT
        * lundquist ** (-3.0 / 5.0)
        * ka ** (2.0 / 5.0)
        * delta_a ** (4.0 / 5.0)
    )
    return float(gamma_tau / tau_a)


@dataclass(frozen=True)
class HarrisSheetTearingModel:
    r"""Reduced resistive-tearing model for :math:`B_y=B_0\tanh(x/a_s)`.

    The ideal outer-region matching result is

    .. math::
        \Delta'_0 a_s = 2\left[(k a_s)^{-1} - k a_s\right].

    ``finite_width_coefficient`` defines the first nonlinear correction
    :math:`\Delta'(w)=\Delta'_0-C_w w/a_s^2`.  It is an explicit equilibrium
    input, not a fitted hidden scale; the corresponding Rutherford saturation
    width is therefore the exact root of that declared finite-width model.

    References
    ----------
    Furth, Killeen & Rosenbluth, *Phys. Fluids* 6, 459 (1963),
    doi:10.1063/1.1706761.
    Rutherford, *Phys. Fluids* 16, 1903 (1973),
    doi:10.1063/1.1694232.
    """

    sheet_half_width_m: float
    wave_number_per_m: float
    finite_width_coefficient: float

    def __post_init__(self) -> None:
        """Validate the declared equilibrium and nonlinear closure."""
        _positive_finite("sheet_half_width_m", self.sheet_half_width_m)
        _positive_finite("wave_number_per_m", self.wave_number_per_m)
        _positive_finite("finite_width_coefficient", self.finite_width_coefficient)

    @property
    def ka(self) -> float:
        """Return the dimensionless current-sheet wave number."""
        return self.wave_number_per_m * self.sheet_half_width_m

    @property
    def delta_prime_zero_per_m(self) -> float:
        """Return the Harris-sheet ideal outer matching index at zero width."""
        return 2.0 * (1.0 / self.ka - self.ka) / self.sheet_half_width_m

    @property
    def linearly_unstable(self) -> bool:
        """Return whether the ideal outer matching index is positive."""
        return self.delta_prime_zero_per_m > 0.0

    def delta_prime_per_m(self, island_width_m: float) -> float:
        """Return the declared finite-width tearing index."""
        width = float(island_width_m)
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("island_width_m must be finite and >= 0")
        correction = self.finite_width_coefficient * width / self.sheet_half_width_m**2
        return self.delta_prime_zero_per_m - correction

    def fkr_growth_rate_per_s(
        self,
        *,
        alfven_time_s: float,
        lundquist_number: float,
    ) -> float:
        r"""Return the constant-psi FKR linear growth rate.

        The matched asymptotic scaling is
        :math:`\gamma\tau_A=C_{FKR}S^{-3/5}(ka_s)^{2/5}
        (\Delta'a_s)^{4/5}`.  Stable sheets return zero rather than taking a
        fractional power of a negative stability index.
        """
        return fkr_constant_psi_growth_rate_per_s(
            delta_prime_per_m=self.delta_prime_zero_per_m,
            sheet_half_width_m=self.sheet_half_width_m,
            wave_number_per_m=self.wave_number_per_m,
            alfven_time_s=alfven_time_s,
            lundquist_number=lundquist_number,
        )

    @property
    def rutherford_saturation_width_m(self) -> float:
        """Return the non-negative root of the finite-width tearing index."""
        if not self.linearly_unstable:
            return 0.0
        return (
            self.delta_prime_zero_per_m * self.sheet_half_width_m**2 / self.finite_width_coefficient
        )

    def rutherford_width_rate_m_per_s(
        self,
        island_width_m: float,
        *,
        magnetic_diffusivity_m2_per_s: float,
    ) -> float:
        r"""Return the normalised Rutherford rate :math:`\dot w=\eta_m\Delta'(w)`.

        The geometry-dependent order-unity Rutherford matching constant is
        absorbed into ``magnetic_diffusivity_m2_per_s``.  This convention does
        not alter the analytic saturation root used by the validation gate.
        """
        diffusivity = _positive_finite(
            "magnetic_diffusivity_m2_per_s", magnetic_diffusivity_m2_per_s
        )
        return diffusivity * self.delta_prime_per_m(island_width_m)

    def analytic_rutherford_width_m(
        self,
        initial_width_m: float,
        time_s: float,
        *,
        magnetic_diffusivity_m2_per_s: float,
    ) -> float:
        """Return the exact finite-width Rutherford solution for an unstable sheet."""
        width0 = float(initial_width_m)
        time = float(time_s)
        if not math.isfinite(width0) or width0 < 0.0:
            raise ValueError("initial_width_m must be finite and >= 0")
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("time_s must be finite and >= 0")
        diffusivity = _positive_finite(
            "magnetic_diffusivity_m2_per_s", magnetic_diffusivity_m2_per_s
        )
        if not self.linearly_unstable:
            return max(
                0.0,
                width0 + diffusivity * self.delta_prime_zero_per_m * time,
            )
        saturation = self.rutherford_saturation_width_m
        decay_rate = diffusivity * self.finite_width_coefficient / self.sheet_half_width_m**2
        return saturation + (width0 - saturation) * math.exp(-decay_rate * time)

    def integrate_rutherford_width(
        self,
        initial_width_m: float,
        *,
        magnetic_diffusivity_m2_per_s: float,
        dt_s: float,
        n_steps: int,
    ) -> FloatArray:
        """Integrate the finite-width Rutherford equation with fourth-order Runge-Kutta."""
        width = float(initial_width_m)
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("initial_width_m must be finite and >= 0")
        dt = _positive_finite("dt_s", dt_s)
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) < 1:
            raise ValueError("n_steps must be an integer >= 1")
        steps = int(n_steps)
        diffusivity = _positive_finite(
            "magnetic_diffusivity_m2_per_s", magnetic_diffusivity_m2_per_s
        )
        trace = np.empty(steps + 1, dtype=np.float64)
        trace[0] = width

        def rate(value: float) -> float:
            return diffusivity * self.delta_prime_per_m(max(value, 0.0))

        for index in range(steps):
            k1 = rate(width)
            k2 = rate(width + 0.5 * dt * k1)
            k3 = rate(width + 0.5 * dt * k2)
            k4 = rate(width + dt * k3)
            width = max(0.0, width + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
            trace[index + 1] = width
        return trace
