# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Adaptive Controller
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Adaptive time-step control for integrated transport runtime."""

from __future__ import annotations

import numpy as np


class AdaptiveTimeController:
    """Richardson-extrapolation adaptive time controller for CN transport.

    Compares one full CN step vs. two half-steps to estimate the local
    truncation error, then uses a PI controller to adjust dt.

    Parameters
    ----------
    dt_init : float — initial time step [s]
    dt_min : float — minimum allowed dt
    dt_max : float — maximum allowed dt
    tol : float — target local error tolerance
    safety : float — safety factor (< 1) for step adjustment
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        dt_min: float = 1e-5,
        dt_max: float = 1.0,
        tol: float = 1e-3,
        safety: float = 0.9,
    ):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol
        self.safety = safety
        self.p = 2  # CN is second-order

        self.dt_history: list[float] = []
        self.error_history: list[float] = []
        self._err_prev: float = tol  # initialise for PI controller

    def estimate_error(
        self,
        solver: "TransportSolver",
        P_aux: float,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> float:
        """Estimate local error via Richardson extrapolation.

        Takes one full CN step of size dt and two half-steps of size dt/2,
        then compares.  The solver state is advanced by the *half-step*
        result (more accurate).

        Returns the estimated error norm.
        """
        Ti_save = solver.Ti.copy()
        Te_save = solver.Te.copy()

        # One full step
        solver.Ti = Ti_save.copy()
        solver.Te = Te_save.copy()
        solver.evolve_profiles(
            self.dt,
            P_aux,
            enforce_numerical_recovery=enforce_numerical_recovery,
            max_numerical_recoveries=max_numerical_recoveries,
        )
        T_full = solver.Ti.copy()

        # Two half steps
        solver.Ti = Ti_save.copy()
        solver.Te = Te_save.copy()
        solver.evolve_profiles(
            self.dt / 2.0,
            P_aux,
            enforce_numerical_recovery=enforce_numerical_recovery,
            max_numerical_recoveries=max_numerical_recoveries,
        )
        solver.evolve_profiles(
            self.dt / 2.0,
            P_aux,
            enforce_numerical_recovery=enforce_numerical_recovery,
            max_numerical_recoveries=max_numerical_recoveries,
        )
        T_half = solver.Ti.copy()

        # Richardson error estimate: ||T_full - T_half|| / (2^p - 1)
        error = float(np.linalg.norm(T_full - T_half)) / (2**self.p - 1)
        error = max(error, 1e-15)

        # Accept the half-step result (more accurate)
        solver.Ti = T_half
        solver.Te = T_half.copy()

        return error

    def adapt_dt(self, error: float) -> None:
        """Adjust dt using a PI controller.

        dt *= min(2, safety * (tol/err)^(0.7/p) * (err_prev/err)^(0.4/p))
        """
        self.error_history.append(error)
        self.dt_history.append(self.dt)

        ratio_i = (self.tol / error) ** (0.7 / self.p)
        ratio_p = (self._err_prev / error) ** (0.4 / self.p)
        factor = self.safety * ratio_i * ratio_p
        factor = min(factor, 2.0)
        factor = max(factor, 0.1)  # don't shrink too aggressively

        self.dt *= factor
        self.dt = max(self.dt, self.dt_min)
        self.dt = min(self.dt, self.dt_max)

        self._err_prev = error

