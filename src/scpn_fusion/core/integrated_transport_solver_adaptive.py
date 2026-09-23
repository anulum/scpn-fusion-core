# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Adaptive Controller
"""Adaptive time-step control for integrated transport runtime."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


_TRIAL_FIELDS = frozenset(
    {
        "Ti",
        "Te",
        "ne",
        "n_D",
        "n_T",
        "n_He",
        "n_impurity",
        "chi_i",
        "chi_e",
        "D_n",
        "q_profile",
        "_Z_eff",
        "T_edge_keV",
        "_dV_cache",
        "pedestal_model",
        "_neural_transport_model",
        "_neural_transport_model_weights_path",
    }
)


def _trial_state(solver: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(solver).items()
        if name in _TRIAL_FIELDS or name.startswith("_last_")
    }


def _install_trial_state(solver: Any, state: dict[str, Any]) -> None:
    for name in _trial_state(solver):
        delattr(solver, name)
    vars(solver).update(state)


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
        self.trial_difference_history: list[dict[str, float]] = []
        self._err_prev: float = tol  # initialise for PI controller

    def estimate_error(
        self,
        solver: Any,
        P_aux: float,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> float:
        """Estimate the Ti error using isolated full and half transport trials.

        Parameters
        ----------
        solver : TransportSolver-compatible object
            Runtime with Ti/Te profiles and ``evolve_profiles``. Transport-owned
            profiles, species, coefficients, boundary values, diagnostics and
            neural/pedestal caches are isolated; kernel geometry is not evolved.
            Mutable state must be stored in ``vars(solver)``. Implementations
            storing profiles only in properties or slots are not supported.
        P_aux : float
            Auxiliary heating power in MW.
        enforce_numerical_recovery : bool, optional
            Propagate the runtime recovery-budget refusal from either trial.
        max_numerical_recoveries : int or None, optional
            Per-evolution recovery limit, passed unchanged to the runtime.

        Returns
        -------
        float
            Historical Ti-only Richardson L2 estimate in keV, floored at 1e-15.

        Notes
        -----
        Only two successful half steps are committed. Any trial exception leaves
        the original runtime objects and controller history unchanged. External
        logs or backend process effects cannot be rolled back by this operation.
        Global fallback telemetry and its budgets count every attempted trial,
        including discarded full steps; restoring solver state does not refund
        those budgets. A fallback-budget refusal is propagated unchanged.
        Nonfinite Ti, Te or ne trial differences refuse the trial and roll back.
        Finite Te/ne differences do not control the timestep.
        Raw Ti/Te and, when present, ne full-versus-half L2 differences are retained
        in ``trial_difference_history`` with units in their keys. They do not
        change the existing Ti-only timestep policy or qualify coupled order.
        """
        original = _trial_state(solver)
        kwargs = {
            "enforce_numerical_recovery": enforce_numerical_recovery,
            "max_numerical_recoveries": max_numerical_recoveries,
        }
        _install_trial_state(solver, deepcopy(original))
        try:
            solver.evolve_profiles(self.dt, P_aux, **kwargs)
            full = {
                name: getattr(solver, name).copy()
                for name in ("Ti", "Te", "ne")
                if name in original
            }
        finally:
            _install_trial_state(solver, original)

        accepted = False
        _install_trial_state(solver, deepcopy(original))
        try:
            solver.evolve_profiles(self.dt / 2.0, P_aux, **kwargs)
            solver.evolve_profiles(self.dt / 2.0, P_aux, **kwargs)
            differences = {
                f"{name.lower()}_l2_{unit}": float(np.linalg.norm(values - getattr(solver, name)))
                for name, values in full.items()
                for unit in ("1e19_m3" if name == "ne" else "kev",)
            }
            if not all(np.isfinite(value) for value in differences.values()):
                raise FloatingPointError("Nonfinite full-versus-half transport difference")
            error = max(differences["ti_l2_kev"] / (2**self.p - 1), 1e-15)
            self.trial_difference_history.append(differences)
            accepted = True
            return float(error)
        finally:
            if not accepted:
                _install_trial_state(solver, original)

    def adapt_dt(self, error: float) -> None:
        """Adjust dt using a PI controller.

        ``dt *= min(2, safety * (tol/err)^(0.7/p) * (err_prev/err)^(0.4/p))``
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
