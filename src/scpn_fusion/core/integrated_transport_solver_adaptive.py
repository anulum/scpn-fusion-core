# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Adaptive Controller
"""Adaptive time-step control for integrated transport runtime."""

from __future__ import annotations

import operator
from copy import deepcopy
from typing import Any, SupportsIndex

import numpy as np

STOP_COMPLETED = "completed"
STOP_MINIMUM_STEP_EXHAUSTED = "minimum_step_exhausted"
STOP_RAISED = "raised"

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
    """Return the solver attributes that one transport trial may change."""
    return {
        name: value
        for name, value in vars(solver).items()
        if name in _TRIAL_FIELDS or name.startswith("_last_")
    }


def _install_trial_state(solver: Any, state: dict[str, Any]) -> None:
    """Replace every trial attribute of the solver with the entries of ``state``."""
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
    ) -> None:
        """Set the timestep bounds, the tolerance and empty diagnostic histories."""
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol
        self.safety = safety
        self.p = 2  # CN is second-order

        self.dt_history: list[float] = []
        self.error_history: list[float] = []
        self.trial_difference_history: list[dict[str, float]] = []
        self.minimum_step_exhausted = False
        self._err_prev: float = tol  # initialise for PI controller

    def diagnostics(self) -> dict[str, Any]:
        """Return copies of the current timestep and the diagnostic histories.

        Returns
        -------
        dict[str, Any]
            ``dt_final`` in seconds, ``dt_history`` in seconds, ``error_history``
            in keV and ``trial_difference_history`` with units in its keys. The
            lists and their entries are copies, so later steps do not change them.
        """
        return {
            "dt_final": self.dt,
            "dt_history": self.dt_history.copy(),
            "error_history": self.error_history.copy(),
            "trial_difference_history": [dict(entry) for entry in self.trial_difference_history],
        }

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
        """Adjust the next timestep from a finite positive Ti error estimate.

        Parameters
        ----------
        error : float
            Positive Richardson L2 estimate for Ti in keV.

        Raises
        ------
        ValueError
            If error is nonfinite or nonpositive. The controller state and
            all diagnostic histories remain unchanged on refusal.

        Notes
        -----
        The PI factor is ``safety * (tol/error)**(0.7/p) *
        (err_prev/error)**(0.4/p)``, clamped to [0.1, 2]. The updated timestep
        is clamped to ``[dt_min, dt_max]`` in seconds. Successful updates record
        the supplied error and the timestep before adjustment.

        ``minimum_step_exhausted`` becomes true when the step just measured was
        taken at or below ``dt_min`` and its error still exceeds ``tol``: the
        controller cannot shrink the step further, so the tolerance cannot be
        met. It is recomputed on every successful update. Steps are never
        rejected, so the measured step has already been committed.
        """
        if not np.isfinite(error) or error <= 0.0:
            raise ValueError("error must be finite and positive")

        self.minimum_step_exhausted = bool(self.dt <= self.dt_min and error > self.tol)
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


def positive_step_count(n_steps: SupportsIndex) -> int:
    """Return a transport step count as a positive ``int``, converted as ``range`` does.

    The count is converted with :func:`operator.index` before its sign is checked.
    Every value without ``__index__``, such as 1.5, -1.5, NaN or infinity, therefore
    raises ``TypeError``, and an object that implements only ``__index__`` is accepted.

    Parameters
    ----------
    n_steps : SupportsIndex
        Requested number of transport steps.

    Returns
    -------
    int
        The converted count, at least 1.

    Raises
    ------
    TypeError
        If ``n_steps`` has no ``__index__``.
    ValueError
        If the converted count is zero or negative.
    """
    count = operator.index(n_steps)
    if count <= 0:
        raise ValueError("n_steps must be positive")
    return count


def run_adaptive_steps(
    solver: Any,
    P_aux: float,
    n_steps: int,
    dt: float,
    tol: float,
    *,
    enforce_numerical_recovery: bool = False,
    max_numerical_recoveries: int | None = None,
) -> dict[str, Any]:
    """Run up to ``n_steps`` adaptive transport steps and report why the run stopped.

    Parameters
    ----------
    solver : TransportSolver-compatible object
        Runtime with ``update_transport_model``, ``evolve_profiles`` and
        ``compute_confinement_time``; see :meth:`AdaptiveTimeController.estimate_error`.
    P_aux : float
        Auxiliary heating power in MW.
    n_steps : int
        Positive maximum number of adaptive steps.
    dt : float
        Initial timestep in seconds.
    tol : float
        Ti Richardson L2 error target in keV.
    enforce_numerical_recovery : bool, optional
        Propagate recovery-budget refusals from profile evolution.
    max_numerical_recoveries : int or None, optional
        Per-evolution recovery limit, passed unchanged to the runtime.

    Returns
    -------
    dict[str, Any]
        Final mean/core Ti in keV, confinement time in seconds, the number of
        executed steps as ``n_steps``, copied Ti/ne profiles, the controller
        diagnostics of :meth:`AdaptiveTimeController.diagnostics` and
        ``stop_reason``: ``"completed"`` after ``n_steps`` steps, or
        ``"minimum_step_exhausted"`` after a step at the minimum timestep
        exceeded ``tol`` (the run stops there instead of continuing at a step
        that cannot meet the tolerance).

    Raises
    ------
    TypeError
        If ``n_steps`` has no ``__index__``, as ``range`` refuses it: fractional
        (including negative fractional), NaN and infinite counts are never
        rounded or run. The conversion precedes the sign check; nothing is
        mutated. See :func:`positive_step_count`.
    ValueError
        If ``n_steps`` converts to zero or a negative integer; nothing is mutated.
    Exception
        Any refusal from the transport model, a trial, the controller or the
        final result (for example the confinement-time estimate) propagates
        unchanged.

    Notes
    -----
    After every run that passes the ``n_steps`` check, ``solver.adaptive_run_diagnostics``
    holds the controller diagnostics, ``steps_completed`` and ``stop_reason``.
    When an exception ends the run, including one raised while the final
    result is built after the last step, ``stop_reason`` is ``"raised"``, the
    entry ``exception`` names its type and message, and the histories cover
    every step completed before it, so a recovery refusal does not discard them.
    """
    # Same contract as range(n_steps): index conversion first, then the sign, before any mutation.
    step_limit = positive_step_count(n_steps)

    controller = AdaptiveTimeController(dt_init=dt, tol=tol)
    steps = 0
    stop_reason = STOP_COMPLETED
    try:
        while steps < step_limit:
            solver.update_transport_model(P_aux)
            error = controller.estimate_error(
                solver,
                P_aux,
                enforce_numerical_recovery=enforce_numerical_recovery,
                max_numerical_recoveries=max_numerical_recoveries,
            )
            controller.adapt_dt(error)
            steps += 1
            if controller.minimum_step_exhausted:
                stop_reason = STOP_MINIMUM_STEP_EXHAUSTED
                break
        result = {
            "T_avg": float(np.mean(solver.Ti)),
            "T_core": float(solver.Ti[0]),
            "tau_e": solver.compute_confinement_time(P_aux),
            "n_steps": steps,
            "Ti_profile": solver.Ti.copy(),
            "ne_profile": solver.ne.copy(),
            **controller.diagnostics(),
            "stop_reason": stop_reason,
        }
    except BaseException as exc:
        solver.adaptive_run_diagnostics = {
            **controller.diagnostics(),
            "steps_completed": steps,
            "stop_reason": STOP_RAISED,
            "exception": f"{type(exc).__name__}: {exc}",
        }
        raise

    solver.adaptive_run_diagnostics = {
        **controller.diagnostics(),
        "steps_completed": steps,
        "stop_reason": stop_reason,
    }
    return result
