# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for AdaptiveTimeController extraction module."""

from __future__ import annotations

import numpy as np

import scpn_fusion.core.integrated_transport_solver_adaptive as adaptive_mod


class _StubSolver:
    def __init__(self) -> None:
        self.Ti = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.Te = self.Ti.copy()
        self.calls: list[float] = []

    def evolve_profiles(
        self,
        dt: float,
        P_aux: float,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> tuple[float, float]:
        _ = (P_aux, enforce_numerical_recovery, max_numerical_recoveries)
        self.calls.append(dt)
        self.Ti = self.Ti + dt
        self.Te = self.Ti.copy()
        return float(np.mean(self.Ti)), float(self.Ti[0])


def test_estimate_error_uses_richardson_half_step_path() -> None:
    atc = adaptive_mod.AdaptiveTimeController(dt_init=0.1, tol=1e-3)
    solver = _StubSolver()

    err = atc.estimate_error(solver, P_aux=5.0)

    assert err >= 0.0
    assert solver.calls == [0.1, 0.05, 0.05]
    np.testing.assert_allclose(solver.Te, solver.Ti)


def test_adapt_dt_respects_bounds() -> None:
    atc = adaptive_mod.AdaptiveTimeController(dt_init=0.01, dt_min=0.001, dt_max=0.02, tol=1e-3)

    atc.adapt_dt(error=1e-12)
    assert 0.001 <= atc.dt <= 0.02
    atc.adapt_dt(error=1e3)
    assert 0.001 <= atc.dt <= 0.02
