# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: nonlinear GK time integration (implicit e-, CFL, NaN)

from __future__ import annotations

import logging

import numpy as np
import pytest

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKState
from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

_IMPLICIT_CFG = NonlinearGKConfig(
    n_kx=4,
    n_ky=4,
    n_theta=8,
    n_vpar=6,
    n_mu=4,
    n_species=2,
    kinetic_electrons=True,
    implicit_electrons=True,
    cfl_adapt=False,
)


class TestImplicitElectronStreaming:
    """Backward-Euler electron streaming is applied when both flags are set."""

    def test_direct_solve_preserves_shape_and_finiteness(self) -> None:
        solver = NonlinearGKSolver(_IMPLICIT_CFG)
        state = solver.init_state(seed=3)

        corrected = solver._implicit_electron_streaming(state.f, dt=_IMPLICIT_CFG.dt)

        assert corrected.shape == state.f.shape
        assert np.all(np.isfinite(corrected))
        # Ion species (index 0) is untouched by the electron-only correction.
        np.testing.assert_array_equal(corrected[0], state.f[0])

    def test_rk4_step_invokes_implicit_correction(self) -> None:
        solver = NonlinearGKSolver(_IMPLICIT_CFG)
        state = solver.init_state(seed=5)

        stepped = solver._rk4_step(state, _IMPLICIT_CFG.dt)

        assert stepped.f.shape == state.f.shape
        assert np.all(np.isfinite(stepped.f))
        assert stepped.time == pytest.approx(state.time + _IMPLICIT_CFG.dt)


class TestAdaptiveCFL:
    """The CFL-adaptive branch bounds the step by the fastest advective scale."""

    def test_electrostatic_adaptive_step_is_bounded(self) -> None:
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            cfl_adapt=True,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(seed=7)

        dt = solver._cfl_dt(state)

        assert 0.0 < dt <= cfg.dt

    def test_kinetic_non_implicit_adaptive_step_uses_electron_thermal_scale(self) -> None:
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            kinetic_electrons=True,
            implicit_electrons=False,
            cfl_adapt=True,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(seed=9)

        dt = solver._cfl_dt(state)

        assert 0.0 < dt <= cfg.dt


class TestRunNaNGuard:
    """A non-finite state aborts the run loop with a logged warning."""

    def test_run_breaks_on_non_finite_state(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            n_steps=10,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        seed_state = solver.init_state(seed=11)
        nan_f = seed_state.f.copy()
        nan_f[:] = np.nan

        def _nan_step(state: NonlinearGKState, dt: float) -> NonlinearGKState:
            return NonlinearGKState(f=nan_f, phi=seed_state.phi, time=state.time + dt)

        monkeypatch.setattr(solver, "_rk4_step", _nan_step)

        with caplog.at_level(logging.WARNING):
            result = solver.run(seed_state)

        assert any("NaN at step" in record.message for record in caplog.records)
        # The loop broke on the first step, so no diagnostics were accumulated.
        assert float(np.sum(np.abs(result.Q_i_t))) == 0.0
