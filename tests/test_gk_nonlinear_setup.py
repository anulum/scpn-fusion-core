# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: nonlinear GK setup (state validation + dealiasing)

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKState
from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

_CFG = NonlinearGKConfig(
    n_kx=4,
    n_ky=4,
    n_theta=8,
    n_vpar=6,
    n_mu=4,
    n_species=2,
    cfl_adapt=False,
)


def _field(solver: NonlinearGKSolver) -> NDArray[np.complex128]:
    """Zero field array shaped to the solver's ``(n_kx, n_ky, n_theta)`` contract."""
    c = solver.cfg
    return np.zeros((c.n_kx, c.n_ky, c.n_theta), dtype=np.complex128)


class TestValidateStateRejectsMalformedFields:
    """validate_state guards every field shape, finiteness, and the scalar time."""

    def test_rejects_phi_shape_mismatch(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=3)
        bad = NonlinearGKState(f=state.f, phi=state.phi[:-1], time=state.time)

        with pytest.raises(ValueError, match="phi shape"):
            solver.validate_state(bad)

    def test_rejects_a_par_shape_mismatch(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=5)
        bad = NonlinearGKState(
            f=state.f,
            phi=state.phi,
            time=state.time,
            A_par=np.zeros((_CFG.n_kx, _CFG.n_ky), dtype=np.complex128),
        )

        with pytest.raises(ValueError, match="A_par shape"):
            solver.validate_state(bad)

    def test_rejects_non_finite_distribution(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=7)
        f = state.f.copy()
        f[0, 0, 0, 0, 0, 0] = np.nan
        bad = NonlinearGKState(f=f, phi=state.phi, time=state.time)

        with pytest.raises(ValueError, match="distribution must contain only finite"):
            solver.validate_state(bad)

    def test_rejects_non_finite_phi(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=9)
        phi = state.phi.copy()
        phi[0, 0, 0] = np.inf
        bad = NonlinearGKState(f=state.f, phi=phi, time=state.time)

        with pytest.raises(ValueError, match="phi must contain only finite"):
            solver.validate_state(bad)

    def test_rejects_non_finite_a_par(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=11)
        a_par = _field(solver)
        a_par[0, 0, 0] = np.nan
        bad = NonlinearGKState(f=state.f, phi=state.phi, time=state.time, A_par=a_par)

        with pytest.raises(ValueError, match="A_par must contain only finite"):
            solver.validate_state(bad)

    def test_rejects_non_finite_b_par(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=13)
        b_par = _field(solver)
        b_par[0, 0, 0] = np.inf
        bad = NonlinearGKState(f=state.f, phi=state.phi, time=state.time, B_par=b_par)

        with pytest.raises(ValueError, match="B_par must contain only finite"):
            solver.validate_state(bad)

    def test_rejects_non_finite_time(self) -> None:
        solver = NonlinearGKSolver(_CFG)
        state = solver.init_state(seed=15)
        bad = NonlinearGKState(f=state.f, phi=state.phi, time=float("inf"))

        with pytest.raises(ValueError, match="time must be finite"):
            solver.validate_state(bad)


class TestDealiasingMask:
    """A dealiasing mode other than ``2/3`` keeps the full spectral mask."""

    def test_non_two_thirds_dealiasing_keeps_full_mask(self) -> None:
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=6,
            n_mu=4,
            n_species=2,
            dealiasing="none",
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)

        assert solver.dealias_mask.shape == (cfg.n_kx, cfg.n_ky)
        assert solver.dealias_mask.dtype == np.bool_
        assert bool(np.all(solver.dealias_mask))
