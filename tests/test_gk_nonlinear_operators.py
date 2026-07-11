# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: nonlinear GK field operators (electrostatic branches)

from __future__ import annotations

import numpy as np

from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

# Small electrostatic grid: ``electromagnetic`` defaults to ``False`` so the
# Ampère and magnetic-compression solves take their zero-field early returns.
_ELECTROSTATIC_CFG = NonlinearGKConfig(
    n_kx=4,
    n_ky=4,
    n_theta=8,
    n_vpar=6,
    n_mu=4,
    n_species=2,
    cfl_adapt=False,
)


class TestElectrostaticFieldSolves:
    """The magnetic field solves collapse to zero when the run is electrostatic."""

    def test_ampere_solve_returns_zero_field_when_electromagnetic_off(self) -> None:
        solver = NonlinearGKSolver(_ELECTROSTATIC_CFG)
        state = solver.init_state(amplitude=1e-5, seed=11)

        a_par = solver.ampere_solve(state.f)

        c = _ELECTROSTATIC_CFG
        assert a_par.shape == (c.n_kx, c.n_ky, c.n_theta)
        assert np.count_nonzero(a_par) == 0
        assert a_par.dtype == np.complex128

    def test_magnetic_compression_solve_returns_zero_field_when_electromagnetic_off(
        self,
    ) -> None:
        solver = NonlinearGKSolver(_ELECTROSTATIC_CFG)
        state = solver.init_state(amplitude=1e-5, seed=17)

        b_par = solver.magnetic_compression_solve(state.f)

        c = _ELECTROSTATIC_CFG
        assert b_par.shape == (c.n_kx, c.n_ky, c.n_theta)
        assert np.count_nonzero(b_par) == 0
        assert b_par.dtype == np.complex128
