# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nonlinear δf Gyrokinetic Solver
"""
Nonlinear delta-f gyrokinetic solver in flux-tube geometry.

Solves the gyrokinetic Vlasov equation for the perturbed distribution
function delta-f(k_x, k_y, theta, v_parallel, mu) with E cross B
nonlinearity computed via dealiased 2D FFT (Orszag 1971 2/3 rule).

Physics:
  - Quasineutrality field solve (adiabatic electrons)
  - E cross B advection: dealiased Arakawa bracket via FFT
  - Parallel streaming: 4th-order compact finite differences
  - Magnetic curvature/grad-B drift
  - Sugama-like collision operator with moment projection
  - 4th-order hyperdiffusion for numerical stability
  - RK4 time stepping with CFL-adaptive dt
"""

from __future__ import annotations

from scpn_fusion.core._gk_nonlinear_operators import NonlinearGKOperatorsMixin
from scpn_fusion.core._gk_nonlinear_setup import NonlinearGKSetupMixin
from scpn_fusion.core._gk_nonlinear_time import NonlinearGKTimeMixin
from scpn_fusion.core._gk_nonlinear_types import (
    NonlinearGKConfig,
    NonlinearGKInvariantDiagnostics,
    NonlinearGKResult,
    NonlinearGKState,
)


class NonlinearGKSolver(
    NonlinearGKSetupMixin,
    NonlinearGKOperatorsMixin,
    NonlinearGKTimeMixin,
):
    """Nonlinear delta-f gyrokinetic solver."""

    def __init__(self, config: NonlinearGKConfig | None = None):
        self.cfg = config or NonlinearGKConfig()
        self._setup_grids()
        self._setup_ballooning()
        self._setup_geometry()
        self._setup_species()


__all__ = [
    "NonlinearGKConfig",
    "NonlinearGKInvariantDiagnostics",
    "NonlinearGKResult",
    "NonlinearGKSolver",
    "NonlinearGKState",
]
