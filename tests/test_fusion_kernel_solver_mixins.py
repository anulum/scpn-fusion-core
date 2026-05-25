# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Linkage and contract tests for FusionKernel solver mixins."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.fusion_kernel_iterative_solver import (
    FusionKernelIterativeSolverMixin,
)
from scpn_fusion.core.fusion_kernel_newton_solver import FusionKernelNewtonSolverMixin
from scpn_fusion.core.fusion_kernel_numerics import (
    NUMERIC_SANITIZE_CAP,
    sanitize_numeric_array,
    stable_rms,
)


def test_fusion_kernel_inherits_solver_mixins() -> None:
    assert issubclass(FusionKernel, FusionKernelIterativeSolverMixin)
    assert issubclass(FusionKernel, FusionKernelNewtonSolverMixin)


def test_numeric_helpers_are_stable_on_non_finite_inputs() -> None:
    arr = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float64)
    sanitized = sanitize_numeric_array(arr)
    assert np.all(np.isfinite(sanitized))
    assert float(np.max(np.abs(sanitized))) <= NUMERIC_SANITIZE_CAP
    assert stable_rms(sanitized) >= 0.0


class _IterativeKernelStub(FusionKernelIterativeSolverMixin):
    def __init__(self) -> None:
        self.dR = 0.25
        self.dZ = 0.25
        r = np.linspace(1.0, 2.0, 5)
        z = np.linspace(-0.5, 0.5, 5)
        self.RR, self.ZZ = np.meshgrid(r, z)


@pytest.mark.parametrize("omega", [0.0, 0.999999, 2.0, np.inf, np.nan])
def test_sor_step_rejects_unstable_relaxation_factor(omega: float) -> None:
    """SOR must reject relaxation factors outside the stable GS interval."""
    kernel = _IterativeKernelStub()
    psi = np.zeros((5, 5), dtype=np.float64)
    source = np.zeros_like(psi)

    with pytest.raises(ValueError, match="omega"):
        kernel._sor_step(psi, source, omega=omega)


@pytest.mark.parametrize("omega", [0.0, 0.999999, 2.0, np.inf, np.nan])
def test_multigrid_smoother_rejects_unstable_relaxation_factor(omega: float) -> None:
    """The multigrid smoother shares the same SOR relaxation contract."""
    kernel = _IterativeKernelStub()
    psi = np.zeros((5, 5), dtype=np.float64)
    source = np.zeros_like(psi)

    with pytest.raises(ValueError, match="omega"):
        kernel._mg_smooth(psi, source, kernel.RR, kernel.dR, kernel.dZ, omega, n_sweeps=1)
