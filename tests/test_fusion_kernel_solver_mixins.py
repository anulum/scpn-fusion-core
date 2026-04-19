# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Linkage and contract tests for FusionKernel solver mixins."""

from __future__ import annotations

import numpy as np

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
