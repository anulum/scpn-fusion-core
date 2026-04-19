# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for integrated transport runtime mixins."""

from __future__ import annotations

import numpy as np

import scpn_fusion.core.integrated_transport_solver_runtime as runtime_mod


def test_runtime_module_exports_core_symbols() -> None:
    assert hasattr(runtime_mod, "TransportSolverRuntimeMixin")
    assert hasattr(runtime_mod, "AdaptiveTimeController")


def test_thomas_solver_matches_known_solution() -> None:
    mixin = runtime_mod.TransportSolverRuntimeMixin
    # Tridiagonal system:
    # [ 2 -1  0][x0]   [1]
    # [-1  2 -1][x1] = [0]
    # [ 0 -1  2][x2]   [1]
    # Exact solution is x=[1, 1, 1].
    a = np.array([-1.0, -1.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    c = np.array([-1.0, -1.0], dtype=np.float64)
    d = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    x = mixin._thomas_solve(a, b, c, d)
    np.testing.assert_allclose(x, np.ones(3), rtol=1e-12, atol=1e-12)


def test_sanitize_with_fallback_replaces_nonfinite_and_clips() -> None:
    mixin = runtime_mod.TransportSolverRuntimeMixin
    arr = np.array([np.nan, np.inf, -np.inf, 5.0], dtype=np.float64)
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    out, recovered = mixin._sanitize_with_fallback(arr, ref, floor=0.5, ceil=4.5)

    assert recovered == 3
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float64))
