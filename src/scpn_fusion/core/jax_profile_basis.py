# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — differentiable compact B-spline profile basis (p'/FF' coefficients → samples)
"""Differentiable compact profile basis: B-spline coefficients → ``p'`` / ``FF'`` samples.

The Integrated Data Analysis (IDA) loop infers a **compact** parameterisation of the profiles
``p'(ψ_N)`` and ``FF'(ψ_N)`` — a handful of B-spline coefficients — not the full per-cell sample
vector the free-boundary solver consumes. This module maps such coefficients to profile samples
on any ``ψ_N`` grid, differentiably.

The key property (the same fixed-basis idea the smooth O-point uses): because the knot vector is
**fixed**, the B-spline design matrix ``B[query, coeff]`` is a *constant* linear map — a
partition of unity, ``C^{degree−1}`` smooth — so the evaluated profile is an **exact-gradient
linear function** of the coefficients, ``profile = B @ coeffs``. ``∂profile/∂coeffs = B`` exactly,
with no interpolation-induced gradient noise. Feeding ``evaluate_profile(coeffs, B)`` as the
solver's ``pprime_vals`` / ``ffprime_vals`` therefore turns the differentiated inputs into the
compact basis coefficients (shrinking the inference dimension and imposing profile smoothness),
and ``jax.grad`` of the equilibrium w.r.t. the coefficients follows through automatically.

A clamped uniform cubic knot vector on ``ψ_N ∈ [0, 1]`` is used by default (the spline is anchored
at the axis and edge); ``n_coeff`` and ``degree`` are configurable so the exact basis (e.g. the
12-coefficient contract) can be matched. The design matrix is built once (via SciPy, outside
tracing) and cached as a concrete NumPy array, then converted per-trace at the call site so no
traced intermediate escapes a ``jit`` scope.
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

DEFAULT_DEGREE = 3
DEFAULT_N_COEFF = 12

_DESIGN_CACHE: dict[tuple[int, int, bytes], NDArray[np.float64]] = {}


def _clamped_uniform_knots(n_coeff: int, degree: int) -> NDArray[np.float64]:
    """Clamped uniform knot vector on [0, 1] for ``n_coeff`` cubic-ish control points.

    ``degree + 1`` repeated knots at each end (so the spline is clamped to the first/last
    coefficient at ``ψ_N = 0`` / ``1``), with ``n_coeff − degree − 1`` uniform interior knots.
    """
    n_interior = n_coeff - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)]).astype(np.float64)
    return cast(NDArray[np.float64], knots)


def bspline_design_matrix(
    psin_query: NDArray[np.float64] | jnp.ndarray,
    n_coeff: int = DEFAULT_N_COEFF,
    degree: int = DEFAULT_DEGREE,
) -> NDArray[np.float64]:
    """Constant B-spline design matrix ``B`` (shape ``(len(psin_query), n_coeff)``).

    ``profile(ψ_N) = B @ coeffs``. Rows sum to one (partition of unity). Built once per
    ``(n_coeff, degree, ψ_N grid)`` and cached as a concrete NumPy array. Query points are
    clipped to ``[0, 1]`` (the spline's clamped support).
    """
    q = np.clip(np.asarray(psin_query, dtype=np.float64), 0.0, 1.0)
    key = (n_coeff, degree, q.tobytes())
    if key not in _DESIGN_CACHE:
        knots = _clamped_uniform_knots(n_coeff, degree)
        design = np.asarray(BSpline.design_matrix(q, knots, degree).todense(), dtype=np.float64)
        _DESIGN_CACHE[key] = design
    return _DESIGN_CACHE[key]


def evaluate_profile(
    coeffs: jnp.ndarray, design_matrix: NDArray[np.float64] | jnp.ndarray
) -> jnp.ndarray:
    """Profile samples ``B @ coeffs`` — differentiable and exact-gradient in ``coeffs``.

    ``design_matrix`` comes from :func:`bspline_design_matrix` (or any fixed basis of matching
    shape). Converts the (concrete) design to a JAX array at the call site so it composes inside
    ``jit`` / ``grad`` without a traced value escaping.
    """
    return jnp.asarray(design_matrix) @ coeffs
