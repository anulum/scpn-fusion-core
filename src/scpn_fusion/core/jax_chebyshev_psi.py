# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — differentiable compact Chebyshev ψ representation (coefficients ⇄ ψ field)
"""Differentiable compact **Chebyshev ψ representation**: a tensor-product Chebyshev expansion of
the poloidal flux ``ψ(R, Z)`` on the equilibrium grid.

A compact **primitive** for the collaboration's ψ-field exchange contract: represent a solved
equilibrium's ψ field not as the full ``NZ·NR`` grid but as a small block of Chebyshev coefficients
(the contract fixes the total mode budget ``16×34`` → 544 numbers on the contract's 65² grid). This
module maps that block both ways:

* **synthesis** — ``ψ = Φ @ c`` (:func:`evaluate_psi`): coefficients → the flux field, and
* **analysis** — ``c = fit(ψ)`` (:func:`fit_psi_coeffs`): a solved field → its compact coefficients.

Scope: this is the differentiable basis primitive only. It is not yet wired into a production
exchange pipeline, and the paper's fuller contract (a paired off-grid diagnostic design and spatial
``∂ψ/∂R``, ``∂ψ/∂Z`` outputs) is deliberately out of scope here — those build on this primitive.

The key property (the same fixed-basis idea the smooth O-point and the B-spline profile basis use):
the Chebyshev polynomials evaluated on the **fixed** grid are a *constant* design matrix
``Φ[grid, mode]``, so the synthesised field is an **exact-gradient linear function** of the
coefficients, ``∂ψ/∂c = Φ`` exactly — no interpolation-induced gradient noise. Feeding
``evaluate_psi(c, Φ)`` into the downstream equilibrium quantities therefore turns the differentiated
unknown into the compact ψ coefficients, and ``jax.grad`` follows through automatically — the same
role for the *field* that :mod:`scpn_fusion.core.jax_profile_basis` plays for the *profiles*.

Convention (matching the free-boundary solver): ``ψ`` has shape ``(NZ, NR)``; its row-major flatten
is ``iz·NR + ir``. The design is the Kronecker structure ``Φ[(iz,ir),(jz,jr)] =
T_{jz}(z_iz)·T_{jr}(r_ir)`` with coefficients ordered ``jz·n_r + jr``. Coordinates are affinely
mapped to the Chebyshev domain ``[-1, 1]`` from each grid's own ``[min, max]``. **Axis-mode split:**
the contract states ``16×34`` but does not pin which axis carries which; the local convention here is
``n_z = 16`` (across Z), ``n_r = 34`` (across R) — both are arguments, so a collaborator's confirmed
orientation is a one-line change. A well-posed (identifiable) fit requires at least as many grid
points as modes on each axis: :func:`chebyshev_psi_design_matrix` fails closed otherwise. The design
is built once (via NumPy, outside tracing), cached read-only, then converted per-trace at the call
site so no traced intermediate escapes a ``jit`` scope.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.typing import NDArray

DEFAULT_N_R = 34  # Chebyshev modes across R (major radius) — local convention for the 16×34 budget
DEFAULT_N_Z = 16  # Chebyshev modes across Z (height)        — local convention for the 16×34 budget

_DESIGN_CACHE: dict[tuple[int, int, bytes, bytes], NDArray[np.float64]] = {}


def _map_to_unit(coord: NDArray[np.float64]) -> NDArray[np.float64]:
    """Affinely map a monotone grid to the Chebyshev domain ``[-1, 1]`` using its own span.

    A degenerate (single-valued) span maps to all-zeros (the domain centre) rather than dividing by
    zero. Note this is *not* an identifiable basis for more than one mode — at ``x = 0`` the even
    Chebyshev polynomials are ``±1`` (``T_0, T_2, T_4, … = 1, −1, 1, …``) so the columns alias;
    :func:`chebyshev_psi_design_matrix` fails closed on a degenerate span with more than one mode.
    """
    lo = float(coord.min())
    hi = float(coord.max())
    span = hi - lo
    if span == 0.0:
        return np.zeros_like(coord)
    return 2.0 * (coord - lo) / span - 1.0


def chebyshev_psi_design_matrix(
    R_grid: NDArray[np.float64] | jnp.ndarray,
    Z_grid: NDArray[np.float64] | jnp.ndarray,
    n_r: int = DEFAULT_N_R,
    n_z: int = DEFAULT_N_Z,
) -> NDArray[np.float64]:
    """Constant tensor-product Chebyshev design ``Φ`` (shape ``(NZ·NR, n_z·n_r)``).

    ``ψ.reshape(-1) = Φ @ c`` with ``c`` ordered ``jz·n_r + jr``. Built once per
    ``(n_r, n_z, R grid, Z grid)`` and cached read-only. ``T_0 ≡ 1`` so the first coefficient
    (``jz = jr = 0``) is the field's constant offset.

    Fails closed (``ValueError``) on an ill-posed request — fewer grid points than modes on an axis,
    or a degenerate span with more than one mode — either of which makes the Chebyshev columns
    rank-deficient so the analysis fit is non-identifiable and synthesis is severely ill-conditioned.
    """
    r = np.asarray(R_grid, dtype=np.float64)
    z = np.asarray(Z_grid, dtype=np.float64)
    if n_r < 1 or n_z < 1:
        raise ValueError(f"mode counts must be ≥ 1, got n_r={n_r}, n_z={n_z}")
    if n_r > r.size or n_z > z.size:
        raise ValueError(
            f"under-resolved: need ≥ n_r points in R and ≥ n_z in Z; "
            f"got NR={r.size} (n_r={n_r}), NZ={z.size} (n_z={n_z})"
        )
    if (n_r > 1 and r.max() == r.min()) or (n_z > 1 and z.max() == z.min()):
        raise ValueError("degenerate grid span with more than one mode is non-identifiable")
    key = (n_r, n_z, r.tobytes(), z.tobytes())
    if key not in _DESIGN_CACHE:
        t_r = chebvander(_map_to_unit(r), n_r - 1)  # (NR, n_r)
        t_z = chebvander(_map_to_unit(z), n_z - 1)  # (NZ, n_z)
        # Φ[(iz,ir),(jz,jr)] = t_z[iz,jz] · t_r[ir,jr]
        design = np.einsum("zZ,rR->zrZR", t_z, t_r).reshape(z.size * r.size, n_z * n_r)
        design = np.ascontiguousarray(design, dtype=np.float64)
        design.flags.writeable = False  # cached array is shared — freeze against mutation
        _DESIGN_CACHE[key] = design
    return _DESIGN_CACHE[key]


def evaluate_psi(
    coeffs: jnp.ndarray, design_matrix: NDArray[np.float64] | jnp.ndarray
) -> jnp.ndarray:
    """Flat ψ field ``Φ @ coeffs`` — differentiable and exact-gradient in ``coeffs``.

    Returns the row-major flat field (length ``NZ·NR``); use :func:`psi_from_coeffs` for the 2-D
    ``(NZ, NR)`` field. ``design_matrix`` comes from :func:`chebyshev_psi_design_matrix`. The design
    is cast to a JAX array at the call site so it composes inside ``jit`` / ``grad`` with no traced
    value escaping.
    """
    return jnp.asarray(design_matrix) @ coeffs


def psi_from_coeffs(
    coeffs: jnp.ndarray,
    design_matrix: NDArray[np.float64] | jnp.ndarray,
    nz: int,
    nr: int,
) -> jnp.ndarray:
    """The 2-D ``(NZ, NR)`` flux field from compact coefficients (differentiable)."""
    return evaluate_psi(coeffs, design_matrix).reshape(nz, nr)


def fit_psi_coeffs(
    psi: NDArray[np.float64] | jnp.ndarray,
    design_matrix: NDArray[np.float64] | jnp.ndarray,
) -> jnp.ndarray:
    """Least-squares compact coefficients of a solved field: ``c = argmin ‖Φ c − ψ‖``.

    ``psi`` may be the 2-D ``(NZ, NR)`` field or its row-major flatten. Returns ``n_z·n_r``
    coefficients; the fit is exact (to round-off) when ``ψ`` lies in the basis's span and is the
    optimal projection otherwise. Uses :func:`jax.numpy.linalg.lstsq` so it composes under ``grad``
    of ψ (the analysis map is itself differentiable in the field).
    """
    phi = jnp.asarray(design_matrix)
    rhs = jnp.asarray(psi).reshape(-1)
    coeffs, *_ = jnp.linalg.lstsq(phi, rhs, rcond=None)
    return coeffs
