# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Numerical field operations for the IDA operator-response diagnostic."""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeAlias, cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_operator = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_operator"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

FloatArray: TypeAlias = NDArray[np.float64]
_array_sha256: Callable[[object], str] = _same_case._array_sha256


def finite_plane(value: object, *, field: str) -> FloatArray:
    """Return a finite, non-trivial float64 plane or fail closed."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite non-trivial 2D array")
    return array


def zero_wall(value: object, *, field: str) -> FloatArray:
    """Copy a residual plane while enforcing zero identity-wall forcing."""
    array = finite_plane(value, field=field).copy()
    array[0, :] = 0.0
    array[-1, :] = 0.0
    array[:, 0] = 0.0
    array[:, -1] = 0.0
    return array


def sum_fields(fields: tuple[FloatArray, ...]) -> FloatArray:
    """Sum matching finite planes without accepting an empty decomposition."""
    if not fields:
        raise ValueError("field decomposition must not be empty")
    shape = fields[0].shape
    if any(field.shape != shape or not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("field decomposition must contain matching finite planes")
    return np.asarray(
        np.sum(np.stack(fields), axis=0, dtype=np.float64),
        dtype=np.float64,
    )


def closure_max_abs(actual: FloatArray, components: tuple[FloatArray, ...]) -> float:
    """Return the maximum absolute reconstruction error."""
    return float(np.max(np.abs(actual - sum_fields(components))))


def forcing_metric(
    field: FloatArray,
    *,
    exact_residual: FloatArray,
) -> dict[str, Any]:
    """Summarise one stationarity residual against the exact-source residual."""
    plane = finite_plane(field, field="forcing component")
    scale = finite_plane(exact_residual, field="exact-source residual")
    if plane.shape != scale.shape:
        raise ValueError("forcing component and exact-source residual must match")
    denominator = max(float(np.linalg.norm(scale)), 1.0e-30)
    return {
        "field_sha256": _array_sha256(plane),
        "l2": float(np.linalg.norm(plane)),
        "linf": float(np.max(np.abs(plane))),
        "relative_l2_to_exact_source_residual": float(np.linalg.norm(plane)) / denominator,
    }


def operator_components(
    *,
    freegs: Any,
    equilibrium: Any,
    reference_current_rz: FloatArray,
    exact_current_rz: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    mu0: float,
) -> dict[str, FloatArray]:
    """Reconstruct the four bound residual components on the full R-Z plane."""
    total_psi_rz = finite_plane(equilibrium.psi(), field="FreeGS total psi")
    plasma_psi_rz = finite_plane(equilibrium.plasma_psi, field="FreeGS plasma psi")
    reference_rhs = _operator._source_field(
        current=reference_current_rz,
        r_grid=r_grid,
        mu0=mu0,
    )
    exact_rhs = _operator._source_field(
        current=exact_current_rz,
        r_grid=r_grid,
        mu0=mu0,
    )
    freegs_lhs = _operator._freegs_fourth_order_lhs(
        freegs=freegs,
        equilibrium=equilibrium,
        plasma_psi_rz=plasma_psi_rz,
    )
    native_plasma_lhs = _operator._native_lhs(
        plasma_psi_rz,
        r_grid=r_grid,
        z_grid=z_grid,
    )
    native_total_lhs = _operator._native_lhs(
        total_psi_rz,
        r_grid=r_grid,
        z_grid=z_grid,
    )
    return {
        "freegs_fourth_order_baseline": zero_wall(
            freegs_lhs - reference_rhs,
            field="fourth-order baseline",
        ),
        "native_second_order_stencil": zero_wall(
            native_plasma_lhs - freegs_lhs,
            field="native second-order stencil",
        ),
        "coil_vacuum_discretisation": zero_wall(
            native_total_lhs - native_plasma_lhs,
            field="coil-vacuum discretisation",
        ),
        "exact_source_convention": zero_wall(
            reference_rhs - exact_rhs,
            field="exact-source convention",
        ),
    }


def verify_operator_binding(
    components: dict[str, FloatArray],
    *,
    reference_current_rz: FloatArray,
    operator_report: dict[str, Any],
) -> None:
    """Verify reconstructed support vectors against the bound operator payload."""
    support = np.asarray(
        np.abs(reference_current_rz[1:-1, 1:-1]) > 0.0,
        dtype=np.bool_,
    )
    report_names = {
        "freegs_fourth_order_baseline": "freegs_fourth_order_baseline",
        "native_second_order_stencil": "second_order_operator",
        "coil_vacuum_discretisation": "vacuum_discretisation",
        "exact_source_convention": "exact_source_convention",
    }
    for name, report_name in report_names.items():
        vector = np.asarray(components[name][1:-1, 1:-1][support], dtype=np.float64)
        expected = operator_report["interior_components"][report_name]["field_sha256"]
        if _array_sha256(vector) != expected:
            raise ValueError(f"{name} reconstruction disagrees with operator evidence")


def native_inverse(
    rhs_zr: FloatArray,
    *,
    r_grid: FloatArray,
    d_r: float,
    d_z: float,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
    x0_zr: FloatArray,
) -> FloatArray:
    """Apply the production native GS inverse with its frozen solver settings."""
    rhs = finite_plane(rhs_zr, field="native inverse rhs")
    x0 = finite_plane(x0_zr, field="native inverse initial state")
    if rhs.shape != x0.shape:
        raise ValueError("native inverse rhs and initial state must match")
    shape = rhs.shape
    r_jax = jnp.asarray(r_grid)

    def operator(value: jnp.ndarray) -> jnp.ndarray:
        return cast(
            jnp.ndarray,
            _predictive._gs_operator_flat(
                value,
                shape,
                r_jax,
                jnp.asarray(d_r),
                jnp.asarray(d_z),
            ),
        )

    solution, _ = _predictive._bicgstab(
        operator,
        jnp.asarray(rhs).reshape(-1),
        x0=jnp.asarray(x0).reshape(-1),
        tol=_predictive._BICGSTAB_TOL,
        maxiter=_predictive._BICGSTAB_MAXITER,
        M=preconditioner,
    )
    solution.block_until_ready()
    return finite_plane(
        np.asarray(solution.reshape(shape), dtype=np.float64),
        field="native inverse solution",
    )
