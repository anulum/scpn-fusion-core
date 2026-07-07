#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Grad-Shafranov Solov'ev analytic-equilibrium validation
"""Validate the production Grad-Shafranov solver stack against Solov'ev.

Ported from SCPN-CONTROL's exact-equilibrium suite (master-plan F-1): the
Solov'ev family admits an exact polynomial solution of the Grad-Shafranov
equation, making it the canonical analytic benchmark for the equilibrium
discretisation — credibility evidence independent of any bundled GEQDSK file.

Manufactured exact solution (constant ``p'`` and ``FF'`` Solov'ev branch):

    ψ(R, Z) = c₁ R⁴/8 + c₂ Z²        ⇒        Δ*ψ = c₁ R² + 2 c₂.

Three **production** code paths are validated against this exact field:

1. **Discrete operator.** ``FusionKernel._apply_gs_operator`` (the Newton/GMRES
   matvec) is applied to the exact ψ sampled on grids of increasing resolution.
   Its truncation error against the analytic ``Δ*ψ`` must vanish at second
   order in the mesh spacing ``h``.
2. **SOR equilibrium solver.** The production ``_sor_step`` smoother is
   iterated to a fixed residual with Dirichlet data taken from the exact ψ.
   The reconstructed flux must converge to the exact field at second order.
3. **Dispatched multigrid full solve.** The canonical ``multigrid_solve``
   dispatch kernel is driven on the same problem twice — once pinned to the
   NumPy floor and once through the fastest available tier — and both
   reconstructions must reach the analytic tolerance. The resolved tier name
   is recorded in the evidence payload.

References:
  Solov'ev L. S. (1968) "The theory of hydromagnetic stability of toroidal
  plasma configurations", *Sov. Phys. JETP* 26, 400.
  Cerfon A. J., Freidberg J. P. (2010) "One size fits all analytic solutions
  to the Grad-Shafranov equation", *Phys. Plasmas* 17, 032502.
  Jardin S. (2010) *Computational Methods in Plasma Physics*, CRC Press, Ch. 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel import FusionKernel

FloatArray = NDArray[np.float64]

GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION = "scpn-fusion-core.grad-shafranov-solovev-validation.v1"

DEFAULT_RESOLUTIONS: tuple[int, ...] = (33, 49, 65, 97)


@dataclass(frozen=True)
class SolovevGeometry:
    """Rectangular ``(R, Z)`` domain and Solov'ev coefficients for the benchmark.

    The exact field is ``ψ = c1 · R⁴/8 + c2 · Z²`` with analytic source
    ``Δ*ψ = c1 · R² + 2 c2``. ``r0`` and ``a`` are the major and minor radii of
    the embedded tokamak; the box is widened by ``half_width_factor`` so the
    plasma sits strictly inside the Dirichlet boundary.
    """

    r0: float
    a: float
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("c1", self.c1)
        _positive_float("c2", self.c2)
        if not self.r_min > 0.0:
            raise ValueError("r_min must be positive (toroidal 1/R term is singular at R=0)")
        if not self.r_max > self.r_min:
            raise ValueError("r_max must exceed r_min")
        if not self.z_max > self.z_min:
            raise ValueError("z_max must exceed z_min")

    @classmethod
    def from_aspect(
        cls,
        *,
        r0: float = 1.7,
        a: float = 0.5,
        half_width_factor: float = 1.5,
        c1: float = 1.0,
        c2: float = 0.5,
    ) -> "SolovevGeometry":
        """Build a centred box of half-width ``half_width_factor · a`` about ``r0``.

        Parameters
        ----------
        r0 : float
            Major radius of the embedded tokamak.
        a : float
            Minor radius of the embedded tokamak.
        half_width_factor : float
            Box half-width in units of ``a``.
        c1 : float
            Solov'ev ``R⁴`` coefficient.
        c2 : float
            Solov'ev ``Z²`` coefficient.

        Returns
        -------
        SolovevGeometry
            Validated benchmark geometry.
        """
        _positive_float("half_width_factor", half_width_factor)
        half = half_width_factor * _positive_float("a", a)
        return cls(
            r0=r0,
            a=a,
            r_min=r0 - half,
            r_max=r0 + half,
            z_min=-half,
            z_max=half,
            c1=c1,
            c2=c2,
        )


def solovev_psi(rr: FloatArray, zz: FloatArray, geometry: SolovevGeometry) -> FloatArray:
    """Exact Solov'ev flux ``ψ = c1 R⁴/8 + c2 Z²`` on the supplied mesh.

    Parameters
    ----------
    rr : FloatArray
        Major-radius mesh.
    zz : FloatArray
        Vertical mesh.
    geometry : SolovevGeometry
        Benchmark geometry carrying the Solov'ev coefficients.

    Returns
    -------
    FloatArray
        Exact flux field sampled on the mesh.
    """
    return np.asarray(geometry.c1 * rr**4 / 8.0 + geometry.c2 * zz**2, dtype=np.float64)


def solovev_source(rr: FloatArray, geometry: SolovevGeometry) -> FloatArray:
    """Analytic Grad-Shafranov source ``Δ*ψ = c1 R² + 2 c2`` for the exact field.

    Parameters
    ----------
    rr : FloatArray
        Major-radius mesh.
    geometry : SolovevGeometry
        Benchmark geometry carrying the Solov'ev coefficients.

    Returns
    -------
    FloatArray
        Analytic source field sampled on the mesh.
    """
    return np.asarray(geometry.c1 * rr**2 + 2.0 * geometry.c2, dtype=np.float64)


def _build_kernel(geometry: SolovevGeometry, n: int) -> FusionKernel:
    """Instantiate the production ``FusionKernel`` on an ``n × n`` Solov'ev mesh."""
    _grid_resolution("n", n)
    config: dict[str, Any] = {
        "reactor_name": "solovev-grad-shafranov-validation",
        "grid_resolution": [n, n],
        "dimensions": {
            "R_min": geometry.r_min,
            "R_max": geometry.r_max,
            "Z_min": geometry.z_min,
            "Z_max": geometry.z_max,
        },
        "physics": {
            "plasma_current_target": 1.0,
            "vacuum_permeability": 1.0,
            "R0": geometry.r0,
            "a": geometry.a,
            "B0": 2.0,
        },
        "coils": [{"name": "PF1", "r": geometry.r_min, "z": geometry.z_max, "current": 1.0}],
        "solver": {
            "max_iterations": 1,
            "convergence_threshold": 1e-10,
            "relaxation_factor": 0.1,
            "solver_method": "sor",
            "sor_omega": 1.6,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "solovev_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return FusionKernel(str(config_path))


def _enforce_dirichlet(psi: FloatArray, psi_exact: FloatArray) -> None:
    """Overwrite the four boundary edges of ``psi`` with the exact field in place."""
    psi[0, :] = psi_exact[0, :]
    psi[-1, :] = psi_exact[-1, :]
    psi[:, 0] = psi_exact[:, 0]
    psi[:, -1] = psi_exact[:, -1]


def _interior_nrmse(numerical: FloatArray, exact: FloatArray) -> float:
    """Interior-node NRMSE normalised by the exact field range."""
    num_int = numerical[1:-1, 1:-1]
    exact_int = exact[1:-1, 1:-1]
    span = float(exact.max() - exact.min())
    rmse = float(np.sqrt(np.mean((num_int - exact_int) ** 2)))
    return rmse / max(span, 1e-15)


def operator_truncation_error(geometry: SolovevGeometry, n: int) -> float:
    """Max interior truncation error of the production discrete ``Δ*`` operator.

    The exact ψ is fed to ``FusionKernel._apply_gs_operator`` and compared to
    the analytic ``Δ*ψ``; the returned value is the maximum absolute residual
    over interior nodes, which must decay as ``O(h²)``.

    Parameters
    ----------
    geometry : SolovevGeometry
        Benchmark geometry.
    n : int
        Grid resolution per axis.

    Returns
    -------
    float
        Maximum interior truncation error.
    """
    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)
    kernel.Psi = psi_exact.copy()
    applied = np.asarray(kernel._apply_gs_operator(psi_exact), dtype=np.float64)
    error = np.abs(applied[1:-1, 1:-1] - source[1:-1, 1:-1])
    return float(error.max())


@dataclass(frozen=True)
class SorReconstruction:
    """Outcome of an SOR reconstruction of the Solov'ev equilibrium."""

    nrmse: float
    iterations: int
    converged: bool
    residual_inf: float


def sor_reconstruction(
    geometry: SolovevGeometry,
    n: int,
    *,
    omega: float = 1.6,
    residual_tol: float = 1e-9,
    max_sweeps: int = 40000,
    check_every: int = 50,
) -> SorReconstruction:
    """Iterate the production ``_sor_step`` smoother to a fixed residual.

    Dirichlet data are taken from the exact Solov'ev field. Iteration stops
    when the infinity-norm of the discrete Grad-Shafranov residual falls below
    ``residual_tol`` (decoupling iteration error from discretisation error) or
    when ``max_sweeps`` is reached.

    Parameters
    ----------
    geometry : SolovevGeometry
        Benchmark geometry.
    n : int
        Grid resolution per axis.
    omega : float
        SOR over-relaxation factor.
    residual_tol : float
        Infinity-norm residual stopping tolerance.
    max_sweeps : int
        Sweep budget.
    check_every : int
        Residual evaluation cadence in sweeps.

    Returns
    -------
    SorReconstruction
        Reconstruction error and convergence record.
    """
    _positive_float("omega", omega)
    _positive_float("residual_tol", residual_tol)
    _positive_int("max_sweeps", max_sweeps)
    _positive_int("check_every", check_every)

    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)

    psi = np.zeros_like(psi_exact)
    _enforce_dirichlet(psi, psi_exact)
    kernel.Psi = psi.copy()

    iterations = 0
    residual_inf = math.inf
    converged = False
    for sweep in range(max_sweeps):
        updated = np.asarray(kernel._sor_step(kernel.Psi, source, omega=omega), dtype=np.float64)
        _enforce_dirichlet(updated, psi_exact)
        kernel.Psi = updated
        iterations = sweep + 1
        if sweep % check_every == 0:
            residual = np.asarray(kernel._apply_gs_operator(kernel.Psi), dtype=np.float64) - source
            residual_inf = float(np.max(np.abs(residual[1:-1, 1:-1])))
            if residual_inf < residual_tol:
                converged = True
                break

    nrmse = _interior_nrmse(kernel.Psi, psi_exact)
    return SorReconstruction(
        nrmse=nrmse, iterations=iterations, converged=converged, residual_inf=residual_inf
    )


@dataclass(frozen=True)
class DispatchedMultigridRecord:
    """Outcome of a dispatched ``multigrid_solve`` reconstruction."""

    tier: str
    resolution: int
    nrmse: float
    residual: float
    cycles: int
    converged: bool
    meets_analytic_tolerance: bool


def dispatched_multigrid_reconstruction(
    geometry: SolovevGeometry,
    n: int,
    *,
    tier: str,
    analytic_tolerance: float,
    tol: float = 1e-9,
    max_cycles: int = 200,
) -> DispatchedMultigridRecord:
    """Reconstruct the Solov'ev field through the ``multigrid_solve`` dispatcher.

    Parameters
    ----------
    geometry : SolovevGeometry
        Benchmark geometry.
    n : int
        Grid resolution per axis.
    tier : str
        ``"numpy"`` pins the NumPy floor; ``"fastest"`` resolves the fastest
        registered tier via the dispatcher.
    analytic_tolerance : float
        NRMSE gate against the exact field.
    tol : float
        GS* residual convergence tolerance handed to the solver.
    max_cycles : int
        V-cycle budget.

    Returns
    -------
    DispatchedMultigridRecord
        Reconstruction record with the resolved tier name.
    """
    from scpn_fusion.core import _multi_compat as multi

    if tier not in ("numpy", "fastest"):
        raise ValueError("tier must be 'numpy' or 'fastest'")
    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)
    psi_bc = np.zeros_like(psi_exact)
    _enforce_dirichlet(psi_bc, psi_exact)

    if tier == "numpy":
        impl = multi._numpy_multigrid_solve
        resolved = "numpy"
    else:
        impl = multi.dispatch("multigrid_solve")
        resolved = multi.dispatch_tier("multigrid_solve")

    psi, residual, cycles, converged = impl(
        source,
        psi_bc,
        geometry.r_min,
        geometry.r_max,
        geometry.z_min,
        geometry.z_max,
        n,
        n,
        tol=tol,
        max_cycles=max_cycles,
    )
    nrmse = _interior_nrmse(np.asarray(psi, dtype=np.float64), psi_exact)
    return DispatchedMultigridRecord(
        tier=resolved,
        resolution=n,
        nrmse=nrmse,
        residual=float(residual),
        cycles=int(cycles),
        converged=bool(converged),
        meets_analytic_tolerance=nrmse < analytic_tolerance,
    )


@dataclass(frozen=True)
class ConvergenceRecord:
    """Per-resolution error sample for a convergence study."""

    resolution: int
    mesh_spacing: float
    error: float


def _log_log_slope(records: Sequence[ConvergenceRecord]) -> float:
    """Least-squares order of accuracy from ``log(error)`` versus ``log(h)``."""
    if len(records) < 2:
        raise ValueError("at least two resolutions are required to estimate an order")
    log_h = np.log(np.array([record.mesh_spacing for record in records], dtype=np.float64))
    log_e = np.log(np.array([record.error for record in records], dtype=np.float64))
    slope, _ = np.polyfit(log_h, log_e, 1)
    return float(slope)


@dataclass(frozen=True)
class GradShafranovValidationResult:
    """Outcome of the Solov'ev Grad-Shafranov solver validation."""

    geometry: SolovevGeometry
    resolutions: tuple[int, ...]
    operator_records: tuple[ConvergenceRecord, ...]
    operator_order: float
    operator_error_finest: float
    operator_passed: bool
    reconstruction_records: tuple[ConvergenceRecord, ...]
    reconstruction_details: tuple[SorReconstruction, ...]
    reconstruction_order: float
    reconstruction_nrmse_finest: float
    reconstruction_passed: bool
    min_order: float
    operator_error_gate: float
    reconstruction_nrmse_gate: float
    multigrid_numpy_record: DispatchedMultigridRecord
    multigrid_fastest_record: DispatchedMultigridRecord
    multigrid_passed: bool
    passed: bool


def validate_grad_shafranov(
    *,
    geometry: SolovevGeometry | None = None,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    omega: float = 1.6,
    residual_tol: float = 1e-9,
    max_sweeps: int = 40000,
    min_order: float = 1.8,
    operator_error_gate: float = 5e-4,
    reconstruction_nrmse_gate: float = 1e-4,
) -> GradShafranovValidationResult:
    """Validate the production Grad-Shafranov stack on the Solov'ev equilibrium.

    Three production code paths are gated:

    1. **Operator.** The truncation error of ``_apply_gs_operator`` against the
       analytic ``Δ*ψ`` decays at order ``≥ min_order`` and the finest-grid
       error is below ``operator_error_gate``.
    2. **SOR reconstruction.** The ``_sor_step`` solver reconstructs the exact
       ψ at order ``≥ min_order`` with finest-grid NRMSE below
       ``reconstruction_nrmse_gate``.
    3. **Dispatched multigrid.** The canonical ``multigrid_solve`` kernel
       reconstructs the exact ψ on the finest grid below
       ``reconstruction_nrmse_gate`` on both the pinned NumPy floor and the
       resolved fastest tier.

    Parameters
    ----------
    geometry : SolovevGeometry, optional
        Benchmark geometry; defaults to ``SolovevGeometry.from_aspect()``.
    resolutions : Sequence[int]
        Grid resolutions for the convergence studies (at least two distinct).
    omega : float
        SOR over-relaxation factor.
    residual_tol : float
        SOR residual stopping tolerance.
    max_sweeps : int
        SOR sweep budget per resolution.
    min_order : float
        Minimum acceptable order of accuracy.
    operator_error_gate : float
        Finest-grid operator truncation-error gate.
    reconstruction_nrmse_gate : float
        Finest-grid reconstruction NRMSE gate.

    Returns
    -------
    GradShafranovValidationResult
        Full validation outcome with per-path pass flags.
    """
    geometry = geometry or SolovevGeometry.from_aspect()
    ordered = tuple(sorted({_grid_resolution("resolution", n) for n in resolutions}))
    if len(ordered) < 2:
        raise ValueError("at least two distinct resolutions are required")

    operator_records: list[ConvergenceRecord] = []
    reconstruction_records: list[ConvergenceRecord] = []
    reconstruction_details: list[SorReconstruction] = []
    for n in ordered:
        h = (geometry.r_max - geometry.r_min) / (n - 1)
        operator_records.append(
            ConvergenceRecord(
                resolution=n, mesh_spacing=h, error=operator_truncation_error(geometry, n)
            )
        )
        reconstruction = sor_reconstruction(
            geometry, n, omega=omega, residual_tol=residual_tol, max_sweeps=max_sweeps
        )
        reconstruction_details.append(reconstruction)
        reconstruction_records.append(
            ConvergenceRecord(resolution=n, mesh_spacing=h, error=reconstruction.nrmse)
        )

    operator_order = _log_log_slope(operator_records)
    operator_error_finest = operator_records[-1].error
    operator_passed = operator_order >= min_order and operator_error_finest < operator_error_gate

    reconstruction_order = _log_log_slope(reconstruction_records)
    reconstruction_nrmse_finest = reconstruction_records[-1].error
    reconstruction_converged = all(detail.converged for detail in reconstruction_details)
    reconstruction_passed = (
        reconstruction_order >= min_order
        and reconstruction_nrmse_finest < reconstruction_nrmse_gate
        and reconstruction_converged
    )

    finest = ordered[-1]
    multigrid_numpy_record = dispatched_multigrid_reconstruction(
        geometry, finest, tier="numpy", analytic_tolerance=reconstruction_nrmse_gate
    )
    multigrid_fastest_record = dispatched_multigrid_reconstruction(
        geometry, finest, tier="fastest", analytic_tolerance=reconstruction_nrmse_gate
    )
    multigrid_passed = (
        multigrid_numpy_record.converged
        and multigrid_numpy_record.meets_analytic_tolerance
        and multigrid_fastest_record.converged
        and multigrid_fastest_record.meets_analytic_tolerance
    )

    return GradShafranovValidationResult(
        geometry=geometry,
        resolutions=ordered,
        operator_records=tuple(operator_records),
        operator_order=operator_order,
        operator_error_finest=operator_error_finest,
        operator_passed=operator_passed,
        reconstruction_records=tuple(reconstruction_records),
        reconstruction_details=tuple(reconstruction_details),
        reconstruction_order=reconstruction_order,
        reconstruction_nrmse_finest=reconstruction_nrmse_finest,
        reconstruction_passed=reconstruction_passed,
        min_order=min_order,
        operator_error_gate=operator_error_gate,
        reconstruction_nrmse_gate=reconstruction_nrmse_gate,
        multigrid_numpy_record=multigrid_numpy_record,
        multigrid_fastest_record=multigrid_fastest_record,
        multigrid_passed=multigrid_passed,
        passed=operator_passed and reconstruction_passed and multigrid_passed,
    )


def _multigrid_record_payload(record: DispatchedMultigridRecord) -> dict[str, Any]:
    """Serialise a dispatched-multigrid record for the evidence payload."""
    return {
        "tier": record.tier,
        "resolution": record.resolution,
        "nrmse": record.nrmse,
        "residual": record.residual,
        "cycles": record.cycles,
        "converged": record.converged,
        "meets_analytic_tolerance": record.meets_analytic_tolerance,
    }


def build_evidence(result: GradShafranovValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload.

    Parameters
    ----------
    result : GradShafranovValidationResult
        Validation outcome to serialise.
    target_id : str
        Identifier of the validated target (recorded verbatim).

    Returns
    -------
    dict
        Sealed evidence payload with a canonical-JSON SHA-256 digest.
    """
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "geometry": {
            "r0": result.geometry.r0,
            "a": result.geometry.a,
            "r_min": result.geometry.r_min,
            "r_max": result.geometry.r_max,
            "z_min": result.geometry.z_min,
            "z_max": result.geometry.z_max,
            "c1": result.geometry.c1,
            "c2": result.geometry.c2,
        },
        "resolutions": list(result.resolutions),
        "min_order": result.min_order,
        "operator_error_gate": result.operator_error_gate,
        "reconstruction_nrmse_gate": result.reconstruction_nrmse_gate,
        "operator_records": [
            {"resolution": rec.resolution, "mesh_spacing": rec.mesh_spacing, "error": rec.error}
            for rec in result.operator_records
        ],
        "operator_order": result.operator_order,
        "operator_error_finest": result.operator_error_finest,
        "operator_passed": result.operator_passed,
        "reconstruction_records": [
            {
                "resolution": rec.resolution,
                "mesh_spacing": rec.mesh_spacing,
                "nrmse": rec.error,
                "iterations": detail.iterations,
                "converged": detail.converged,
                "residual_inf": detail.residual_inf,
            }
            for rec, detail in zip(result.reconstruction_records, result.reconstruction_details)
        ],
        "reconstruction_order": result.reconstruction_order,
        "reconstruction_nrmse_finest": result.reconstruction_nrmse_finest,
        "reconstruction_passed": result.reconstruction_passed,
        "multigrid_numpy_record": _multigrid_record_payload(result.multigrid_numpy_record),
        "multigrid_fastest_record": _multigrid_record_payload(result.multigrid_fastest_record),
        "multigrid_passed": result.multigrid_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing.

    Parameters
    ----------
    payload : Mapping
        Evidence payload produced by :func:`build_evidence`.

    Returns
    -------
    bool
        ``True`` when the payload passes and its seal verifies.

    Raises
    ------
    ValueError
        If the schema version is foreign or the SHA-256 seal does not match.
    """
    if payload.get("schema_version") != GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION:
        raise ValueError("unsupported grad-shafranov solovev evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _grid_resolution(name: str, value: object) -> int:
    result = _positive_int(name, value)
    if result < 3:
        raise ValueError(f"{name} must be at least 3 (need interior nodes)")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Grad-Shafranov Solov'ev Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Discrete operator (`FusionKernel._apply_gs_operator`)",
        "",
        f"- Order of accuracy: {evidence['operator_order']:.3f} (gate ≥ {evidence['min_order']})",
        f"- Finest-grid max truncation error: {evidence['operator_error_finest']:.3e} "
        f"(gate < {evidence['operator_error_gate']:.1e})",
        f"- Passed: {evidence['operator_passed']}",
        "",
        "| resolution | h | max |Δ* error| |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['error']:.4e} |"
        for rec in evidence["operator_records"]
    ]
    lines += [
        "",
        "## SOR reconstruction (`FusionKernel._sor_step`)",
        "",
        f"- Order of accuracy: {evidence['reconstruction_order']:.3f} "
        f"(gate ≥ {evidence['min_order']})",
        f"- Finest-grid NRMSE: {evidence['reconstruction_nrmse_finest']:.3e} "
        f"(gate < {evidence['reconstruction_nrmse_gate']:.1e})",
        f"- Passed: {evidence['reconstruction_passed']}",
        "",
        "| resolution | h | NRMSE | sweeps | converged |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['nrmse']:.4e} | "
        f"{rec['iterations']} | {rec['converged']} |"
        for rec in evidence["reconstruction_records"]
    ]
    lines += [
        "",
        "## Dispatched multigrid full solve (`multigrid_solve` kernel)",
        "",
        "| tier | resolution | NRMSE | residual | cycles | converged | meets tolerance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in ("multigrid_numpy_record", "multigrid_fastest_record"):
        rec = evidence[key]
        lines.append(
            f"| {rec['tier']} | {rec['resolution']} | {rec['nrmse']:.4e} | "
            f"{rec['residual']:.4e} | {rec['cycles']} | {rec['converged']} | "
            f"{rec['meets_analytic_tolerance']} |"
        )
    lines += [
        "",
        f"- Multigrid gate passed: {evidence['multigrid_passed']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Solov'ev validation and emit schema-versioned evidence.

    Parameters
    ----------
    argv : Sequence[str], optional
        CLI argument list; defaults to process arguments.

    Returns
    -------
    int
        ``0`` when all gates pass, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fusion-kernel Grad-Shafranov solver stack against the "
            "Solov'ev analytic equilibrium"
        )
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--target-id", type=str, default="local-grad-shafranov-solovev")
    parser.add_argument(
        "--operator-error-gate",
        type=float,
        default=5e-4,
        help="finest-grid operator truncation-error gate (default tuned to 97² grids)",
    )
    parser.add_argument(
        "--reconstruction-nrmse-gate",
        type=float,
        default=1e-4,
        help="finest-grid reconstruction NRMSE gate (default tuned to 97² grids)",
    )
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="write sealed JSON evidence and a Markdown summary to this path",
    )
    args = parser.parse_args(argv)

    result = validate_grad_shafranov(
        resolutions=args.resolutions,
        operator_error_gate=args.operator_error_gate,
        reconstruction_nrmse_gate=args.reconstruction_nrmse_gate,
    )
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print(
            f"Grad-Shafranov Solov'ev validation "
            f"(R0={result.geometry.r0}, a={result.geometry.a}, "
            f"box=[{result.geometry.r_min:.3f},{result.geometry.r_max:.3f}]×"
            f"[{result.geometry.z_min:.3f},{result.geometry.z_max:.3f}])"
        )
        print(
            f"  operator:           order={result.operator_order:.3f} "
            f"finest_err={result.operator_error_finest:.3e} "
            f"{'ok' if result.operator_passed else 'FAIL'}"
        )
        print(
            f"  SOR solver:         order={result.reconstruction_order:.3f} "
            f"finest_nrmse={result.reconstruction_nrmse_finest:.3e} "
            f"{'ok' if result.reconstruction_passed else 'FAIL'}"
        )
        for label, record in (
            ("multigrid (numpy)", result.multigrid_numpy_record),
            (
                f"multigrid ({result.multigrid_fastest_record.tier})",
                result.multigrid_fastest_record,
            ),
        ):
            print(
                f"  {label}: nrmse={record.nrmse:.3e} cycles={record.cycles} "
                f"{'ok' if record.meets_analytic_tolerance and record.converged else 'FAIL'}"
            )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
