#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — same-case FreeGSNKE parity on three public machine descriptions
"""Run FreeGSNKE's public inverse problems and compare SCPN on the same cases.

The pinned FreeGSNKE 3.0.1 checkout ships three machine descriptions with a static inverse
problem each: the MAST-U-like regression case used by
:mod:`validation.benchmark_freegsnke_inverse`, and the SPARC-like and ITER-like problems of its
examples 07 and 08.  For every case this benchmark

1. runs the FreeGSNKE inverse solve and a forward refinement to a relative tolerance of 1e-9;
2. maps the resolved circuit currents onto SCPN's filament Green's functions and compares the
   vacuum flux inside the limiter;
3. samples the converged FreeGSNKE profile onto SCPN's ``pprime``/``FFprime`` source surface;
4. solves SCPN's self-consistent predictive equilibrium from FreeGSNKE's flux (warm start), with
   SCPN's own smooth topology and with FreeGSNKE's support frozen, and checks implicit gradients;
5. repeats the SCPN solve from the vacuum field (cold start) and records whether it reaches the
   same fixed point.

The comparison is bounded to these three synthetic public machine descriptions.  It makes no
facility, control, safety or speed claim and compares nothing with measured discharges.  A cold start that does not converge is
recorded as a robustness result of the SCPN solver; it does not fail the parity gates, which
concern the warm-started fixed point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# The base benchmark enables JAX float64 at import time, before any array is created here.
import jax.numpy as jnp

from scpn_fusion.core.jax_free_boundary_gs import vacuum_field_si
from validation import benchmark_freegsnke_inverse as base

DEFAULT_SOURCE = base.DEFAULT_SOURCE
REPORT_PATH = ROOT / "validation/reports/freegsnke_multimachine_parity.json"
MARKDOWN_PATH = ROOT / "validation/reports/freegsnke_multimachine_parity.md"
SCHEMA_VERSION = "scpn-fusion.freegsnke-multimachine-parity.v1"
BENCHMARK_ID = "freegsnke-public-machines-same-case-parity"

VACUUM_RELATIVE_MAX = 1.0e-10
COLD_START_N_ITER = 600
COLD_START_IP_RAMP = 100
COLD_START_PSI_N_RMSE_MATCH = 1.0e-6
REFINEMENT_TOLERANCE = 1.0e-9
CLAIM_BOUNDARY = {
    "control_admission": False,
    "experimental_validation": False,
    "facility_validation": False,
    "same_case_cross_solver_parity": True,
    "speed_comparison": False,
}

FloatArray = base.FloatArray
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — FreeGSNKE same-case parity on three public machines -->",
)


@dataclass(frozen=True)
class MachineCase:
    """One FreeGSNKE public static inverse problem, written as upstream states it."""

    name: str
    machine_dir: str
    file_prefix: str
    upstream_reference: str
    domain: tuple[float, float, float, float]
    grid: tuple[int, int]
    ip_a: float
    profile: dict[str, float]
    profile_class: str
    null_points: tuple[tuple[float, ...], tuple[float, ...]]
    isoflux: tuple[tuple[float, ...], tuple[float, ...]]
    xpoint_targets: tuple[tuple[float, float], ...]
    inverse_kwargs: dict[str, Any] = field(default_factory=dict)


ITER_ISOFLUX_R = (
    4.25455147, 4.1881875, 4.2625, 4.45683769, 4.78746942, 5.31835938, 5.91875, 6.44275166,
    6.92285156, 7.35441013, 7.73027344, 8.03046875, 8.19517497, 8.05673414, 7.75308013,
    7.37358451, 6.94355469, 6.47773438, 5.99121094, 5.49433594, 5.02, 4.79042969, 4.56269531,
    4.36038615,
)  # fmt: skip
ITER_ISOFLUX_Z = (
    0.0, 1.13554688, 2.2565134, 3.16757813, 3.80507812, 4.0499021, 3.93089003, 3.665625,
    3.31076604, 2.86875, 2.3199601, 1.62832118, 0.657421875, -0.35859375, -1.05585937,
    -1.59375, -2.03492727, -2.41356403, -2.75622016, -3.08699278, -3.23, -2.40548044,
    -1.55982142, -0.67734375,
)  # fmt: skip

CASES: dict[str, MachineCase] = {
    "mastu_like": MachineCase(
        name="mastu_like",
        machine_dir="MAST-U",
        file_prefix="MAST-U_like",
        upstream_reference="freegsnke/tests/test_inverse_static_solver.py",
        domain=(0.1, 2.0, -2.2, 2.2),
        grid=(65, 129),
        ip_a=6.0e5,
        profile={"paxis": 8.0e3, "fvac": 0.5, "alpha_m": 1.8, "alpha_n": 1.2},
        profile_class="ConstrainPaxisIp",
        null_points=((0.6, 0.6), (1.1, -1.1)),
        isoflux=(
            (0.6, 0.6, 0.34, 1.4, 1.0, 1.0, 0.8, 0.8),
            (1.1, -1.1, 0.0, 0.0, 2.0, -2.0, 1.62, -1.62),
        ),
        xpoint_targets=((0.6, 1.1), (0.6, -1.1)),
    ),
    "sparc_like": MachineCase(
        name="sparc_like",
        machine_dir="SPARC",
        file_prefix="SPARC",
        upstream_reference="examples/example07 - static_inverse_solve_SPARC.ipynb",
        domain=(1.1, 2.7, -1.8, 1.8),
        grid=(129, 129),
        ip_a=8.7e6,
        profile={"paxis": 5.0e4, "fvac": 0.5, "alpha_m": 1.8, "alpha_n": 1.2},
        profile_class="ConstrainPaxisIp",
        null_points=((1.55, 1.55), (1.15, -1.15)),
        isoflux=((1.55, 1.55, 2.4, 1.3, 1.7, 1.7), (1.15, -1.15, 0.0, 0.0, 1.5, -1.5)),
        xpoint_targets=((1.55, 1.15), (1.55, -1.15)),
        inverse_kwargs={
            "target_relative_tolerance": 1.0e-6,
            "target_relative_psit_update": 1.0e-3,
            "l2_reg": [1.0e-16] * 10 + [1.0e-5],
        },
    ),
    "iter_like": MachineCase(
        name="iter_like",
        machine_dir="ITER",
        file_prefix="ITER",
        upstream_reference="examples/example08 - static_inverse_solve_ITER.ipynb",
        domain=(3.2, 8.8, -5.0, 5.0),
        grid=(129, 129),
        ip_a=15.0e6,
        profile={"Beta0": 0.5978, "fvac": 0.5, "Raxis": 6.2, "alpha_m": 2.0, "alpha_n": 1.395},
        profile_class="Fiesta_Topeol",
        null_points=((5.02, 6.34), (-3.23, 0.66)),
        isoflux=(ITER_ISOFLUX_R, ITER_ISOFLUX_Z),
        xpoint_targets=((5.02, -3.23),),
        inverse_kwargs={
            "target_relative_tolerance": 1.0e-4,
            "target_relative_psit_update": 1.0e-3,
            "l2_reg": 1.0e-14,
            "Picard_handover": 1.0e-4,
            "full_jacobian_handover": [1.0e-3, 1.0e-2],
        },
    ),
}


def machine_paths(source: Path, case: MachineCase) -> dict[str, Path]:
    """Return the pinned machine-description files of ``case``."""
    folder = source / "machine_configs" / case.machine_dir
    return {
        kind: folder / f"{case.file_prefix}_{kind}.pickle"
        for kind in ("active_coils", "passive_coils", "limiter", "wall")
    }


def solve_freegsnke_case(source: Path, case: MachineCase) -> tuple[Any, Any, dict[str, Any]]:
    """Run one upstream inverse problem and refine it with a forward solve."""
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainPaxisIp, Fiesta_Topeol

    if case.name == "mastu_like":
        started = time.perf_counter()
        eq, profiles, solver, _limits = base.solve_inverse_case_with_profiles(source)
        return _refine(eq, profiles, solver, started)
    paths = machine_paths(source, case)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned machine files: {missing}")
    tokamak = build_machine.tokamak(
        active_coils_path=str(paths["active_coils"]),
        passive_coils_path=str(paths["passive_coils"]),
        limiter_path=str(paths["limiter"]),
        wall_path=str(paths["wall"]),
    )
    r_min, r_max, z_min, z_max = case.domain
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=r_min,
        Rmax=r_max,
        Zmin=z_min,
        Zmax=z_max,
        nx=case.grid[0],
        ny=case.grid[1],
    )
    profile_type: Callable[..., Any] = (
        ConstrainPaxisIp if case.profile_class == "ConstrainPaxisIp" else Fiesta_Topeol
    )
    profiles = profile_type(eq=eq, Ip=case.ip_a, **case.profile)
    constrain = Inverse_optimizer(
        null_points=[list(case.null_points[0]), list(case.null_points[1])],
        isoflux_set=np.array([[list(case.isoflux[0]), list(case.isoflux[1])]], dtype=np.float64),
    )
    solver = GSstaticsolver.NKGSsolver(eq)
    kwargs = dict(case.inverse_kwargs)
    if isinstance(kwargs.get("l2_reg"), list):
        kwargs["l2_reg"] = np.asarray(kwargs["l2_reg"], dtype=np.float64)
    started = time.perf_counter()
    solver.inverse_solve(eq=eq, profiles=profiles, constrain=constrain, verbose=False, **kwargs)
    return _refine(eq, profiles, solver, started)


def _refine(eq: Any, profiles: Any, solver: Any, started: float) -> tuple[Any, Any, dict[str, Any]]:
    """Refine an inverse solution with a forward solve at fixed currents, as upstream example 07."""
    inverse_change = float(solver.relative_change)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=REFINEMENT_TOLERANCE,
        verbose=False,
    )
    return (
        eq,
        profiles,
        {
            "forward_refinement_relative_change": float(solver.relative_change),
            "forward_refinement_tolerance": REFINEMENT_TOLERANCE,
            "inverse_relative_change": inverse_change,
            "seconds_orientation_only": time.perf_counter() - started,
        },
    )


def reference_xpoints(eq: Any, case: MachineCase) -> FloatArray:
    """Select the converged FreeGSNKE X-point nearest each upstream X-point target."""
    from freegs4e.critical import find_critical

    _opt, xpt = find_critical(eq.R, eq.Z, eq.psi(), eq.mask_inside_limiter.astype(bool), None)
    if len(xpt) == 0:
        raise ValueError(f"FreeGSNKE flux of {case.name} has no X-point")
    candidates = np.asarray(xpt, dtype=np.float64)[:, :2]
    selected = [
        candidates[int(np.argmin(np.linalg.norm(candidates - np.asarray(target), axis=1)))]
        for target in case.xpoint_targets
    ]
    return np.asarray(selected, dtype=np.float64)


def magnetic_axis_found(eq: Any, psi_rz: FloatArray) -> bool:
    """Return whether ``psi_rz`` (FreeGSNKE (R, Z) order) has an O-point inside the limiter.

    freegs4e's ``find_critical`` raises ``ValueError("No opoints found!")`` for a flux without an
    O-point; that case returns False. Any other error propagates.
    """
    from freegs4e.critical import find_critical

    try:
        opt, _xpt = find_critical(eq.R, eq.Z, psi_rz, eq.mask_inside_limiter.astype(bool), None)
    except ValueError as error:
        if "No opoints found" not in str(error):
            raise
        return False
    return len(opt) > 0


def cold_start(
    eq: Any,
    profiles: Any,
    active_currents: FloatArray,
    coil_r: FloatArray,
    coil_z: FloatArray,
    incidence: FloatArray,
    ip_a: float,
    warm_psi_n_rmse: float,
) -> dict[str, Any]:
    """Solve SCPN's predictive equilibrium from the vacuum field and compare it with FreeGSNKE."""
    from scpn_fusion.core.jax_free_boundary_predictive import (
        build_response_matrix,
        solve_predictive_equilibrium,
    )

    r_grid = jnp.asarray(eq.R[:, 0])
    z_grid = jnp.asarray(eq.Z[0, :])
    knots = np.linspace(0.0, 1.0, base.PROFILE_SAMPLE_COUNT, dtype=np.float64)
    response, wall_idx, source_idx = build_response_matrix(r_grid, z_grid)
    iterations: list[tuple[int, bool]] = []
    started = time.perf_counter()
    candidate = solve_predictive_equilibrium(
        jnp.asarray(incidence @ active_currents),
        jnp.asarray(profiles.pprime(knots)),
        jnp.asarray(profiles.ffprime(knots)),
        r_grid,
        z_grid,
        jnp.asarray(coil_r),
        jnp.asarray(coil_z),
        jnp.asarray(knots),
        ip_a,
        response,
        wall_idx,
        source_idx,
        n_iter=COLD_START_N_ITER,
        ip_ramp=COLD_START_IP_RAMP,
        tol=1.0e-9,
        decompose_coil_field=True,
        iteration_observer=lambda snapshot: iterations.append(
            (snapshot.iteration_index, snapshot.converged)
        ),
    )
    seconds = time.perf_counter() - started
    converged = bool(iterations and iterations[-1][1])
    return {
        "continuation": {"ip_ramp": COLD_START_IP_RAMP, "n_iter_max": COLD_START_N_ITER},
        "converged": converged,
        "iteration_count": len(iterations),
        "seconds_orientation_only": seconds,
        **classify_cold_start(
            eq, profiles, np.asarray(candidate, dtype=np.float64), converged, warm_psi_n_rmse
        ),
    }


def classify_cold_start(
    eq: Any,
    profiles: Any,
    psi: FloatArray,
    converged: bool,
    warm_psi_n_rmse: float,
) -> dict[str, Any]:
    """Classify a cold-start flux ``psi`` (SCPN (Z, R) order) against FreeGSNKE's equilibrium.

    The outcome is ``no_magnetic_axis_inside_limiter`` without an O-point,
    ``same_fixed_point_as_warm_start`` when the solve converged and its ψ_N RMSE matches the
    warm-started one within ``COLD_START_PSI_N_RMSE_MATCH``, and ``different_state`` otherwise.
    """
    from scpn_fusion.core.jax_o_point import smooth_axis_flux
    from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

    if not magnetic_axis_found(eq, psi.T):
        return {"outcome": "no_magnetic_axis_inside_limiter"}
    r_grid = jnp.asarray(eq.R[:, 0])
    z_grid = jnp.asarray(eq.Z[0, :])
    candidate = jnp.asarray(psi)
    reference = np.asarray(eq.psi(), dtype=np.float64).T
    limiter = np.asarray(eq.mask_inside_limiter, dtype=bool).T
    axis_flux = float(profiles.inputs[0])
    boundary_flux = float(profiles.inputs[1])
    candidate_axis = float(smooth_axis_flux(candidate))
    candidate_boundary = float(smooth_xpoint_flux(candidate, r_grid, z_grid))
    reference_n = (reference - axis_flux) / (boundary_flux - axis_flux)
    candidate_n = (psi - candidate_axis) / (candidate_boundary - candidate_axis)
    rmse = float(np.sqrt(np.mean(np.square(candidate_n[limiter] - reference_n[limiter]))))
    same_fixed_point = converged and abs(rmse - warm_psi_n_rmse) <= COLD_START_PSI_N_RMSE_MATCH
    return {
        "outcome": "same_fixed_point_as_warm_start" if same_fixed_point else "different_state",
        "psi_n_rmse_inside_limiter": rmse,
    }


def run_case(source: Path, case: MachineCase) -> dict[str, Any]:
    """Execute one machine case and return its comparison block."""
    eq, profiles, freegsnke_run = solve_freegsnke_case(source, case)
    paths = machine_paths(source, case)
    circuit_count = int(eq.tokamak.n_active_coils)
    currents = np.asarray(eq.tokamak.getCurrentsVec(), dtype=np.float64)
    active_currents = currents[:circuit_count]
    passive_currents = currents[circuit_count:]
    coil_r, coil_z, filament_i, incidence, circuits = base._active_filaments(
        source, active_currents, paths["active_coils"]
    )
    scpn_vacuum = np.asarray(
        vacuum_field_si(
            jnp.asarray(eq.R[:, 0]),
            jnp.asarray(eq.Z[0, :]),
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(filament_i),
        ),
        dtype=np.float64,
    ).T
    freegsnke_vacuum = np.asarray(eq.tokamak_psi, dtype=np.float64)
    limiter = np.asarray(eq.mask_inside_limiter, dtype=bool)
    vacuum_difference = scpn_vacuum[limiter] - freegsnke_vacuum[limiter]
    vacuum_scale = float(np.max(np.abs(freegsnke_vacuum[limiter])))
    vacuum_max_abs = float(np.max(np.abs(vacuum_difference)))
    profile_bridge = base._profile_source_bridge(eq, profiles)
    xpoints = reference_xpoints(eq, case)
    total_psi = base._total_psi_comparison(
        eq,
        profiles,
        active_currents,
        coil_r,
        coil_z,
        incidence,
        ip_target_a=case.ip_a,
        reference_xpoints_m=xpoints,
    )
    warm_rmse = float(total_psi["production_smooth"]["psi_n_rmse_inside_limiter"])
    cold = cold_start(
        eq, profiles, active_currents, coil_r, coil_z, incidence, case.ip_a, warm_rmse
    )
    checks = {
        "profile_source_bridge": bool(profile_bridge["passed"]),
        "same_case_total_psi": bool(total_psi["passed"]),
        "vacuum_parity_inside_limiter": vacuum_max_abs <= VACUUM_RELATIVE_MAX * vacuum_scale,
    }
    return {
        "checks": checks,
        "circuits": circuits,
        "cold_start": cold,
        "freegsnke_run": {
            **freegsnke_run,
            "passive_current_max_abs_a": (
                float(np.max(np.abs(passive_currents))) if passive_currents.size else 0.0
            ),
            "upstream_reference": case.upstream_reference,
        },
        "grid_shape_r_z": list(case.grid),
        "ip_target_a": case.ip_a,
        "machine_files": {
            kind: {"path": str(path.relative_to(source)), "sha256": base._file_sha256(path)}
            for kind, path in sorted(paths.items())
        },
        "passed": all(checks.values()),
        "profile": {"class": case.profile_class, **case.profile},
        "profile_source_bridge": profile_bridge,
        "reference_xpoints_m": xpoints.tolist(),
        "total_psi_comparison": total_psi,
        "vacuum_bridge": {
            "active_filament_count": int(coil_r.size),
            "freegsnke_vacuum_max_abs_wb": vacuum_scale,
            "max_abs_error_wb": vacuum_max_abs,
            "relative_threshold": VACUUM_RELATIVE_MAX,
            "rmse_wb": float(np.sqrt(np.mean(np.square(vacuum_difference)))),
        },
    }


def host_identity(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    """Name the executing host: the cold-start outcome and FreeGSNKE's stopping point vary by CPU.

    ``cpuinfo`` is the Linux processor table; without it, or without a ``model name`` line, the
    model falls back to :func:`platform.processor` and then to ``"unknown"``.
    """
    cpu_model = ""
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu_model": cpu_model or platform.processor() or "unknown",
        "logical_cpus": os.cpu_count() or 0,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "system": platform.system(),
    }


def build_report(
    source: Path = DEFAULT_SOURCE,
    case_names: tuple[str, ...] = tuple(CASES),
) -> dict[str, Any]:
    """Run every selected case and return the signed fail-closed report."""
    source = source.resolve()
    versions = base.validate_source(source)
    unknown = [name for name in case_names if name not in CASES]
    if unknown or not case_names:
        raise ValueError(f"unknown or empty case selection: {unknown or case_names}")
    cases = {name: run_case(source, CASES[name]) for name in case_names}
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "cases": cases,
        "claim_boundary": CLAIM_BOUNDARY,
        "generator": {
            "path": "validation/benchmark_freegsnke_multimachine.py",
            "sha256": base._file_sha256(Path(__file__).resolve()),
        },
        "provenance": {
            "machine_sources_per_upstream_readme": {
                "ITER": "FUSE.jl",
                "MAST-U": "UKAEA",
                "SPARC": "SPARCPublic",
            },
            "host": host_identity(),
            "source_commit": versions["commit"],
            "source_url": "https://github.com/FusionComputingLab/freegsnke",
            "versions": versions,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(case["passed"] for case in cases.values()) else "fail",
    }
    report["payload_sha256"] = base._payload_sha256(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the per-case table and the claim boundary."""
    lines = [
        *MARKDOWN_HEADER,
        "# FreeGSNKE same-case parity on three public machine descriptions",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "Synthetic public machine descriptions only. No facility, control, safety or speed "
        "claim; no comparison with measured discharges.",
        "",
        "| Case | Grid | Vacuum max (Wb) | Source rel. L2 | Warm ψ_N RMSE | Frozen ψ_N RMSE "
        "| Cold start |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, case in report["cases"].items():
        total = case["total_psi_comparison"]
        cold = case["cold_start"]
        lines.append(
            f"| {name} | {case['grid_shape_r_z'][0]}×{case['grid_shape_r_z'][1]} "
            f"| {case['vacuum_bridge']['max_abs_error_wb']:.3g} "
            f"| {case['profile_source_bridge']['source_parity']['relative_l2_error']:.3g} "
            f"| {total['production_smooth']['psi_n_rmse_inside_limiter']:.4g} "
            f"| {total['frozen_topology_control']['psi_n_rmse_inside_limiter']:.4g} "
            f"| {cold['outcome']} ({cold['iteration_count']} iterations) |"
        )
    lines += ["", f"Payload SHA-256: `{report['payload_sha256']}`", ""]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the source checkout, output paths, case selection and strict flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 under ``--strict`` when a gate fails."""
    args = _parse_args(argv)
    report = build_report(args.source, tuple(args.case) if args.case else tuple(CASES))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
