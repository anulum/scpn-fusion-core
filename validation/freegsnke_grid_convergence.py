#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — grid convergence of the same-case FreeGSNKE comparison
"""Repeat the MAST-U-like same-case comparison with FreeGSNKE on finer grids.

:mod:`validation.benchmark_freegsnke_multimachine` compares SCPN's predictive free-boundary
solver with FreeGSNKE 3.0.1 on one grid per case. This study asks how the differences behave
when both codes solve the MAST-U-like case on a finer grid. For every grid it

1. builds the MAST-U-like inverse problem exactly as
   :func:`validation.benchmark_freegsnke_inverse.build_inverse_case` does (machine files,
   domain, profile, isoflux and null points, current limits, Tikhonov weights, the solenoid
   held at 5 kA), with only the grid changed, solves it with FreeGSNKE and refines it with a
   forward solve;
2. compares the vacuum flux, the profile source and SCPN's warm-started and frozen-support
   equilibria with FreeGSNKE's, through the same functions as the multi-machine benchmark.

The cold start is not repeated. On the base grid the study first checks that its own build
reproduces the base benchmark's FreeGSNKE currents exactly. The gates concern completeness
only; the differences are reported, not gated. The case is a synthetic machine description:
there is no facility, control, safety or speed claim and no comparison with measured
discharges.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp
from scpn_fusion.core.jax_free_boundary_gs import vacuum_field_si
from validation import benchmark_freegsnke_inverse as base
from validation import benchmark_freegsnke_multimachine as machines

REPORT_PATH = ROOT / "validation/reports/freegsnke_grid_convergence.json"
MARKDOWN_PATH = ROOT / "validation/reports/freegsnke_grid_convergence.md"
SCHEMA_VERSION = "scpn-fusion.freegsnke-grid-convergence.v1"
BENCHMARK_ID = "freegsnke-mastu-like-grid-convergence"
GRIDS: tuple[tuple[int, int], ...] = ((65, 129), (129, 257))
CLAIM_BOUNDARY = {
    "control_admission": False,
    "experimental_validation": False,
    "facility_validation": False,
    "same_case_grid_study": True,
    "speed_comparison": False,
}
SOURCE_FILES = (
    "validation/freegsnke_grid_convergence.py",
    "validation/benchmark_freegsnke_inverse.py",
    "validation/benchmark_freegsnke_multimachine.py",
    "src/scpn_fusion/core/jax_free_boundary_predictive.py",
)
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — grid convergence of the same-case FreeGSNKE comparison -->",
)


def build_mastu_case(source: Path, grid: tuple[int, int]) -> tuple[Any, Any, Any]:
    """Build the MAST-U-like inverse problem of the base benchmark on ``grid`` (nR, nZ)."""
    paths = base._case_paths(source)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned case artifacts: {missing}")
    case = machines.CASES["mastu_like"]
    tokamak = build_machine.tokamak(
        active_coils_path=str(paths["active_coils"]),
        passive_coils_path=str(paths["passive_coils"]),
        limiter_path=str(paths["limiter"]),
        wall_path=str(paths["wall"]),
    )
    r_min, r_max, z_min, z_max = case.domain
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=r_min, Rmax=r_max, Zmin=z_min, Zmax=z_max, nx=grid[0], ny=grid[1]
    )
    profiles = ConstrainPaxisIp(eq=eq, Ip=case.ip_a, **case.profile)
    constrain = Inverse_optimizer(
        null_points=[list(case.null_points[0]), list(case.null_points[1])],
        isoflux_set=np.array([[list(case.isoflux[0]), list(case.isoflux[1])]], dtype=np.float64),
        coil_current_limits=[list(row) for row in base.MASTU_COIL_LIMITS],
    )
    constrain.mu_coils = 1.0e5
    eq.tokamak.set_coil_current("Solenoid", 5000.0)
    eq.tokamak["Solenoid"].control = False
    return eq, profiles, constrain


def solve_mastu_case(source: Path, grid: tuple[int, int]) -> tuple[Any, Any, dict[str, Any]]:
    """Solve the MAST-U-like case on ``grid`` with the base settings and refine it."""
    eq, profiles, constrain = build_mastu_case(source, grid)
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    started = time.perf_counter()
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=constrain,
        target_relative_tolerance=1.0e-6,
        target_relative_psit_update=1.0e-3,
        verbose=False,
        l2_reg=np.array(base.MASTU_L2_REG, dtype=np.float64),
    )
    return machines._refine(eq, profiles, solver, started)


def active_currents(eq: Any) -> np.ndarray:
    """Return a copy of the active circuit currents of a FreeGSNKE equilibrium (A)."""
    count = int(eq.tokamak.n_active_coils)
    return np.array(eq.tokamak.getCurrentsVec(), dtype=np.float64)[:count].copy()


def grid_record(source: Path, grid: tuple[int, int]) -> dict[str, Any]:
    """Run both codes on one grid and return the comparison block."""
    eq, profiles, freegsnke_run = solve_mastu_case(source, grid)
    currents = active_currents(eq)
    paths = base._case_paths(source)
    coil_r, coil_z, filament_i, incidence, _circuits = base._active_filaments(
        source, currents, paths["active_coils"]
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
    limiter = np.asarray(eq.mask_inside_limiter, dtype=bool)
    freegsnke_vacuum = np.asarray(eq.tokamak_psi, dtype=np.float64)
    vacuum_max = float(np.max(np.abs(scpn_vacuum[limiter] - freegsnke_vacuum[limiter])))
    bridge = base._profile_source_bridge(eq, profiles)
    case = machines.CASES["mastu_like"]
    total = base._total_psi_comparison(
        eq,
        profiles,
        currents,
        coil_r,
        coil_z,
        incidence,
        ip_target_a=case.ip_a,
        reference_xpoints_m=machines.reference_xpoints(eq, case),
    )
    smooth, frozen = total["production_smooth"], total["frozen_topology_control"]
    return {
        "grid_shape_r_z": list(grid),
        "freegsnke_run": freegsnke_run,
        "freegsnke_currents_a": currents.tolist(),
        "vacuum_max_abs_error_wb": vacuum_max,
        "source_relative_l2": float(bridge["source_parity"]["relative_l2_error"]),
        "warm": {
            "converged": bool(smooth["converged"]),
            "psi_n_rmse_inside_limiter": float(smooth["psi_n_rmse_inside_limiter"]),
            "axis_position_error_m": float(smooth["axis_position_error_m"]),
            "worst_xpoint_error_m": float(max(smooth["xpoint_position_errors_m"])),
            "support_current_relative_l2": float(smooth["support_current_relative_l2"]),
        },
        "frozen": {
            "converged": bool(frozen["converged"]),
            "psi_n_rmse_inside_limiter": float(frozen["psi_n_rmse_inside_limiter"]),
        },
    }


def grid_checks(records: list[dict[str, Any]], baseline_reproduced: bool) -> dict[str, bool]:
    """Return the completeness gates: converged solves on every grid, reproduced baseline."""
    return {
        "base_grid_reproduces_base_benchmark_currents": baseline_reproduced,
        "freegsnke_refinements_converged": all(
            record["freegsnke_run"]["forward_refinement_relative_change"]
            <= machines.REFINEMENT_TOLERANCE
            for record in records
        ),
        "scpn_solves_converged": all(
            record["warm"]["converged"] and record["frozen"]["converged"] for record in records
        ),
    }


def grid_changes(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return, for each grid after the first, the change of both codes against the first grid."""
    first = records[0]
    reference = np.asarray(first["freegsnke_currents_a"], dtype=np.float64)
    changes = []
    for record in records[1:]:
        currents = np.asarray(record["freegsnke_currents_a"], dtype=np.float64)
        changes.append(
            {
                "freegsnke_current_relative_l2": float(
                    np.linalg.norm(currents - reference) / np.linalg.norm(reference)
                ),
                "warm_rmse_ratio": record["warm"]["psi_n_rmse_inside_limiter"]
                / first["warm"]["psi_n_rmse_inside_limiter"],
                "frozen_rmse_ratio": record["frozen"]["psi_n_rmse_inside_limiter"]
                / first["frozen"]["psi_n_rmse_inside_limiter"],
            }
        )
    return changes


def build_report(
    source: Path = base.DEFAULT_SOURCE, grids: tuple[tuple[int, int], ...] = GRIDS
) -> dict[str, Any]:
    """Run every grid and return the signed report."""
    if len(grids) < 2 or grids[0] != GRIDS[0]:
        raise ValueError(f"the study needs the base grid {GRIDS[0]} first and at least one more")
    versions = base.validate_source(source)
    reference_eq, _profiles, _solver, _limits = base.solve_inverse_case_with_profiles(source)
    records = [grid_record(source, grid) for grid in grids]
    reproduced = bool(
        np.array_equal(active_currents(reference_eq), records[0]["freegsnke_currents_a"])
    )
    checks = grid_checks(records, reproduced)
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "changes_against_base_grid": grid_changes(records),
        "checks": checks,
        "claim_boundary": CLAIM_BOUNDARY,
        "grids": records,
        "provenance": {
            "freegsnke_source_commit": versions["commit"],
            "host": machines.host_identity(),
            "scpn_source_revision": base._git_head(ROOT),
            "scpn_source_sha256": {path: base._file_sha256(ROOT / path) for path in SOURCE_FILES},
            "versions": versions,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
    }
    report["payload_sha256"] = base._payload_sha256(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render one row per grid and the changes against the base grid."""
    lines = [
        *MARKDOWN_HEADER,
        "# Grid convergence of the MAST-U-like same-case FreeGSNKE comparison",
        "",
        f"Status: **{report['status'].upper()}** (completeness gates; differences are reported)",
        "",
        "| Grid (nR × nZ) | Warm ψ_N RMSE | Frozen ψ_N RMSE | Axis error (m) | "
        "Worst X-point error (m) | Source rel. L2 | Vacuum max (Wb/rad) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["grids"]:
        warm = record["warm"]
        nr, nz = record["grid_shape_r_z"]
        lines.append(
            f"| {nr} × {nz} | {warm['psi_n_rmse_inside_limiter']:.4g} "
            f"| {record['frozen']['psi_n_rmse_inside_limiter']:.4g} "
            f"| {warm['axis_position_error_m']:.3g} | {warm['worst_xpoint_error_m']:.3g} "
            f"| {record['source_relative_l2']:.3g} | {record['vacuum_max_abs_error_wb']:.3g} |"
        )
    lines += ["", "Changes against the base grid:", ""]
    for record, change in zip(report["grids"][1:], report["changes_against_base_grid"]):
        nr, nz = record["grid_shape_r_z"]
        lines.append(
            f"- {nr} × {nz}: FreeGSNKE currents change by "
            f"{change['freegsnke_current_relative_l2']:.3g} (relative L2); warm RMSE ratio "
            f"{change['warm_rmse_ratio']:.3g}; frozen RMSE ratio {change['frozen_rmse_ratio']:.3g}."
        )
    lines += [
        "",
        "Synthetic public machine description only. No facility, control, safety or speed claim; "
        "no comparison with measured discharges.",
        "",
        f"Payload SHA-256: `{report['payload_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def parse_grid(text: str) -> tuple[int, int]:
    """Parse ``NRxNZ`` into a grid; both sizes must be odd and at least 33."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"grid must look like 129x257, got {text!r}")
    grid = (int(parts[0]), int(parts[1]))
    if any(size < 33 or size % 2 == 0 for size in grid):
        raise argparse.ArgumentTypeError(f"grid sizes must be odd and at least 33, got {text!r}")
    return grid


def main(argv: list[str] | None = None) -> int:
    """Run the study and write the JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--source", type=Path, default=base.DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--grid", action="append", type=parse_grid, help="extra grid, NRxNZ")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    grids = (GRIDS[0], *(args.grid or GRIDS[1:]))
    report = build_report(args.source, grids)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
