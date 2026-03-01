# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Unified Experimental Validation Runner
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Unified validation runner for all 8 SPARC GEQDSK/EQDSK equilibrium files.

Runs each file through two validation modes:

1. **Topology mode** — Validates equilibrium structure:
   - Magnetic axis position matches file header
   - Safety factor q profile is monotonic
   - GS operator sign is consistent
   - Boundary/limiter data is present

2. **Solver mode** (if Rust backend available) — Validates Picard+SOR solver:
   - Solve equilibrium from GEQDSK-derived config
   - Check convergence
   - Report axis position error and solve time

Available SPARC files (8 total):
   3 × .geqdsk  (lmode_vv, lmode_vh, lmode_hv) — 129×129 L-mode equilibria
   5 × .eqdsk   (sparc_1300..1349) — 61×129 equilibrium entries

Requires: numpy, scpn_fusion.core.eqdsk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Locate project root
BASE = Path(__file__).resolve().parent.parent
EXPERIMENTAL_ACK_TOKEN = "I_UNDERSTAND_EXPERIMENTAL"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def require_experimental_opt_in(
    *,
    allow_experimental: bool,
    experimental_ack: str | None,
) -> None:
    if not (allow_experimental or _env_enabled("SCPN_EXPERIMENTAL")):
        raise SystemExit(
            "Experimental validation is locked; pass --experimental or set SCPN_EXPERIMENTAL=1."
        )
    ack_env = os.environ.get("SCPN_EXPERIMENTAL_ACK", "").strip()
    ack = (experimental_ack or "").strip() or ack_env
    if ack != EXPERIMENTAL_ACK_TOKEN:
        raise SystemExit(
            "Experimental acknowledgement missing; pass "
            f"--experimental-ack {EXPERIMENTAL_ACK_TOKEN} "
            "or set SCPN_EXPERIMENTAL_ACK."
        )


def find_sparc_files() -> list[Path]:
    """Find all SPARC GEQDSK and EQDSK files."""
    sparc_dir = BASE / "validation" / "reference_data" / "sparc"
    if not sparc_dir.exists():
        print(f"ERROR: SPARC data directory not found: {sparc_dir}")
        sys.exit(1)

    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    if not files:
        print(f"ERROR: No .geqdsk or .eqdsk files found in {sparc_dir}")
        sys.exit(1)

    return files


def validate_topology(eq, label: str) -> dict:
    """
    Mode 1: Validate equilibrium topology from GEQDSK data.

    Checks axis position, q-profile monotonicity, GS operator sign,
    and boundary/limiter presence.
    """
    r = eq.r
    z = eq.z

    # Find computed magnetic axis from psi extremum
    if eq.simag < eq.sibry:
        idx = np.argmin(eq.psirz)
    else:
        idx = np.argmax(eq.psirz)
    iz, ir = np.unravel_index(idx, eq.psirz.shape)
    r_axis_computed = r[ir]
    z_axis_computed = z[iz]

    axis_dr = abs(r_axis_computed - eq.rmaxis)
    axis_dz = abs(z_axis_computed - eq.zmaxis)

    # q-profile monotonicity
    q_abs = np.abs(eq.qpsi)
    q_diff = np.diff(q_abs)
    q_monotonic = bool(np.all(q_diff >= -1e-6))  # allow tiny noise

    # Safety factor at 95% normalised flux
    q95_idx = int(0.95 * len(eq.qpsi))
    q_95 = float(np.abs(eq.qpsi[q95_idx])) if q95_idx < len(eq.qpsi) else float("nan")

    # GS operator sign check (interior Laplacian)
    psi = eq.psirz
    dR = r[1] - r[0]
    dZ = z[1] - z[0]
    lap_sum = 0.0
    count = 0
    for i in range(eq.nh // 4, 3 * eq.nh // 4):
        for j in range(eq.nw // 4, 3 * eq.nw // 4):
            if r[j] > 0:
                d2psi_dR2 = (psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1]) / dR**2
                dpsi_dR = (psi[i, j + 1] - psi[i, j - 1]) / (2 * dR)
                d2psi_dZ2 = (psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j]) / dZ**2
                gs_op = d2psi_dR2 - dpsi_dR / r[j] + d2psi_dZ2
                lap_sum += gs_op
                count += 1
    gs_sign = np.sign(lap_sum / count) if count > 0 else 0

    has_boundary = len(eq.rbdry) > 0
    has_limiter = len(eq.rlim) > 0

    # Overall pass/fail
    topology_ok = (
        axis_dr < 0.05  # axis R within 5 cm
        and axis_dz < 0.05  # axis Z within 5 cm
        and q_monotonic
        and has_boundary
    )

    return {
        "label": label,
        "grid": f"{eq.nw}x{eq.nh}",
        "B_T": round(eq.bcentr, 2),
        "Ip_MA": round(eq.current / 1e6, 2),
        "R_axis_ref": round(eq.rmaxis, 4),
        "Z_axis_ref": round(eq.zmaxis, 4),
        "R_axis_psi": round(r_axis_computed, 4),
        "Z_axis_psi": round(z_axis_computed, 4),
        "axis_error_R_m": round(axis_dr, 5),
        "axis_error_Z_m": round(axis_dz, 5),
        "q_95": round(q_95, 2),
        "q_monotonic": q_monotonic,
        "gs_sign": int(gs_sign),
        "has_boundary": has_boundary,
        "has_limiter": has_limiter,
        "boundary_pts": len(eq.rbdry),
        "limiter_pts": len(eq.rlim),
        "topology_pass": topology_ok,
    }


def validate_solver(eq, label: str) -> dict:
    """
    Mode 2: Validate the Picard+SOR solver against GEQDSK reference.

    Converts the GEQDSK to a config and runs the solver, comparing
    the solved axis position against the reference.
    """
    try:
        from scpn_fusion.core._rust_compat import RustAcceleratedKernel, RUST_BACKEND
    except ImportError:
        return {"label": label, "solver_available": False, "note": "Rust backend not available"}

    if not RUST_BACKEND:
        return {"label": label, "solver_available": False, "note": "Rust backend not available"}

    # Convert GEQDSK to config
    config = eq.to_config(name=label)

    # Write temp config
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    try:
        kernel = RustAcceleratedKernel(config_path)
        result = kernel.solve_equilibrium()

        return {
            "label": label,
            "solver_available": True,
            "converged": result.converged,
            "iterations": result.iterations,
            "residual": float(result.residual),
            "solve_time_ms": round(result.solve_time_ms, 1),
            "axis_r": round(result.axis_r, 4),
            "axis_z": round(result.axis_z, 4),
        }
    except Exception as e:
        return {
            "label": label,
            "solver_available": True,
            "error": str(e),
        }
    finally:
        import os
        os.unlink(config_path)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Unlock experimental validation lane.",
    )
    parser.add_argument(
        "--experimental-ack",
        default="",
        help=f"Acknowledgement token for experimental lane ({EXPERIMENTAL_ACK_TOKEN}).",
    )
    args = parser.parse_args(argv)
    require_experimental_opt_in(
        allow_experimental=bool(args.experimental),
        experimental_ack=str(args.experimental_ack),
    )

    from scpn_fusion.core.eqdsk import read_geqdsk

    print("=" * 80)
    print("  SCPN Fusion Core — Unified Experimental Validation")
    print("=" * 80)

    files = find_sparc_files()
    print(f"\nFound {len(files)} SPARC equilibrium files:\n")
    for f in files:
        print(f"  {f.name}")

    # ── Mode 1: Topology Validation ──
    print("\n" + "-" * 80)
    print("  Mode 1: Topology Validation (GEQDSK structure checks)")
    print("-" * 80 + "\n")

    topology_results = []
    for gfile in files:
        eq = read_geqdsk(gfile)
        result = validate_topology(eq, gfile.stem)
        topology_results.append(result)

        status = "PASS" if result["topology_pass"] else "FAIL"
        print(
            f"  [{status}] {result['label']:20s}  "
            f"Grid={result['grid']:>7s}  "
            f"B_T={result['B_T']:>5.1f}T  "
            f"I_p={result['Ip_MA']:>5.1f}MA  "
            f"q95={result['q_95']:>5.1f}  "
            f"AxisErr(R)={result['axis_error_R_m']:.4f}m"
        )

    passed = sum(1 for r in topology_results if r["topology_pass"])
    print(f"\n  Topology: {passed}/{len(topology_results)} passed")

    # ── Mode 2: Solver Validation ──
    print("\n" + "-" * 80)
    print("  Mode 2: Solver Validation (Picard+SOR against GEQDSK)")
    print("-" * 80 + "\n")

    solver_results = []
    for gfile in files:
        eq = read_geqdsk(gfile)
        result = validate_solver(eq, gfile.stem)
        solver_results.append(result)

        if not result.get("solver_available", False):
            print(f"  [SKIP] {result['label']:20s}  {result.get('note', '')}")
        elif "error" in result:
            print(f"  [ERR ] {result['label']:20s}  {result['error']}")
        else:
            conv = "CONV" if result["converged"] else "NCON"
            print(
                f"  [{conv}] {result['label']:20s}  "
                f"iters={result['iterations']:>4d}  "
                f"residual={result['residual']:.2e}  "
                f"time={result['solve_time_ms']:>7.1f}ms  "
                f"axis=({result['axis_r']:.3f}, {result['axis_z']:.3f})"
            )

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"  Files validated:     {len(files)}")
    print(f"  Topology pass:       {passed}/{len(topology_results)}")
    solver_conv = sum(1 for r in solver_results if r.get("converged", False))
    solver_run = sum(1 for r in solver_results if r.get("solver_available", False) and "error" not in r)
    if solver_run > 0:
        print(f"  Solver converged:    {solver_conv}/{solver_run}")
    else:
        print(f"  Solver:              skipped (Rust backend not available)")

    # ── Save JSON ──
    output_path = BASE / "validation" / "experimental_validation_results.json"
    summary = {
        "files_count": len(files),
        "topology": topology_results,
        "solver": solver_results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
