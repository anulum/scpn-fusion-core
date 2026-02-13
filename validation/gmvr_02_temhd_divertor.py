# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GMVR-02 TEMHD Divertor Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GMVR-02: TEMHD divertor MHD/evaporation validation with 3D toroidal sweep."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D


def run_campaign() -> dict[str, Any]:
    t0 = time.perf_counter()

    lab = DivertorLab(P_sol_MW=35.0, R_major=1.4, B_pol=2.3)
    slow = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=0.001, expansion_factor=40.0)
    fast = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=10.0, expansion_factor=40.0)

    eq = VMECStyleEquilibrium3D(
        r_axis=1.4,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.75,
        triangularity=0.32,
        nfp=3,
        modes=[FourierMode3D(m=1, n=1, r_cos=0.05, z_sin=0.04)],
    )

    phis = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    thetas = np.full_like(phis, 0.8 * np.pi)
    rho = np.ones_like(phis)
    r_vals, _, _ = eq.flux_to_cylindrical(rho, thetas, phis)
    modulation = 1.0 + 0.08 * (r_vals - np.mean(r_vals)) / max(np.mean(r_vals), 1e-9)

    indices = []
    for factor in modulation:
        q_mod = fast["surface_heat_flux_w_m2"] * float(factor)
        idx = (
            q_mod / 45.0e6
            + fast["pressure_loss_pa"] / 8.0e5
            + fast["evaporation_rate_kg_m2_s"] / 1.0e-3
        )
        indices.append(float(idx))
    indices_arr = np.asarray(indices, dtype=float)
    toroidal_stability_rate = float(np.mean(indices_arr <= 1.0))

    pressure_ratio = float(
        fast["pressure_loss_pa"] / max(slow["pressure_loss_pa"], 1e-12)
    )
    evap_ratio = float(
        fast["evaporation_rate_kg_m2_s"] / max(slow["evaporation_rate_kg_m2_s"], 1e-12)
    )

    passes = bool(
        slow["is_stable"]
        and fast["is_stable"]
        and pressure_ratio >= 1000.0
        and evap_ratio < 1.0
        and toroidal_stability_rate >= 0.95
    )

    return {
        "slow_flow": slow,
        "fast_flow": fast,
        "pressure_ratio_fast_to_slow": pressure_ratio,
        "evap_ratio_fast_to_slow": evap_ratio,
        "toroidal_stability_rate": toroidal_stability_rate,
        "thresholds": {
            "min_pressure_ratio_fast_to_slow": 1000.0,
            "max_evap_ratio_fast_to_slow": 1.0,
            "min_toroidal_stability_rate": 0.95,
        },
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gmvr_02": run_campaign(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gmvr_02"]
    th = g["thresholds"]
    lines = [
        "# GMVR-02 TEMHD Divertor Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        "",
        "## Slow vs Fast Flow",
        "",
        f"- Slow (0.001 m/s) stability index: `{g['slow_flow']['stability_index']:.4f}`",
        f"- Fast (10 m/s) stability index: `{g['fast_flow']['stability_index']:.4f}`",
        f"- Pressure ratio (fast/slow): `{g['pressure_ratio_fast_to_slow']:.1f}` (threshold `>= {th['min_pressure_ratio_fast_to_slow']:.1f}`)",
        f"- Evaporation ratio (fast/slow): `{g['evap_ratio_fast_to_slow']:.4f}` (threshold `< {th['max_evap_ratio_fast_to_slow']:.1f}`)",
        "",
        "## 3D Toroidal Stability",
        "",
        f"- Stability rate over toroidal sweep: `{g['toroidal_stability_rate']:.3f}` (threshold `>= {th['min_toroidal_stability_rate']:.2f}`)",
        f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gmvr_02_temhd_divertor.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gmvr_02_temhd_divertor.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gmvr_02"]
    print("GMVR-02 TEMHD divertor validation complete.")
    print(
        f"pressure_ratio={g['pressure_ratio_fast_to_slow']:.1f}, "
        f"evap_ratio={g['evap_ratio_fast_to_slow']:.4f}, "
        f"toroidal_stability_rate={g['toroidal_stability_rate']:.3f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
