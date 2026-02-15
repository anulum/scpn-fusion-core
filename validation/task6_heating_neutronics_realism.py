# ----------------------------------------------------------------------
# SCPN Fusion Core -- Task 6 Heating + Neutronics Realism
# ----------------------------------------------------------------------
"""Task 6: GENRAY-like heating proxies and MCNP-lite TBR optimization lane."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.core.heating_neutronics_contracts import (
    quick_candidate,
    refine_candidate_tbr,
    require_int,
)
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


ROOT = Path(__file__).resolve().parents[1]


def run_campaign(
    *,
    seed: int = 42,
    scan_candidates: int = 96,
    target_optimized_configs: int = 10,
    shortlist_size: int = 20,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = require_int("seed", seed, 0)
    n_candidates = require_int("scan_candidates", scan_candidates, 20)
    target = require_int("target_optimized_configs", target_optimized_configs, 1)
    shortlist = require_int("shortlist_size", shortlist_size, target)
    if target < 5:
        raise ValueError("target_optimized_configs must be >= 5.")
    if n_candidates < target:
        raise ValueError("scan_candidates must be >= target_optimized_configs.")

    rng = np.random.default_rng(seed_i)
    explorer = GlobalDesignExplorer("dummy")
    base_tbr = float(
        BreedingBlanket(thickness_cm=260.0, li6_enrichment=1.0)
        .calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
            incident_flux=1e14,
        )
        .tbr
    )

    quick = [
        quick_candidate(
            rng=rng,
            idx=i,
            base_tbr=base_tbr,
            explorer=explorer,
        )
        for i in range(n_candidates)
    ]
    quick_sorted = sorted(quick, key=lambda row: float(row["objective"]), reverse=True)
    refine_count = min(max(shortlist, target * 3), len(quick_sorted))
    refined = [refine_candidate_tbr(row) for row in quick_sorted[:refine_count]]
    refined_sorted = sorted(refined, key=lambda row: float(row["objective"]), reverse=True)

    valid = [
        row for row in refined_sorted if float(row["q_proxy"]) >= 10.0 and float(row["tbr_final"]) >= 1.05
    ]
    selected = valid[:target]

    q_arr = np.asarray([float(row["q_proxy"]) for row in selected], dtype=np.float64)
    t_arr = np.asarray([float(row["tbr_final"]) for row in selected], dtype=np.float64)
    qr_arr = np.asarray([float(row["q_aries_at_proxy"]) for row in selected], dtype=np.float64)
    rf_arr = np.asarray([float(row["rf_absorption_eff"]) for row in selected], dtype=np.float64)
    nbi_arr = np.asarray([float(row["nbi_absorption_eff"]) for row in selected], dtype=np.float64)
    rf_reflect_arr = np.asarray([float(row["rf_reflection_rate"]) for row in selected], dtype=np.float64)
    nbi_reflect_arr = np.asarray([float(row["nbi_reflection_rate"]) for row in selected], dtype=np.float64)
    leak_arr = np.asarray([float(row.get("neutron_leakage_rate", 0.0)) for row in selected], dtype=np.float64)
    tbr_mc_arr = np.asarray([float(row.get("tbr_mc", 0.0)) for row in selected], dtype=np.float64)
    if q_arr.size > 0:
        rel_err = np.abs((q_arr - qr_arr) / np.maximum(np.abs(qr_arr), 1e-9))
        aries_parity_pct = float(np.clip(100.0 * (1.0 - 0.55 * np.mean(rel_err)), 0.0, 100.0))
    else:
        aries_parity_pct = 0.0

    thresholds = {
        "min_optimized_config_count": 10,
        "min_q": 10.0,
        "min_tbr": 1.05,
        "min_aries_at_parity_pct": 75.0,
        "min_rf_absorption_eff": 0.55,
        "min_nbi_absorption_eff": 0.45,
        "max_rf_reflection_rate": 0.55,
        "max_nbi_reflection_rate": 0.55,
        "max_neutron_leakage_rate": 0.50,
    }

    metrics = {
        "scan_candidates": int(n_candidates),
        "shortlist_size": int(refine_count),
        "optimized_config_count": int(len(selected)),
        "valid_config_count": int(len(valid)),
        "mean_q": float(np.mean(q_arr)) if q_arr.size else 0.0,
        "min_q": float(np.min(q_arr)) if q_arr.size else 0.0,
        "mean_tbr": float(np.mean(t_arr)) if t_arr.size else 0.0,
        "min_tbr": float(np.min(t_arr)) if t_arr.size else 0.0,
        "mean_rf_absorption_eff": float(np.mean(rf_arr)) if rf_arr.size else 0.0,
        "mean_nbi_absorption_eff": float(np.mean(nbi_arr)) if nbi_arr.size else 0.0,
        "mean_rf_reflection_rate": float(np.mean(rf_reflect_arr)) if rf_reflect_arr.size else 0.0,
        "mean_nbi_reflection_rate": float(np.mean(nbi_reflect_arr)) if nbi_reflect_arr.size else 0.0,
        "mean_neutron_leakage_rate": float(np.mean(leak_arr)) if leak_arr.size else 0.0,
        "mean_tbr_mc": float(np.mean(tbr_mc_arr)) if tbr_mc_arr.size else 0.0,
        "aries_at_parity_pct": float(aries_parity_pct),
    }

    failure_reasons: list[str] = []
    if metrics["optimized_config_count"] < thresholds["min_optimized_config_count"]:
        failure_reasons.append("optimized_config_count")
    if metrics["min_q"] < thresholds["min_q"]:
        failure_reasons.append("min_q")
    if metrics["min_tbr"] < thresholds["min_tbr"]:
        failure_reasons.append("min_tbr")
    if metrics["aries_at_parity_pct"] < thresholds["min_aries_at_parity_pct"]:
        failure_reasons.append("aries_at_parity_pct")
    if metrics["mean_rf_absorption_eff"] < thresholds["min_rf_absorption_eff"]:
        failure_reasons.append("rf_absorption_eff")
    if metrics["mean_nbi_absorption_eff"] < thresholds["min_nbi_absorption_eff"]:
        failure_reasons.append("nbi_absorption_eff")
    if metrics["mean_rf_reflection_rate"] > thresholds["max_rf_reflection_rate"]:
        failure_reasons.append("rf_reflection_rate")
    if metrics["mean_nbi_reflection_rate"] > thresholds["max_nbi_reflection_rate"]:
        failure_reasons.append("nbi_reflection_rate")
    if metrics["mean_neutron_leakage_rate"] > thresholds["max_neutron_leakage_rate"]:
        failure_reasons.append("neutron_leakage_rate")

    selected_out = [
        {
            "candidate_id": int(row["candidate_id"]),
            "major_radius_m": float(row["major_radius_m"]),
            "b_t": float(row["b_t"]),
            "ip_ma": float(row["ip_ma"]),
            "q_proxy": float(row["q_proxy"]),
            "q_aries_at_proxy": float(row["q_aries_at_proxy"]),
            "tbr_final": float(row["tbr_final"]),
            "rf_absorption_eff": float(row["rf_absorption_eff"]),
            "nbi_absorption_eff": float(row["nbi_absorption_eff"]),
            "rf_reflection_rate": float(row["rf_reflection_rate"]),
            "nbi_reflection_rate": float(row["nbi_reflection_rate"]),
            "absorbed_heating_mw": float(row["absorbed_heating_mw"]),
            "tbr_mc": float(row["tbr_mc"]),
            "neutron_leakage_rate": float(row["neutron_leakage_rate"]),
        }
        for row in selected
    ]

    return {
        "task6_heating_neutronics_realism": {
            "metrics": metrics,
            "thresholds": thresholds,
            "optimized_configs": selected_out,
            "failure_reasons": failure_reasons,
            "passes_thresholds": bool(len(failure_reasons) == 0),
            "runtime_seconds": float(time.perf_counter() - t0),
        },
        "seed": seed_i,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    out = run_campaign(**kwargs)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task6_heating_neutronics_realism"]
    m = g["metrics"]
    th = g["thresholds"]
    lines = [
        "# Task 6 Heating + Neutronics Realism",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## GENRAY-Like Heating Proxies",
        "",
        f"- Mean RF absorption efficiency: `{m['mean_rf_absorption_eff']:.3f}` (threshold `>= {th['min_rf_absorption_eff']:.2f}`)",
        f"- Mean NBI absorption efficiency: `{m['mean_nbi_absorption_eff']:.3f}` (threshold `>= {th['min_nbi_absorption_eff']:.2f}`)",
        f"- Mean RF reflection rate: `{m['mean_rf_reflection_rate']:.3f}` (threshold `<= {th['max_rf_reflection_rate']:.2f}`)",
        f"- Mean NBI reflection rate: `{m['mean_nbi_reflection_rate']:.3f}` (threshold `<= {th['max_nbi_reflection_rate']:.2f}`)",
        "",
        "## MCNP-Lite Neutronics Optimization (MVR-0.96 Lane)",
        "",
        f"- Optimized configs meeting Q/TBR gate: `{m['optimized_config_count']}` (threshold `>= {th['min_optimized_config_count']}`)",
        f"- Min Q in optimized set: `{m['min_q']:.3f}` (threshold `>= {th['min_q']:.1f}`)",
        f"- Min TBR in optimized set: `{m['min_tbr']:.3f}` (threshold `>= {th['min_tbr']:.2f}`)",
        f"- Mean MC TBR (history transport): `{m['mean_tbr_mc']:.3f}`",
        f"- Mean neutron leakage rate: `{m['mean_neutron_leakage_rate']:.3f}` (threshold `<= {th['max_neutron_leakage_rate']:.2f}`)",
        f"- Mean Q: `{m['mean_q']:.3f}`",
        f"- Mean TBR: `{m['mean_tbr']:.3f}`",
        "",
        "## ARIES-AT Scaling Parity",
        "",
        f"- Q parity score: `{m['aries_at_parity_pct']:.2f}%` (threshold `>= {th['min_aries_at_parity_pct']:.1f}%`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-candidates", type=int, default=96)
    parser.add_argument("--target-optimized-configs", type=int, default=10)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task6_heating_neutronics_realism.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task6_heating_neutronics_realism.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        scan_candidates=args.scan_candidates,
        target_optimized_configs=args.target_optimized_configs,
        shortlist_size=args.shortlist_size,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task6_heating_neutronics_realism"]
    print("Task 6 heating/neutronics realism validation complete.")
    print(
        "Summary -> "
        f"optimized={g['metrics']['optimized_config_count']}, "
        f"min_q={g['metrics']['min_q']:.3f}, "
        f"min_tbr={g['metrics']['min_tbr']:.3f}, "
        f"parity={g['metrics']['aries_at_parity_pct']:.2f}%, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
