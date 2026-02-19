# ----------------------------------------------------------------------
# SCPN Fusion Core -- Task 5 Disruption + Mitigation Integration
# ----------------------------------------------------------------------
"""Task 5: SPI/impurity disruption mitigation with halo/RE and RL objective gates."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_contracts import require_int, run_disruption_episode
from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


ROOT = Path(__file__).resolve().parents[1]


def run_campaign(
    *,
    seed: int = 42,
    ensemble_runs: int = 50,
    mpc_steps_per_episode: int = 220,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = require_int("seed", seed, 0)
    runs = require_int("ensemble_runs", ensemble_runs, 10)
    steps = require_int("mpc_steps_per_episode", mpc_steps_per_episode, 64)

    rng = np.random.default_rng(seed_i)
    rl_agent = FusionAIAgent(epsilon=0.08)
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
        )
        .tbr
    )

    episodes: list[dict[str, float | bool]] = []
    for _ in range(runs):
        episodes.append(
            run_disruption_episode(
                rng=rng,
                rl_agent=rl_agent,
                base_tbr=base_tbr,
                explorer=explorer,
            )
        )

    prevented = np.asarray([float(e["prevented"]) for e in episodes], dtype=np.float64)
    objective = np.asarray([float(e["objective_success"]) for e in episodes], dtype=np.float64)
    qv = np.asarray([float(e["q_proxy"]) for e in episodes], dtype=np.float64)
    tv = np.asarray([float(e["tbr_proxy"]) for e in episodes], dtype=np.float64)
    halo = np.asarray([float(e["halo_current_ma"]) for e in episodes], dtype=np.float64)
    halo_peak = np.asarray([float(e["halo_peak_ma"]) for e in episodes], dtype=np.float64)
    runaway = np.asarray([float(e["runaway_beam_ma"]) for e in episodes], dtype=np.float64)
    runaway_peak = np.asarray([float(e["runaway_peak_ma"]) for e in episodes], dtype=np.float64)
    wall = np.asarray([float(e["wall_damage_index"]) for e in episodes], dtype=np.float64)
    zeff = np.asarray([float(e["zeff"]) for e in episodes], dtype=np.float64)
    tau_imp = np.asarray([float(e["impurity_decay_tau_ms"]) for e in episodes], dtype=np.float64)
    risk_hi = np.asarray([float(e["risk_p95_high"]) for e in episodes], dtype=np.float64)
    wall_hi = np.asarray([float(e["wall_damage_p95_high"]) for e in episodes], dtype=np.float64)
    unc = np.asarray([float(e["uncertainty_envelope"]) for e in episodes], dtype=np.float64)
    prevented_robust = np.asarray([float(e["prevented_robust"]) for e in episodes], dtype=np.float64)

    hybrid = run_nstxu_torax_hybrid_campaign(
        seed=seed_i + 911,
        episodes=runs,
        steps_per_episode=steps,
    )

    physics = {
        "ensemble_runs": int(runs),
        "disruption_prevention_rate": float(np.mean(prevented)),
        "mean_halo_current_ma": float(np.mean(halo)),
        "p95_halo_current_ma": float(np.percentile(halo, 95)),
        "p95_halo_peak_ma": float(np.percentile(halo_peak, 95)),
        "mean_runaway_beam_ma": float(np.mean(runaway)),
        "p95_runaway_beam_ma": float(np.percentile(runaway, 95)),
        "p95_runaway_peak_ma": float(np.percentile(runaway_peak, 95)),
        "mean_wall_damage_index": float(np.mean(wall)),
        "no_wall_damage_rate": float(np.mean(wall < 1.0)),
        "mean_zeff": float(np.mean(zeff)),
        "mean_impurity_decay_tau_ms": float(np.mean(tau_imp)),
        "p95_risk_upper": float(np.percentile(risk_hi, 95)),
        "p95_wall_damage_upper": float(np.percentile(wall_hi, 95)),
        "mean_uncertainty_envelope": float(np.mean(unc)),
    }
    rl_summary = {
        "multiobjective_success_rate": float(np.mean(objective)),
        "q_ge_10_rate": float(np.mean(qv >= 10.0)),
        "tbr_ge_1_rate": float(np.mean(tv >= 1.0)),
        "mean_q_proxy": float(np.mean(qv)),
        "mean_tbr_proxy": float(np.mean(tv)),
        "robust_prevention_rate": float(np.mean(prevented_robust)),
        "q_table_max_abs": float(np.max(np.abs(rl_agent.q_table))),
        "total_reward": float(rl_agent.total_reward),
    }
    mpc_summary = {
        "elm_rejection_rate": float(hybrid.disruption_avoidance_rate),
        "torax_parity_pct": float(hybrid.torax_parity_pct),
        "p95_loop_latency_ms": float(hybrid.p95_loop_latency_ms),
    }

    thresholds = {
        "min_disruption_prevention_rate": 0.90,
        "max_mean_halo_current_ma": 2.60,
        "max_p95_halo_peak_ma": 3.40,
        "max_p95_runaway_beam_ma": 1.00,
        "max_p95_runaway_peak_ma": 1.20,
        "min_mpc_elm_rejection_rate": 0.90,
        "min_multiobjective_success_rate": 0.75,
        "min_q_ge_10_rate": 0.90,
        "min_tbr_ge_1_rate": 0.90,
    }

    failure_reasons: list[str] = []
    if physics["disruption_prevention_rate"] < thresholds["min_disruption_prevention_rate"]:
        failure_reasons.append("disruption_prevention_rate")
    if physics["mean_halo_current_ma"] > thresholds["max_mean_halo_current_ma"]:
        failure_reasons.append("mean_halo_current_ma")
    if physics["p95_halo_peak_ma"] > thresholds["max_p95_halo_peak_ma"]:
        failure_reasons.append("p95_halo_peak_ma")
    if physics["p95_runaway_beam_ma"] > thresholds["max_p95_runaway_beam_ma"]:
        failure_reasons.append("p95_runaway_beam_ma")
    if physics["p95_runaway_peak_ma"] > thresholds["max_p95_runaway_peak_ma"]:
        failure_reasons.append("p95_runaway_peak_ma")
    if mpc_summary["elm_rejection_rate"] < thresholds["min_mpc_elm_rejection_rate"]:
        failure_reasons.append("mpc_elm_rejection_rate")
    if rl_summary["multiobjective_success_rate"] < thresholds["min_multiobjective_success_rate"]:
        failure_reasons.append("multiobjective_success_rate")
    if rl_summary["q_ge_10_rate"] < thresholds["min_q_ge_10_rate"]:
        failure_reasons.append("q_ge_10_rate")
    if rl_summary["tbr_ge_1_rate"] < thresholds["min_tbr_ge_1_rate"]:
        failure_reasons.append("tbr_ge_1_rate")

    return {
        "seed": seed_i,
        "task5_disruption_mitigation": {
            "physics_mitigation": physics,
            "mpc_elm_lane": mpc_summary,
            "rl_multiobjective": rl_summary,
            "thresholds": thresholds,
            "failure_reasons": failure_reasons,
            "passes_thresholds": bool(len(failure_reasons) == 0),
            "runtime_seconds": float(time.perf_counter() - t0),
        },
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    out = run_campaign(**kwargs)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task5_disruption_mitigation"]
    p = g["physics_mitigation"]
    m = g["mpc_elm_lane"]
    r = g["rl_multiobjective"]
    th = g["thresholds"]
    lines = [
        "# Task 5 Disruption + Mitigation Integration",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Ensemble runs: `{p['ensemble_runs']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## SPI / Impurity + Post-Disruption Physics",
        "",
        f"- Disruption prevention rate: `{p['disruption_prevention_rate']:.3f}` (threshold `>= {th['min_disruption_prevention_rate']:.2f}`)",
        f"- Mean halo current: `{p['mean_halo_current_ma']:.3f} MA` (threshold `<= {th['max_mean_halo_current_ma']:.2f} MA`)",
        f"- P95 halo peak current: `{p['p95_halo_peak_ma']:.3f} MA` (threshold `<= {th['max_p95_halo_peak_ma']:.2f} MA`)",
        f"- P95 runaway beam: `{p['p95_runaway_beam_ma']:.3f} MA` (threshold `<= {th['max_p95_runaway_beam_ma']:.2f} MA`)",
        f"- P95 runaway peak beam: `{p['p95_runaway_peak_ma']:.3f} MA` (threshold `<= {th['max_p95_runaway_peak_ma']:.2f} MA`)",
        f"- Mean impurity decay tau: `{p['mean_impurity_decay_tau_ms']:.3f} ms`",
        f"- Mean wall-damage index: `{p['mean_wall_damage_index']:.3f}`",
        f"- P95 risk upper bound: `{p['p95_risk_upper']:.3f}`",
        f"- P95 wall-damage upper bound: `{p['p95_wall_damage_upper']:.3f}`",
        f"- Mean uncertainty envelope: `{p['mean_uncertainty_envelope']:.3f}`",
        "",
        "## MPC ELM Disturbance Rejection",
        "",
        f"- ELM rejection rate: `{m['elm_rejection_rate']:.3f}` (threshold `>= {th['min_mpc_elm_rejection_rate']:.2f}`)",
        f"- TORAX parity: `{m['torax_parity_pct']:.2f}%`",
        f"- P95 loop latency: `{m['p95_loop_latency_ms']:.4f} ms`",
        "",
        "## RL Multi-Objective Optimization",
        "",
        f"- Success rate (`Q>10`, `TBR>1`, no wall damage): `{r['multiobjective_success_rate']:.3f}` (threshold `>= {th['min_multiobjective_success_rate']:.2f}`)",
        f"- Q>=10 rate: `{r['q_ge_10_rate']:.3f}` (threshold `>= {th['min_q_ge_10_rate']:.2f}`)",
        f"- TBR>=1 rate: `{r['tbr_ge_1_rate']:.3f}` (threshold `>= {th['min_tbr_ge_1_rate']:.2f}`)",
        f"- Robust prevention rate (p95 bounds): `{r['robust_prevention_rate']:.3f}`",
        f"- Mean Q proxy: `{r['mean_q_proxy']:.3f}`",
        f"- Mean TBR proxy: `{r['mean_tbr_proxy']:.3f}`",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-runs", type=int, default=50)
    parser.add_argument("--mpc-steps-per-episode", type=int, default=220)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "task5_disruption_mitigation_integration.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "task5_disruption_mitigation_integration.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        ensemble_runs=args.ensemble_runs,
        mpc_steps_per_episode=args.mpc_steps_per_episode,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task5_disruption_mitigation"]
    print("Task 5 disruption/mitigation integration complete.")
    print(
        "Summary -> "
        f"prevention_rate={g['physics_mitigation']['disruption_prevention_rate']:.3f}, "
        f"elm_rejection={g['mpc_elm_lane']['elm_rejection_rate']:.3f}, "
        f"multiobjective_success={g['rl_multiobjective']['multiobjective_success_rate']:.3f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
