# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 5 Disruption + Mitigation Integration
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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
from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection
from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_fraction(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def _synthetic_disruption_signal(
    *,
    rng: np.random.Generator,
    disturbance: float,
    window: int = 220,
) -> tuple[np.ndarray, dict[str, float]]:
    t = np.linspace(0.0, 1.0, int(window), dtype=np.float64)
    base = 0.68 + 0.10 * np.sin(2.0 * np.pi * 2.4 * t + rng.uniform(-0.4, 0.4))
    elm = disturbance * (0.30 * np.exp(-((t - 0.78) / 0.10) ** 2))
    signal = np.clip(base + elm + rng.normal(0.0, 0.018, size=t.shape), 0.01, None)
    n1 = float(0.08 + 0.55 * disturbance + rng.uniform(0.00, 0.05))
    n2 = float(0.05 + 0.32 * disturbance + rng.uniform(0.00, 0.04))
    n3 = float(0.02 + 0.15 * disturbance + rng.uniform(0.00, 0.03))
    toroidal = {
        "toroidal_n1_amp": n1,
        "toroidal_n2_amp": n2,
        "toroidal_n3_amp": n3,
        "toroidal_asymmetry_index": float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)),
        "toroidal_radial_spread": float(0.02 + 0.08 * disturbance),
    }
    return signal, toroidal


def _mcnp_lite_tbr(
    *,
    base_tbr: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> tuple[float, float]:
    factor = float(
        1.15
        + 0.20 * float(np.clip(be_multiplier_fraction, 0.0, 1.0))
        + 0.10 * float(np.clip(li6_enrichment, 0.0, 1.0))
        + 0.05 * float(np.clip(reflector_albedo, 0.0, 1.0))
    )
    return float(base_tbr * factor), factor


def _run_episode(
    *,
    rng: np.random.Generator,
    rl_agent: FusionAIAgent,
    base_tbr: float,
    explorer: GlobalDesignExplorer,
) -> dict[str, float | bool]:
    disturbance = float(rng.uniform(0.0, 1.0))
    pre_energy_mj = float(rng.uniform(240.0, 420.0))
    pre_current_ma = float(rng.uniform(11.0, 16.5))
    signal, toroidal = _synthetic_disruption_signal(rng=rng, disturbance=disturbance)
    risk_before = float(predict_disruption_risk(signal, toroidal))

    rl_state = rl_agent.discretize_state(12.0 * risk_before, 4.0 * disturbance)
    rl_action = int(rl_agent.choose_action(rl_state, rng))
    rl_action_bias = {-1: -1.0, 0: 0.0, 1: 1.0}[rl_action - 1]

    neon_quantity_mol = float(
        np.clip(
            0.05
            + 0.15 * risk_before
            + 0.07 * disturbance
            + 0.02 * rl_action_bias,
            0.03,
            0.24,
        )
    )
    spi = ShatteredPelletInjection(
        Plasma_Energy_MJ=pre_energy_mj,
        Plasma_Current_MA=pre_current_ma,
    )
    _, _, _, spi_diag = spi.trigger_mitigation(
        neon_quantity_mol=neon_quantity_mol,
        return_diagnostics=True,
        duration_s=0.03,
        dt_s=5e-5,
        verbose=False,
    )

    zeff = float(spi_diag["z_eff"])
    tau_cq_s = float(spi_diag["tau_cq_ms_mean"]) * 1e-3
    final_current_ma = float(spi_diag["final_current_MA"])
    quench_fraction = float(np.clip((pre_current_ma - final_current_ma) / pre_current_ma, 0.0, 1.0))
    impurity_radiation_mw = float(40.0 * zeff * (0.6 + quench_fraction))

    halo_current_ma = float(
        pre_current_ma
        * np.clip(
            0.06
            + 0.18 * disturbance
            + 0.12 * quench_fraction
            - 0.025 * (10.0 * neon_quantity_mol),
            0.03,
            0.36,
        )
    )
    runaway_beam_ma = float(
        pre_current_ma
        * np.clip(
            0.02
            + 0.045 * disturbance
            + 0.02 * min(tau_cq_s / 0.02, 2.0)
            - 0.10 * neon_quantity_mol
            - 0.003 * zeff,
            0.0,
            0.075,
        )
    )

    mitigation_strength = float(
        np.clip(
            1.60 * neon_quantity_mol + 0.03 * zeff + 0.10 * (rl_action_bias + 1.0),
            0.08,
            0.95,
        )
    )
    post_toroidal = {
        "toroidal_n1_amp": float(max(0.0, toroidal["toroidal_n1_amp"] * (1.0 - 0.75 * mitigation_strength))),
        "toroidal_n2_amp": float(max(0.0, toroidal["toroidal_n2_amp"] * (1.0 - 0.70 * mitigation_strength))),
        "toroidal_n3_amp": float(max(0.0, toroidal["toroidal_n3_amp"] * (1.0 - 0.65 * mitigation_strength))),
        "toroidal_asymmetry_index": float(
            max(0.0, toroidal["toroidal_asymmetry_index"] * (1.0 - 0.72 * mitigation_strength))
        ),
        "toroidal_radial_spread": float(max(0.0, toroidal["toroidal_radial_spread"] * (1.0 - 0.60 * mitigation_strength))),
    }
    post_signal = np.clip(signal * (1.0 - 0.60 * mitigation_strength), 0.01, None)
    risk_after_model = float(predict_disruption_risk(post_signal, post_toroidal))
    risk_after = float(
        np.clip(
            0.45 * risk_after_model
            + 0.55 * (risk_before * (1.0 - 0.80 * mitigation_strength) + 0.03 * disturbance),
            0.0,
            1.0,
        )
    )

    wall_damage_index = float(
        np.clip(
            0.18 * halo_current_ma
            + 0.55 * runaway_beam_ma
            + 5.0e-4 * impurity_radiation_mw
            + 0.10 * disturbance,
            0.0,
            3.0,
        )
    )

    r_maj = float(rng.uniform(1.2, 1.6))
    b_t = float(rng.uniform(9.0, 12.0))
    ip = float(rng.uniform(3.5, 8.0))
    design = explorer.evaluate_design(r_maj, b_t, ip)
    q_proxy = float(
        7.5
        + 0.10 * np.sqrt(max(float(design["Q"]), 0.0))
        * (1.0 - 0.25 * disturbance)
    )
    li6_enrichment = float(rng.uniform(0.85, 1.0))
    be_multiplier_fraction = float(rng.uniform(0.35, 0.95))
    reflector_albedo = float(rng.uniform(0.30, 0.90))
    tbr_proxy, _ = _mcnp_lite_tbr(
        base_tbr=base_tbr,
        li6_enrichment=li6_enrichment,
        be_multiplier_fraction=be_multiplier_fraction,
        reflector_albedo=reflector_albedo,
    )

    no_wall_damage = bool(wall_damage_index < 1.10)
    objective_success = bool(q_proxy >= 10.0 and tbr_proxy >= 1.0 and no_wall_damage)
    prevented = bool(risk_after < 0.88 and no_wall_damage and runaway_beam_ma < 1.00)

    reward = (
        2.0 * float(q_proxy >= 10.0)
        + 2.0 * float(tbr_proxy >= 1.0)
        + 1.5 * float(no_wall_damage)
        + 1.2 * float(prevented)
        - 1.4 * wall_damage_index
        - 1.1 * risk_after
    )
    next_state = rl_agent.discretize_state(12.0 * risk_after, 4.0 * max(0.0, disturbance - mitigation_strength))
    rl_agent.learn(rl_state, rl_action, next_state, reward)

    return {
        "disturbance": disturbance,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "neon_quantity_mol": neon_quantity_mol,
        "zeff": zeff,
        "halo_current_ma": halo_current_ma,
        "runaway_beam_ma": runaway_beam_ma,
        "impurity_radiation_mw": impurity_radiation_mw,
        "wall_damage_index": wall_damage_index,
        "q_proxy": q_proxy,
        "tbr_proxy": tbr_proxy,
        "no_wall_damage": no_wall_damage,
        "objective_success": objective_success,
        "prevented": prevented,
    }


def run_campaign(
    *,
    seed: int = 42,
    ensemble_runs: int = 50,
    mpc_steps_per_episode: int = 220,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = _require_int("seed", seed, 0)
    runs = _require_int("ensemble_runs", ensemble_runs, 10)
    steps = _require_int("mpc_steps_per_episode", mpc_steps_per_episode, 64)

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
            _run_episode(
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
    runaway = np.asarray([float(e["runaway_beam_ma"]) for e in episodes], dtype=np.float64)
    wall = np.asarray([float(e["wall_damage_index"]) for e in episodes], dtype=np.float64)
    zeff = np.asarray([float(e["zeff"]) for e in episodes], dtype=np.float64)

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
        "mean_runaway_beam_ma": float(np.mean(runaway)),
        "p95_runaway_beam_ma": float(np.percentile(runaway, 95)),
        "mean_wall_damage_index": float(np.mean(wall)),
        "no_wall_damage_rate": float(np.mean(wall < 1.0)),
        "mean_zeff": float(np.mean(zeff)),
    }
    rl_summary = {
        "multiobjective_success_rate": float(np.mean(objective)),
        "q_ge_10_rate": float(np.mean(qv >= 10.0)),
        "tbr_ge_1_rate": float(np.mean(tv >= 1.0)),
        "mean_q_proxy": float(np.mean(qv)),
        "mean_tbr_proxy": float(np.mean(tv)),
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
        "max_p95_runaway_beam_ma": 1.00,
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
    if physics["p95_runaway_beam_ma"] > thresholds["max_p95_runaway_beam_ma"]:
        failure_reasons.append("p95_runaway_beam_ma")
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
        f"- P95 runaway beam: `{p['p95_runaway_beam_ma']:.3f} MA` (threshold `<= {th['max_p95_runaway_beam_ma']:.2f} MA`)",
        f"- Mean wall-damage index: `{p['mean_wall_damage_index']:.3f}`",
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
