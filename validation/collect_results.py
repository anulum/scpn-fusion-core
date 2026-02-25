#!/usr/bin/env python3
# SCPN Fusion Core
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2023-2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
"""Run all key benchmarks and generate RESULTS.md at the repository root.

Usage::

    python validation/collect_results.py          # full suite
    python validation/collect_results.py --quick   # skip heavy benchmarks
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_PATH = REPO_ROOT / "RESULTS.md"
ARTIFACTS = REPO_ROOT / "artifacts"
WEIGHTS = REPO_ROOT / "weights"


# ── Utility ──────────────────────────────────────────────────────────

def _hw_header() -> str:
    lines = [
        f"- **CPU:** {platform.processor() or platform.machine()}",
        f"- **Architecture:** {platform.machine()}",
        f"- **OS:** {platform.platform()}",
        f"- **Python:** {platform.python_version()}",
        f"- **NumPy:** {np.__version__}",
    ]
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        lines.append(f"- **RAM:** {ram_gb:.1f} GB")
    except ImportError:
        lines.append("- **RAM:** (install psutil for auto-detection)")
    return "\n".join(lines)


def _safe_run(label: str, fn, *args, **kwargs) -> dict[str, Any] | None:
    print(f"  [{label}] running...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s)")
        return result
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"FAILED ({elapsed:.1f}s)")
        traceback.print_exc()
        return None


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: Any, fmt: str = ".4f") -> str:
    if val is None:
        return "—"
    if isinstance(val, (bool, np.bool_)):
        return "Yes" if val else "No"
    if isinstance(val, (float, np.floating)):
        return f"{float(val):{fmt}}"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return str(val)


# ── Live benchmark runners (compute on the fly) ─────────────────────

def run_hil(quick: bool) -> dict[str, Any] | None:
    from scpn_fusion.control.hil_harness import run_hil_benchmark
    iters = 200 if quick else 1000
    result = run_hil_benchmark(iterations=iters)
    m = result.control_metrics
    return {
        "iterations": m.iterations,
        "p50_us": m.p50_latency_us,
        "p95_us": m.p95_latency_us,
        "p99_us": m.p99_latency_us,
        "sub_ms": m.sub_ms_achieved,
        "total_loop_us": result.total_loop_latency_us,
    }


def run_disruption(quick: bool) -> dict[str, Any] | None:
    from scpn_fusion.control.halo_re_physics import run_disruption_ensemble
    n = 10 if quick else 50
    r = run_disruption_ensemble(ensemble_runs=n, seed=42)
    return {
        "ensemble_runs": r.ensemble_runs,
        "prevention_rate": r.prevention_rate,
        "mean_halo_ma": r.mean_halo_peak_ma,
        "p95_halo_ma": r.p95_halo_peak_ma,
        "mean_re_ma": r.mean_re_peak_ma,
        "p95_re_ma": r.p95_re_peak_ma,
        "passes_iter": r.passes_iter_limits,
    }


def run_q10() -> dict[str, Any] | None:
    from scpn_fusion.core.fusion_ignition_sim import DynamicBurnModel
    r = DynamicBurnModel.find_q10_operating_point(
        R0=6.2, a=2.0, B_t=5.3, I_p=15.0, kappa=1.7,
    )
    best = r["best"]
    return {
        "q10_achieved": r["q10_achieved"],
        "best_Q": best["Q_peak"],
        "P_aux_mw": best["P_aux_MW"],
        "P_fus_mw": best["P_fus_final_MW"],
        "T_keV": best["T_final_keV"],
        "n_e20": best["n_e20"],
    }


def run_tbr() -> dict[str, Any] | None:
    from scpn_fusion.nuclear.blanket_neutronics import MultiGroupBlanket
    blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
    r = blanket.solve_transport()
    return {
        "tbr_total": r["tbr"],
        "tbr_fast": r["tbr_by_group"]["fast"],
        "tbr_epi": r["tbr_by_group"]["epithermal"],
        "tbr_therm": r["tbr_by_group"]["thermal"],
    }


def run_ecrh() -> dict[str, Any] | None:
    from scpn_fusion.core.rf_heating import ECRHHeatingSystem
    ecrh = ECRHHeatingSystem(b0_tesla=5.3, r0_major=6.2, freq_ghz=170.0)
    _rho, _pdep, efficiency = ecrh.compute_deposition(P_ecrh_mw=20.0)
    return {
        "P_ecrh_mw": 20.0,
        "absorption_eff": efficiency,
        "absorption_pct": efficiency * 100,
    }


def run_force_balance_3d() -> dict[str, Any] | None:
    from scpn_fusion.core.equilibrium_3d import (
        VMECStyleEquilibrium3D,
        ForceBalance3D,
    )
    eq = VMECStyleEquilibrium3D(
        r_axis=6.2, z_axis=0.0, a_minor=2.0, kappa=1.7,
        triangularity=0.3, nfp=1,
    )
    fb = ForceBalance3D(eq, b0_tesla=5.3, r0_major=6.2, p0_pa=5e5)
    result = fb.solve(
        max_iterations=20, n_rho=6, n_theta=12, n_phi=8,
    )
    return {
        "iterations": result.iterations,
        "initial_residual": result.initial_residual,
        "final_residual": result.residual_norm,
        "converged": result.converged,
        "reduction_factor": result.initial_residual / max(result.residual_norm, 1e-30),
    }


def run_surrogates() -> dict[str, Any] | None:
    from scpn_fusion.core.pretrained_surrogates import (
        evaluate_pretrained_mlp,
        evaluate_pretrained_fno,
    )
    results: dict[str, Any] = {}
    try:
        mlp = evaluate_pretrained_mlp()
        results["mlp_rmse_s"] = mlp["rmse_s"]
        results["mlp_rmse_pct"] = mlp["rmse_pct"]
        results["mlp_samples"] = mlp["samples"]
    except Exception:
        traceback.print_exc()
        results["mlp_rmse_s"] = None

    try:
        fno = evaluate_pretrained_fno()
        results["fno_rel_l2_mean"] = fno["eval_relative_l2_mean"]
        results["fno_rel_l2_p95"] = fno["eval_relative_l2_p95"]
        results["fno_samples"] = fno["eval_samples"]
    except Exception:
        traceback.print_exc()
        results["fno_rel_l2_mean"] = None

    return results


def run_neural_eq() -> dict[str, Any] | None:
    from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumAccelerator
    weights_path = WEIGHTS / "neural_equilibrium_sparc.npz"
    if not weights_path.exists():
        print("  [neural_eq] weights not found, skipping")
        return None
    model = NeuralEquilibriumAccelerator()
    model.load_weights(weights_path)
    features = np.array([8.7, 12.2, 1.85, 0.0, 1.0, 1.0, -5.0, 0.0])
    psi = model.predict(features)
    bench = model.benchmark(features, n_runs=50)
    return {
        "grid_shape": f"{psi.shape[0]}x{psi.shape[1]}",
        "inference_mean_ms": bench["mean_ms"],
        "inference_p95_ms": bench["p95_ms"],
        "psi_range": f"[{psi.min():.3f}, {psi.max():.3f}]",
    }


# ── Artifact-based benchmark loaders (read pre-computed JSON) ────────

def load_qlknn_metrics() -> dict[str, Any] | None:
    primary = _load_artifact(WEIGHTS / "neural_transport_qlknn.metrics.json")
    if primary is None:
        return None
    result = {"primary": primary}
    backup = _load_artifact(WEIGHTS / "neural_transport_qlknn_run11.metrics.json")
    if backup is not None:
        result["backup"] = backup
    return result


def load_real_shot_validation() -> dict[str, Any] | None:
    return _load_artifact(ARTIFACTS / "real_shot_validation.json")


def load_confinement_itpa() -> dict[str, Any] | None:
    return _load_artifact(ARTIFACTS / "rmse_dashboard_ci.json")


def load_disturbance_rejection() -> dict[str, Any] | None:
    return _load_artifact(ARTIFACTS / "disturbance_rejection_benchmark.json")


def load_freegs_benchmark() -> dict[str, Any] | None:
    return _load_artifact(ARTIFACTS / "freegs_benchmark.json")


def load_disruption_threshold() -> dict[str, Any] | None:
    return _load_artifact(ARTIFACTS / "disruption_threshold_sweep.json")


# ── RESULTS.md generation ────────────────────────────────────────────

def generate_results_md(
    hw: str,
    hil: dict | None,
    disruption: dict | None,
    q10: dict | None,
    tbr: dict | None,
    ecrh: dict | None,
    fb3d: dict | None,
    surrogates: dict | None,
    neural_eq: dict | None,
    qlknn: dict | None,
    real_shots: dict | None,
    confinement: dict | None,
    disturbance: dict | None,
    freegs: dict | None,
    threshold_sweep: dict | None,
    elapsed_s: float,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    sections.append(f"""# SCPN Fusion Core — Benchmark Results

> **Auto-generated** by `validation/collect_results.py` on {now}.
> Re-run the script to refresh these numbers on your hardware.

## Environment

{hw}
- **Generated:** {now}
- **Wall-clock:** {elapsed_s:.0f}s
""")

    # ── Equilibrium & Transport ──
    rows: list[str] = []
    if fb3d:
        rows.append(f"| 3D Force-Balance initial residual | {_fmt(fb3d['initial_residual'], '.4e')} | — | Spectral variational method |")
        rows.append(f"| 3D Force-Balance final residual | {_fmt(fb3d['final_residual'], '.4e')} | — | After {fb3d.get('iterations', '?')} iterations |")
        rows.append(f"| 3D Force-Balance reduction factor | {_fmt(fb3d['reduction_factor'], '.1f')}× | — | initial / final |")
    if neural_eq:
        rows.append(f"| Neural Equilibrium inference (mean) | {_fmt(neural_eq['inference_mean_ms'], '.2f')} | ms | PCA+MLP surrogate on {neural_eq['grid_shape']} grid |")
        rows.append(f"| Neural Equilibrium inference (P95) | {_fmt(neural_eq['inference_p95_ms'], '.2f')} | ms | {neural_eq['grid_shape']} grid |")
    if rows:
        sections.append("## Equilibrium & Transport\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        sections.extend(rows)
        sections.append("")

    # ── QLKNN Neural Transport Surrogate ──
    if qlknn:
        p = qlknn["primary"]
        dims = "×".join(str(d) for d in p["hidden_dims"])
        sections.append("## QLKNN Neural Transport Surrogate\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        sections.append(f"| Test relative L2 | {_fmt(p['test_relative_l2'])} | — | Hard-fail gate < 0.10 |")
        sections.append(f"| Val relative L2 | {_fmt(p['val_relative_l2'])} | — | |")
        sections.append(f"| Train relative L2 | {_fmt(p['train_relative_l2'])} | — | val/train = {p['val_relative_l2'] / max(p['train_relative_l2'], 1e-30):.2f} |")
        sections.append(f"| Best val MSE | {_fmt(p['best_val_mse'], '.6f')} | — | |")
        sections.append(f"| Architecture | {dims} | — | MLP hidden dims |")
        sections.append(f"| Epochs | {p['epochs_run']} | — | Early-stopped |")
        sections.append(f"| Training time | {p['training_time_s'] / 3600:.1f} | h | {p['platform']} |")
        sections.append(f"| Data source | {p['data_source']} | — | |")
        if "backup" in qlknn:
            b = qlknn["backup"]
            bdims = "×".join(str(d) for d in b["hidden_dims"])
            sections.append(f"| Backup test relative L2 | {_fmt(b['test_relative_l2'])} | — | {bdims} architecture |")
        sections.append("")

    # ── Confinement ITPA (20 machines) ──
    if confinement:
        ci = confinement.get("confinement_itpa", {})
        cs = confinement.get("confinement_iter_sparc", {})
        bn = confinement.get("beta_iter_sparc", {})
        fd = confinement.get("forward_diagnostics", {})
        sections.append("## Confinement Scaling (ITPA H-mode)\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        if ci:
            sections.append(f"| Machines validated | {ci.get('count', '—')} | — | ITER, JET, DIII-D, ASDEX-U, C-Mod, JT-60U, NSTX, MAST, KSTAR, EAST, SPARC, ARC |")
            sections.append(f"| τ_E RMSE | {_fmt(ci.get('tau_rmse_s'), '.4f')} | s | |")
            sections.append(f"| τ_E relative RMSE | {_fmt(ci.get('tau_mae_rel_pct'), '.1f')} | % | |")
            sections.append(f"| H98 RMSE | {_fmt(ci.get('h98_rmse'), '.4f')} | — | |")
        if cs and cs.get("rows"):
            for row in cs["rows"]:
                sections.append(f"| {row['scenario']} τ_E error | {_fmt(row.get('relative_error_pct'), '.1f')} | % | τ_pred={_fmt(row.get('tau_pred_s'), '.3f')} s |")
        if bn and bn.get("rows"):
            sections.append(f"| β_N RMSE | {_fmt(bn.get('beta_n_rmse'), '.4f')} | — | |")
            for row in bn["rows"]:
                sections.append(f"| {row['scenario']} β_N error | {_fmt(row.get('relative_error_pct'), '.1f')} | % | Q={_fmt(row.get('model_q'), '.0f')}, P_fus={_fmt(row.get('model_p_fusion_mw'), '.0f')} MW |")
        if fd:
            sections.append(f"| Interferometer phase RMSE | {_fmt(fd.get('phase_rmse_rad'), '.6f')} | rad | {fd.get('count_interferometer_channels', '?')} channels |")
            sections.append(f"| Neutron rate relative error | {_fmt(fd.get('neutron_rate_rel_error_pct'), '.1f')} | % | |")
            sections.append(f"| Thomson voltage RMSE | {_fmt(fd.get('thomson_voltage_rmse_v'), '.2e')} | V | {fd.get('count_thomson_channels', '?')} channels |")
        sections.append("")

    # ── Heating & Neutronics ──
    rows = []
    if q10:
        rows.append(f"| Best Q (ITER-like scan) | {_fmt(q10['best_Q'], '.2f')} | — | Target: Q ≥ 10 |")
        rows.append(f"| Q ≥ 10 achieved | {_fmt(q10['q10_achieved'])} | — | {_fmt(q10['n_e20'], '.2f')} × 10²⁰ m⁻³ |")
        rows.append(f"| P_aux at best Q | {_fmt(q10['P_aux_mw'], '.1f')} | MW | Auxiliary heating |")
        rows.append(f"| P_fus at best Q | {_fmt(q10['P_fus_mw'], '.1f')} | MW | Fusion power |")
        rows.append(f"| T at best Q | {_fmt(q10['T_keV'], '.1f')} | keV | Ion temperature |")
    if ecrh:
        rows.append(f"| ECRH absorption efficiency | {_fmt(ecrh['absorption_pct'], '.1f')} | % | 170 GHz, 1st harmonic, {_fmt(ecrh['P_ecrh_mw'], '.0f')} MW |")
    if tbr:
        rows.append(f"| Tritium Breeding Ratio (total) | {_fmt(tbr['tbr_total'], '.4f')} | — | 3-group, 80 cm, 90% ⁶Li |")
        rows.append(f"| TBR fast group | {_fmt(tbr['tbr_fast'], '.4f')} | — | 14.1 MeV neutrons |")
        rows.append(f"| TBR epithermal group | {_fmt(tbr['tbr_epi'], '.4f')} | — | Slowed neutrons |")
        rows.append(f"| TBR thermal group | {_fmt(tbr['tbr_therm'], '.4f')} | — | Thermalized |")
    if rows:
        sections.append("## Heating & Neutronics\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        sections.extend(rows)
        sections.append("")

    # ── Disruption & Control ──
    rows = []
    if disruption:
        n = disruption["ensemble_runs"]
        rows.append(f"| Disruption prevention rate | {_fmt(disruption['prevention_rate'] * 100, '.1f')} | % | {n}-run ensemble |")
        rows.append(f"| Mean halo current peak | {_fmt(disruption['mean_halo_ma'], '.3f')} | MA | |")
        rows.append(f"| P95 halo current peak | {_fmt(disruption['p95_halo_ma'], '.3f')} | MA | |")
        rows.append(f"| Mean RE current peak | {_fmt(disruption['mean_re_ma'], '.3f')} | MA | |")
        rows.append(f"| P95 RE current peak | {_fmt(disruption['p95_re_ma'], '.3f')} | MA | |")
        rows.append(f"| Passes ITER limits | {_fmt(disruption['passes_iter'])} | — | Halo + RE constraints |")
    if hil:
        rows.append(f"| HIL control-loop P50 latency | {_fmt(hil['p50_us'], '.1f')} | μs | {hil['iterations']} iterations |")
        rows.append(f"| HIL control-loop P95 latency | {_fmt(hil['p95_us'], '.1f')} | μs | |")
        rows.append(f"| HIL control-loop P99 latency | {_fmt(hil['p99_us'], '.1f')} | μs | |")
        rows.append(f"| Sub-ms achieved | {_fmt(hil['sub_ms'])} | — | Total loop: {_fmt(hil['total_loop_us'], '.1f')} μs |")
    if rows:
        sections.append("## Disruption & Control\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        sections.extend(rows)
        sections.append("")

    # ── Real-Shot Disruption Detection ──
    if real_shots:
        d = real_shots.get("disruption", {})
        t = real_shots.get("transport", {})
        eq = real_shots.get("equilibrium", {})
        sections.append("## Real-Shot Validation\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        if d:
            sections.append(f"| Disruption recall | {_fmt(d.get('recall'), '.2f')} | — | {d.get('true_positives', '?')}/{d.get('n_disruptions', '?')} disruptions detected |")
            sections.append(f"| Disruption FPR | {_fmt(d.get('false_positive_rate'), '.2f')} | — | {d.get('false_positives', '?')}/{d.get('n_safe', '?')} false alarms |")
            sections.append(f"| Disruption detection | {_fmt(d.get('passes'))} | — | recall ≥ 0.6 and FPR ≤ 0.4 |")
        if t:
            sections.append(f"| Transport τ_E RMSE | {_fmt(t.get('rmse_s'), '.4f')} | s | {t.get('n_shots', '?')} shots |")
            sections.append(f"| Transport within 2σ | {_fmt((t.get('within_2sigma_fraction', 0)) * 100, '.0f')} | % | Gate ≥ 80% |")
            sections.append(f"| Transport validation | {_fmt(t.get('passes'))} | — | |")
        if eq:
            sections.append(f"| Equilibrium ψ pass fraction | {_fmt((eq.get('psi_pass_fraction', 0)) * 100, '.0f')} | % | {eq.get('n_psi_pass', '?')}/{eq.get('n_files', '?')} files |")
            sections.append(f"| Equilibrium q95 pass fraction | {_fmt((eq.get('q95_pass_fraction', 0)) * 100, '.0f')} | % | {eq.get('n_q95_pass', '?')}/{eq.get('n_files', '?')} files |")
            sections.append(f"| Equilibrium validation | {_fmt(eq.get('passes'))} | — | |")
        sections.append(f"| Overall real-shot pass | {_fmt(real_shots.get('overall_pass'))} | — | |")
        sections.append("")

    # ── Disturbance Rejection (SNN vs PID vs MPC) ──
    if disturbance:
        sections.append("## Disturbance Rejection\n")
        sections.append("| Controller | Scenario | ISE | Settling (s) | Overshoot | Stable |")
        sections.append("|-----------|----------|-----|-------------|-----------|--------|")
        for r in disturbance.get("results", []):
            sections.append(
                f"| {r['controller']} | {r['scenario']} "
                f"| {_fmt(r['ise'], '.2e')} "
                f"| {_fmt(r['settling_time_s'], '.4f')} "
                f"| {_fmt(r['peak_overshoot'], '.4f')} "
                f"| {_fmt(r['stable'])} |"
            )
        sections.append("")

    # ── FreeGS Equilibrium Benchmark ──
    if freegs:
        sections.append("## FreeGS Equilibrium Benchmark\n")
        sections.append("| Case | ψ NRMSE | q NRMSE | Axis error (m) | Passes |")
        sections.append("|------|---------|---------|---------------|--------|")
        for c in freegs.get("cases", []):
            sections.append(
                f"| {c['name']} "
                f"| {_fmt(c['psi_nrmse'], '.3f')} "
                f"| {_fmt(c['q_profile_nrmse'], '.3f')} "
                f"| {_fmt(c['axis_error_m'], '.3f')} "
                f"| {_fmt(c['passes'])} |"
            )
        sections.append(f"\n*Overall ψ NRMSE: {_fmt(freegs.get('overall_psi_nrmse'), '.3f')} "
                        f"(threshold: {freegs.get('psi_nrmse_threshold', '?')}). "
                        f"Overall: {'PASS' if freegs.get('passes') else 'FAIL'}*\n")

    # ── Disruption Threshold Sweep ──
    if threshold_sweep:
        opt = threshold_sweep.get("optimal", {})
        if opt:
            sections.append("## Disruption Threshold Optimization\n")
            sections.append("| Metric | Value | Notes |")
            sections.append("|--------|-------|-------|")
            sections.append(f"| Optimal bias | {_fmt(opt.get('bias'), '.1f')} | |")
            sections.append(f"| Optimal threshold | {_fmt(opt.get('threshold'), '.2f')} | |")
            sections.append(f"| Recall | {_fmt(opt.get('recall'), '.2f')} | {opt.get('true_positives', '?')}/{opt.get('n_disruptions', '?')} |")
            sections.append(f"| FPR | {_fmt(opt.get('fpr'), '.2f')} | {opt.get('false_positives', '?')}/{opt.get('n_safe', '?')} |")
            sections.append(f"| Pareto score | {_fmt(opt.get('pareto_score'), '.2f')} | recall − FPR |")
            cfg = threshold_sweep.get("config", {})
            sections.append(f"| Shots evaluated | {cfg.get('n_shots', '?')} | {cfg.get('n_disruptions', '?')} disruptions, {cfg.get('n_safe', '?')} safe |")
            sections.append("")

    # ── Legacy Surrogates ──
    rows = []
    if surrogates:
        if surrogates.get("mlp_rmse_s") is not None:
            rows.append(f"| MLP (ITPA H-mode) RMSE | {_fmt(surrogates['mlp_rmse_s'], '.4f')} | s | τ_E confinement time |")
            rows.append(f"| MLP (ITPA H-mode) RMSE % | {_fmt(surrogates['mlp_rmse_pct'], '.1f')} | % | {int(surrogates.get('mlp_samples', 0))} samples |")
        if surrogates.get("fno_rel_l2_mean") is not None:
            rows.append(f"| FNO (EUROfusion JET) relative L2 (mean) | {_fmt(surrogates['fno_rel_l2_mean'], '.4f')} | — | ψ(R,Z) reconstruction |")
            rows.append(f"| FNO (EUROfusion JET) relative L2 (P95) | {_fmt(surrogates['fno_rel_l2_p95'], '.4f')} | — | {int(surrogates.get('fno_samples', 0))} samples |")
    if rows:
        sections.append("## Legacy Surrogates\n")
        sections.append("| Metric | Value | Unit | Notes |")
        sections.append("|--------|-------|------|-------|")
        sections.extend(rows)
        sections.append("")

    # ── Validation Summary ──
    sections.append("## Validation Summary\n")
    sections.append("| Lane | Status | Key metric |")
    sections.append("|------|--------|------------|")

    def _lane(name: str, data, pass_key: str | None, metric_str: str):
        if data is None:
            sections.append(f"| {name} | — | not available |")
            return
        if pass_key:
            passed = data.get(pass_key)
            if isinstance(data, dict) and pass_key not in data:
                passed = None
            status = "PASS" if passed else ("FAIL" if passed is not None else "—")
        else:
            status = "RUN"
        sections.append(f"| {name} | {status} | {metric_str} |")

    qlknn_pass = None
    qlknn_metric = "—"
    if qlknn:
        v = qlknn["primary"]["test_relative_l2"]
        qlknn_pass = v < 0.10
        qlknn_metric = f"test_rel_l2 = {v:.4f}"
    _lane("QLKNN Transport", {"passes": qlknn_pass} if qlknn else None, "passes", qlknn_metric)
    _lane("Real-shot validation", real_shots, "overall_pass",
          f"recall={real_shots['disruption']['recall']:.0%}, FPR={real_shots['disruption']['false_positive_rate']:.0%}" if real_shots and "disruption" in real_shots else "—")

    conf_metric = "—"
    conf_pass = None
    if confinement and "confinement_itpa" in confinement:
        ci = confinement["confinement_itpa"]
        conf_metric = f"RMSE = {ci.get('tau_rmse_s', 0):.4f} s"
    _lane("Confinement ITPA", {"passes": conf_pass} if confinement else None, None, conf_metric)

    _lane("3D Force Balance", fb3d, None,
          f"reduction = {fb3d['reduction_factor']:.1f}×" if fb3d else "—")
    _lane("Q ≥ 10", q10, "q10_achieved",
          f"Q = {q10['best_Q']:.1f}" if q10 else "—")
    _lane("TBR > 1.05", {"passes": tbr and tbr["tbr_total"] > 1.05} if tbr else None, "passes",
          f"TBR = {tbr['tbr_total']:.4f}" if tbr else "—")
    _lane("ECRH absorption", ecrh, None,
          f"{ecrh['absorption_pct']:.1f}%" if ecrh else "—")
    _lane("Disruption detection", real_shots.get("disruption") if real_shots else None, "passes",
          f"recall={real_shots['disruption']['recall']:.0%}" if real_shots and "disruption" in real_shots else "—")
    _lane("HIL sub-ms", hil, "sub_ms",
          f"P50 = {hil['p50_us']:.1f} μs" if hil else "—")
    _lane("FreeGS analytic", freegs, "passes",
          f"ψ NRMSE = {freegs['overall_psi_nrmse']:.3f}" if freegs else "—")
    _lane("FNO EUROfusion", {"passes": surrogates and surrogates.get("fno_rel_l2_mean") is not None and surrogates["fno_rel_l2_mean"] < 0.10} if surrogates else None, "passes",
          f"rel_L2 = {surrogates['fno_rel_l2_mean']:.4f}" if surrogates and surrogates.get("fno_rel_l2_mean") is not None else "—")
    sections.append("")

    # ── Footer ──
    sections.append("""---

*All benchmarks run on the environment listed above.
Artifact-based lanes load pre-computed JSON from `artifacts/` and `weights/`.
Timings are wall-clock and may vary between machines.
Re-run with `python validation/collect_results.py` to reproduce.*
""")

    return "\n".join(sections)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SCPN Fusion Core benchmark results")
    parser.add_argument("--quick", action="store_true", help="Run reduced iterations for faster turnaround")
    args = parser.parse_args()

    print("=" * 60)
    print("SCPN Fusion Core — Benchmark Results Collector")
    print("=" * 60)

    t_start = time.perf_counter()
    hw = _hw_header()

    # Live benchmarks (8)
    print("\n[1/14] HIL Control Loop")
    hil = _safe_run("hil", run_hil, args.quick)

    print("[2/14] Disruption Ensemble")
    disruption = _safe_run("disruption", run_disruption, args.quick)

    print("[3/14] Q-10 Operating Point Scan")
    q10 = _safe_run("q10", run_q10)

    print("[4/14] 3-Group Tritium Breeding")
    tbr = _safe_run("tbr", run_tbr)

    print("[5/14] ECRH Absorption")
    ecrh = _safe_run("ecrh", run_ecrh)

    print("[6/14] 3D Force Balance")
    fb3d = _safe_run("fb3d", run_force_balance_3d)

    print("[7/14] Pretrained Surrogates")
    surrogates = _safe_run("surrogates", run_surrogates)

    print("[8/14] Neural Equilibrium")
    neural_eq = _safe_run("neural_eq", run_neural_eq)

    # Artifact-based lanes (6)
    print("[9/14] QLKNN Transport Surrogate")
    qlknn = _safe_run("qlknn", load_qlknn_metrics)

    print("[10/14] Real-Shot Validation")
    real_shots = _safe_run("real_shots", load_real_shot_validation)

    print("[11/14] Confinement ITPA Dashboard")
    confinement = _safe_run("confinement", load_confinement_itpa)

    print("[12/14] Disturbance Rejection")
    disturbance = _safe_run("disturbance_rej", load_disturbance_rejection)

    print("[13/14] FreeGS Benchmark")
    freegs = _safe_run("freegs", load_freegs_benchmark)

    print("[14/14] Disruption Threshold Sweep")
    threshold_sweep = _safe_run("threshold", load_disruption_threshold)

    elapsed = time.perf_counter() - t_start

    print(f"\nGenerating {RESULTS_PATH.name}...")
    md = generate_results_md(
        hw=hw,
        hil=hil,
        disruption=disruption,
        q10=q10,
        tbr=tbr,
        ecrh=ecrh,
        fb3d=fb3d,
        surrogates=surrogates,
        neural_eq=neural_eq,
        qlknn=qlknn,
        real_shots=real_shots,
        confinement=confinement,
        disturbance=disturbance,
        freegs=freegs,
        threshold_sweep=threshold_sweep,
        elapsed_s=elapsed,
    )
    RESULTS_PATH.write_text(md, encoding="utf-8")
    print(f"  -> {RESULTS_PATH} written ({len(md)} bytes)")

    all_lanes = [hil, disruption, q10, tbr, ecrh, fb3d, surrogates, neural_eq,
                 qlknn, real_shots, confinement, disturbance, freegs, threshold_sweep]
    n_ok = sum(1 for x in all_lanes if x is not None)
    print(f"\nDone: {n_ok}/14 benchmarks succeeded in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
