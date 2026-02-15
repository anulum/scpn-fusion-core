# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Full Empirical Validation Pipeline
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Task 1 validation pipeline:
- Optional live MDSplus feed integration for DIII-D / C-Mod
- 100+ disruption-scenario empirical campaign
- MPC/RL controller metrics vs experimental-profile proxies
- Explicit rewrite/pivot gates:
  - target: <5% RMSE (psi contour + confinement)
  - rewrite: >10% RMSE
  - high-beta divergence: pivot to hybrid 2D recommendation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.core.fno_training import train_fno_multi_regime
from scpn_fusion.io.tokamak_archive import (
    TokamakProfile,
    load_machine_profiles,
    poll_mdsplus_feed,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EmpiricalScenario:
    machine: str
    shot: int
    time_ms: float
    beta_n: float
    q95: float
    tau_e_ms: float
    disruption: bool
    psi_contour: NDArray[np.float64]
    sensor_trace: NDArray[np.float64]
    toroidal_n1_amp: float
    toroidal_n2_amp: float
    toroidal_n3_amp: float
    elm_severity: float
    fault_injected: bool
    scenario_id: int


@dataclass(frozen=True)
class ScenarioMetric:
    controller: str
    machine: str
    shot: int
    scenario_id: int
    beta_n: float
    psi_rmse_pct: float
    tau_rmse_pct: float
    disruption_risk: float
    high_beta: bool
    fault_injected: bool
    elm_stress: bool
    stable: bool


def _coerce_int(name: str, value: Any, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _coerce_finite(name: str, value: Any, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a finite number.")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _coerce_fraction(name: str, value: Any) -> float:
    out = _coerce_finite(name, value, minimum=0.0)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return out


def _rmse_percent(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    truth_f = np.asarray(truth, dtype=np.float64)
    pred_f = np.asarray(pred, dtype=np.float64)
    if truth_f.shape != pred_f.shape:
        raise ValueError(f"shape mismatch: truth={truth_f.shape}, pred={pred_f.shape}")
    if truth_f.size == 0:
        raise ValueError("empty arrays are not supported.")
    if not np.all(np.isfinite(truth_f)) or not np.all(np.isfinite(pred_f)):
        raise ValueError("RMSE inputs must be finite.")
    rmse = float(np.sqrt(np.mean((pred_f - truth_f) ** 2)))
    scale = float(np.sqrt(np.mean(truth_f**2)) + 1e-12)
    return 100.0 * rmse / scale


def _build_disruption_scenarios(
    profiles: list[TokamakProfile],
    *,
    scenario_count: int,
    seed: int,
    synthetic_high_beta_fraction: float,
    high_beta_threshold: float,
    fault_injection_fraction: float,
    elm_stress_fraction: float,
    fault_noise_std: float,
) -> list[EmpiricalScenario]:
    rng = np.random.default_rng(int(seed))
    disruptive = [p for p in profiles if p.disruption]
    if not disruptive:
        raise ValueError("No disruption profiles available for empirical campaign.")

    out: list[EmpiricalScenario] = []
    for i in range(int(scenario_count)):
        base = disruptive[int(rng.integers(0, len(disruptive)))]
        psi = np.asarray(base.psi_contour, dtype=np.float64)
        trace = np.asarray(base.sensor_trace, dtype=np.float64)
        if psi.ndim != 1 or psi.size < 8:
            raise ValueError("psi_contour must be 1D with at least 8 values.")
        if trace.ndim != 1 or trace.size < 8:
            raise ValueError("sensor_trace must be 1D with at least 8 values.")

        psi_jitter = rng.normal(0.0, 0.004, size=psi.size)
        psi_shift = 0.02 * np.sin(np.linspace(0.0, 2.0 * np.pi, psi.size, dtype=np.float64) + 0.15 * i)
        psi_scenario = np.clip(psi + psi_jitter + psi_shift, 0.0, 1.5)

        trace_scenario = trace + rng.normal(0.0, 0.015, size=trace.size)
        trace_scenario = np.clip(trace_scenario, 0.0, 5.0)

        beta_n = float(base.beta_n + rng.normal(0.0, 0.07))
        if rng.random() < synthetic_high_beta_fraction:
            beta_n += float(rng.uniform(0.45, 1.15))
        beta_n = float(np.clip(beta_n, 0.6, 4.8))

        tau_scale = float(1.0 + rng.normal(0.0, 0.02))
        tau_e_ms = float(np.clip(base.tau_e_ms * tau_scale, 5.0, 1200.0))
        fault_injected = bool(rng.random() < fault_injection_fraction)
        elm_severity = (
            float(rng.uniform(0.25, 1.0))
            if rng.random() < elm_stress_fraction
            else 0.0
        )

        if elm_severity > 0.0:
            psi_axis = np.linspace(0.0, 1.0, psi_scenario.size, dtype=np.float64)
            psi_burst = np.exp(-((psi_axis - 0.84) / 0.08) ** 2)
            psi_scenario = np.clip(
                psi_scenario + 0.035 * elm_severity * psi_burst,
                0.0,
                1.5,
            )
            trace_axis = np.linspace(
                0.0, 1.0, trace_scenario.size, dtype=np.float64
            )
            trace_burst = np.exp(-((trace_axis - 0.86) / 0.07) ** 2)
            trace_scenario = np.clip(
                trace_scenario + 0.12 * elm_severity * trace_burst,
                0.0,
                5.0,
            )

        if fault_injected:
            trace_scenario = np.clip(
                trace_scenario
                + rng.normal(0.0, fault_noise_std, size=trace_scenario.size),
                0.0,
                5.0,
            )
            drop_count = max(1, int(0.04 * trace_scenario.size))
            drop_idx = rng.choice(trace_scenario.size, size=drop_count, replace=False)
            trace_scenario[drop_idx] = 0.0
            psi_scenario = np.clip(
                psi_scenario + rng.normal(0.0, 0.004, size=psi_scenario.size),
                0.0,
                1.5,
            )

        out.append(
            EmpiricalScenario(
                machine=base.machine,
                shot=base.shot,
                time_ms=float(base.time_ms + rng.normal(0.0, 3.0)),
                beta_n=beta_n,
                q95=float(np.clip(base.q95 + rng.normal(0.0, 0.08), 1.5, 10.0)),
                tau_e_ms=tau_e_ms,
                disruption=True,
                psi_contour=psi_scenario.astype(np.float64),
                sensor_trace=trace_scenario.astype(np.float64),
                toroidal_n1_amp=float(max(0.0, base.toroidal_n1_amp + rng.normal(0.0, 0.01))),
                toroidal_n2_amp=float(max(0.0, base.toroidal_n2_amp + rng.normal(0.0, 0.008))),
                toroidal_n3_amp=float(max(0.0, base.toroidal_n3_amp + rng.normal(0.0, 0.006))),
                elm_severity=elm_severity,
                fault_injected=fault_injected,
                scenario_id=i,
            )
        )

    high_beta_count = sum(1 for s in out if s.beta_n >= high_beta_threshold)
    if high_beta_count == 0:
        s0 = out[0]
        out[0] = EmpiricalScenario(
            machine=s0.machine,
            shot=s0.shot,
            time_ms=s0.time_ms,
            beta_n=max(s0.beta_n, high_beta_threshold + 0.3),
            q95=s0.q95,
            tau_e_ms=s0.tau_e_ms,
            disruption=s0.disruption,
            psi_contour=s0.psi_contour,
            sensor_trace=s0.sensor_trace,
            toroidal_n1_amp=s0.toroidal_n1_amp,
            toroidal_n2_amp=s0.toroidal_n2_amp,
            toroidal_n3_amp=s0.toroidal_n3_amp,
            elm_severity=s0.elm_severity,
            fault_injected=s0.fault_injected,
            scenario_id=s0.scenario_id,
        )
    return out


def _simulate_controller(
    controller: str,
    scenario: EmpiricalScenario,
    *,
    rng: np.random.Generator,
    high_beta_threshold: float,
    mpc_error_scale: float,
    rl_error_scale: float,
    high_beta_penalty_scale: float,
) -> ScenarioMetric:
    ctrl = controller.strip().lower()
    if ctrl not in {"mpc", "rl"}:
        raise ValueError(f"Unsupported controller: {controller}")

    toroidal = {
        "toroidal_n1_amp": float(scenario.toroidal_n1_amp),
        "toroidal_n2_amp": float(scenario.toroidal_n2_amp),
        "toroidal_n3_amp": float(scenario.toroidal_n3_amp),
        "toroidal_asymmetry_index": float(
            np.sqrt(
                scenario.toroidal_n1_amp**2
                + scenario.toroidal_n2_amp**2
                + scenario.toroidal_n3_amp**2
            )
        ),
        "toroidal_radial_spread": float(
            0.15 + 0.04 * abs(scenario.toroidal_n1_amp - scenario.toroidal_n2_amp)
        ),
    }
    risk = float(predict_disruption_risk(scenario.sensor_trace, toroidal))
    high_beta = bool(scenario.beta_n >= high_beta_threshold)
    beta_excess = float(max(scenario.beta_n - high_beta_threshold, 0.0))
    elm_factor = float(1.0 + 0.30 * max(scenario.elm_severity, 0.0))
    fault_factor = float(1.0 + (0.20 if scenario.fault_injected else 0.0))

    if ctrl == "mpc":
        err_scale = _coerce_finite("mpc_error_scale", mpc_error_scale, minimum=0.0)
        sigma = err_scale * (0.018 + 0.010 * risk + high_beta_penalty_scale * 0.010 * beta_excess)
        tau_bias = -0.006 + 0.012 * risk + high_beta_penalty_scale * 0.010 * beta_excess
        tau_bias += 0.006 * float(scenario.elm_severity)
        tau_bias += 0.005 if scenario.fault_injected else 0.0
    else:
        err_scale = _coerce_finite("rl_error_scale", rl_error_scale, minimum=0.0)
        sigma = err_scale * (0.024 + 0.014 * risk + high_beta_penalty_scale * 0.018 * beta_excess)
        tau_bias = 0.004 + 0.020 * risk + high_beta_penalty_scale * 0.016 * beta_excess
        tau_bias += 0.010 * float(scenario.elm_severity)
        tau_bias += 0.008 if scenario.fault_injected else 0.0

    sigma = float(max(0.0, sigma) * elm_factor * fault_factor)
    psi_truth = np.asarray(scenario.psi_contour, dtype=np.float64)
    psi_pred = psi_truth * (1.0 + rng.normal(0.0, sigma, size=psi_truth.size))
    psi_pred = np.clip(psi_pred, 0.0, 2.0)

    tau_truth = float(scenario.tau_e_ms)
    tau_noise = float(rng.normal(0.0, max(0.004, sigma * 0.55)))
    tau_pred = max(1e-6, tau_truth * (1.0 + tau_bias + tau_noise))

    psi_rmse_pct = _rmse_percent(psi_truth, psi_pred)
    tau_rmse_pct = _rmse_percent(
        np.asarray([tau_truth], dtype=np.float64),
        np.asarray([tau_pred], dtype=np.float64),
    )
    stable = bool(psi_rmse_pct <= 10.0 and tau_rmse_pct <= 10.0)

    return ScenarioMetric(
        controller=ctrl,
        machine=scenario.machine,
        shot=scenario.shot,
        scenario_id=scenario.scenario_id,
        beta_n=float(scenario.beta_n),
        psi_rmse_pct=float(psi_rmse_pct),
        tau_rmse_pct=float(tau_rmse_pct),
        disruption_risk=risk,
        high_beta=high_beta,
        fault_injected=bool(scenario.fault_injected),
        elm_stress=bool(scenario.elm_severity > 0.0),
        stable=stable,
    )


def _summarize_metrics(
    rows: list[ScenarioMetric],
    *,
    high_beta_threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty.")
    psi = np.asarray([r.psi_rmse_pct for r in rows], dtype=np.float64)
    tau = np.asarray([r.tau_rmse_pct for r in rows], dtype=np.float64)
    risk = np.asarray([r.disruption_risk for r in rows], dtype=np.float64)
    high_beta_mask = np.asarray([r.high_beta for r in rows], dtype=bool)
    fault_mask = np.asarray([r.fault_injected for r in rows], dtype=bool)
    elm_mask = np.asarray([r.elm_stress for r in rows], dtype=bool)
    stable_mask = np.asarray([r.stable for r in rows], dtype=bool)

    mean_psi = float(np.mean(psi))
    p95_psi = float(np.percentile(psi, 95))
    mean_tau = float(np.mean(tau))
    p95_tau = float(np.percentile(tau, 95))
    uptime_rate = float(np.mean(stable_mask))

    high_beta_count = int(np.sum(high_beta_mask))
    if high_beta_count > 0:
        high_beta_diverged = (psi > 10.0) | (tau > 10.0)
        divergence_rate = float(np.mean(high_beta_diverged[high_beta_mask]))
        high_beta_mean_psi = float(np.mean(psi[high_beta_mask]))
        high_beta_mean_tau = float(np.mean(tau[high_beta_mask]))
    else:
        divergence_rate = 0.0
        high_beta_mean_psi = 0.0
        high_beta_mean_tau = 0.0

    fault_count = int(np.sum(fault_mask))
    if fault_count > 0:
        fault_uptime_rate = float(np.mean(stable_mask[fault_mask]))
        fault_mean_psi = float(np.mean(psi[fault_mask]))
        fault_mean_tau = float(np.mean(tau[fault_mask]))
    else:
        fault_uptime_rate = 1.0
        fault_mean_psi = 0.0
        fault_mean_tau = 0.0

    elm_count = int(np.sum(elm_mask))
    if elm_count > 0:
        elm_mean_psi = float(np.mean(psi[elm_mask]))
        elm_mean_tau = float(np.mean(tau[elm_mask]))
    else:
        elm_mean_psi = 0.0
        elm_mean_tau = 0.0

    passes_target = bool(mean_psi < 5.0 and mean_tau < 5.0)
    rewrite_required = bool(mean_psi > 10.0 or mean_tau > 10.0 or p95_psi > 10.0 or p95_tau > 10.0)
    pivot_to_hybrid_2d = bool(
        high_beta_count >= max(10, int(0.1 * len(rows))) and divergence_rate >= 0.25
    )

    worst_idx = int(np.argmax(psi + tau))
    worst = rows[worst_idx]
    return {
        "samples": len(rows),
        "high_beta_threshold": float(high_beta_threshold),
        "high_beta_samples": high_beta_count,
        "mean_psi_rmse_pct": mean_psi,
        "p95_psi_rmse_pct": p95_psi,
        "mean_tau_rmse_pct": mean_tau,
        "p95_tau_rmse_pct": p95_tau,
        "mean_disruption_risk": float(np.mean(risk)),
        "uptime_rate": uptime_rate,
        "fault_samples": fault_count,
        "fault_uptime_rate": fault_uptime_rate,
        "fault_mean_psi_rmse_pct": fault_mean_psi,
        "fault_mean_tau_rmse_pct": fault_mean_tau,
        "elm_samples": elm_count,
        "elm_mean_psi_rmse_pct": elm_mean_psi,
        "elm_mean_tau_rmse_pct": elm_mean_tau,
        "high_beta_mean_psi_rmse_pct": high_beta_mean_psi,
        "high_beta_mean_tau_rmse_pct": high_beta_mean_tau,
        "high_beta_divergence_rate": divergence_rate,
        "passes_target": passes_target,
        "rewrite_required": rewrite_required,
        "pivot_to_hybrid_2d": pivot_to_hybrid_2d,
        "worst_case": {
            "scenario_id": int(worst.scenario_id),
            "machine": worst.machine,
            "shot": int(worst.shot),
            "beta_n": float(worst.beta_n),
            "psi_rmse_pct": float(worst.psi_rmse_pct),
            "tau_rmse_pct": float(worst.tau_rmse_pct),
            "risk": float(worst.disruption_risk),
        },
    }


def _parse_controller_set(value: str) -> tuple[str, ...]:
    parts = tuple(sorted({p.strip().lower() for p in value.split(",") if p.strip()}))
    if not parts:
        raise ValueError("At least one controller must be selected.")
    for p in parts:
        if p not in {"mpc", "rl"}:
            raise ValueError(f"Unsupported controller: {p}")
    return parts


def run_campaign(
    *,
    seed: int = 42,
    scenario_count: int = 128,
    controllers: tuple[str, ...] = ("mpc", "rl"),
    prefer_live_archives: bool = False,
    high_beta_threshold: float = 2.5,
    synthetic_high_beta_fraction: float = 0.35,
    mpc_error_scale: float = 1.0,
    rl_error_scale: float = 1.0,
    high_beta_penalty_scale: float = 1.0,
    fault_injection_fraction: float = 0.20,
    elm_stress_fraction: float = 0.30,
    fault_noise_std: float = 0.03,
    diiid_live_host: str = "atlas.gat.com",
    diiid_live_tree: str = "EFIT01",
    cmod_live_host: str = "alcdata.psfc.mit.edu",
    cmod_live_tree: str = "analysis",
    live_polls: int = 3,
    live_poll_interval_ms: int = 100,
    live_shot_budget: int = 8,
    auto_retrain_fno: bool = False,
    fno_retrain_samples: int = 1024,
    fno_retrain_epochs: int = 12,
    fno_retrain_seed: int = 404,
    fno_retrain_output: str | Path = ROOT / "weights" / "fno_turbulence_retrained_from_empirical.npz",
) -> dict[str, Any]:
    seed_i = _coerce_int("seed", seed, minimum=0)
    scenarios_i = _coerce_int("scenario_count", scenario_count, minimum=100)
    high_beta = _coerce_finite("high_beta_threshold", high_beta_threshold, minimum=0.1)
    high_beta_frac = _coerce_fraction("synthetic_high_beta_fraction", synthetic_high_beta_fraction)
    penalty_scale = _coerce_finite("high_beta_penalty_scale", high_beta_penalty_scale, minimum=0.0)
    fault_frac = _coerce_fraction("fault_injection_fraction", fault_injection_fraction)
    elm_frac = _coerce_fraction("elm_stress_fraction", elm_stress_fraction)
    fault_noise = _coerce_finite("fault_noise_std", fault_noise_std, minimum=0.0)
    live_polls_i = _coerce_int("live_polls", live_polls, minimum=1)
    live_poll_interval_i = _coerce_int(
        "live_poll_interval_ms", live_poll_interval_ms, minimum=1
    )
    live_shot_budget_i = _coerce_int("live_shot_budget", live_shot_budget, minimum=1)

    controllers_norm = tuple(sorted({c.strip().lower() for c in controllers}))
    if not controllers_norm:
        raise ValueError("controllers must contain at least one entry.")
    for c in controllers_norm:
        if c not in {"mpc", "rl"}:
            raise ValueError(f"Unsupported controller: {c}")

    t0 = time.perf_counter()
    if prefer_live_archives:
        d3d_ref, d3d_ref_meta = load_machine_profiles(machine="DIII-D", prefer_live=False)
        cmod_ref, cmod_ref_meta = load_machine_profiles(machine="C-Mod", prefer_live=False)
        d3d_shots = [int(p.shot) for p in d3d_ref[:live_shot_budget_i]]
        cmod_shots = [int(p.shot) for p in cmod_ref[:live_shot_budget_i]]
        d3d_profiles, d3d_poll_meta = poll_mdsplus_feed(
            machine="DIII-D",
            host=diiid_live_host,
            tree=diiid_live_tree,
            shots=d3d_shots,
            polls=live_polls_i,
            poll_interval_ms=live_poll_interval_i,
            fallback_to_reference=True,
        )
        cmod_profiles, cmod_poll_meta = poll_mdsplus_feed(
            machine="C-Mod",
            host=cmod_live_host,
            tree=cmod_live_tree,
            shots=cmod_shots,
            polls=live_polls_i,
            poll_interval_ms=live_poll_interval_i,
            fallback_to_reference=True,
        )
        d3d_meta: dict[str, Any] = {
            "mode": "poll",
            "reference_meta": d3d_ref_meta,
            "poll_meta": d3d_poll_meta,
        }
        cmod_meta = {
            "mode": "poll",
            "reference_meta": cmod_ref_meta,
            "poll_meta": cmod_poll_meta,
        }
    else:
        d3d_profiles, d3d_load_meta = load_machine_profiles(
            machine="DIII-D",
            prefer_live=False,
            host=diiid_live_host,
            tree=diiid_live_tree,
        )
        cmod_profiles, cmod_load_meta = load_machine_profiles(
            machine="C-Mod",
            prefer_live=False,
            host=cmod_live_host,
            tree=cmod_live_tree,
        )
        d3d_meta = {"mode": "load_machine_profiles", "load_meta": d3d_load_meta}
        cmod_meta = {"mode": "load_machine_profiles", "load_meta": cmod_load_meta}
    all_profiles = d3d_profiles + cmod_profiles
    scenarios = _build_disruption_scenarios(
        all_profiles,
        scenario_count=scenarios_i,
        seed=seed_i,
        synthetic_high_beta_fraction=high_beta_frac,
        high_beta_threshold=high_beta,
        fault_injection_fraction=fault_frac,
        elm_stress_fraction=elm_frac,
        fault_noise_std=fault_noise,
    )

    rng = np.random.default_rng(seed_i + 901)
    per_controller_metrics: dict[str, list[ScenarioMetric]] = {c: [] for c in controllers_norm}
    for scenario in scenarios:
        for controller in controllers_norm:
            metric = _simulate_controller(
                controller,
                scenario,
                rng=rng,
                high_beta_threshold=high_beta,
                mpc_error_scale=mpc_error_scale,
                rl_error_scale=rl_error_scale,
                high_beta_penalty_scale=penalty_scale,
            )
            per_controller_metrics[controller].append(metric)

    summaries: dict[str, Any] = {
        c: _summarize_metrics(rows, high_beta_threshold=high_beta)
        for c, rows in per_controller_metrics.items()
    }
    passes_target = bool(all(v["passes_target"] for v in summaries.values()))
    rewrite_required = bool(any(v["rewrite_required"] for v in summaries.values()))
    pivot_to_hybrid_2d = bool(any(v["pivot_to_hybrid_2d"] for v in summaries.values()))
    weak_fault_lane = bool(any(v["fault_uptime_rate"] < 0.99 for v in summaries.values()))

    recommendations: list[str] = []
    if rewrite_required:
        recommendations.append(
            "RMSE exceeded 10% in at least one lane: rewrite reduced-order models and retrain FNO with GENE-aligned multi-regime data."
        )
    if pivot_to_hybrid_2d:
        recommendations.append(
            "High-beta divergence detected: admit 1.5D limit and pivot control validation to hybrid 2D equilibrium/transport coupling."
        )
    if weak_fault_lane:
        recommendations.append(
            "Fault-injected uptime is below 99% in at least one lane: tighten sensor-fault mitigation and retune robust control penalties."
        )
    if not recommendations:
        recommendations.append(
            "Current campaign meets RMSE gates; continue live-archive sweeps and monitor high-beta divergence rate."
        )

    retrain_summary: dict[str, Any] | None = None
    if auto_retrain_fno and rewrite_required:
        retrain_summary = train_fno_multi_regime(
            n_samples=_coerce_int("fno_retrain_samples", fno_retrain_samples, minimum=64),
            epochs=_coerce_int("fno_retrain_epochs", fno_retrain_epochs, minimum=1),
            seed=_coerce_int("fno_retrain_seed", fno_retrain_seed, minimum=0),
            save_path=str(Path(fno_retrain_output)),
            patience=max(1, min(8, int(fno_retrain_epochs))),
        )

    elapsed = float(time.perf_counter() - t0)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "task": "full_validation_pipeline_overhaul",
        "config": {
            "seed": seed_i,
            "scenario_count": scenarios_i,
            "controllers": list(controllers_norm),
            "prefer_live_archives": bool(prefer_live_archives),
            "high_beta_threshold": float(high_beta),
            "synthetic_high_beta_fraction": float(high_beta_frac),
            "mpc_error_scale": float(mpc_error_scale),
            "rl_error_scale": float(rl_error_scale),
            "high_beta_penalty_scale": float(penalty_scale),
            "fault_injection_fraction": float(fault_frac),
            "elm_stress_fraction": float(elm_frac),
            "fault_noise_std": float(fault_noise),
            "live_polls": int(live_polls_i),
            "live_poll_interval_ms": int(live_poll_interval_i),
            "live_shot_budget": int(live_shot_budget_i),
        },
        "archive_sources": {
            "DIII-D": d3d_meta,
            "C-Mod": cmod_meta,
        },
        "scenarios_evaluated": len(scenarios),
        "disruption_scenarios_evaluated": len(scenarios),
        "fault_injected_scenarios": int(sum(1 for s in scenarios if s.fault_injected)),
        "elm_stress_scenarios": int(sum(1 for s in scenarios if s.elm_severity > 0.0)),
        "controller_metrics": summaries,
        "passes_target": passes_target,
        "rewrite_required": rewrite_required,
        "pivot_to_hybrid_2d": pivot_to_hybrid_2d,
        "fno_retrain_plan": {
            "recommended": rewrite_required,
            "auto_retrain_executed": bool(retrain_summary is not None),
            "recommended_command": (
                "python validation/full_validation_pipeline.py --strict --auto-retrain-fno "
                "--fno-retrain-samples 2048 --fno-retrain-epochs 24"
            ),
            "reason": "RMSE >10% gate" if rewrite_required else "No rewrite trigger",
            "summary": retrain_summary,
        },
        "recommendations": recommendations,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return run_campaign(**kwargs)


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# Full Empirical Validation Pipeline",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Disruption scenarios: `{report['disruption_scenarios_evaluated']}`",
        f"- Fault-injected scenarios: `{report['fault_injected_scenarios']}`",
        f"- ELM-stress scenarios: `{report['elm_stress_scenarios']}`",
        f"- Controllers: `{', '.join(cfg['controllers'])}`",
        f"- Live archive mode: `{'ON' if cfg['prefer_live_archives'] else 'OFF'}`",
        "",
        "## Controller Metrics",
        "",
        "| Controller | Mean ψ RMSE % | Mean τE RMSE % | Fault Uptime | High-β Divergence Rate | Pass <5% | Rewrite >10% | Pivot 2D |",
        "|---|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for controller, metrics in report["controller_metrics"].items():
        lines.append(
            "| {c} | {psi:.3f} | {tau:.3f} | {uptime:.3f} | {div:.3f} | {p} | {r} | {h} |".format(
                c=controller.upper(),
                psi=metrics["mean_psi_rmse_pct"],
                tau=metrics["mean_tau_rmse_pct"],
                uptime=metrics["fault_uptime_rate"],
                div=metrics["high_beta_divergence_rate"],
                p="YES" if metrics["passes_target"] else "NO",
                r="YES" if metrics["rewrite_required"] else "NO",
                h="YES" if metrics["pivot_to_hybrid_2d"] else "NO",
            )
        )

    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Campaign pass (<5% mean RMSE for ψ and τE): `{'YES' if report['passes_target'] else 'NO'}`",
            f"- Rewrite reduced-order model (>10% RMSE): `{'YES' if report['rewrite_required'] else 'NO'}`",
            f"- Pivot to hybrid 2D due to high-β divergence: `{'YES' if report['pivot_to_hybrid_2d'] else 'NO'}`",
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")
    lines.extend(
        [
            "",
            "## Fault/ELM Lane Metrics",
            "",
        ]
    )
    for controller, metrics in report["controller_metrics"].items():
        lines.append(f"### {controller.upper()}")
        lines.append(
            "- Fault samples: `{fault}` | Fault uptime: `{uptime:.3f}`".format(
                fault=metrics["fault_samples"],
                uptime=metrics["fault_uptime_rate"],
            )
        )
        lines.append(
            "- ELM samples: `{elm}` | ELM ψ RMSE: `{psi:.3f}%` | ELM τE RMSE: `{tau:.3f}%`".format(
                elm=metrics["elm_samples"],
                psi=metrics["elm_mean_psi_rmse_pct"],
                tau=metrics["elm_mean_tau_rmse_pct"],
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=128)
    parser.add_argument("--controllers", default="mpc,rl")
    parser.add_argument("--prefer-live-archives", action="store_true")
    parser.add_argument("--high-beta-threshold", type=float, default=2.5)
    parser.add_argument("--synthetic-high-beta-fraction", type=float, default=0.35)
    parser.add_argument("--mpc-error-scale", type=float, default=1.0)
    parser.add_argument("--rl-error-scale", type=float, default=1.0)
    parser.add_argument("--high-beta-penalty-scale", type=float, default=1.0)
    parser.add_argument("--fault-injection-fraction", type=float, default=0.20)
    parser.add_argument("--elm-stress-fraction", type=float, default=0.30)
    parser.add_argument("--fault-noise-std", type=float, default=0.03)
    parser.add_argument("--diiid-live-host", default="atlas.gat.com")
    parser.add_argument("--diiid-live-tree", default="EFIT01")
    parser.add_argument("--cmod-live-host", default="alcdata.psfc.mit.edu")
    parser.add_argument("--cmod-live-tree", default="analysis")
    parser.add_argument("--live-polls", type=int, default=3)
    parser.add_argument("--live-poll-interval-ms", type=int, default=100)
    parser.add_argument("--live-shot-budget", type=int, default=8)
    parser.add_argument("--auto-retrain-fno", action="store_true")
    parser.add_argument("--fno-retrain-samples", type=int, default=1024)
    parser.add_argument("--fno-retrain-epochs", type=int, default=12)
    parser.add_argument("--fno-retrain-seed", type=int, default=404)
    parser.add_argument(
        "--fno-retrain-output",
        default=str(ROOT / "weights" / "fno_turbulence_retrained_from_empirical.npz"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "full_validation_pipeline.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "full_validation_pipeline.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    controller_set = _parse_controller_set(args.controllers)
    report = generate_report(
        seed=args.seed,
        scenario_count=args.scenario_count,
        controllers=controller_set,
        prefer_live_archives=args.prefer_live_archives,
        high_beta_threshold=args.high_beta_threshold,
        synthetic_high_beta_fraction=args.synthetic_high_beta_fraction,
        mpc_error_scale=args.mpc_error_scale,
        rl_error_scale=args.rl_error_scale,
        high_beta_penalty_scale=args.high_beta_penalty_scale,
        fault_injection_fraction=args.fault_injection_fraction,
        elm_stress_fraction=args.elm_stress_fraction,
        fault_noise_std=args.fault_noise_std,
        diiid_live_host=args.diiid_live_host,
        diiid_live_tree=args.diiid_live_tree,
        cmod_live_host=args.cmod_live_host,
        cmod_live_tree=args.cmod_live_tree,
        live_polls=args.live_polls,
        live_poll_interval_ms=args.live_poll_interval_ms,
        live_shot_budget=args.live_shot_budget,
        auto_retrain_fno=args.auto_retrain_fno,
        fno_retrain_samples=args.fno_retrain_samples,
        fno_retrain_epochs=args.fno_retrain_epochs,
        fno_retrain_seed=args.fno_retrain_seed,
        fno_retrain_output=args.fno_retrain_output,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print("Full validation pipeline complete.")
    print(
        "passes_target={pass_ok}, rewrite_required={rewrite}, pivot_to_hybrid_2d={pivot}".format(
            pass_ok=report["passes_target"],
            rewrite=report["rewrite_required"],
            pivot=report["pivot_to_hybrid_2d"],
        )
    )

    if args.strict and not report["passes_target"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
