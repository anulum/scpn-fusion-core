# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Replay Contracts
# ----------------------------------------------------------------------
"""Real-shot replay contract extracted from ``disruption_contracts`` monolith."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_contract_primitives import (
    require_1d_array,
    require_fraction,
    require_int,
    require_positive_float,
)
from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.control.replay_pipeline import (
    apply_actuator_lag,
    load_replay_pipeline_config,
    preprocess_sensor_trace,
)
from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection


def run_real_shot_replay(
    *,
    shot_data: dict[str, Any],
    rl_agent: FusionAIAgent,
    base_tbr: float = 1.15,
    risk_threshold: float = 0.65,
    spi_trigger_risk: float = 0.80,
    window_size: int = 128,
    replay_pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a real tokamak shot through the disruption mitigation pipeline."""
    _ = require_positive_float("base_tbr", base_tbr)
    _ = rl_agent
    risk_threshold = require_fraction("risk_threshold", risk_threshold)
    spi_trigger_risk = require_fraction("spi_trigger_risk", spi_trigger_risk)
    if spi_trigger_risk < risk_threshold:
        raise ValueError("spi_trigger_risk must be >= risk_threshold.")
    window_size = require_int("window_size", window_size, 8)
    pipeline_cfg = load_replay_pipeline_config(replay_pipeline)

    time_s = require_1d_array(
        "shot_data.time_s",
        shot_data.get("time_s", []),
        minimum_size=16,
    )
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("shot_data.time_s must be strictly increasing.")

    n_steps = int(time_s.size)
    if window_size > n_steps:
        raise ValueError(
            f"window_size must be <= number of samples ({n_steps}), got {window_size}."
        )
    n1_amp = require_1d_array(
        "shot_data.n1_amp",
        shot_data.get("n1_amp", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    n2_amp = require_1d_array(
        "shot_data.n2_amp",
        shot_data.get("n2_amp", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    dBdt = require_1d_array(
        "shot_data.dBdt_gauss_per_s",
        shot_data.get("dBdt_gauss_per_s", np.zeros(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    beta_N = require_1d_array(
        "shot_data.beta_N",
        shot_data.get("beta_N", np.ones(n_steps, dtype=np.float64) * 2.0),
        expected_size=n_steps,
    )
    Ip_MA = require_1d_array(
        "shot_data.Ip_MA",
        shot_data.get("Ip_MA", np.ones(n_steps, dtype=np.float64)),
        expected_size=n_steps,
    )
    n3_amp = n2_amp * 0.4
    signal_proc, mean_abs_sensor_delta = preprocess_sensor_trace(
        dBdt.astype(np.float64),
        config=pipeline_cfg,
    )

    is_disruption = bool(shot_data.get("is_disruption", False))
    disruption_time_idx = int(shot_data.get("disruption_time_idx", -1))
    if disruption_time_idx >= n_steps:
        raise ValueError(
            f"disruption_time_idx must be < number of samples ({n_steps}), got {disruption_time_idx}."
        )
    if disruption_time_idx < -1:
        raise ValueError("disruption_time_idx must be >= -1.")

    risk_raw_series = np.zeros(n_steps, dtype=np.float64)
    for t in range(min(window_size, n_steps), n_steps):
        signal_window = signal_proc[t - window_size : t] if t >= window_size else signal_proc[:t]
        if signal_window.size < 8:
            continue

        toroidal = {
            "toroidal_n1_amp": float(np.clip(n1_amp[t], 0, 10)),
            "toroidal_n2_amp": float(np.clip(n2_amp[t], 0, 10)),
            "toroidal_n3_amp": float(np.clip(n3_amp[t], 0, 10)),
            "toroidal_asymmetry_index": float(
                np.sqrt(n1_amp[t] ** 2 + n2_amp[t] ** 2 + n3_amp[t] ** 2)
            ),
            "toroidal_radial_spread": float(0.02 + 0.05 * n1_amp[t]),
        }

        risk_raw = float(
            np.clip(
                predict_disruption_risk(signal_window, toroidal),
                0.0,
                1.0,
            )
        )
        risk_raw_series[t] = risk_raw

    dt_nominal_s = 3e-3
    if time_s.size > 1:
        dt_nominal_s = float(np.median(np.diff(time_s)))
    risk_series, mean_abs_actuator_lag = apply_actuator_lag(
        risk_raw_series,
        dt_s=max(dt_nominal_s, 1e-6),
        config=pipeline_cfg,
    )

    alarm_series = risk_series > risk_threshold
    first_alarm_idx = int(np.argmax(alarm_series)) if bool(np.any(alarm_series)) else -1
    spi_mask = risk_series > spi_trigger_risk
    spi_triggered = bool(np.any(spi_mask))
    spi_trigger_idx = int(np.argmax(spi_mask)) if spi_triggered else -1

    if spi_triggered:
        pre_energy_mj = float(np.clip(300 + 50 * np.mean(beta_N), 200, 500))
        pre_current_ma = float(np.clip(np.mean(Ip_MA), 0.5, 20))
        risk_now = float(np.clip(risk_series[spi_trigger_idx], 0.0, 1.0))
        disturbance_now = float(np.clip(risk_now + 0.10 * np.mean(n1_amp), 0.0, 1.0))
        cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=risk_now,
            disturbance=disturbance_now,
            action_bias=0.0,
        )
        neon_mol = float(cocktail["neon_quantity_mol"])
        argon_mol = float(cocktail["argon_quantity_mol"])
        xenon_mol = float(cocktail["xenon_quantity_mol"])
        total_impurity_mol = float(cocktail["total_quantity_mol"])

        spi = ShatteredPelletInjection(
            Plasma_Energy_MJ=pre_energy_mj,
            Plasma_Current_MA=pre_current_ma,
        )
        _, _, _, spi_diag = spi.trigger_mitigation(
            neon_quantity_mol=neon_mol,
            argon_quantity_mol=argon_mol,
            xenon_quantity_mol=xenon_mol,
            return_diagnostics=True,
            duration_s=0.03,
            dt_s=5e-5,
            verbose=False,
        )
        tau_cq_ms = float(spi_diag["tau_cq_ms_mean"])
        final_current = float(spi_diag["final_current_MA"])
        z_eff = float(spi_diag["z_eff"])
    else:
        tau_cq_ms = 0.0
        final_current = float(np.mean(Ip_MA))
        z_eff = 1.0
        neon_mol = 0.0
        argon_mol = 0.0
        xenon_mol = 0.0
        total_impurity_mol = 0.0

    detection_lead_ms = -1.0
    if is_disruption and disruption_time_idx > 0 and first_alarm_idx > 0:
        dt_arr = time_s if time_s.size > 0 else np.arange(n_steps) * 0.001
        detection_lead_ms = float((dt_arr[disruption_time_idx] - dt_arr[first_alarm_idx]) * 1000)

    prevented = False
    if is_disruption:
        if spi_triggered and spi_trigger_idx < disruption_time_idx:
            post_risk = np.mean(risk_series[spi_trigger_idx : min(spi_trigger_idx + 50, n_steps)])
            prevented = bool(post_risk < 0.88 and tau_cq_ms > 0)
    else:
        prevented = not spi_triggered

    return {
        "n_steps": n_steps,
        "is_disruption": is_disruption,
        "disruption_time_idx": disruption_time_idx,
        "first_alarm_idx": first_alarm_idx,
        "spi_triggered": spi_triggered,
        "spi_trigger_idx": spi_trigger_idx,
        "detection_lead_ms": round(detection_lead_ms, 1),
        "prevented": prevented,
        "neon_mol": round(neon_mol, 4),
        "argon_mol": round(argon_mol, 4),
        "xenon_mol": round(xenon_mol, 4),
        "total_impurity_mol": round(total_impurity_mol, 4),
        "tau_cq_ms": round(tau_cq_ms, 2),
        "final_current_MA": round(final_current, 3),
        "z_eff": round(z_eff, 2),
        "peak_risk": round(float(np.max(risk_series)), 4),
        "peak_risk_raw": round(float(np.max(risk_raw_series)), 4),
        "mean_risk": round(float(np.mean(risk_series)), 4),
        "pipeline": {
            "config": dict(pipeline_cfg),
            "sensor_preprocess_enabled": bool(pipeline_cfg["sensor_preprocess_enabled"]),
            "actuator_lag_enabled": bool(pipeline_cfg["actuator_lag_enabled"]),
            "mean_abs_sensor_delta": round(float(mean_abs_sensor_delta), 6),
            "mean_abs_actuator_lag": round(float(mean_abs_actuator_lag), 6),
        },
        "risk_series": risk_series.tolist(),
    }


__all__ = ["run_real_shot_replay"]
