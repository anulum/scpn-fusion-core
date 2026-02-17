# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Mitigation Contracts
# ----------------------------------------------------------------------
"""Reusable disruption-mitigation physics contracts for Task 5 lanes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


def require_finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def require_fraction(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def require_1d_array(
    name: str,
    value: Any,
    *,
    minimum_size: int = 1,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size < minimum_size:
        raise ValueError(f"{name} must have at least {minimum_size} samples.")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} samples (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def synthetic_disruption_signal(
    *,
    rng: np.random.Generator,
    disturbance: float,
    window: int = 220,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    t = np.linspace(0.0, 1.0, int(window), dtype=np.float64)
    base = 0.68 + 0.10 * np.sin(2.0 * np.pi * 2.4 * t + rng.uniform(-0.4, 0.4))
    elm = disturbance * (0.30 * np.exp(-(((t - 0.78) / 0.10) ** 2)))
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


def mcnp_lite_tbr(
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


def impurity_transport_response(
    *,
    neon_quantity_mol: float,
    argon_quantity_mol: float,
    xenon_quantity_mol: float,
    disturbance: float,
    seed_shift: int,
) -> dict[str, float]:
    n_steps = 240
    dt = 1.25e-4
    t = np.arange(n_steps, dtype=np.float64) * dt
    neon = max(float(neon_quantity_mol), 0.0)
    argon = max(float(argon_quantity_mol), 0.0)
    xenon = max(float(xenon_quantity_mol), 0.0)
    source_strength = float(1.00 * neon + 1.35 * argon + 1.90 * xenon)
    source = source_strength * np.exp(-t / 0.004)
    sink_rate = 120.0 + 35.0 * float(disturbance) + 45.0 * (argon + 1.2 * xenon)
    n_imp = np.zeros_like(t)
    n_imp[0] = source[0]
    for i in range(1, n_steps):
        dn = source[i] - sink_rate * n_imp[i - 1]
        n_imp[i] = max(0.0, n_imp[i - 1] + dt * dn)
    weighted = float(np.mean(n_imp[-80:]))
    cocktail_zeff = ShatteredPelletInjection.estimate_z_eff_cocktail(
        neon_quantity_mol=neon,
        argon_quantity_mol=argon,
        xenon_quantity_mol=xenon,
    )
    zeff_eff = float(
        np.clip(
            0.65 * cocktail_zeff + 0.35 * (1.05 + 42.0 * weighted + 0.22 * disturbance),
            1.0,
            12.0,
        )
    )
    rad_mw = float(
        (24.0 + 95.0 * weighted)
        * (1.0 + 0.15 * disturbance)
        * (1.0 + 0.035 * max(cocktail_zeff - 1.0, 0.0))
    )
    return {
        "zeff_eff": zeff_eff,
        "impurity_radiation_mw": rad_mw,
        "impurity_decay_tau_ms": float(1e3 / max(sink_rate, 1e-9)),
        "total_impurity_mol": float(neon + argon + xenon),
        "seed_shift": float(seed_shift),
    }


def post_disruption_halo_runaway(
    *,
    pre_current_ma: float,
    tau_cq_s: float,
    disturbance: float,
    mitigation_strength: float,
    zeff_eff: float,
) -> dict[str, float]:
    dt = 1.0e-4
    steps = 320
    ip = float(pre_current_ma)
    halo = 0.0
    runaway = 0.0
    halo_hist: list[float] = []
    re_hist: list[float] = []

    tau_ip = max(float(tau_cq_s), 0.004)
    tau_halo = 0.006 + 0.008 * disturbance
    for _ in range(steps):
        d_ip = -ip / tau_ip
        ip = max(0.0, ip + dt * d_ip)
        e_norm = float(np.clip((-d_ip) / max(pre_current_ma / 0.01, 1e-9), 0.0, 8.0))
        halo_drive = 0.28 * abs(d_ip) * (1.0 + 0.4 * disturbance)
        halo = max(0.0, halo + dt * (halo_drive - halo / max(tau_halo, 1e-4)))

        re_source = max(e_norm - 1.0, 0.0) * (1.0 + 0.7 * disturbance)
        impurity_damping = (0.14 + 0.015 * zeff_eff) * (1.0 + 0.9 * mitigation_strength)
        runaway = max(
            0.0,
            runaway
            + dt
            * (
                0.22 * re_source
                + 0.48 * runaway * max(e_norm - 0.6, 0.0)
                - impurity_damping * runaway
            ),
        )
        halo_hist.append(float(halo))
        re_hist.append(float(runaway))

    halo_arr = np.asarray(halo_hist, dtype=np.float64)
    re_arr = np.asarray(re_hist, dtype=np.float64)
    return {
        "halo_current_ma": float(np.percentile(halo_arr, 95)),
        "runaway_beam_ma": float(np.percentile(re_arr, 95)),
        "halo_peak_ma": float(np.max(halo_arr)),
        "runaway_peak_ma": float(np.max(re_arr)),
    }


def run_disruption_episode(
    *,
    rng: np.random.Generator,
    rl_agent: FusionAIAgent,
    base_tbr: float,
    explorer: GlobalDesignExplorer,
) -> dict[str, float | bool]:
    disturbance = float(rng.uniform(0.0, 1.0))
    pre_energy_mj = float(rng.uniform(240.0, 420.0))
    pre_current_ma = float(rng.uniform(11.0, 16.5))
    signal, toroidal = synthetic_disruption_signal(rng=rng, disturbance=disturbance)
    risk_before = float(predict_disruption_risk(signal, toroidal))

    rl_state = rl_agent.discretize_state(12.0 * risk_before, 4.0 * disturbance)
    rl_action = int(rl_agent.choose_action(rl_state, rng))
    rl_action_bias = {-1: -1.0, 0: 0.0, 1: 1.0}[rl_action - 1]

    cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=risk_before,
        disturbance=disturbance,
        action_bias=rl_action_bias,
    )
    neon_quantity_mol = float(cocktail["neon_quantity_mol"])
    argon_quantity_mol = float(cocktail["argon_quantity_mol"])
    xenon_quantity_mol = float(cocktail["xenon_quantity_mol"])
    total_impurity_mol = float(cocktail["total_quantity_mol"])
    spi = ShatteredPelletInjection(
        Plasma_Energy_MJ=pre_energy_mj,
        Plasma_Current_MA=pre_current_ma,
    )
    _, _, _, spi_diag = spi.trigger_mitigation(
        neon_quantity_mol=neon_quantity_mol,
        argon_quantity_mol=argon_quantity_mol,
        xenon_quantity_mol=xenon_quantity_mol,
        return_diagnostics=True,
        duration_s=0.03,
        dt_s=5e-5,
        verbose=False,
    )
    tau_cq_s = float(spi_diag["tau_cq_ms_mean"]) * 1e-3
    final_current_ma = float(spi_diag["final_current_MA"])
    quench_fraction = float(
        np.clip((pre_current_ma - final_current_ma) / pre_current_ma, 0.0, 1.0)
    )
    mitigation_strength = float(
        np.clip(
            1.60 * total_impurity_mol
            + 0.03 * float(spi_diag["z_eff"])
            + 0.10 * (rl_action_bias + 1.0),
            0.08,
            0.95,
        )
    )
    impurity = impurity_transport_response(
        neon_quantity_mol=neon_quantity_mol,
        argon_quantity_mol=argon_quantity_mol,
        xenon_quantity_mol=xenon_quantity_mol,
        disturbance=disturbance,
        seed_shift=rl_action,
    )
    zeff = float(0.6 * float(spi_diag["z_eff"]) + 0.4 * impurity["zeff_eff"])
    impurity_radiation_mw = float(
        impurity["impurity_radiation_mw"] * (0.72 + 0.28 * quench_fraction)
    )
    post_dyn = post_disruption_halo_runaway(
        pre_current_ma=pre_current_ma,
        tau_cq_s=tau_cq_s,
        disturbance=disturbance,
        mitigation_strength=mitigation_strength,
        zeff_eff=zeff,
    )
    halo_current_ma = float(post_dyn["halo_current_ma"])
    runaway_beam_ma = float(post_dyn["runaway_beam_ma"])
    post_toroidal = {
        "toroidal_n1_amp": float(
            max(0.0, toroidal["toroidal_n1_amp"] * (1.0 - 0.75 * mitigation_strength))
        ),
        "toroidal_n2_amp": float(
            max(0.0, toroidal["toroidal_n2_amp"] * (1.0 - 0.70 * mitigation_strength))
        ),
        "toroidal_n3_amp": float(
            max(0.0, toroidal["toroidal_n3_amp"] * (1.0 - 0.65 * mitigation_strength))
        ),
        "toroidal_asymmetry_index": float(
            max(
                0.0,
                toroidal["toroidal_asymmetry_index"]
                * (1.0 - 0.72 * mitigation_strength),
            )
        ),
        "toroidal_radial_spread": float(
            max(
                0.0,
                toroidal["toroidal_radial_spread"] * (1.0 - 0.60 * mitigation_strength),
            )
        ),
    }
    post_signal = np.clip(signal * (1.0 - 0.60 * mitigation_strength), 0.01, None)
    risk_after_model = float(predict_disruption_risk(post_signal, post_toroidal))
    risk_after = float(
        np.clip(
            0.45 * risk_after_model
            + 0.55
            * (risk_before * (1.0 - 0.80 * mitigation_strength) + 0.03 * disturbance),
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
        7.5 + 0.10 * np.sqrt(max(float(design["Q"]), 0.0)) * (1.0 - 0.25 * disturbance)
    )
    li6_enrichment = float(rng.uniform(0.85, 1.0))
    be_multiplier_fraction = float(rng.uniform(0.35, 0.95))
    reflector_albedo = float(rng.uniform(0.30, 0.90))
    tbr_proxy, _ = mcnp_lite_tbr(
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
    next_state = rl_agent.discretize_state(
        12.0 * risk_after, 4.0 * max(0.0, disturbance - mitigation_strength)
    )
    rl_agent.learn(rl_state, rl_action, next_state, reward)

    return {
        "disturbance": disturbance,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "neon_quantity_mol": neon_quantity_mol,
        "argon_quantity_mol": argon_quantity_mol,
        "xenon_quantity_mol": xenon_quantity_mol,
        "total_impurity_mol": total_impurity_mol,
        "zeff": zeff,
        "impurity_decay_tau_ms": float(impurity["impurity_decay_tau_ms"]),
        "halo_current_ma": halo_current_ma,
        "halo_peak_ma": float(post_dyn["halo_peak_ma"]),
        "runaway_beam_ma": runaway_beam_ma,
        "runaway_peak_ma": float(post_dyn["runaway_peak_ma"]),
        "impurity_radiation_mw": impurity_radiation_mw,
        "wall_damage_index": wall_damage_index,
        "q_proxy": q_proxy,
        "tbr_proxy": tbr_proxy,
        "no_wall_damage": no_wall_damage,
        "objective_success": objective_success,
        "prevented": prevented,
    }


def run_real_shot_replay(
    *,
    shot_data: dict[str, Any],
    rl_agent: FusionAIAgent,
    base_tbr: float = 1.15,
    risk_threshold: float = 0.65,
    spi_trigger_risk: float = 0.80,
    window_size: int = 128,
) -> dict[str, Any]:
    """Replay a real tokamak shot through the disruption mitigation pipeline.

    Parameters
    ----------
    shot_data : dict
        NPZ-loaded shot data with keys: time_s, Ip_MA, BT_T, beta_N, q95,
        ne_1e19, n1_amp, n2_amp, locked_mode_amp, dBdt_gauss_per_s,
        vertical_position_m, is_disruption, disruption_time_idx, disruption_type.
    rl_agent : FusionAIAgent
        Reinforcement learning agent for SPI action selection.
    base_tbr : float
        Baseline tritium breeding ratio.
    risk_threshold : float
        Risk level above which alarm is raised.
    spi_trigger_risk : float
        Risk level above which SPI mitigation is triggered.
    window_size : int
        Sliding window size for disruption predictor.

    Returns
    -------
    dict with replay results including risk time-series and mitigation outcomes.
    """
    risk_threshold = require_fraction("risk_threshold", risk_threshold)
    spi_trigger_risk = require_fraction("spi_trigger_risk", spi_trigger_risk)
    if spi_trigger_risk < risk_threshold:
        raise ValueError("spi_trigger_risk must be >= risk_threshold.")
    window_size = require_int("window_size", window_size, 8)

    time_s = require_1d_array("shot_data.time_s", shot_data.get("time_s", []), minimum_size=16)
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("shot_data.time_s must be strictly increasing.")

    n_steps = int(time_s.size)
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
    n3_amp = n2_amp * 0.4  # approximate n3 from n2

    is_disruption = bool(shot_data.get("is_disruption", False))
    disruption_time_idx = int(shot_data.get("disruption_time_idx", -1))
    if disruption_time_idx >= n_steps:
        raise ValueError(
            f"disruption_time_idx must be < number of samples ({n_steps}), got {disruption_time_idx}."
        )
    if disruption_time_idx < -1:
        raise ValueError("disruption_time_idx must be >= -1.")

    risk_series = np.zeros(n_steps, dtype=np.float64)
    alarm_series = np.zeros(n_steps, dtype=bool)
    first_alarm_idx = -1
    spi_triggered = False
    spi_trigger_idx = -1

    # Slide through the shot
    for t in range(min(window_size, n_steps), n_steps):
        # Build signal window from dB/dt
        signal_window = dBdt[t - window_size : t] if t >= window_size else dBdt[:t]
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

        risk = float(predict_disruption_risk(signal_window, toroidal))
        risk_series[t] = risk

        if risk > risk_threshold:
            alarm_series[t] = True
            if first_alarm_idx < 0:
                first_alarm_idx = t

        # SPI mitigation trigger
        if risk > spi_trigger_risk and not spi_triggered:
            spi_triggered = True
            spi_trigger_idx = t

    # Compute mitigation outcome
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

    # Detection timing
    detection_lead_ms = -1.0
    if is_disruption and disruption_time_idx > 0 and first_alarm_idx > 0:
        dt_arr = time_s if time_s.size > 0 else np.arange(n_steps) * 0.001
        detection_lead_ms = float(
            (dt_arr[disruption_time_idx] - dt_arr[first_alarm_idx]) * 1000
        )

    # Prevention determination
    prevented = False
    if is_disruption:
        if spi_triggered and spi_trigger_idx < disruption_time_idx:
            # SPI triggered before disruption — check if it mitigated
            post_risk = np.mean(
                risk_series[spi_trigger_idx : min(spi_trigger_idx + 50, n_steps)]
            )
            prevented = bool(post_risk < 0.88 and tau_cq_ms > 0)
    else:
        prevented = not spi_triggered  # For safe shots, not triggering SPI = correct

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
        "mean_risk": round(float(np.mean(risk_series)), 4),
        "risk_series": risk_series.tolist(),
    }


__all__ = [
    "impurity_transport_response",
    "mcnp_lite_tbr",
    "post_disruption_halo_runaway",
    "require_fraction",
    "require_int",
    "run_disruption_episode",
    "run_real_shot_replay",
    "synthetic_disruption_signal",
]
