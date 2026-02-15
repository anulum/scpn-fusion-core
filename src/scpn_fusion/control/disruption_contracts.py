# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Mitigation Contracts
# ----------------------------------------------------------------------
"""Reusable disruption-mitigation physics contracts for Task 5 lanes."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


def require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def require_fraction(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def synthetic_disruption_signal(
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
    disturbance: float,
    seed_shift: int,
) -> dict[str, float]:
    n_steps = 240
    dt = 1.25e-4
    t = np.arange(n_steps, dtype=np.float64) * dt
    source = float(neon_quantity_mol) * np.exp(-t / 0.004)
    sink_rate = 120.0 + 35.0 * float(disturbance)
    n_imp = np.zeros_like(t)
    n_imp[0] = source[0]
    for i in range(1, n_steps):
        dn = source[i] - sink_rate * n_imp[i - 1]
        n_imp[i] = max(0.0, n_imp[i - 1] + dt * dn)
    weighted = float(np.mean(n_imp[-80:]))
    zeff_eff = float(1.05 + 42.0 * weighted + 0.22 * disturbance)
    rad_mw = float((24.0 + 95.0 * weighted) * (1.0 + 0.15 * disturbance))
    return {
        "zeff_eff": zeff_eff,
        "impurity_radiation_mw": rad_mw,
        "impurity_decay_tau_ms": float(1e3 / max(sink_rate, 1e-9)),
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
    tau_cq_s = float(spi_diag["tau_cq_ms_mean"]) * 1e-3
    final_current_ma = float(spi_diag["final_current_MA"])
    quench_fraction = float(
        np.clip((pre_current_ma - final_current_ma) / pre_current_ma, 0.0, 1.0)
    )
    mitigation_strength = float(
        np.clip(
            1.60 * neon_quantity_mol
            + 0.03 * float(spi_diag["z_eff"])
            + 0.10 * (rl_action_bias + 1.0),
            0.08,
            0.95,
        )
    )
    impurity = impurity_transport_response(
        neon_quantity_mol=neon_quantity_mol,
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
                toroidal["toroidal_asymmetry_index"] * (1.0 - 0.72 * mitigation_strength),
            )
        ),
        "toroidal_radial_spread": float(
            max(0.0, toroidal["toroidal_radial_spread"] * (1.0 - 0.60 * mitigation_strength))
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


__all__ = [
    "impurity_transport_response",
    "mcnp_lite_tbr",
    "post_disruption_halo_runaway",
    "require_fraction",
    "require_int",
    "run_disruption_episode",
    "synthetic_disruption_signal",
]
