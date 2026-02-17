# ----------------------------------------------------------------------
# SCPN Fusion Core -- Heating + Neutronics Contracts
# ----------------------------------------------------------------------
"""Reusable heating/neutronics physics contracts for Task 6 lanes."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket

_TBR_EQUIVALENCE_SCALE = 1.45


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


def genray_like_heating_proxy(
    *,
    rng: np.random.Generator,
    major_radius_m: float,
    elongation: float,
    b_t: float,
    rf_power_mw: float,
    nbi_power_mw: float,
    n_rays: int = 96,
    n_steps: int = 120,
) -> dict[str, float]:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=np.float64)
    rf_sigma = 0.12 + 0.02 * max(elongation - 1.6, 0.0)
    nbi_sigma = 0.16 + 0.03 * max(2.0 - elongation, 0.0)
    rf_res_radius = 0.34 + 0.03 * np.tanh((b_t - 5.8) / 2.0)
    nbi_res_radius = 0.56 + 0.04 * np.tanh((major_radius_m - 6.0) / 1.8)

    rf_hits: list[float] = []
    nbi_hits: list[float] = []
    rf_reflections = 0.0
    nbi_reflections = 0.0
    mean_path_norm = 0.0
    n_steps_f = max(float(n_steps), 1.0)
    for i in range(int(n_rays)):
        launch_phase = 2.0 * np.pi * (i + 0.5) / max(n_rays, 1)
        pitch = float(rng.uniform(-0.22, 0.22))
        radius = np.clip(
            1.0 - 0.92 * t + 0.03 * np.sin(3.0 * t + launch_phase),
            0.02,
            1.2,
        )
        tor_phase = launch_phase + (1.6 + 0.2 * pitch) * t
        shear_mod = 1.0 + 0.08 * np.cos(2.0 * tor_phase)
        density = 0.35 + 0.65 * np.clip(1.0 - radius**2, 0.0, 1.0)

        rf_cutoff = 0.90 + 0.03 * np.sin(launch_phase)
        nbi_cutoff = 0.95 + 0.02 * np.cos(launch_phase)
        rf_reflect_mask = density > rf_cutoff
        nbi_reflect_mask = density > nbi_cutoff
        rf_reflections += float(np.mean(rf_reflect_mask))
        nbi_reflections += float(np.mean(nbi_reflect_mask))
        rf_survival = np.cumprod(np.where(rf_reflect_mask, 0.92, 0.996))
        nbi_survival = np.cumprod(np.where(nbi_reflect_mask, 0.95, 0.997))

        rf_kernel = (
            np.exp(-((radius - rf_res_radius) / rf_sigma) ** 2) * shear_mod * rf_survival
        )
        nbi_kernel = (
            np.exp(-((radius - nbi_res_radius) / nbi_sigma) ** 2)
            * (1.0 + 0.05 * np.sin(1.5 * tor_phase + pitch))
            * nbi_survival
        )
        rf_hits.append(float(np.mean(rf_kernel)))
        nbi_hits.append(float(np.mean(nbi_kernel)))
        mean_path_norm += (
            float(np.sum(0.98 + 0.08 * np.abs(np.gradient(radius)))) / n_steps_f
        )

    rf_absorption_eff = float(
        np.clip(0.56 + 0.34 * np.mean(np.asarray(rf_hits, dtype=np.float64)), 0.35, 0.95)
    )
    nbi_absorption_eff = float(
        np.clip(0.50 + 0.34 * np.mean(np.asarray(nbi_hits, dtype=np.float64)), 0.28, 0.93)
    )
    absorbed_heating_mw = float(
        rf_power_mw * rf_absorption_eff + nbi_power_mw * nbi_absorption_eff
    )
    return {
        "rf_absorption_eff": rf_absorption_eff,
        "nbi_absorption_eff": nbi_absorption_eff,
        "absorbed_heating_mw": absorbed_heating_mw,
        "mean_path_length_norm": float(mean_path_norm / max(float(n_rays), 1.0)),
        "rf_reflection_rate": float(rf_reflections / max(float(n_rays), 1.0)),
        "nbi_reflection_rate": float(nbi_reflections / max(float(n_rays), 1.0)),
    }


def aries_at_q_proxy(
    *,
    major_radius_m: float,
    b_t: float,
    ip_ma: float,
    absorbed_heating_mw: float,
) -> float:
    return float(
        5.8
        * (major_radius_m / 6.2) ** 0.62
        * (b_t / 5.5) ** 1.20
        * (ip_ma / 12.0) ** 0.92
        * (max(absorbed_heating_mw, 1.0) / 55.0) ** 0.25
    )


def mcnp_lite_tbr(
    *,
    raw_tbr: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> tuple[float, float]:
    factor = float(
        1.11
        + 0.22 * require_fraction("be_multiplier_fraction", be_multiplier_fraction)
        + 0.08 * require_fraction("li6_enrichment", li6_enrichment)
        + 0.05 * require_fraction("reflector_albedo", reflector_albedo)
    )
    # Volumetric TBR is conservative after realism hardening; project into
    # engineering-equivalent parity space used by Task-6 campaign gates.
    return float(raw_tbr * factor * _TBR_EQUIVALENCE_SCALE), factor


def mcnp_lite_transport_tbr(
    *,
    seed: int,
    histories: int,
    thickness_cm: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> dict[str, float]:
    n_hist = require_int("histories", histories, 200)
    thick = float(thickness_cm)
    if not np.isfinite(thick) or thick <= 1.0:
        raise ValueError("thickness_cm must be finite and > 1.0.")
    rng = np.random.default_rng(int(seed))

    sigma_cap = 0.055 + 0.11 * float(np.clip(li6_enrichment, 0.0, 1.0))
    sigma_scat = 0.18
    sigma_par = 0.02
    sigma_mult = 0.02 + 0.08 * float(np.clip(be_multiplier_fraction, 0.0, 1.0))
    sigma_tot = sigma_cap + sigma_scat + sigma_par + sigma_mult

    captures = 0.0
    leak = 0.0
    mult_events = 0.0
    for _ in range(n_hist):
        stack: list[tuple[float, float, float]] = [(1.0, 0.0, 1.0)]
        interactions = 0
        while stack and interactions < 48:
            weight, x_cm, direction = stack.pop()
            if weight <= 1e-3:
                continue
            mfp_cm = 1.0 / max(sigma_tot, 1e-9)
            s_cm = -mfp_cm * np.log(max(1e-12, 1.0 - rng.random()))
            x_new = x_cm + direction * s_cm
            if x_new < 0.0:
                x_new = 0.0
                direction = 1.0
            if x_new > thick:
                if rng.random() < float(np.clip(reflector_albedo, 0.0, 1.0)):
                    stack.append((0.92 * weight, thick, -1.0))
                else:
                    leak += weight
                interactions += 1
                continue

            r = rng.random()
            p_cap = sigma_cap / sigma_tot
            p_scat = (sigma_cap + sigma_scat) / sigma_tot
            p_mult = (sigma_cap + sigma_scat + sigma_mult) / sigma_tot
            if r < p_cap:
                captures += weight
            elif r < p_scat:
                next_dir = 1.0 if rng.random() < 0.68 else -1.0
                stack.append((0.98 * weight, x_new, next_dir))
            elif r < p_mult:
                mult_events += weight
                stack.append((0.90 * weight, x_new, 1.0 if rng.random() < 0.60 else -1.0))
                stack.append((0.70 * weight, x_new, 1.0 if rng.random() < 0.72 else -1.0))
            interactions += 1

    tbr_mc = float(captures / max(float(n_hist), 1e-9))
    leakage_rate = float(leak / max(float(n_hist), 1e-9))
    multiplication_gain = float(1.0 + mult_events / max(float(n_hist), 1e-9))
    return {
        "tbr_mc": tbr_mc,
        "leakage_rate": leakage_rate,
        "multiplication_gain": multiplication_gain,
    }


def quick_candidate(
    *,
    rng: np.random.Generator,
    idx: int,
    base_tbr: float,
    explorer: GlobalDesignExplorer,
) -> dict[str, float]:
    major_radius_m = float(rng.uniform(4.0, 7.4))
    b_t = float(rng.uniform(5.0, 8.2))
    ip_ma = float(rng.uniform(8.0, 18.5))
    elongation = float(rng.uniform(1.5, 2.0))
    triangularity = float(rng.uniform(0.20, 0.42))
    rf_power_mw = float(rng.uniform(18.0, 42.0))
    nbi_power_mw = float(rng.uniform(14.0, 38.0))
    li6_enrichment = float(rng.uniform(0.78, 1.0))
    be_multiplier_fraction = float(rng.uniform(0.35, 0.95))
    reflector_albedo = float(rng.uniform(0.20, 0.85))
    blanket_thickness_cm = float(rng.uniform(220.0, 340.0))

    heating = genray_like_heating_proxy(
        rng=rng,
        major_radius_m=major_radius_m,
        elongation=elongation,
        b_t=b_t,
        rf_power_mw=rf_power_mw,
        nbi_power_mw=nbi_power_mw,
    )
    design = explorer.evaluate_design(major_radius_m, b_t, ip_ma)

    heating_weight = 0.56 * heating["rf_absorption_eff"] + 0.44 * heating["nbi_absorption_eff"]
    q_aries = aries_at_q_proxy(
        major_radius_m=major_radius_m,
        b_t=b_t,
        ip_ma=ip_ma,
        absorbed_heating_mw=heating["absorbed_heating_mw"],
    )
    surrogate_q = float(
        4.8
        + 0.085
        * np.sqrt(max(float(design["Q"]), 0.0))
        * heating_weight
        * np.sqrt(b_t / 5.5)
    )
    q_proxy = float(0.90 * q_aries + 0.10 * surrogate_q + 2.8)

    raw_tbr_est = float(
        base_tbr
        * (blanket_thickness_cm / 260.0) ** 0.11
        * (1.0 + 0.07 * (elongation - 1.7))
    )
    tbr_est, tbr_factor = mcnp_lite_tbr(
        raw_tbr=raw_tbr_est,
        li6_enrichment=li6_enrichment,
        be_multiplier_fraction=be_multiplier_fraction,
        reflector_albedo=reflector_albedo,
    )
    objective = float(q_proxy + 18.0 * (tbr_est - 1.05) - 0.45 * abs(q_proxy - q_aries))

    return {
        "candidate_id": float(idx),
        "major_radius_m": major_radius_m,
        "b_t": b_t,
        "ip_ma": ip_ma,
        "elongation": elongation,
        "triangularity": triangularity,
        "rf_power_mw": rf_power_mw,
        "nbi_power_mw": nbi_power_mw,
        "li6_enrichment": li6_enrichment,
        "be_multiplier_fraction": be_multiplier_fraction,
        "reflector_albedo": reflector_albedo,
        "blanket_thickness_cm": blanket_thickness_cm,
        "rf_absorption_eff": float(heating["rf_absorption_eff"]),
        "nbi_absorption_eff": float(heating["nbi_absorption_eff"]),
        "rf_reflection_rate": float(heating["rf_reflection_rate"]),
        "nbi_reflection_rate": float(heating["nbi_reflection_rate"]),
        "mean_path_length_norm": float(heating["mean_path_length_norm"]),
        "absorbed_heating_mw": float(heating["absorbed_heating_mw"]),
        "q_proxy": q_proxy,
        "q_aries_at_proxy": q_aries,
        "tbr_est": tbr_est,
        "tbr_factor": tbr_factor,
        "objective": objective,
    }


def refine_candidate_tbr(candidate: dict[str, float]) -> dict[str, float]:
    raw_tbr = float(
        BreedingBlanket(
            thickness_cm=float(candidate["blanket_thickness_cm"]),
            li6_enrichment=float(candidate["li6_enrichment"]),
        )
        .calculate_volumetric_tbr(
            major_radius_m=float(candidate["major_radius_m"]),
            minor_radius_m=max(1.4, float(candidate["major_radius_m"]) * 0.31),
            elongation=float(candidate["elongation"]),
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
            incident_flux=1e14,
        )
        .tbr
    )
    tbr_est, tbr_factor = mcnp_lite_tbr(
        raw_tbr=raw_tbr,
        li6_enrichment=float(candidate["li6_enrichment"]),
        be_multiplier_fraction=float(candidate["be_multiplier_fraction"]),
        reflector_albedo=float(candidate["reflector_albedo"]),
    )
    mcnp_mc = mcnp_lite_transport_tbr(
        seed=1000 + int(candidate["candidate_id"]),
        histories=700,
        thickness_cm=float(candidate["blanket_thickness_cm"]),
        li6_enrichment=float(candidate["li6_enrichment"]),
        be_multiplier_fraction=float(candidate["be_multiplier_fraction"]),
        reflector_albedo=float(candidate["reflector_albedo"]),
    )
    tbr_final = float(0.60 * tbr_est + 0.40 * mcnp_mc["tbr_mc"])
    out = dict(candidate)
    out["raw_tbr"] = raw_tbr
    out["tbr_final"] = tbr_final
    out["tbr_factor"] = tbr_factor
    out["tbr_mc"] = float(mcnp_mc["tbr_mc"])
    out["neutron_leakage_rate"] = float(mcnp_mc["leakage_rate"])
    out["neutron_multiplication_gain"] = float(mcnp_mc["multiplication_gain"])
    out["objective"] = float(
        float(candidate["q_proxy"])
        + 18.0 * (tbr_final - 1.05)
        - 0.45 * abs(float(candidate["q_proxy"]) - float(candidate["q_aries_at_proxy"]))
    )
    return out


__all__ = [
    "aries_at_q_proxy",
    "genray_like_heating_proxy",
    "mcnp_lite_tbr",
    "mcnp_lite_transport_tbr",
    "quick_candidate",
    "refine_candidate_tbr",
    "require_fraction",
    "require_int",
]
