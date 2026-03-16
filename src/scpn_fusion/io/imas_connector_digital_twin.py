# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector Digital Twin Mappings
"""IDS mapping/validation for digital-twin summary and state payloads."""

from __future__ import annotations

import math
from typing import Any, Mapping

from scpn_fusion.io.imas_connector_common import (
    REQUIRED_DIGITAL_TWIN_STATE_KEYS,
    REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS,
    REQUIRED_IDS_KEYS,
    _coerce_finite_real,
    _coerce_int,
    _coerce_profiles_1d,
    _missing_required_keys,
)


def validate_ids_payload(payload: Mapping[str, Any]) -> None:
    if isinstance(payload, bool) or not isinstance(payload, Mapping):
        raise ValueError("IDS payload must be a mapping.")
    missing = [key for key in REQUIRED_IDS_KEYS if key not in payload]
    if missing:
        raise ValueError(f"IDS payload missing keys: {missing}")
    if payload.get("schema") != "ids_equilibrium_v1":
        raise ValueError("Unsupported IDS schema.")
    machine = payload.get("machine")
    if not isinstance(machine, str) or not machine.strip():
        raise ValueError("IDS machine must be a non-empty string.")
    if "shot" in payload:
        _coerce_int("shot", payload["shot"], minimum=0)
    if "run" in payload:
        _coerce_int("run", payload["run"], minimum=0)
    if not isinstance(payload.get("time_slice"), Mapping):
        raise ValueError("IDS time_slice must be a mapping.")
    if not isinstance(payload.get("equilibrium"), Mapping):
        raise ValueError("IDS equilibrium must be a mapping.")
    if not isinstance(payload.get("performance"), Mapping):
        raise ValueError("IDS performance must be a mapping.")

    time_slice = payload["time_slice"]
    equilibrium = payload["equilibrium"]
    performance = payload["performance"]

    missing_time_slice = _missing_required_keys(time_slice, ("index", "time_s"))
    if missing_time_slice:
        raise ValueError(f"IDS time_slice missing keys: {missing_time_slice}")
    _coerce_int("time_slice.index", time_slice.get("index", 0), minimum=0)
    time_s = _coerce_finite_real("time_slice.time_s", time_slice.get("time_s", 0.0), minimum=0.0)
    time_ms = time_s * 1.0e3
    rounded_ms = round(time_ms)
    if not math.isclose(time_ms, rounded_ms, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("time_slice.time_s must map to an integer millisecond count.")

    missing_equilibrium = _missing_required_keys(equilibrium, ("axis", "islands_px"))
    if missing_equilibrium:
        raise ValueError(f"IDS equilibrium missing keys: {missing_equilibrium}")
    axis = equilibrium.get("axis")
    if not isinstance(axis, Mapping):
        raise ValueError("equilibrium.axis must be a mapping.")
    missing_axis = _missing_required_keys(axis, ("r_m", "z_m"))
    if missing_axis:
        raise ValueError(f"IDS equilibrium.axis missing keys: {missing_axis}")
    _coerce_finite_real("equilibrium.axis.r_m", axis.get("r_m", 0.0))
    _coerce_finite_real("equilibrium.axis.z_m", axis.get("z_m", 0.0))
    _coerce_int("equilibrium.islands_px", equilibrium.get("islands_px", 0), minimum=0)

    missing_performance = _missing_required_keys(
        performance,
        ("final_reward", "reward_mean_last_50", "final_avg_temp_keV"),
    )
    if missing_performance:
        raise ValueError(f"IDS performance missing keys: {missing_performance}")
    _coerce_finite_real("performance.final_reward", performance.get("final_reward", 0.0))
    _coerce_finite_real(
        "performance.reward_mean_last_50",
        performance.get("reward_mean_last_50", 0.0),
    )
    _coerce_finite_real(
        "performance.final_avg_temp_keV",
        performance.get("final_avg_temp_keV", 0.0),
    )
    if "profiles_1d" in equilibrium:
        profiles = equilibrium.get("profiles_1d")
        if not isinstance(profiles, Mapping):
            raise ValueError("equilibrium.profiles_1d must be a mapping.")
        _coerce_profiles_1d(profiles, name="equilibrium.profiles_1d")


def digital_twin_summary_to_ids(
    summary: Mapping[str, Any],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Map internal digital-twin summary into IDS-like payload."""
    if isinstance(summary, bool) or not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping.")
    missing_summary = _missing_required_keys(summary, REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS)
    if missing_summary:
        raise ValueError(f"digital twin summary missing keys: {missing_summary}")

    machine_name = str(machine).strip()
    if not machine_name:
        raise ValueError("machine must be a non-empty string.")
    shot_i = _coerce_int("shot", shot, minimum=0)
    run_i = _coerce_int("run", run, minimum=0)
    steps = _coerce_int("summary.steps", summary.get("steps", 0), minimum=0)
    axis_r = _coerce_finite_real("summary.final_axis_r", summary.get("final_axis_r", 0.0))
    axis_z = _coerce_finite_real("summary.final_axis_z", summary.get("final_axis_z", 0.0))
    islands = _coerce_int(
        "summary.final_islands_px",
        summary.get("final_islands_px", 0),
        minimum=0,
    )
    final_reward = _coerce_finite_real(
        "summary.final_reward",
        summary.get("final_reward", 0.0),
    )
    reward_mean = _coerce_finite_real(
        "summary.reward_mean_last_50",
        summary.get("reward_mean_last_50", 0.0),
    )
    final_avg_temp = _coerce_finite_real(
        "summary.final_avg_temp",
        summary.get("final_avg_temp", 0.0),
    )

    return {
        "schema": "ids_equilibrium_v1",
        "machine": machine_name,
        "shot": shot_i,
        "run": run_i,
        "time_slice": {
            "index": 0,
            "time_s": steps * 1.0e-3,
        },
        "equilibrium": {
            "axis": {
                "r_m": axis_r,
                "z_m": axis_z,
            },
            "islands_px": islands,
        },
        "performance": {
            "final_reward": final_reward,
            "reward_mean_last_50": reward_mean,
            "final_avg_temp_keV": final_avg_temp,
        },
    }


def digital_twin_state_to_ids(
    state: Mapping[str, Any],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Map detailed digital-twin state + profiles into IDS-like payload."""
    if isinstance(state, bool) or not isinstance(state, Mapping):
        raise ValueError("state must be a mapping.")
    missing_state = _missing_required_keys(state, REQUIRED_DIGITAL_TWIN_STATE_KEYS)
    if missing_state:
        raise ValueError(f"digital twin state missing keys: {missing_state}")

    payload = digital_twin_summary_to_ids(
        state,
        machine=machine,
        shot=shot,
        run=run,
    )
    payload["equilibrium"]["profiles_1d"] = _coerce_profiles_1d(
        state,
        name="state",
    )
    validate_ids_payload(payload)
    return payload


def ids_to_digital_twin_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map IDS-like payload back to internal digital-twin summary shape."""
    validate_ids_payload(payload)
    equilibrium = payload["equilibrium"]
    performance = payload["performance"]
    axis = equilibrium.get("axis", {})
    time_slice = payload.get("time_slice", {})

    if not isinstance(axis, Mapping):
        raise ValueError("equilibrium.axis must be a mapping.")

    time_s = _coerce_finite_real("time_slice.time_s", time_slice.get("time_s", 0.0), minimum=0.0)
    time_ms = time_s * 1.0e3
    rounded_ms = round(time_ms)
    if not math.isclose(time_ms, rounded_ms, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("time_slice.time_s must map to an integer millisecond count.")

    _coerce_int("time_slice.index", time_slice.get("index", 0), minimum=0)
    steps = int(rounded_ms)
    final_axis_r = _coerce_finite_real("equilibrium.axis.r_m", axis.get("r_m", 0.0))
    final_axis_z = _coerce_finite_real("equilibrium.axis.z_m", axis.get("z_m", 0.0))
    final_islands_px = _coerce_int(
        "equilibrium.islands_px", equilibrium.get("islands_px", 0), minimum=0
    )
    final_reward = _coerce_finite_real(
        "performance.final_reward", performance.get("final_reward", 0.0)
    )
    reward_mean_last_50 = _coerce_finite_real(
        "performance.reward_mean_last_50",
        performance.get("reward_mean_last_50", 0.0),
    )
    final_avg_temp = _coerce_finite_real(
        "performance.final_avg_temp_keV",
        performance.get("final_avg_temp_keV", 0.0),
    )

    return {
        "steps": steps,
        "final_axis_r": final_axis_r,
        "final_axis_z": final_axis_z,
        "final_islands_px": final_islands_px,
        "final_reward": final_reward,
        "reward_mean_last_50": reward_mean_last_50,
        "final_avg_temp": final_avg_temp,
    }


def ids_to_digital_twin_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map IDS payload back to detailed digital-twin state with optional profiles."""
    summary = ids_to_digital_twin_summary(payload)
    equilibrium = payload.get("equilibrium", {})
    if not isinstance(equilibrium, Mapping):
        raise ValueError("IDS equilibrium must be a mapping.")

    profiles = equilibrium.get("profiles_1d")
    if profiles is None:
        return summary
    if not isinstance(profiles, Mapping):
        raise ValueError("equilibrium.profiles_1d must be a mapping.")

    out = dict(summary)
    out.update(_coerce_profiles_1d(profiles, name="equilibrium.profiles_1d"))
    return out


__all__ = [
    "validate_ids_payload",
    "digital_twin_summary_to_ids",
    "digital_twin_state_to_ids",
    "ids_to_digital_twin_summary",
    "ids_to_digital_twin_state",
]
