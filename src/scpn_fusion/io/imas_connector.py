# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""IMAS/IDS adapter pattern for digital-twin state interchange."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numbers
from typing import Any, Mapping


REQUIRED_IDS_KEYS = (
    "schema",
    "machine",
    "time_slice",
    "equilibrium",
    "performance",
)

REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS = (
    "steps",
    "final_islands_px",
    "final_reward",
    "reward_mean_last_50",
    "final_avg_temp",
)

REQUIRED_PROFILE_1D_KEYS = (
    "rho_norm",
    "electron_temp_keV",
    "electron_density_1e20_m3",
)

REQUIRED_DIGITAL_TWIN_STATE_KEYS = REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS + REQUIRED_PROFILE_1D_KEYS


def _missing_required_keys(mapping: Mapping[str, Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in mapping]


def _coerce_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _coerce_finite_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _coerce_finite_real_sequence(
    name: str,
    value: Any,
    *,
    minimum_len: int = 1,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_increasing: bool = False,
) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of finite numbers.")
    parsed: list[float] = []
    for idx, item in enumerate(value):
        parsed_item = _coerce_finite_real(f"{name}[{idx}]", item, minimum=minimum)
        if maximum is not None and parsed_item > maximum:
            raise ValueError(f"{name}[{idx}] must be <= {maximum}.")
        parsed.append(parsed_item)
    if len(parsed) < minimum_len:
        raise ValueError(f"{name} must contain at least {minimum_len} values.")
    if strictly_increasing:
        for idx in range(1, len(parsed)):
            if parsed[idx] <= parsed[idx - 1]:
                raise ValueError(f"{name} must be strictly increasing.")
    return parsed


def _coerce_profiles_1d(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, list[float]]:
    missing = _missing_required_keys(payload, REQUIRED_PROFILE_1D_KEYS)
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")

    rho = _coerce_finite_real_sequence(
        f"{name}.rho_norm",
        payload.get("rho_norm"),
        minimum_len=2,
        minimum=0.0,
        maximum=1.0,
        strictly_increasing=True,
    )
    temp = _coerce_finite_real_sequence(
        f"{name}.electron_temp_keV",
        payload.get("electron_temp_keV"),
        minimum_len=2,
        minimum=0.0,
    )
    dens = _coerce_finite_real_sequence(
        f"{name}.electron_density_1e20_m3",
        payload.get("electron_density_1e20_m3"),
        minimum_len=2,
        minimum=0.0,
    )
    n = len(rho)
    if len(temp) != n:
        raise ValueError(
            f"{name}.electron_temp_keV length must match {name}.rho_norm."
        )
    if len(dens) != n:
        raise ValueError(
            f"{name}.electron_density_1e20_m3 length must match {name}.rho_norm."
        )

    return {
        "rho_norm": rho,
        "electron_temp_keV": temp,
        "electron_density_1e20_m3": dens,
    }


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
    final_islands_px = _coerce_int("equilibrium.islands_px", equilibrium.get("islands_px", 0), minimum=0)
    final_reward = _coerce_finite_real("performance.final_reward", performance.get("final_reward", 0.0))
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
