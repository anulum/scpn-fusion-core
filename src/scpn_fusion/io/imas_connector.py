# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""IMAS/IDS adapter pattern for digital-twin state interchange."""

from __future__ import annotations

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

    _coerce_int("time_slice.index", time_slice.get("index", 0), minimum=0)
    _coerce_finite_real("time_slice.time_s", time_slice.get("time_s", 0.0), minimum=0.0)

    axis = equilibrium.get("axis")
    if not isinstance(axis, Mapping):
        raise ValueError("equilibrium.axis must be a mapping.")
    _coerce_finite_real("equilibrium.axis.r_m", axis.get("r_m", 0.0))
    _coerce_finite_real("equilibrium.axis.z_m", axis.get("z_m", 0.0))
    _coerce_int("equilibrium.islands_px", equilibrium.get("islands_px", 0), minimum=0)

    _coerce_finite_real("performance.final_reward", performance.get("final_reward", 0.0))
    _coerce_finite_real(
        "performance.reward_mean_last_50",
        performance.get("reward_mean_last_50", 0.0),
    )
    _coerce_finite_real(
        "performance.final_avg_temp_keV",
        performance.get("final_avg_temp_keV", 0.0),
    )


def digital_twin_summary_to_ids(
    summary: Mapping[str, Any],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Map internal digital-twin summary into IDS-like payload."""
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
