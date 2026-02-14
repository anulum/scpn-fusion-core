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
from typing import Any, Mapping


REQUIRED_IDS_KEYS = (
    "schema",
    "machine",
    "time_slice",
    "equilibrium",
    "performance",
)


def validate_ids_payload(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_IDS_KEYS if key not in payload]
    if missing:
        raise ValueError(f"IDS payload missing keys: {missing}")
    if payload.get("schema") != "ids_equilibrium_v1":
        raise ValueError("Unsupported IDS schema.")
    if not isinstance(payload.get("time_slice"), Mapping):
        raise ValueError("IDS time_slice must be a mapping.")
    if not isinstance(payload.get("equilibrium"), Mapping):
        raise ValueError("IDS equilibrium must be a mapping.")
    if not isinstance(payload.get("performance"), Mapping):
        raise ValueError("IDS performance must be a mapping.")


def digital_twin_summary_to_ids(
    summary: Mapping[str, Any],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Map internal digital-twin summary into IDS-like payload."""
    return {
        "schema": "ids_equilibrium_v1",
        "machine": str(machine),
        "shot": int(shot),
        "run": int(run),
        "time_slice": {
            "index": 0,
            "time_s": float(summary.get("steps", 0)) * 1.0e-3,
        },
        "equilibrium": {
            "axis": {
                "r_m": float(summary.get("final_axis_r", 0.0)),
                "z_m": float(summary.get("final_axis_z", 0.0)),
            },
            "islands_px": int(summary.get("final_islands_px", 0)),
        },
        "performance": {
            "final_reward": float(summary.get("final_reward", 0.0)),
            "reward_mean_last_50": float(summary.get("reward_mean_last_50", 0.0)),
            "final_avg_temp_keV": float(summary.get("final_avg_temp", 0.0)),
        },
    }


def ids_to_digital_twin_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map IDS-like payload back to internal digital-twin summary shape."""
    validate_ids_payload(payload)
    equilibrium = payload["equilibrium"]
    performance = payload["performance"]
    axis = equilibrium.get("axis", {})
    time_slice = payload.get("time_slice", {})

    time_s = float(time_slice.get("time_s", 0.0))
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_slice.time_s must be finite and >= 0.")
    steps = int(round(time_s * 1.0e3))

    return {
        "steps": steps,
        "final_axis_r": float(axis.get("r_m", 0.0)),
        "final_axis_z": float(axis.get("z_m", 0.0)),
        "final_islands_px": int(equilibrium.get("islands_px", 0)),
        "final_reward": float(performance.get("final_reward", 0.0)),
        "reward_mean_last_50": float(performance.get("reward_mean_last_50", 0.0)),
        "final_avg_temp": float(performance.get("final_avg_temp_keV", 0.0)),
    }
