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

REQUIRED_IDS_PULSE_KEYS = (
    "schema",
    "machine",
    "shot",
    "run",
    "time_slices",
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


def _has_all_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in mapping for key in keys)


def validate_ids_payload_sequence(payloads: Sequence[Mapping[str, Any]]) -> None:
    """Validate a deterministic sequence of IDS-like payloads."""
    if isinstance(payloads, (str, bytes, bytearray)) or not isinstance(payloads, Sequence):
        raise ValueError("payloads must be a sequence of IDS payload mappings.")
    if len(payloads) == 0:
        raise ValueError("payloads must contain at least one IDS payload.")

    baseline_machine: str | None = None
    baseline_shot: int | None = None
    baseline_run: int | None = None
    prev_index: int | None = None
    prev_time_s: float | None = None

    for idx, payload in enumerate(payloads):
        if isinstance(payload, bool) or not isinstance(payload, Mapping):
            raise ValueError(f"payloads[{idx}] must be a mapping.")
        validate_ids_payload(payload)

        machine = str(payload.get("machine"))
        shot = _coerce_int(f"payloads[{idx}].shot", payload.get("shot", 0), minimum=0)
        run = _coerce_int(f"payloads[{idx}].run", payload.get("run", 0), minimum=0)
        time_slice = payload.get("time_slice")
        if not isinstance(time_slice, Mapping):
            raise ValueError(f"payloads[{idx}].time_slice must be a mapping.")
        time_index = _coerce_int(
            f"payloads[{idx}].time_slice.index",
            time_slice.get("index", 0),
            minimum=0,
        )
        time_s = _coerce_finite_real(
            f"payloads[{idx}].time_slice.time_s",
            time_slice.get("time_s", 0.0),
            minimum=0.0,
        )

        if baseline_machine is None:
            baseline_machine = machine
            baseline_shot = shot
            baseline_run = run
        else:
            if machine != baseline_machine:
                raise ValueError(
                    "All IDS payloads in sequence must share the same machine."
                )
            if shot != baseline_shot:
                raise ValueError("All IDS payloads in sequence must share the same shot.")
            if run != baseline_run:
                raise ValueError("All IDS payloads in sequence must share the same run.")

        if prev_index is not None and time_index <= prev_index:
            raise ValueError("IDS payload sequence requires strictly increasing time_slice.index.")
        if prev_time_s is not None and time_s <= prev_time_s:
            raise ValueError("IDS payload sequence requires strictly increasing time_slice.time_s.")
        prev_index = time_index
        prev_time_s = time_s


def digital_twin_history_to_ids(
    history: Sequence[Mapping[str, Any]],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> list[dict[str, Any]]:
    """Map digital-twin history snapshots to an IDS-like payload sequence."""
    if isinstance(history, (str, bytes, bytearray)) or not isinstance(history, Sequence):
        raise ValueError("history must be a sequence of digital twin snapshots.")
    if len(history) == 0:
        raise ValueError("history must contain at least one snapshot.")

    out: list[dict[str, Any]] = []
    prev_time_ms: int | None = None
    for idx, snapshot in enumerate(history):
        if isinstance(snapshot, bool) or not isinstance(snapshot, Mapping):
            raise ValueError(f"history[{idx}] must be a mapping.")
        if _has_all_keys(snapshot, REQUIRED_PROFILE_1D_KEYS):
            payload = digital_twin_state_to_ids(
                snapshot,
                machine=machine,
                shot=shot,
                run=run,
            )
        else:
            payload = digital_twin_summary_to_ids(
                snapshot,
                machine=machine,
                shot=shot,
                run=run,
            )
        time_slice = payload.get("time_slice")
        if not isinstance(time_slice, Mapping):
            raise ValueError(f"history[{idx}] produced invalid IDS time_slice mapping.")
        time_ms = round(
            _coerce_finite_real(
                f"history[{idx}].time_slice.time_s",
                time_slice.get("time_s", 0.0),
                minimum=0.0,
            )
            * 1.0e3
        )
        if prev_time_ms is not None and time_ms <= prev_time_ms:
            time_ms = prev_time_ms + 1
        payload["time_slice"]["index"] = idx
        payload["time_slice"]["time_s"] = time_ms * 1.0e-3
        prev_time_ms = time_ms
        out.append(payload)

    validate_ids_payload_sequence(out)
    return out


def ids_to_digital_twin_history(
    payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map IDS payload sequence back to digital-twin history snapshots."""
    validate_ids_payload_sequence(payloads)
    out: list[dict[str, Any]] = []
    for payload in payloads:
        equilibrium = payload.get("equilibrium", {})
        if isinstance(equilibrium, Mapping) and "profiles_1d" in equilibrium:
            out.append(ids_to_digital_twin_state(payload))
        else:
            out.append(ids_to_digital_twin_summary(payload))
    return out


def validate_ids_pulse_payload(pulse: Mapping[str, Any]) -> None:
    """Validate pulse-style IDS container with a deterministic time-slice sequence."""
    if isinstance(pulse, bool) or not isinstance(pulse, Mapping):
        raise ValueError("IDS pulse payload must be a mapping.")
    missing = _missing_required_keys(pulse, REQUIRED_IDS_PULSE_KEYS)
    if missing:
        raise ValueError(f"IDS pulse payload missing keys: {missing}")
    if pulse.get("schema") != "ids_equilibrium_pulse_v1":
        raise ValueError("Unsupported IDS pulse schema.")
    machine = pulse.get("machine")
    if not isinstance(machine, str) or not machine.strip():
        raise ValueError("IDS pulse machine must be a non-empty string.")
    shot = _coerce_int("pulse.shot", pulse.get("shot", 0), minimum=0)
    run = _coerce_int("pulse.run", pulse.get("run", 0), minimum=0)

    slices = pulse.get("time_slices")
    if isinstance(slices, (str, bytes, bytearray)) or not isinstance(slices, Sequence):
        raise ValueError("IDS pulse time_slices must be a sequence.")
    if len(slices) == 0:
        raise ValueError("IDS pulse time_slices must contain at least one payload.")

    validate_ids_payload_sequence(slices)
    for idx, payload in enumerate(slices):
        if str(payload.get("machine")) != machine.strip():
            raise ValueError(f"pulse.time_slices[{idx}] machine must match pulse.machine.")
        if _coerce_int(f"pulse.time_slices[{idx}].shot", payload.get("shot", 0), minimum=0) != shot:
            raise ValueError(f"pulse.time_slices[{idx}] shot must match pulse.shot.")
        if _coerce_int(f"pulse.time_slices[{idx}].run", payload.get("run", 0), minimum=0) != run:
            raise ValueError(f"pulse.time_slices[{idx}] run must match pulse.run.")


def digital_twin_history_to_ids_pulse(
    history: Sequence[Mapping[str, Any]],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Map digital-twin history snapshots to a pulse-style IDS container."""
    payloads = digital_twin_history_to_ids(
        history,
        machine=machine,
        shot=shot,
        run=run,
    )
    pulse = {
        "schema": "ids_equilibrium_pulse_v1",
        "machine": str(machine).strip(),
        "shot": int(shot),
        "run": int(run),
        "time_slices": payloads,
    }
    validate_ids_pulse_payload(pulse)
    return pulse


def ids_pulse_to_digital_twin_history(pulse: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map pulse-style IDS container back to digital-twin history snapshots."""
    validate_ids_pulse_payload(pulse)
    slices = pulse["time_slices"]
    if isinstance(slices, (str, bytes, bytearray)) or not isinstance(slices, Sequence):
        raise ValueError("IDS pulse time_slices must be a sequence.")
    return ids_to_digital_twin_history(slices)


# ──────────────────────────────────────────────────────────────────────
# IMAS Data Dictionary v3 compliant converters
# ──────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import GEqdsk


# Required top-level keys for IMAS DD-compliant IDS dictionaries
IMAS_DD_EQUILIBRIUM_KEYS = (
    "ids_properties",
    "time",
    "time_slice",
)
IMAS_DD_CORE_PROFILES_KEYS = (
    "ids_properties",
    "time",
    "profiles_1d",
)
IMAS_DD_SUMMARY_KEYS = (
    "ids_properties",
    "time",
    "global_quantities",
)


def geqdsk_to_imas_equilibrium(
    eq: GEqdsk,
    *,
    time_s: float = 0.0,
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Convert a GEqdsk equilibrium to an IMAS Data Dictionary ``equilibrium`` IDS.

    The output follows the IMAS DD v3 structure:
    - ``ids_properties`` with ``homogeneous_time`` and ``comment``
    - ``time`` array (single time point)
    - ``time_slice[0]`` with ``global_quantities``, ``profiles_1d``, ``profiles_2d``
    """
    if eq.nw < 2 or eq.nh < 2:
        raise ValueError("GEqdsk must have nw >= 2 and nh >= 2 for IMAS conversion.")
    if eq.psirz.size == 0:
        raise ValueError("GEqdsk psirz must be non-empty for IMAS conversion.")

    time_val = float(time_s)
    r_grid = eq.r.tolist()
    z_grid = eq.z.tolist()
    psi_norm = eq.psi_norm.tolist()

    # Midplane slice of psirz for 1D psi profile
    midplane_idx = eq.nh // 2
    psi_1d = eq.psirz[midplane_idx, :].tolist()

    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": f"SCPN Fusion Core IMAS export (shot={shot}, run={run})",
        },
        "time": [time_val],
        "time_slice": [
            {
                "time": time_val,
                "global_quantities": {
                    "ip": float(eq.current),
                    "magnetic_axis": {
                        "r": float(eq.rmaxis),
                        "z": float(eq.zmaxis),
                    },
                    "psi_axis": float(eq.simag),
                    "psi_boundary": float(eq.sibry),
                    "vacuum_toroidal_field": {
                        "r0": float(eq.rcentr),
                        "b0": float(eq.bcentr),
                    },
                },
                "profiles_1d": {
                    "psi": psi_1d,
                    "q": eq.qpsi.tolist() if eq.qpsi.size > 0 else [],
                    "pressure": eq.pres.tolist() if eq.pres.size > 0 else [],
                    "f": eq.fpol.tolist() if eq.fpol.size > 0 else [],
                    "psi_norm": psi_norm,
                },
                "profiles_2d": [
                    {
                        "psi": eq.psirz.tolist(),
                        "grid": {
                            "dim1": r_grid,
                            "dim2": z_grid,
                        },
                        "grid_type": {"index": 1, "name": "rectangular"},
                    }
                ],
                "boundary": {
                    "outline": {
                        "r": eq.rbdry.tolist() if eq.rbdry.size > 0 else [],
                        "z": eq.zbdry.tolist() if eq.zbdry.size > 0 else [],
                    }
                },
            }
        ],
        "code": {
            "name": "SCPN-Fusion-Core",
            "version": "2.0.0",
        },
    }


def imas_equilibrium_to_geqdsk(ids: Mapping[str, Any]) -> GEqdsk:
    """Convert an IMAS Data Dictionary ``equilibrium`` IDS back to a GEqdsk.

    Expects the structure produced by :func:`geqdsk_to_imas_equilibrium`.
    """
    missing = _missing_required_keys(ids, IMAS_DD_EQUILIBRIUM_KEYS)
    if missing:
        raise ValueError(f"IMAS equilibrium IDS missing keys: {missing}")

    time_slices = ids["time_slice"]
    if not isinstance(time_slices, Sequence) or len(time_slices) == 0:
        raise ValueError("IMAS equilibrium must have at least one time_slice.")

    ts = time_slices[0]
    gq = ts.get("global_quantities", {})
    p1d = ts.get("profiles_1d", {})
    p2d_list = ts.get("profiles_2d", [])
    boundary = ts.get("boundary", {})

    axis = gq.get("magnetic_axis", {})
    vtf = gq.get("vacuum_toroidal_field", {})

    # Extract 2D grid
    if not p2d_list:
        raise ValueError("IMAS equilibrium must have at least one profiles_2d entry.")
    p2d = p2d_list[0]
    grid = p2d.get("grid", {})
    r_grid = np.asarray(grid.get("dim1", []), dtype=np.float64)
    z_grid = np.asarray(grid.get("dim2", []), dtype=np.float64)
    psirz = np.asarray(p2d.get("psi", []), dtype=np.float64)

    nw = r_grid.size
    nh = z_grid.size
    if nw < 2 or nh < 2:
        raise ValueError("IMAS profiles_2d grid must have at least 2 points per dimension.")
    if psirz.ndim != 2 or psirz.shape != (nh, nw):
        raise ValueError(
            f"IMAS profiles_2d psi shape {psirz.shape} does not match grid ({nh}, {nw})."
        )

    rdim = float(r_grid[-1] - r_grid[0])
    zdim = float(z_grid[-1] - z_grid[0])
    rleft = float(r_grid[0])
    zmid = float(0.5 * (z_grid[0] + z_grid[-1]))

    outline = boundary.get("outline", {})

    return GEqdsk(
        description="IMAS import",
        nw=nw,
        nh=nh,
        rdim=rdim,
        zdim=zdim,
        rcentr=float(vtf.get("r0", 0.0)),
        rleft=rleft,
        zmid=zmid,
        rmaxis=float(axis.get("r", 0.0)),
        zmaxis=float(axis.get("z", 0.0)),
        simag=float(gq.get("psi_axis", 0.0)),
        sibry=float(gq.get("psi_boundary", 0.0)),
        bcentr=float(vtf.get("b0", 0.0)),
        current=float(gq.get("ip", 0.0)),
        fpol=np.asarray(p1d.get("f", []), dtype=np.float64),
        pres=np.asarray(p1d.get("pressure", []), dtype=np.float64),
        ffprime=np.zeros(nw, dtype=np.float64),
        pprime=np.zeros(nw, dtype=np.float64),
        qpsi=np.asarray(p1d.get("q", []), dtype=np.float64),
        psirz=psirz,
        rbdry=np.asarray(outline.get("r", []), dtype=np.float64),
        zbdry=np.asarray(outline.get("z", []), dtype=np.float64),
        rlim=np.array([], dtype=np.float64),
        zlim=np.array([], dtype=np.float64),
    )


def state_to_imas_core_profiles(
    state: Mapping[str, Any],
    *,
    time_s: float = 0.0,
) -> dict[str, Any]:
    """Convert a plasma state dict to an IMAS ``core_profiles`` IDS.

    Expects keys: ``rho_norm``, ``electron_temp_keV``, ``electron_density_1e20_m3``.
    Converts units: keV -> eV, 1e20 m^-3 -> m^-3.
    """
    profiles = _coerce_profiles_1d(state, name="state")
    rho = profiles["rho_norm"]
    te_kev = profiles["electron_temp_keV"]
    ne_1e20 = profiles["electron_density_1e20_m3"]

    # Unit conversions
    te_ev = [v * 1.0e3 for v in te_kev]
    ne_m3 = [v * 1.0e20 for v in ne_1e20]

    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": "SCPN Fusion Core core_profiles export",
        },
        "time": [float(time_s)],
        "profiles_1d": [
            {
                "time": float(time_s),
                "grid": {
                    "rho_tor_norm": rho,
                },
                "electrons": {
                    "temperature": te_ev,
                    "density": ne_m3,
                },
            }
        ],
    }


def state_to_imas_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a performance/state dict to an IMAS ``summary`` IDS.

    Accepts optional keys: ``power_fusion_MW``, ``q_sci``, ``beta_n``,
    ``li``, ``plasma_current_MA``, ``confinement_time_s``.
    """
    if isinstance(state, bool) or not isinstance(state, Mapping):
        raise ValueError("state must be a mapping.")

    gq: dict[str, Any] = {}
    for key, imas_key in [
        ("power_fusion_MW", "power_fusion"),
        ("q_sci", "q"),
        ("beta_n", "beta_n"),
        ("li", "li"),
        ("plasma_current_MA", "ip"),
        ("confinement_time_s", "tau_e"),
    ]:
        val = state.get(key)
        if val is not None:
            gq[imas_key] = float(val)

    return {
        "ids_properties": {
            "homogeneous_time": 0,
            "comment": "SCPN Fusion Core summary export",
        },
        "time": [0.0],
        "global_quantities": gq,
    }


# ── File-level IDS I/O ────────────────────────────────────────────────

_VALID_IDS_TYPES = ("equilibrium", "core_profiles", "summary")


def _numpy_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_ids(ids_dict: Mapping[str, Any], path: str | Path) -> None:
    """Write an IDS dict to a JSON file with schema validation.

    The dict must contain ``ids_properties`` with a valid ``homogeneous_time`` field.
    """
    if not isinstance(ids_dict, Mapping):
        raise ValueError("ids_dict must be a mapping.")
    if "ids_properties" not in ids_dict:
        raise ValueError("ids_dict must contain 'ids_properties' key.")
    props = ids_dict["ids_properties"]
    if not isinstance(props, Mapping) or "homogeneous_time" not in props:
        raise ValueError("ids_properties must contain 'homogeneous_time'.")

    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(ids_dict), f, indent=2, default=_numpy_serializer)


def read_ids(path: str | Path) -> dict[str, Any]:
    """Read an IDS JSON file and validate minimal schema.

    Returns the parsed dict. Raises ValueError on corrupt or invalid data.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt IDS JSON file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("IDS file must contain a JSON object at top level.")
    if "ids_properties" not in data:
        raise ValueError("IDS file missing 'ids_properties' key.")
    props = data["ids_properties"]
    if not isinstance(props, dict) or "homogeneous_time" not in props:
        raise ValueError("ids_properties must contain 'homogeneous_time'.")
    return data


# ── OMAS compatibility layer ──────────────────────────────────────────
# When the `omas` package is available, provides bidirectional conversion
# between our IDS dict format and OMAS ODS objects.
# Reference: https://gafusion.github.io/omas/

try:
    import omas  # type: ignore[import-untyped]
    HAS_OMAS = True
except ImportError:
    HAS_OMAS = False


def ids_to_omas_equilibrium(ids_dict: Mapping[str, Any]) -> Any:
    """Convert our IMAS equilibrium IDS dict to an OMAS ODS.

    Requires the ``omas`` package. Raises ImportError if not installed.

    Parameters
    ----------
    ids_dict : Mapping
        An IMAS DD-compliant equilibrium IDS dict (as produced by
        ``geqdsk_to_imas_equilibrium``).

    Returns
    -------
    omas.ODS
        An OMAS ODS with ``equilibrium`` populated.
    """
    if not HAS_OMAS:
        raise ImportError(
            "The 'omas' package is required for OMAS conversion. "
            "Install with: pip install omas"
        )

    ods = omas.ODS()

    # ids_properties
    props = ids_dict.get("ids_properties", {})
    ods["equilibrium.ids_properties.homogeneous_time"] = props.get("homogeneous_time", 1)
    ods["equilibrium.ids_properties.comment"] = props.get("comment", "")

    # time array
    time_arr = ids_dict.get("time", [0.0])
    ods["equilibrium.time"] = np.asarray(time_arr, dtype=np.float64)

    # time_slice
    for i, ts in enumerate(ids_dict.get("time_slice", [])):
        prefix = f"equilibrium.time_slice.{i}"
        ods[f"{prefix}.time"] = ts.get("time", 0.0)

        gq = ts.get("global_quantities", {})
        ods[f"{prefix}.global_quantities.ip"] = gq.get("ip", 0.0)
        axis = gq.get("magnetic_axis", {})
        ods[f"{prefix}.global_quantities.magnetic_axis.r"] = axis.get("r", 0.0)
        ods[f"{prefix}.global_quantities.magnetic_axis.z"] = axis.get("z", 0.0)
        ods[f"{prefix}.global_quantities.psi_axis"] = gq.get("psi_axis", 0.0)
        ods[f"{prefix}.global_quantities.psi_boundary"] = gq.get("psi_boundary", 0.0)

        vtf = gq.get("vacuum_toroidal_field", {})
        ods[f"{prefix}.global_quantities.vacuum_toroidal_field.r0"] = vtf.get("r0", 0.0)
        ods[f"{prefix}.global_quantities.vacuum_toroidal_field.b0"] = vtf.get("b0", 0.0)

        # profiles_1d
        p1d = ts.get("profiles_1d", {})
        if p1d:
            ods[f"{prefix}.profiles_1d.psi"] = np.asarray(p1d.get("psi", []), dtype=np.float64)
            ods[f"{prefix}.profiles_1d.q"] = np.asarray(p1d.get("q", []), dtype=np.float64)
            ods[f"{prefix}.profiles_1d.pressure"] = np.asarray(p1d.get("pressure", []), dtype=np.float64)
            ods[f"{prefix}.profiles_1d.f"] = np.asarray(p1d.get("f", []), dtype=np.float64)

        # boundary
        bdry = ts.get("boundary", {})
        outline = bdry.get("outline", {})
        if outline:
            ods[f"{prefix}.boundary.outline.r"] = np.asarray(outline.get("r", []), dtype=np.float64)
            ods[f"{prefix}.boundary.outline.z"] = np.asarray(outline.get("z", []), dtype=np.float64)

    return ods


def omas_equilibrium_to_ids(ods: Any) -> dict[str, Any]:
    """Convert an OMAS ODS equilibrium back to our IDS dict format.

    Parameters
    ----------
    ods : omas.ODS
        An OMAS ODS with ``equilibrium`` populated.

    Returns
    -------
    dict — IMAS DD-compliant equilibrium IDS dict.
    """
    if not HAS_OMAS:
        raise ImportError("The 'omas' package is required for OMAS conversion.")

    ids_dict: dict[str, Any] = {}

    # ids_properties
    ids_dict["ids_properties"] = {
        "homogeneous_time": int(ods.get("equilibrium.ids_properties.homogeneous_time", 1)),
        "comment": str(ods.get("equilibrium.ids_properties.comment", "")),
    }

    # time
    time_val = ods.get("equilibrium.time", np.array([0.0]))
    if hasattr(time_val, "tolist"):
        ids_dict["time"] = time_val.tolist()
    else:
        ids_dict["time"] = [float(time_val)]

    # time_slices
    time_slices = []
    i = 0
    while True:
        prefix = f"equilibrium.time_slice.{i}"
        try:
            t = float(ods[f"{prefix}.time"])
        except (KeyError, IndexError, TypeError):
            break

        ts: dict[str, Any] = {"time": t}

        # global_quantities
        gq: dict[str, Any] = {}
        gq["ip"] = float(ods.get(f"{prefix}.global_quantities.ip", 0.0))
        gq["magnetic_axis"] = {
            "r": float(ods.get(f"{prefix}.global_quantities.magnetic_axis.r", 0.0)),
            "z": float(ods.get(f"{prefix}.global_quantities.magnetic_axis.z", 0.0)),
        }
        gq["psi_axis"] = float(ods.get(f"{prefix}.global_quantities.psi_axis", 0.0))
        gq["psi_boundary"] = float(ods.get(f"{prefix}.global_quantities.psi_boundary", 0.0))
        gq["vacuum_toroidal_field"] = {
            "r0": float(ods.get(f"{prefix}.global_quantities.vacuum_toroidal_field.r0", 0.0)),
            "b0": float(ods.get(f"{prefix}.global_quantities.vacuum_toroidal_field.b0", 0.0)),
        }
        ts["global_quantities"] = gq

        # profiles_1d
        p1d: dict[str, Any] = {}
        for key in ("psi", "q", "pressure", "f"):
            val = ods.get(f"{prefix}.profiles_1d.{key}", None)
            if val is not None:
                p1d[key] = np.asarray(val).tolist() if hasattr(val, "tolist") else val
        if p1d:
            ts["profiles_1d"] = p1d

        # boundary
        bdry_r = ods.get(f"{prefix}.boundary.outline.r", None)
        bdry_z = ods.get(f"{prefix}.boundary.outline.z", None)
        if bdry_r is not None and bdry_z is not None:
            ts["boundary"] = {
                "outline": {
                    "r": np.asarray(bdry_r).tolist() if hasattr(bdry_r, "tolist") else bdry_r,
                    "z": np.asarray(bdry_z).tolist() if hasattr(bdry_z, "tolist") else bdry_z,
                }
            }

        time_slices.append(ts)
        i += 1

    ids_dict["time_slice"] = time_slices
    return ids_dict
