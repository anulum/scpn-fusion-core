# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS History Payloads
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""History/pulse payload helpers split from ``imas_connector`` monolith."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

REQUIRED_IDS_PULSE_KEYS = (
    "schema",
    "machine",
    "shot",
    "run",
    "time_slices",
)


def _connector():
    # Local import keeps this module acyclic with ``imas_connector``.
    from scpn_fusion.io import imas_connector

    return imas_connector


def _has_all_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in mapping for key in keys)


def validate_ids_payload_sequence(payloads: Sequence[Mapping[str, Any]]) -> None:
    c = _connector()
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
        c.validate_ids_payload(payload)

        machine = str(payload.get("machine"))
        shot = c._coerce_int(f"payloads[{idx}].shot", payload.get("shot", 0), minimum=0)
        run = c._coerce_int(f"payloads[{idx}].run", payload.get("run", 0), minimum=0)
        time_slice = payload.get("time_slice")
        if not isinstance(time_slice, Mapping):
            raise ValueError(f"payloads[{idx}].time_slice must be a mapping.")
        time_index = c._coerce_int(
            f"payloads[{idx}].time_slice.index",
            time_slice.get("index", 0),
            minimum=0,
        )
        time_s = c._coerce_finite_real(
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
                raise ValueError("All IDS payloads in sequence must share the same machine.")
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
    c = _connector()
    if isinstance(history, (str, bytes, bytearray)) or not isinstance(history, Sequence):
        raise ValueError("history must be a sequence of digital twin snapshots.")
    if len(history) == 0:
        raise ValueError("history must contain at least one snapshot.")

    out: list[dict[str, Any]] = []
    prev_time_ms: int | None = None
    for idx, snapshot in enumerate(history):
        if isinstance(snapshot, bool) or not isinstance(snapshot, Mapping):
            raise ValueError(f"history[{idx}] must be a mapping.")
        if _has_all_keys(snapshot, c.REQUIRED_PROFILE_1D_KEYS):
            payload = c.digital_twin_state_to_ids(
                snapshot,
                machine=machine,
                shot=shot,
                run=run,
            )
        else:
            payload = c.digital_twin_summary_to_ids(
                snapshot,
                machine=machine,
                shot=shot,
                run=run,
            )
        time_slice = payload.get("time_slice")
        if not isinstance(time_slice, Mapping):
            raise ValueError(f"history[{idx}] produced invalid IDS time_slice mapping.")
        time_ms = round(
            c._coerce_finite_real(
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
    c = _connector()
    validate_ids_payload_sequence(payloads)
    out: list[dict[str, Any]] = []
    for payload in payloads:
        equilibrium = payload.get("equilibrium", {})
        if isinstance(equilibrium, Mapping) and "profiles_1d" in equilibrium:
            out.append(c.ids_to_digital_twin_state(payload))
        else:
            out.append(c.ids_to_digital_twin_summary(payload))
    return out


def validate_ids_pulse_payload(pulse: Mapping[str, Any]) -> None:
    c = _connector()
    if not isinstance(pulse, Mapping):
        raise ValueError("IDS pulse payload must be a mapping.")
    missing = c._missing_required_keys(pulse, REQUIRED_IDS_PULSE_KEYS)
    if missing:
        raise ValueError(f"IDS pulse payload missing keys: {', '.join(missing)}")

    schema = str(pulse.get("schema", "")).strip()
    if schema != "ids_equilibrium_pulse_v1":
        raise ValueError(
            f"Unsupported IDS pulse schema '{schema}'. Expected 'ids_equilibrium_pulse_v1'.",
        )

    machine = str(pulse.get("machine", "")).strip()
    if not machine:
        raise ValueError("pulse.machine must be a non-empty string.")
    shot = c._coerce_int("pulse.shot", pulse.get("shot", 0), minimum=0)
    run = c._coerce_int("pulse.run", pulse.get("run", 0), minimum=0)

    slices = pulse.get("time_slices")
    if not isinstance(slices, Sequence):
        raise ValueError("pulse.time_slices must be a sequence.")
    if len(slices) == 0:
        raise ValueError("pulse.time_slices must contain at least one payload.")

    validate_ids_payload_sequence(slices)
    for idx, payload in enumerate(slices):
        if str(payload.get("machine", "")).strip() != machine:
            raise ValueError(f"pulse.time_slices[{idx}].machine does not match pulse.machine.")
        if (
            c._coerce_int(f"pulse.time_slices[{idx}].shot", payload.get("shot", 0), minimum=0)
            != shot
        ):
            raise ValueError(f"pulse.time_slices[{idx}].shot does not match pulse.shot.")
        if c._coerce_int(f"pulse.time_slices[{idx}].run", payload.get("run", 0), minimum=0) != run:
            raise ValueError(f"pulse.time_slices[{idx}].run does not match pulse.run.")


def digital_twin_history_to_ids_pulse(
    history: Sequence[Mapping[str, Any]],
    *,
    machine: str = "ITER",
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    payloads = digital_twin_history_to_ids(
        history,
        machine=machine,
        shot=shot,
        run=run,
    )
    pulse = {
        "schema": "ids_equilibrium_pulse_v1",
        "machine": machine,
        "shot": int(shot),
        "run": int(run),
        "time_slices": payloads,
    }
    validate_ids_pulse_payload(pulse)
    return pulse


def ids_pulse_to_digital_twin_history(pulse: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_ids_pulse_payload(pulse)
    slices = pulse.get("time_slices")
    if not isinstance(slices, Sequence):
        raise ValueError("pulse.time_slices must be a sequence.")
    return ids_to_digital_twin_history(slices)
