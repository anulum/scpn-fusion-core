# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Validation-contract tests for IMAS/IDS adapter payloads."""

from __future__ import annotations

import pytest

from scpn_fusion.io.imas_connector import (
    digital_twin_history_to_ids,
    digital_twin_state_to_ids,
    digital_twin_summary_to_ids,
    ids_to_digital_twin_history,
    ids_to_digital_twin_state,
    ids_to_digital_twin_summary,
    validate_ids_payload_sequence,
    validate_ids_payload,
)


def _sample_payload() -> dict[str, object]:
    summary = {
        "steps": 12,
        "final_axis_r": 6.2,
        "final_axis_z": 0.03,
        "final_islands_px": 4,
        "final_reward": 1.1,
        "reward_mean_last_50": 0.9,
        "final_avg_temp": 14.0,
    }
    return digital_twin_summary_to_ids(summary, machine="ITER", shot=1, run=2)


def _sample_state() -> dict[str, object]:
    state: dict[str, object] = {
        "steps": 24,
        "final_axis_r": 6.22,
        "final_axis_z": -0.01,
        "final_islands_px": 3,
        "final_reward": 1.4,
        "reward_mean_last_50": 1.1,
        "final_avg_temp": 15.2,
        "rho_norm": [0.0, 0.25, 0.5, 0.75, 1.0],
        "electron_temp_keV": [18.0, 17.1, 15.4, 12.2, 8.8],
        "electron_density_1e20_m3": [1.4, 1.35, 1.24, 1.11, 0.94],
    }
    return state


def test_validate_ids_payload_accepts_nominal_payload() -> None:
    payload = _sample_payload()
    validate_ids_payload(payload)
    recovered = ids_to_digital_twin_summary(payload)
    assert recovered["steps"] == 12
    assert recovered["final_islands_px"] == 4


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (lambda p: p.__setitem__("schema", "ids_equilibrium_v2"), "Unsupported IDS schema"),
        (lambda p: p.__setitem__("machine", ""), "machine"),
        (lambda p: p.__setitem__("time_slice", []), "time_slice"),
        (
            lambda p: p["time_slice"].__setitem__("index", 1.5),
            "time_slice.index",
        ),
        (
            lambda p: p["time_slice"].__setitem__("time_s", -0.001),
            "time_slice.time_s",
        ),
        (
            lambda p: p["time_slice"].__setitem__("time_s", 0.0125),
            "integer millisecond",
        ),
        (lambda p: p["equilibrium"].__setitem__("axis", []), "equilibrium.axis"),
        (
            lambda p: p["equilibrium"]["axis"].__setitem__("r_m", float("nan")),
            "equilibrium.axis.r_m",
        ),
        (
            lambda p: p["equilibrium"].__setitem__("islands_px", -1),
            "equilibrium.islands_px",
        ),
        (
            lambda p: p["performance"].__setitem__("final_reward", float("inf")),
            "performance.final_reward",
        ),
        (
            lambda p: p["performance"].__setitem__("reward_mean_last_50", "bad"),
            "performance.reward_mean_last_50",
        ),
        (
            lambda p: p["performance"].__setitem__("final_avg_temp_keV", float("nan")),
            "performance.final_avg_temp_keV",
        ),
    ],
)
def test_validate_ids_payload_rejects_invalid_nested_fields(
    mutate,
    msg: str,
) -> None:
    payload = _sample_payload()
    mutate(payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=msg):
        validate_ids_payload(payload)


def test_validate_ids_payload_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_ids_payload([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (lambda p: p["time_slice"].pop("time_s"), "time_slice missing keys"),
        (lambda p: p["time_slice"].pop("index"), "time_slice missing keys"),
        (lambda p: p["equilibrium"].pop("islands_px"), "equilibrium missing keys"),
        (lambda p: p["equilibrium"]["axis"].pop("r_m"), "equilibrium.axis missing keys"),
        (lambda p: p["performance"].pop("final_avg_temp_keV"), "performance missing keys"),
    ],
)
def test_validate_ids_payload_rejects_missing_nested_keys(mutate, msg: str) -> None:
    payload = _sample_payload()
    mutate(payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=msg):
        validate_ids_payload(payload)


def test_digital_twin_summary_to_ids_rejects_non_mapping_summary() -> None:
    with pytest.raises(ValueError, match="summary must be a mapping"):
        digital_twin_summary_to_ids([], machine="ITER", shot=1, run=2)  # type: ignore[arg-type]


def test_digital_twin_summary_to_ids_rejects_missing_required_summary_keys() -> None:
    summary = {
        "steps": 12,
        "final_axis_r": 6.2,
        "final_axis_z": 0.03,
        "final_islands_px": 4,
        "final_reward": 1.1,
        "reward_mean_last_50": 0.9,
    }
    with pytest.raises(ValueError, match="digital twin summary missing keys"):
        digital_twin_summary_to_ids(summary, machine="ITER", shot=1, run=2)


def test_validate_ids_payload_accepts_profiles_1d_block() -> None:
    payload = _sample_payload()
    payload["equilibrium"]["profiles_1d"] = {  # type: ignore[index]
        "rho_norm": [0.0, 0.5, 1.0],
        "electron_temp_keV": [15.0, 12.0, 8.0],
        "electron_density_1e20_m3": [1.2, 1.0, 0.8],
    }
    validate_ids_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (
            lambda p: p["equilibrium"].__setitem__("profiles_1d", []),
            "equilibrium.profiles_1d must be a mapping",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].pop("rho_norm"),
            "equilibrium.profiles_1d missing keys",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].__setitem__("rho_norm", [0.0, 0.0, 1.0]),
            "strictly increasing",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].__setitem__("rho_norm", [0.0, 1.2]),
            "<= 1.0",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].__setitem__("electron_temp_keV", [5.0, -1.0, 2.0]),
            "electron_temp_keV",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].__setitem__("electron_density_1e20_m3", [1.0, 0.8]),
            "length must match",
        ),
        (
            lambda p: p["equilibrium"]["profiles_1d"].__setitem__("electron_density_1e20_m3", [1.0, float("nan"), 0.8]),
            "electron_density_1e20_m3",
        ),
    ],
)
def test_validate_ids_payload_rejects_invalid_profiles_1d(mutate, msg: str) -> None:
    payload = _sample_payload()
    payload["equilibrium"]["profiles_1d"] = {  # type: ignore[index]
        "rho_norm": [0.0, 0.5, 1.0],
        "electron_temp_keV": [15.0, 12.0, 8.0],
        "electron_density_1e20_m3": [1.2, 1.0, 0.8],
    }
    mutate(payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=msg):
        validate_ids_payload(payload)


def test_digital_twin_state_to_ids_roundtrip_with_profiles() -> None:
    state = _sample_state()
    payload = digital_twin_state_to_ids(state, machine="ITER", shot=7, run=3)
    validate_ids_payload(payload)
    recovered = ids_to_digital_twin_state(payload)
    assert recovered["steps"] == state["steps"]
    assert recovered["final_islands_px"] == state["final_islands_px"]
    assert recovered["rho_norm"] == state["rho_norm"]
    assert recovered["electron_temp_keV"] == state["electron_temp_keV"]
    assert recovered["electron_density_1e20_m3"] == state["electron_density_1e20_m3"]


def test_ids_to_digital_twin_state_without_profiles_returns_summary() -> None:
    payload = _sample_payload()
    recovered = ids_to_digital_twin_state(payload)
    assert "rho_norm" not in recovered
    assert recovered["steps"] == 12


def test_digital_twin_state_to_ids_rejects_missing_profile_keys() -> None:
    state = _sample_state()
    state.pop("electron_density_1e20_m3")
    with pytest.raises(ValueError, match="digital twin state missing keys"):
        digital_twin_state_to_ids(state, machine="ITER", shot=7, run=3)


def test_digital_twin_history_roundtrip_mixed_summary_and_state() -> None:
    summary = {
        "steps": 12,
        "final_axis_r": 6.2,
        "final_axis_z": 0.03,
        "final_islands_px": 4,
        "final_reward": 1.1,
        "reward_mean_last_50": 0.9,
        "final_avg_temp": 14.0,
    }
    state = _sample_state()
    history = [summary, state]

    payloads = digital_twin_history_to_ids(history, machine="ITER", shot=9, run=1)
    validate_ids_payload_sequence(payloads)
    recovered = ids_to_digital_twin_history(payloads)
    assert len(recovered) == 2
    assert recovered[0]["steps"] == summary["steps"]
    assert recovered[1]["steps"] == state["steps"]
    assert recovered[1]["rho_norm"] == state["rho_norm"]


def test_validate_ids_payload_sequence_rejects_non_increasing_time() -> None:
    payloads = digital_twin_history_to_ids(
        [_sample_state(), _sample_state()],
        machine="ITER",
        shot=11,
        run=4,
    )
    payloads[1]["time_slice"]["time_s"] = payloads[0]["time_slice"]["time_s"]  # type: ignore[index]
    with pytest.raises(ValueError, match="strictly increasing time_slice.time_s"):
        validate_ids_payload_sequence(payloads)


def test_validate_ids_payload_sequence_rejects_machine_mismatch() -> None:
    payloads = digital_twin_history_to_ids(
        [_sample_state(), _sample_state()],
        machine="ITER",
        shot=11,
        run=4,
    )
    payloads[1]["machine"] = "JET"
    with pytest.raises(ValueError, match="same machine"):
        validate_ids_payload_sequence(payloads)


def test_ids_history_helpers_reject_empty_sequences() -> None:
    with pytest.raises(ValueError, match="at least one"):
        digital_twin_history_to_ids([], machine="ITER", shot=1, run=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        validate_ids_payload_sequence([])  # type: ignore[arg-type]
