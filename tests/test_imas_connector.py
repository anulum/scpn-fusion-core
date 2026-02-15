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
    digital_twin_summary_to_ids,
    ids_to_digital_twin_summary,
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
