# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: IMAS history/pulse payload validation branches

from __future__ import annotations

import copy
from typing import Any

import pytest

from scpn_fusion.io.imas_connector_digital_twin import digital_twin_summary_to_ids
from scpn_fusion.io.imas_history_payloads import (
    digital_twin_history_to_ids,
    digital_twin_history_to_ids_pulse,
    ids_pulse_to_digital_twin_history,
    validate_ids_payload_sequence,
    validate_ids_pulse_payload,
)


def _summary() -> dict[str, Any]:
    """A digital-twin summary satisfying every required key."""
    return {
        "steps": 5,
        "final_islands_px": 2,
        "final_reward": 0.5,
        "reward_mean_last_50": 0.4,
        "final_avg_temp": 7.0,
        "final_axis_r": 1.6,
        "final_axis_z": 0.0,
    }


def _payload(
    *, machine: str = "ITER", shot: int = 0, run: int = 0, index: int = 0, time_s: float = 0.0
) -> dict[str, Any]:
    """A valid single IDS payload with a caller-controlled time slice/identity."""
    payload = digital_twin_summary_to_ids(_summary(), machine=machine, shot=shot, run=run)
    payload["time_slice"]["index"] = index
    payload["time_slice"]["time_s"] = time_s
    return payload


def _valid_pulse() -> dict[str, Any]:
    """A schema-valid IDS pulse built from a two-snapshot history."""
    return digital_twin_history_to_ids_pulse(
        [_summary(), _summary()], machine="ITER", shot=5, run=1
    )


class TestValidateIdsPayloadSequence:
    """validate_ids_payload_sequence enforces container, identity, and monotonicity."""

    def test_payloads_must_be_sequence(self) -> None:
        with pytest.raises(ValueError, match="must be a sequence"):
            validate_ids_payload_sequence("not-a-sequence")  # type: ignore[arg-type]

    def test_payload_entry_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"payloads\[0\] must be a mapping"):
            validate_ids_payload_sequence([42])  # type: ignore[list-item]

    def test_shot_must_be_shared(self) -> None:
        seq = [_payload(shot=1, index=0, time_s=0.0), _payload(shot=2, index=1, time_s=0.001)]
        with pytest.raises(ValueError, match="same shot"):
            validate_ids_payload_sequence(seq)

    def test_run_must_be_shared(self) -> None:
        seq = [_payload(run=1, index=0, time_s=0.0), _payload(run=2, index=1, time_s=0.001)]
        with pytest.raises(ValueError, match="same run"):
            validate_ids_payload_sequence(seq)

    def test_time_index_must_strictly_increase(self) -> None:
        seq = [_payload(index=0, time_s=0.0), _payload(index=0, time_s=0.001)]
        with pytest.raises(ValueError, match="strictly increasing time_slice.index"):
            validate_ids_payload_sequence(seq)


class TestDigitalTwinHistoryToIds:
    """digital_twin_history_to_ids validates its history container."""

    def test_history_must_be_sequence(self) -> None:
        with pytest.raises(ValueError, match="must be a sequence"):
            digital_twin_history_to_ids("not-a-sequence")  # type: ignore[arg-type]

    def test_history_entry_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"history\[0\] must be a mapping"):
            digital_twin_history_to_ids([42])  # type: ignore[list-item]


class TestValidateIdsPulsePayload:
    """validate_ids_pulse_payload guards the pulse envelope and slice consistency."""

    def test_pulse_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match="pulse payload must be a mapping"):
            validate_ids_pulse_payload(42)  # type: ignore[arg-type]

    def test_machine_must_be_non_empty(self) -> None:
        pulse = _valid_pulse()
        pulse["machine"] = ""
        with pytest.raises(ValueError, match="machine must be a non-empty string"):
            validate_ids_pulse_payload(pulse)

    def test_time_slices_must_be_sequence(self) -> None:
        pulse = _valid_pulse()
        pulse["time_slices"] = 123
        with pytest.raises(ValueError, match="time_slices must be a sequence"):
            validate_ids_pulse_payload(pulse)

    def test_slice_machine_must_match_pulse(self) -> None:
        pulse = _valid_pulse()
        pulse["machine"] = "OTHER"
        with pytest.raises(ValueError, match="machine does not match"):
            validate_ids_pulse_payload(pulse)

    def test_slice_shot_must_match_pulse(self) -> None:
        pulse = _valid_pulse()
        pulse["shot"] = 999
        with pytest.raises(ValueError, match="shot does not match"):
            validate_ids_pulse_payload(pulse)

    def test_slice_run_must_match_pulse(self) -> None:
        pulse = _valid_pulse()
        pulse["run"] = 999
        with pytest.raises(ValueError, match="run does not match"):
            validate_ids_pulse_payload(pulse)


class TestPulseRoundTrip:
    """A history survives conversion to a pulse and back."""

    def test_history_to_pulse_to_history(self) -> None:
        pulse = _valid_pulse()
        restored = ids_pulse_to_digital_twin_history(copy.deepcopy(pulse))
        assert len(restored) == 2
        assert all(isinstance(snapshot, dict) for snapshot in restored)
