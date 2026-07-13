# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: IMAS digital-twin IDS validation branches

from __future__ import annotations

from typing import Any

import pytest

from scpn_fusion.io.imas_connector_digital_twin import (
    digital_twin_state_to_ids,
    digital_twin_summary_to_ids,
    validate_ids_payload,
)


def _valid_summary() -> dict[str, Any]:
    """A digital-twin summary satisfying every required key."""
    return {
        "steps": 12,
        "final_islands_px": 3,
        "final_reward": 0.87,
        "reward_mean_last_50": 0.72,
        "final_avg_temp": 8.4,
        "final_axis_r": 1.65,
        "final_axis_z": 0.02,
    }


def _valid_payload() -> dict[str, Any]:
    """A schema-valid IDS payload built from a valid summary."""
    return digital_twin_summary_to_ids(_valid_summary(), machine="ITER", shot=1, run=2)


class TestValidateIdsPayload:
    """validate_ids_payload rejects each malformed top-level field."""

    def test_missing_required_keys(self) -> None:
        payload = _valid_payload()
        del payload["performance"]
        with pytest.raises(ValueError, match="missing keys"):
            validate_ids_payload(payload)

    def test_equilibrium_must_be_mapping(self) -> None:
        payload = _valid_payload()
        payload["equilibrium"] = "not-a-mapping"
        with pytest.raises(ValueError, match="equilibrium must be a mapping"):
            validate_ids_payload(payload)

    def test_performance_must_be_mapping(self) -> None:
        payload = _valid_payload()
        payload["performance"] = "not-a-mapping"
        with pytest.raises(ValueError, match="performance must be a mapping"):
            validate_ids_payload(payload)

    def test_valid_payload_passes(self) -> None:
        # The builder output is schema-valid, so validation is silent.
        validate_ids_payload(_valid_payload())


class TestDigitalTwinStateToIds:
    """digital_twin_state_to_ids validates the state container type."""

    def test_state_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match="state must be a mapping"):
            digital_twin_state_to_ids(True)  # type: ignore[arg-type]


class TestDigitalTwinSummaryToIds:
    """digital_twin_summary_to_ids validates the machine label."""

    def test_rejects_blank_machine(self) -> None:
        with pytest.raises(ValueError, match="machine must be a non-empty string"):
            digital_twin_summary_to_ids(_valid_summary(), machine="   ")
