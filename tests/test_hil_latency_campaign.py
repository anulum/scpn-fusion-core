# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the simulated HIL sensor-to-actuator latency campaign."""

from typing import Any

import numpy as np
import pytest

from scpn_fusion.control.hil_latency_campaign import (
    _hil_measurement_for_step,
    _latency_summary_us,
    _run_sensor_to_actuator_hil_scenario,
    run_sensor_to_actuator_hil_latency_campaign,
)
from scpn_fusion.control.hil_sensors import SensorInterface


class TestSensorToActuatorHILLatencyCampaign:
    def test_campaign_reports_simulated_hil_boundary_and_256_actuators(self) -> None:
        out = run_sensor_to_actuator_hil_latency_campaign(n_steps=64, actuator_count=256)
        assert out["schema"] == "scpn-fusion-core.simulated_hil_sensor_to_actuator_latency.v1"
        assert out["status"] == "measured_simulated_hil"
        assert out["hardware_status"] == "simulated_host_adc_dac_loop"
        assert out["actuator_count"] == 256
        assert out["passes_thresholds"] is True
        assert out["nominal_latency"]["p95_us"] > 0.0
        assert "not a physical HIL rig" in out["claim_boundary"]

    def test_degraded_scenarios_fail_closed_with_safe_outputs(self) -> None:
        out = run_sensor_to_actuator_hil_latency_campaign(n_steps=64, actuator_count=32)
        for name, rec in out["scenarios"].items():
            assert rec["safe_output_rate"] == 1.0
            assert rec["passes_semantics"] is True
            if name != "nominal":
                assert rec["fallback_count"] > 0

    def test_campaign_rejects_invalid_inputs(self) -> None:
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"n_steps": 31}, "n_steps"),
            ({"state_dim": 3}, "state_dim"),
            ({"actuator_count": 0}, "actuator_count"),
        )
        for kwargs, match in cases:
            with pytest.raises(ValueError, match=match):
                run_sensor_to_actuator_hil_latency_campaign(**kwargs)

    def test_campaign_supports_extended_state_dim(self) -> None:
        # state_dim > 4 exercises the padded synthetic-state branch.
        out = run_sensor_to_actuator_hil_latency_campaign(n_steps=32, state_dim=6, actuator_count=8)
        assert out["state_dim"] == 6
        assert all(rec["passes_semantics"] for rec in out["scenarios"].values())


class TestCampaignInternals:
    def test_scenario_runner_rejects_unknown_scenario(self) -> None:
        with pytest.raises(ValueError, match="Unknown HIL latency scenario"):
            _run_sensor_to_actuator_hil_scenario(
                scenario="bogus",
                n_steps=32,
                state_dim=4,
                actuator_count=4,
                rng_seed=1,
            )

    def test_latency_summary_reports_percentiles(self) -> None:
        values = np.linspace(1.0, 100.0, 100, dtype=np.float64)
        summary = _latency_summary_us(values)
        assert set(summary.keys()) == {"p50_us", "p95_us", "p99_us", "max_us", "mean_us"}
        assert summary["max_us"] == 100.0
        assert summary["mean_us"] == pytest.approx(float(np.mean(values)))

    def test_measurement_step_flags_unsafe_output_on_nonfinite_actuator(self) -> None:
        # A non-finite previous actuator makes the slew-limited output non-finite,
        # exercising the "safe_outputs stays 0" arc that the full campaign (which
        # always produces safe outputs) never reaches.
        sensor = SensorInterface(rng_seed=0)
        controller = np.zeros((4, 4), dtype=np.float64)
        result = _hil_measurement_for_step(
            sensor=sensor,
            true_state=np.zeros(4, dtype=np.float64),
            previous_state=np.zeros(4, dtype=np.float64),
            estimator=np.eye(4, dtype=np.float64) * 0.82,
            controller=controller,
            previous_actuator=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float64),
            scenario="nominal",
            step_index=1,
            rng=np.random.default_rng(0),
        )
        safe_outputs = result[5]
        slew_limited = result[2]
        assert safe_outputs == 0
        assert not np.all(np.isfinite(slew_limited))
