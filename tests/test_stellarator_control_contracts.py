# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct contract tests for the geometry-neutral stellarator control contracts.

Covers every validation guard in the frozen dataclasses (finiteness, non-empty
strings, ordering, uniqueness, and replay-bound constraints), the actuator
clamp/slew behaviour, and the deterministic serialisation of each contract.
"""

from __future__ import annotations

import math
from typing import cast

import pytest

from scpn_fusion.control.stellarator_control_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)


def _config() -> MagneticConfiguration:
    return MagneticConfiguration(
        name="W7-X",
        device_class="stellarator",
        field_periods=5,
        coordinate_system="Boozer",
        reference="Pedersen 2022",
    )


def _channel(name: str = "coil_current") -> ActuatorChannel:
    return ActuatorChannel(
        name=name,
        unit="A",
        min_value=-1000.0,
        max_value=1000.0,
        slew_rate_per_s=500.0,
        latency_steps=1,
        failure_mode="none",
    )


def _diagnostic(name: str = "beta") -> DiagnosticChannel:
    return DiagnosticChannel(
        name=name, value=0.04, unit="dimensionless", sigma=1e-3, provenance="thomson"
    )


def _objective() -> ControlObjective:
    return ControlObjective(
        target_metrics={"beta": 0.04},
        weights={"beta": 1.0},
        constraints={"beta_max": 0.05},
    )


def _scenario() -> ReplayScenario:
    return ReplayScenario(
        name="w7x_beta_hold",
        seed=7,
        steps=10,
        dt_s=1e-3,
        magnetic_configuration=_config(),
        actuator_set=ActuatorSet(channels=(_channel(),)),
        objective=_objective(),
        initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
        fault_schedule={3: {"coil_current": "stuck"}},
    )


class TestMagneticConfiguration:
    """Device geometry metadata validation and serialisation."""

    def test_valid_round_trips(self) -> None:
        """A valid configuration serialises with all fields preserved."""
        data = _config().to_dict()
        assert data["name"] == "W7-X"
        assert data["field_periods"] == 5

    @pytest.mark.parametrize("field", ["name", "device_class", "coordinate_system", "reference"])
    def test_empty_string_field_raises(self, field: str) -> None:
        """Any blank required string field is rejected."""
        kwargs = {
            "name": "W7-X",
            "device_class": "stellarator",
            "field_periods": 5,
            "coordinate_system": "Boozer",
            "reference": "Pedersen 2022",
        }
        kwargs[field] = "   "
        with pytest.raises(ValueError, match="must be non-empty"):
            MagneticConfiguration(**kwargs)  # type: ignore[arg-type]

    def test_zero_field_periods_raises(self) -> None:
        """A device with fewer than one field period is rejected."""
        with pytest.raises(ValueError, match="field_periods must be >= 1"):
            MagneticConfiguration(
                name="X",
                device_class="stellarator",
                field_periods=0,
                coordinate_system="Boozer",
                reference="ref",
            )


class TestActuatorChannel:
    """Actuator bounds, slew, latency validation and clamp/slew behaviour."""

    def test_valid_round_trips(self) -> None:
        """A valid actuator serialises all limit and metadata fields."""
        data = _channel().to_dict()
        assert data["min_value"] == -1000.0
        assert data["latency_steps"] == 1

    @pytest.mark.parametrize("field", ["name", "unit"])
    def test_empty_string_field_raises(self, field: str) -> None:
        """A blank name or unit is rejected."""
        kwargs = {
            "name": "coil",
            "unit": "A",
            "min_value": -1.0,
            "max_value": 1.0,
            "slew_rate_per_s": 1.0,
        }
        kwargs[field] = ""
        with pytest.raises(ValueError, match="must be non-empty"):
            ActuatorChannel(**kwargs)  # type: ignore[arg-type]

    def test_non_finite_bound_raises(self) -> None:
        """A non-finite actuator bound is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            ActuatorChannel(
                name="coil", unit="A", min_value=math.nan, max_value=1.0, slew_rate_per_s=1.0
            )

    def test_inverted_bounds_raise(self) -> None:
        """A max not strictly greater than min is rejected."""
        with pytest.raises(ValueError, match="greater than min_value"):
            ActuatorChannel(
                name="coil", unit="A", min_value=1.0, max_value=1.0, slew_rate_per_s=1.0
            )

    def test_non_positive_slew_raises(self) -> None:
        """A non-positive slew rate is rejected."""
        with pytest.raises(ValueError, match="slew_rate_per_s must be > 0"):
            ActuatorChannel(
                name="coil", unit="A", min_value=-1.0, max_value=1.0, slew_rate_per_s=0.0
            )

    def test_negative_latency_raises(self) -> None:
        """A negative latency step count is rejected."""
        with pytest.raises(ValueError, match="latency_steps must be >= 0"):
            ActuatorChannel(
                name="coil",
                unit="A",
                min_value=-1.0,
                max_value=1.0,
                slew_rate_per_s=1.0,
                latency_steps=-1,
            )

    def test_clamp_bounds_value(self) -> None:
        """Clamping saturates requests outside the configured range."""
        channel = _channel()
        assert channel.clamp(5000.0) == 1000.0
        assert channel.clamp(-5000.0) == -1000.0
        assert channel.clamp(0.0) == 0.0

    def test_apply_slew_limits_delta(self) -> None:
        """Slew limiting caps the per-step change to slew_rate * dt."""
        channel = _channel()
        # requested +1000 but slew allows only 500 * 0.1 = 50 per step
        assert channel.apply_slew(previous=0.0, requested=1000.0, dt_s=0.1) == 50.0
        # a small request within the slew budget passes through
        assert channel.apply_slew(previous=0.0, requested=10.0, dt_s=0.1) == 10.0

    def test_apply_slew_non_positive_dt_raises(self) -> None:
        """A non-positive time step in slew limiting is rejected."""
        with pytest.raises(ValueError, match="dt_s must be > 0"):
            _channel().apply_slew(previous=0.0, requested=1.0, dt_s=0.0)


class TestActuatorSet:
    """Actuator collection uniqueness and lookup."""

    def test_empty_set_raises(self) -> None:
        """An actuator set with no channels is rejected."""
        with pytest.raises(ValueError, match="at least one channel"):
            ActuatorSet(channels=())

    def test_duplicate_names_raise(self) -> None:
        """Duplicate actuator channel names are rejected."""
        with pytest.raises(ValueError, match="names must be unique"):
            ActuatorSet(channels=(_channel("a"), _channel("a")))

    def test_by_name_returns_channel(self) -> None:
        """Lookup returns the channel with the requested name."""
        actuator_set = ActuatorSet(channels=(_channel("a"), _channel("b")))
        assert actuator_set.by_name("b").name == "b"

    def test_by_name_unknown_raises_key_error(self) -> None:
        """Lookup of an unknown channel name raises KeyError."""
        with pytest.raises(KeyError, match="unknown actuator channel"):
            ActuatorSet(channels=(_channel("a"),)).by_name("missing")

    def test_to_dict_lists_channels(self) -> None:
        """Serialisation lists every channel in order."""
        data = ActuatorSet(channels=(_channel("a"), _channel("b"))).to_dict()
        channels = cast("list[dict[str, object]]", data["channels"])
        assert [c["name"] for c in channels] == ["a", "b"]


class TestDiagnosticChannel:
    """Diagnostic value validation and serialisation."""

    def test_valid_round_trips(self) -> None:
        """A valid diagnostic serialises value and uncertainty."""
        data = _diagnostic().to_dict()
        assert data["value"] == 0.04
        assert data["sigma"] == 1e-3

    @pytest.mark.parametrize("field", ["name", "unit", "provenance"])
    def test_empty_string_field_raises(self, field: str) -> None:
        """A blank name, unit, or provenance is rejected."""
        kwargs = {
            "name": "beta",
            "value": 0.04,
            "unit": "dimensionless",
            "sigma": 1e-3,
            "provenance": "thomson",
        }
        kwargs[field] = ""
        with pytest.raises(ValueError, match="must be non-empty"):
            DiagnosticChannel(**kwargs)  # type: ignore[arg-type]

    def test_non_finite_value_raises(self) -> None:
        """A non-finite diagnostic value is rejected."""
        with pytest.raises(ValueError, match="value must be finite"):
            DiagnosticChannel(
                name="beta", value=math.inf, unit="dimensionless", sigma=1e-3, provenance="t"
            )

    def test_negative_sigma_raises(self) -> None:
        """A negative uncertainty is rejected."""
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            DiagnosticChannel(
                name="beta", value=0.04, unit="dimensionless", sigma=-1.0, provenance="t"
            )


class TestDiagnosticFrame:
    """Diagnostic frame timing and channel uniqueness."""

    def test_valid_as_mapping_and_to_dict(self) -> None:
        """A valid frame maps channel values and serialises them."""
        frame = DiagnosticFrame(step=2, time_s=0.5, channels=(_diagnostic("a"), _diagnostic("b")))
        assert frame.as_mapping() == {"a": 0.04, "b": 0.04}
        assert frame.to_dict()["step"] == 2

    def test_negative_step_raises(self) -> None:
        """A negative replay step index is rejected."""
        with pytest.raises(ValueError, match="step must be >= 0"):
            DiagnosticFrame(step=-1, time_s=0.0, channels=(_diagnostic(),))

    def test_non_finite_time_raises(self) -> None:
        """A non-finite frame time is rejected."""
        with pytest.raises(ValueError, match="time_s must be finite"):
            DiagnosticFrame(step=0, time_s=math.nan, channels=(_diagnostic(),))

    def test_negative_time_raises(self) -> None:
        """A negative frame time is rejected."""
        with pytest.raises(ValueError, match="time_s must be >= 0"):
            DiagnosticFrame(step=0, time_s=-1.0, channels=(_diagnostic(),))

    def test_empty_channels_raise(self) -> None:
        """A frame with no channels is rejected."""
        with pytest.raises(ValueError, match="at least one channel"):
            DiagnosticFrame(step=0, time_s=0.0, channels=())

    def test_duplicate_channel_names_raise(self) -> None:
        """Duplicate diagnostic channel names are rejected."""
        with pytest.raises(ValueError, match="names must be unique"):
            DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic("a"), _diagnostic("a")))


class TestControlObjective:
    """Objective targets, weights, and constraints validation."""

    def test_valid_round_trips(self) -> None:
        """A valid objective serialises all three metric groups."""
        data = _objective().to_dict()
        assert data["target_metrics"] == {"beta": 0.04}
        assert data["constraints"] == {"beta_max": 0.05}

    def test_empty_targets_raise(self) -> None:
        """An objective without any target metric is rejected."""
        with pytest.raises(ValueError, match="target_metrics must not be empty"):
            ControlObjective(target_metrics={}, weights={}, constraints={})

    def test_blank_key_raises(self) -> None:
        """A blank metric key is rejected."""
        with pytest.raises(ValueError, match="must be non-empty"):
            ControlObjective(target_metrics={"  ": 1.0}, weights={}, constraints={})

    def test_non_finite_value_raises(self) -> None:
        """A non-finite metric value is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            ControlObjective(target_metrics={"beta": math.inf}, weights={}, constraints={})


class TestReplayScenario:
    """End-to-end replay scenario validation and serialisation."""

    def test_valid_round_trips(self) -> None:
        """A valid scenario serialises every nested contract."""
        data = _scenario().to_dict()
        assert data["name"] == "w7x_beta_hold"
        assert data["fault_schedule"] == {"3": {"coil_current": "stuck"}}
        actuator_set = cast("dict[str, object]", data["actuator_set"])
        channels = cast("list[dict[str, object]]", actuator_set["channels"])
        assert channels[0]["name"] == "coil_current"

    def test_blank_name_raises(self) -> None:
        """A blank scenario name is rejected."""
        with pytest.raises(ValueError, match="must be non-empty"):
            ReplayScenario(
                name=" ",
                seed=1,
                steps=10,
                dt_s=1e-3,
                magnetic_configuration=_config(),
                actuator_set=ActuatorSet(channels=(_channel(),)),
                objective=_objective(),
                initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
                fault_schedule={},
            )

    def test_too_few_steps_raise(self) -> None:
        """A scenario shorter than two steps is rejected."""
        with pytest.raises(ValueError, match="steps must be >= 2"):
            ReplayScenario(
                name="x",
                seed=1,
                steps=1,
                dt_s=1e-3,
                magnetic_configuration=_config(),
                actuator_set=ActuatorSet(channels=(_channel(),)),
                objective=_objective(),
                initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
                fault_schedule={},
            )

    def test_non_positive_dt_raises(self) -> None:
        """A non-positive scenario time step is rejected."""
        with pytest.raises(ValueError, match="dt_s must be > 0"):
            ReplayScenario(
                name="x",
                seed=1,
                steps=10,
                dt_s=0.0,
                magnetic_configuration=_config(),
                actuator_set=ActuatorSet(channels=(_channel(),)),
                objective=_objective(),
                initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
                fault_schedule={},
            )

    def test_fault_step_out_of_bounds_raises(self) -> None:
        """A fault scheduled outside the replay window is rejected."""
        with pytest.raises(ValueError, match="fault schedule step out of replay bounds"):
            ReplayScenario(
                name="x",
                seed=1,
                steps=10,
                dt_s=1e-3,
                magnetic_configuration=_config(),
                actuator_set=ActuatorSet(channels=(_channel(),)),
                objective=_objective(),
                initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
                fault_schedule={99: {"coil_current": "stuck"}},
            )

    def test_fault_unknown_channel_raises(self) -> None:
        """A fault referencing an unknown actuator channel is rejected."""
        with pytest.raises(KeyError, match="unknown actuator channel"):
            ReplayScenario(
                name="x",
                seed=1,
                steps=10,
                dt_s=1e-3,
                magnetic_configuration=_config(),
                actuator_set=ActuatorSet(channels=(_channel(),)),
                objective=_objective(),
                initial_frame=DiagnosticFrame(step=0, time_s=0.0, channels=(_diagnostic(),)),
                fault_schedule={2: {"missing_channel": "stuck"}},
            )
