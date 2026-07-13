# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

from typing import cast

import pytest

from scpn_fusion.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    InitialState,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    get_artifact_json_schema,
)
from scpn_fusion.scpn.artifact_validation import validate_artifact


def _build_valid_artifact() -> Artifact:
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="unit-test",
            dt_control_s=1.0e-4,
            stream_length=128,
            fixed_point=FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="seed", hash_fn="sha256", rng_family="pcg64"),
            created_utc="2026-03-04T00:00:00Z",
            compiler=CompilerInfo(name="scpn", version="3.9.3", git_sha="deadbee"),
            notes=None,
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="p0")],
            transitions=[TransitionSpec(id=0, name="t0", threshold=0.5, margin=0.0, delay_ticks=0)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 1], data=[0.0]),
            w_out=WeightMatrix(shape=[1, 1], data=[0.75]),
            packed=None,
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="a0", pos_place=0, neg_place=0)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[10.0],
        ),
        initial_state=InitialState(
            marking=[0.1],
            place_injections=[
                PlaceInjection(place_id=0, source="sensor_x", scale=1.0, offset=0.0, clamp_0_1=True)
            ],
        ),
    )


def test_validate_artifact_accepts_valid_contract() -> None:
    artifact = _build_valid_artifact()
    validate_artifact(artifact, error_type=ValueError)


def test_validate_artifact_rejects_non_finite_dt() -> None:
    artifact = _build_valid_artifact()
    artifact.meta.dt_control_s = float("nan")
    with pytest.raises(ValueError, match="dt_control_s"):
        validate_artifact(artifact, error_type=ValueError)


def test_validate_artifact_rejects_out_of_bounds_action_place() -> None:
    artifact = _build_valid_artifact()
    artifact.readout.actions[0].pos_place = 3
    with pytest.raises(ValueError, match="out of bounds"):
        validate_artifact(artifact, error_type=ValueError)


def test_get_artifact_json_schema_contains_required_sections() -> None:
    schema = get_artifact_json_schema()
    assert schema["type"] == "object"
    for section in ("meta", "topology", "weights", "readout", "initial_state"):
        assert section in schema["properties"]


class TestValidateMetaGuards:
    """Reject malformed ``meta`` fields."""

    def test_rejects_unknown_firing_mode(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.firing_mode = "ternary"
        with pytest.raises(ValueError, match="firing_mode"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_positive_data_width(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.fixed_point.data_width = 0
        with pytest.raises(ValueError, match="data_width must be >= 1"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_fraction_bits(self) -> None:
        artifact = _build_valid_artifact()
        # bool is an int subtype; the dedicated guard rejects it explicitly.
        artifact.meta.fixed_point.fraction_bits = True
        with pytest.raises(ValueError, match="fraction_bits must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_positive_stream_length(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.stream_length = 0
        with pytest.raises(ValueError, match="stream_length must be >= 1"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_positive_dt(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.dt_control_s = 0.0
        with pytest.raises(ValueError, match="dt_control_s must be > 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_data_width(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.fixed_point.data_width = True
        with pytest.raises(ValueError, match="data_width must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_fraction_bits(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.fixed_point.fraction_bits = -1
        with pytest.raises(ValueError, match="fraction_bits must be >= 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_fraction_bits_not_below_data_width(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.fixed_point.fraction_bits = artifact.meta.fixed_point.data_width
        with pytest.raises(ValueError, match="fraction_bits must be < fixed_point.data_width"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_boolean_signed(self) -> None:
        artifact = _build_valid_artifact()
        # ``signed`` is typed ``bool``; inject a non-bool to exercise the guard.
        artifact.meta.fixed_point.signed = cast("bool", 1)
        with pytest.raises(ValueError, match="signed must be a boolean"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_stream_length(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.stream_length = True
        with pytest.raises(ValueError, match="stream_length must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_dt(self) -> None:
        artifact = _build_valid_artifact()
        artifact.meta.dt_control_s = True
        with pytest.raises(ValueError, match="dt_control_s must be finite"):
            validate_artifact(artifact, error_type=ValueError)


class TestValidateWeightGuards:
    """Reject out-of-range weights and shape mismatches."""

    def test_rejects_w_in_weight_out_of_range(self) -> None:
        artifact = _build_valid_artifact()
        artifact.weights.w_in.data = [2.0]
        with pytest.raises(ValueError, match="w_in weight"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_w_out_weight_out_of_range(self) -> None:
        artifact = _build_valid_artifact()
        artifact.weights.w_out.data = [2.0]
        with pytest.raises(ValueError, match="w_out weight"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_w_in_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.weights.w_in.data = [0.0, 0.0]
        with pytest.raises(ValueError, match="w_in data length"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_w_out_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.weights.w_out.data = [0.75, 0.75]
        with pytest.raises(ValueError, match="w_out data length"):
            validate_artifact(artifact, error_type=ValueError)


class TestValidateTransitionGuards:
    """Reject malformed transition thresholds and delays."""

    def test_rejects_threshold_out_of_range(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].threshold = 1.5
        with pytest.raises(ValueError, match=r"threshold .* outside"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_delay_ticks(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].delay_ticks = -1
        with pytest.raises(ValueError, match=r"delay_ticks.*must be >= 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_threshold(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].threshold = True
        with pytest.raises(ValueError, match=r"threshold .* must be finite and in"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_finite_threshold(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].threshold = float("inf")
        with pytest.raises(ValueError, match=r"threshold .* must be finite and in"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_margin(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].margin = True
        with pytest.raises(ValueError, match=r"margin .* must be finite and >= 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_margin(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].margin = -1.0
        with pytest.raises(ValueError, match=r"margin .* must be finite and >= 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_delay_ticks(self) -> None:
        artifact = _build_valid_artifact()
        artifact.topology.transitions[0].delay_ticks = True
        with pytest.raises(ValueError, match="delay_ticks.*must be an integer"):
            validate_artifact(artifact, error_type=ValueError)


class TestValidateInitialStateGuards:
    """Reject malformed initial marking and place injections."""

    def test_rejects_marking_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.marking = [0.1, 0.2]
        with pytest.raises(ValueError, match="marking length"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_marking_value_out_of_range(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.marking = [1.5]
        with pytest.raises(ValueError, match="initial marking"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_place_id(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.place_injections[0].place_id = True
        with pytest.raises(ValueError, match="place_id must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_empty_injection_source(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.place_injections[0].source = ""
        with pytest.raises(ValueError, match="source must be a non-empty string"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_place_id_out_of_bounds(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.place_injections[0].place_id = 5
        with pytest.raises(ValueError, match=r"place_id .* out of bounds"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_finite_injection_scale(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.place_injections[0].scale = float("inf")
        with pytest.raises(ValueError, match="scale must be finite numeric"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_finite_injection_offset(self) -> None:
        artifact = _build_valid_artifact()
        artifact.initial_state.place_injections[0].offset = float("inf")
        with pytest.raises(ValueError, match="offset must be finite numeric"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_boolean_clamp(self) -> None:
        artifact = _build_valid_artifact()
        # ``clamp_0_1`` is typed ``bool``; inject a non-bool to exercise the guard.
        artifact.initial_state.place_injections[0].clamp_0_1 = cast("bool", 1)
        with pytest.raises(ValueError, match="clamp_0_1 must be a boolean"):
            validate_artifact(artifact, error_type=ValueError)


class TestValidateReadoutGuards:
    """Reject malformed readout actions and gain/limit vectors."""

    def test_rejects_boolean_pos_place(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.actions[0].pos_place = True
        with pytest.raises(ValueError, match="pos_place must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_neg_place_out_of_bounds(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.actions[0].neg_place = 5
        with pytest.raises(ValueError, match=r"neg_place .* out of bounds"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_action_id(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.actions[0].id = -1
        with pytest.raises(ValueError, match="actions.id must be an integer >= 0"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_empty_action_name(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.actions[0].name = ""
        with pytest.raises(ValueError, match="actions.name must be a non-empty string"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_boolean_neg_place(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.actions[0].neg_place = True
        with pytest.raises(ValueError, match="neg_place must be an integer"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_non_finite_gain(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.gains = [float("inf")]
        with pytest.raises(ValueError, match="gains must contain finite"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_abs_max(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.abs_max = [-1.0]
        with pytest.raises(ValueError, match="abs_max must contain finite"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_negative_slew(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.slew_per_s = [-1.0]
        with pytest.raises(ValueError, match="slew_per_s must contain finite"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_gains_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.gains = [1.0, 2.0]
        with pytest.raises(ValueError, match="gains length must equal"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_abs_max_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.abs_max = [1.0, 2.0]
        with pytest.raises(ValueError, match="abs_max length must equal"):
            validate_artifact(artifact, error_type=ValueError)

    def test_rejects_slew_per_s_length_mismatch(self) -> None:
        artifact = _build_valid_artifact()
        artifact.readout.slew_per_s = [10.0, 20.0]
        with pytest.raises(ValueError, match="slew_per_s length must equal"):
            validate_artifact(artifact, error_type=ValueError)
