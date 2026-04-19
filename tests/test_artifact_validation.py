# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

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
