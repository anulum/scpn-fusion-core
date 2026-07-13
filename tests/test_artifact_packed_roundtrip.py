# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: packed-weight artifact save/load round trip

from __future__ import annotations

from pathlib import Path

from scpn_fusion.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    InitialState,
    PackedWeights,
    PackedWeightsGroup,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    load_artifact,
    save_artifact,
)


def _build_packed_artifact(*, notes: str | None) -> Artifact:
    """A minimal valid artifact carrying expanded (data_u64) packed weights."""
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="packed-roundtrip",
            dt_control_s=1.0e-4,
            stream_length=128,
            fixed_point=FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="seed", hash_fn="sha256", rng_family="pcg64"),
            created_utc="2026-03-04T00:00:00Z",
            compiler=CompilerInfo(name="scpn", version="3.9.3", git_sha="deadbee"),
            notes=notes,
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="p0")],
            transitions=[TransitionSpec(id=0, name="t0", threshold=0.5, margin=0.0, delay_ticks=0)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 1], data=[0.0]),
            w_out=WeightMatrix(shape=[1, 1], data=[0.75]),
            packed=PackedWeightsGroup(
                words_per_stream=2,
                w_in_packed=PackedWeights(shape=[1, 1, 2], data_u64=[123, 456]),
                w_out_packed=PackedWeights(shape=[1, 1, 2], data_u64=[789, 1011]),
            ),
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


class TestPackedArtifactRoundTrip:
    """Saving expanded packed weights (compact_packed=False) survives a reload."""

    def test_expanded_packed_weights_and_notes_round_trip(self, tmp_path: Path) -> None:
        artifact = _build_packed_artifact(notes="calibration run 7")
        path = tmp_path / "packed.scpnctl.json"
        save_artifact(artifact, path)  # compact_packed=False → data_u64 stays expanded

        text = path.read_text(encoding="utf-8")
        assert '"data_u64"' in text  # expanded, not compact zlib
        assert '"notes"' in text

        loaded = load_artifact(path)
        assert loaded.meta.notes == "calibration run 7"
        assert loaded.weights.packed is not None
        assert loaded.weights.packed.words_per_stream == 2
        assert loaded.weights.packed.w_in_packed.data_u64 == [123, 456]
        assert loaded.weights.packed.w_in_packed.shape == [1, 1, 2]
        assert loaded.weights.packed.w_out_packed is not None
        assert loaded.weights.packed.w_out_packed.data_u64 == [789, 1011]

    def test_round_trip_without_notes_omits_notes_key(self, tmp_path: Path) -> None:
        artifact = _build_packed_artifact(notes=None)
        path = tmp_path / "packed_no_notes.scpnctl.json"
        save_artifact(artifact, path)

        assert '"notes"' not in path.read_text(encoding="utf-8")
        loaded = load_artifact(path)
        assert loaded.meta.notes is None
        assert loaded.weights.packed is not None
