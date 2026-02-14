# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for Packet C — contracts, artifact, controller, fractional firing.

Verification matrix:
    Level 0 — Static: artifact loads, weights in [0,1], thresholds in [0,1],
              firing_mode declared
    Level 1 — Determinism: same seed_base + k → identical actions
    Level 2 — Primitives: encode mean accuracy, AND product accuracy
              (skipif no sc_neurocore)
    Level 3 — Petri semantics: marking bounds [0,1] over 200 steps,
              fractional firing in [0,1]
    Integration — sinusoidal disturbance → bounded actions + slew;
                  step disturbance → nonzero response
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion import __version__ as PACKAGE_VERSION
from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn.compiler import FusionCompiler, CompiledNet, _HAS_SC_NEUROCORE
from scpn_fusion.scpn.contracts import (
    ActionSpec,
    ControlAction,
    ControlObservation,
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
    _clip01,
    decode_actions,
    extract_features,
)
from scpn_fusion.scpn.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    Artifact,
    ArtifactValidationError,
    PackedWeights,
    PackedWeightsGroup,
    PlaceInjection,
    decode_u64_compact,
    encode_u64_compact,
    load_artifact,
    save_artifact,
)
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn import controller as controller_mod
from scpn_fusion.scpn import artifact as artifact_mod


# ── Fixture: 8-place controller net ─────────────────────────────────────────
# 4 feature inputs (x_R_pos, x_R_neg, x_Z_pos, x_Z_neg)
# 4 action outputs (a_R_pos, a_R_neg, a_Z_pos, a_Z_neg)
# 4 pass-through transitions: T_Rp, T_Rn, T_Zp, T_Zn
# Each transition reads one input place and writes one output place.


def _build_controller_net(transition_delay_ticks: int = 0) -> StochasticPetriNet:
    """8-place, 4-transition pass-through controller net."""
    net = StochasticPetriNet()

    # Input places (injected from features)
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)

    # Output places (read by action decoder)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    # Transitions (low threshold for pass-through)
    net.add_transition("T_Rp", threshold=0.1, delay_ticks=transition_delay_ticks)
    net.add_transition("T_Rn", threshold=0.1, delay_ticks=transition_delay_ticks)
    net.add_transition("T_Zp", threshold=0.1, delay_ticks=transition_delay_ticks)
    net.add_transition("T_Zn", threshold=0.1, delay_ticks=transition_delay_ticks)

    # Input arcs: input place → transition
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)

    # Output arcs: transition → output place
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)

    net.compile()
    return net


def _build_artifact_path(
    firing_mode: str = "binary",
    firing_margin: float = 0.05,
    transition_delay_ticks: int = 0,
) -> str:
    """Compile the 8-place net, export artifact, save to temp file."""
    net = _build_controller_net(transition_delay_ticks=transition_delay_ticks)
    compiler = FusionCompiler(bitstream_length=1024, seed=42)
    compiled = compiler.compile(
        net, firing_mode=firing_mode, firing_margin=firing_margin
    )

    readout_config = {
        "actions": [
            {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
            {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]

    artifact = compiled.export_artifact(
        name="test_controller",
        dt_control_s=0.001,
        readout_config=readout_config,
        injection_config=injection_config,
    )

    fd, path = tempfile.mkstemp(suffix=".scpnctl.json")
    os.close(fd)
    save_artifact(artifact, path)
    return path


@pytest.fixture
def artifact_path() -> str:
    path = _build_artifact_path("binary")
    yield path
    os.unlink(path)


@pytest.fixture
def artifact_path_fractional() -> str:
    path = _build_artifact_path("fractional", 0.05)
    yield path
    os.unlink(path)


@pytest.fixture
def controller(artifact_path: str) -> NeuroSymbolicController:
    art = load_artifact(artifact_path)
    return NeuroSymbolicController(
        artifact=art,
        seed_base=123456789,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
    )


@pytest.fixture
def controller_fractional(artifact_path_fractional: str) -> NeuroSymbolicController:
    art = load_artifact(artifact_path_fractional)
    return NeuroSymbolicController(
        artifact=art,
        seed_base=123456789,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Level 0 — Static validation
# ═════════════════════════════════════════════════════════════════════════════


class TestLevel0Static:
    def test_artifact_loads(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        assert art.nP == 8
        assert art.nT == 4

    def test_weights_in_unit_range(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        for v in art.weights.w_in.data:
            assert 0.0 <= v <= 1.0
        for v in art.weights.w_out.data:
            assert 0.0 <= v <= 1.0

    def test_thresholds_in_unit_range(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        for t in art.topology.transitions:
            assert 0.0 <= t.threshold <= 1.0

    def test_firing_mode_declared(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        assert art.meta.firing_mode in ("binary", "fractional")

    def test_artifact_schema_version_matches_constant(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        assert art.meta.artifact_version == ARTIFACT_SCHEMA_VERSION

    def test_compiler_version_matches_package(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        assert art.meta.compiler.version == PACKAGE_VERSION

    def test_marking_in_unit_range(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        for v in art.initial_state.marking:
            assert 0.0 <= v <= 1.0

    def test_artifact_roundtrip(self, artifact_path: str) -> None:
        """Load → save → reload produces identical artifact."""
        art1 = load_artifact(artifact_path)
        fd, path2 = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            save_artifact(art1, path2)
            art2 = load_artifact(path2)
            assert art2.nP == art1.nP
            assert art2.nT == art1.nT
            assert art2.meta.firing_mode == art1.meta.firing_mode
            assert art2.weights.w_in.data == art1.weights.w_in.data
            assert art2.weights.w_out.data == art1.weights.w_out.data
        finally:
            os.unlink(path2)

    def test_artifact_roundtrip_compact_packed(self, artifact_path: str) -> None:
        """Compact packed serialization should roundtrip equivalently."""
        art1 = load_artifact(artifact_path)
        if art1.weights.packed is None:
            pytest.skip("packed weights unavailable in this environment")

        fd, path2 = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            save_artifact(art1, path2, compact_packed=True)
            text = Path(path2).read_text(encoding="utf-8")
            assert "data_u64_b64_zlib" in text

            art2 = load_artifact(path2)
            assert art2.weights.packed is not None
            assert (
                art2.weights.packed.w_in_packed.data_u64
                == art1.weights.packed.w_in_packed.data_u64
            )
            if art1.weights.packed.w_out_packed is not None:
                assert art2.weights.packed.w_out_packed is not None
                assert (
                    art2.weights.packed.w_out_packed.data_u64
                    == art1.weights.packed.w_out_packed.data_u64
                )
        finally:
            os.unlink(path2)

    def test_compact_u64_codec_roundtrip_deterministic(self) -> None:
        values = [
            0,
            1,
            2,
            0x0123456789ABCDEF,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
        ]
        enc1 = encode_u64_compact(values)
        enc2 = encode_u64_compact(values)
        assert enc1 == enc2
        decoded = decode_u64_compact(enc1)
        assert decoded == values

    def test_compact_u64_codec_rejects_invalid_base64(self) -> None:
        bad = {
            "encoding": "u64-le-zlib-base64",
            "count": 1,
            "data_u64_b64_zlib": "$$$not-base64$$$",
        }
        with pytest.raises(ArtifactValidationError, match="Invalid base64 payload"):
            decode_u64_compact(bad)

    def test_compact_u64_codec_rejects_oversized_compressed_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(artifact_mod, "MAX_COMPRESSED_BYTES", 8)
        payload = base64.b64encode(b"x" * 32).decode("ascii")
        bad = {
            "encoding": "u64-le-zlib-base64",
            "count": 1,
            "data_u64_b64_zlib": payload,
        }
        with pytest.raises(ArtifactValidationError, match="Compressed payload too large"):
            decode_u64_compact(bad)

    def test_compact_u64_codec_rejects_oversized_decompressed_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(artifact_mod, "MAX_PACKED_WORDS", 1)
        monkeypatch.setattr(artifact_mod, "MAX_DECOMPRESSED_BYTES", 8)
        raw = (1).to_bytes(8, "little") + (2).to_bytes(8, "little")
        payload = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
        bad = {
            "encoding": "u64-le-zlib-base64",
            "data_u64_b64_zlib": payload,
        }
        with pytest.raises(
            ArtifactValidationError, match="Decompressed packed payload exceeds configured limit"
        ):
            decode_u64_compact(bad)

    def test_compact_u64_codec_rejects_invalid_count_type(self) -> None:
        good = encode_u64_compact([1, 2, 3])
        good["count"] = "3"
        with pytest.raises(ArtifactValidationError, match="count type"):
            decode_u64_compact(good)

    def test_artifact_roundtrip_compact_packed_synthetic(self, artifact_path: str) -> None:
        art1 = load_artifact(artifact_path)
        w_in_u64 = [0, 1, 2, 3, 4, 5]
        w_out_u64 = [9, 8, 7, 6]
        art1.weights.packed = PackedWeightsGroup(
            words_per_stream=2,
            w_in_packed=PackedWeights(shape=[3, 1, 2], data_u64=w_in_u64),
            w_out_packed=PackedWeights(shape=[2, 1, 2], data_u64=w_out_u64),
        )

        fd, path2 = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            save_artifact(art1, path2, compact_packed=True)
            payload_obj = json.loads(Path(path2).read_text(encoding="utf-8"))
            packed_obj = payload_obj["weights"]["packed"]
            assert packed_obj["w_in_packed"]["encoding"] == "u64-le-zlib-base64"
            assert packed_obj["w_out_packed"]["encoding"] == "u64-le-zlib-base64"

            art2 = load_artifact(path2)
            assert art2.weights.packed is not None
            assert art2.weights.packed.w_in_packed.data_u64 == w_in_u64
            assert art2.weights.packed.w_out_packed is not None
            assert art2.weights.packed.w_out_packed.data_u64 == w_out_u64
        finally:
            os.unlink(path2)

    def test_load_artifact_rejects_non_integer_delay_ticks(
        self, artifact_path: str
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        obj["topology"]["transitions"][0]["delay_ticks"] = 1.5
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match="delay_ticks"):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    def test_load_artifact_rejects_non_integer_stream_length(
        self, artifact_path: str
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        obj["meta"]["stream_length"] = 1024.5
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match="stream_length"):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    @pytest.mark.parametrize(
        "dt_control_s", [float("nan"), float("inf"), "0.01", True]
    )
    def test_load_artifact_rejects_non_finite_dt_control(
        self, artifact_path: str, dt_control_s: object
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        obj["meta"]["dt_control_s"] = dt_control_s
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match="dt_control_s"):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    @pytest.mark.parametrize(
        ("field_path", "value", "match"),
        [
            (("meta", "fixed_point", "data_width"), 16.5, "fixed_point.data_width"),
            (("meta", "fixed_point", "fraction_bits"), -1, "fixed_point.fraction_bits"),
            (("meta", "fixed_point", "fraction_bits"), 16, "fixed_point.fraction_bits"),
            (("meta", "fixed_point", "signed"), "true", "fixed_point.signed"),
        ],
    )
    def test_load_artifact_rejects_invalid_fixed_point_metadata(
        self, artifact_path: str, field_path: tuple[str, ...], value: object, match: str
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        target: dict[str, object] = obj
        for key in field_path[:-1]:
            target = target[key]  # type: ignore[index]
        target[field_path[-1]] = value
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match=match):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    @pytest.mark.parametrize(
        ("field_path", "value", "match"),
        [
            (("topology", "transitions", 0, "threshold"), "0.5", "threshold"),
            (("topology", "transitions", 0, "threshold"), float("nan"), "threshold"),
            (("topology", "transitions", 0, "margin"), -0.1, "margin"),
            (("topology", "transitions", 0, "margin"), "0.1", "margin"),
        ],
    )
    def test_load_artifact_rejects_invalid_transition_threshold_or_margin(
        self,
        artifact_path: str,
        field_path: tuple[str | int, ...],
        value: object,
        match: str,
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        target: object = obj
        for key in field_path[:-1]:
            if isinstance(target, list):
                target = target[key]  # type: ignore[index]
            else:
                target = target[key]  # type: ignore[index]
        if isinstance(target, list):
            target[field_path[-1]] = value  # type: ignore[index]
        else:
            target[field_path[-1]] = value  # type: ignore[index]
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match=match):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    @pytest.mark.parametrize(
        ("field_path", "value", "match"),
        [
            (
                ("readout", "limits", "per_action_abs_max", 0),
                "10.0",
                "readout.abs_max",
            ),
            (("readout", "limits", "slew_per_s", 0), -1.0, "readout.slew_per_s"),
            (("readout", "gains", "per_action", 0), float("nan"), "readout.gains"),
        ],
    )
    def test_load_artifact_rejects_invalid_readout_vectors(
        self,
        artifact_path: str,
        field_path: tuple[str | int, ...],
        value: object,
        match: str,
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        target: object = obj
        for key in field_path[:-1]:
            if isinstance(target, list):
                target = target[key]  # type: ignore[index]
            else:
                target = target[key]  # type: ignore[index]
        if isinstance(target, list):
            target[field_path[-1]] = value  # type: ignore[index]
        else:
            target[field_path[-1]] = value  # type: ignore[index]
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match=match):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)

    @pytest.mark.parametrize(
        ("field_path", "value", "match"),
        [
            (
                ("initial_state", "place_injections", 0, "place_id"),
                999,
                "place_injections.place_id",
            ),
            (
                ("initial_state", "place_injections", 0, "scale"),
                "1.0",
                "place_injections.scale",
            ),
            (
                ("initial_state", "place_injections", 0, "offset"),
                float("nan"),
                "place_injections.offset",
            ),
            (
                ("initial_state", "place_injections", 0, "clamp_0_1"),
                "true",
                "place_injections.clamp_0_1",
            ),
        ],
    )
    def test_load_artifact_rejects_invalid_place_injection_fields(
        self,
        artifact_path: str,
        field_path: tuple[str | int, ...],
        value: object,
        match: str,
    ) -> None:
        obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        target: object = obj
        for key in field_path[:-1]:
            if isinstance(target, list):
                target = target[key]  # type: ignore[index]
            else:
                target = target[key]  # type: ignore[index]
        if isinstance(target, list):
            target[field_path[-1]] = value  # type: ignore[index]
        else:
            target[field_path[-1]] = value  # type: ignore[index]
        fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
        os.close(fd)
        try:
            Path(bad_path).write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(ArtifactValidationError, match=match):
                load_artifact(bad_path)
        finally:
            os.unlink(bad_path)


# ═════════════════════════════════════════════════════════════════════════════
# Level 1 — Determinism
# ═════════════════════════════════════════════════════════════════════════════


class TestLevel1Determinism:
    def test_deterministic_replay(self, artifact_path: str) -> None:
        """Same seed_base + k sequence → identical actions."""
        art = load_artifact(artifact_path)
        kwargs = dict(
            artifact=art,
            seed_base=123456789,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )
        c1 = NeuroSymbolicController(**kwargs)
        c2 = NeuroSymbolicController(**kwargs)

        obs: ControlObservation = {"R_axis_m": 6.4, "Z_axis_m": 0.1}
        out1 = [c1.step(obs, k) for k in range(20)]
        out2 = [c2.step(obs, k) for k in range(20)]
        assert out1 == out2

    def test_deterministic_after_reset(
        self, controller: NeuroSymbolicController
    ) -> None:
        obs: ControlObservation = {"R_axis_m": 6.3, "Z_axis_m": -0.05}
        run1 = [controller.step(obs, k) for k in range(10)]

        controller.reset()
        run2 = [controller.step(obs, k) for k in range(10)]
        assert run1 == run2

    def test_deterministic_stochastic_fractional_replay(self, artifact_path_fractional: str) -> None:
        art = load_artifact(artifact_path_fractional)
        kwargs = dict(
            artifact=art,
            seed_base=987654321,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=16,
            sc_bitflip_rate=0.0,
        )
        c1 = NeuroSymbolicController(**kwargs)
        c2 = NeuroSymbolicController(**kwargs)
        obs: ControlObservation = {"R_axis_m": 6.35, "Z_axis_m": -0.08}
        out1 = [c1.step(obs, k) for k in range(20)]
        out2 = [c2.step(obs, k) for k in range(20)]
        assert out1 == out2

    def test_binary_threshold_mode_matches_oracle_by_default(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        controller = NeuroSymbolicController(
            artifact=art,
            seed_base=99,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=32,
            sc_binary_margin=0.0,
        )
        obs: ControlObservation = {"R_axis_m": 6.15, "Z_axis_m": 0.0}
        controller.step(obs, 0)
        np.testing.assert_allclose(
            controller.last_sc_firing,
            controller.last_oracle_firing,
            atol=0.0,
        )

    def test_binary_threshold_mode_matches_oracle_for_deterministic_profile(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        controller = NeuroSymbolicController(
            artifact=art,
            seed_base=100,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=64,
            runtime_profile="deterministic",
        )
        obs: ControlObservation = {"R_axis_m": 6.15, "Z_axis_m": 0.0}
        controller.step(obs, 0)
        np.testing.assert_allclose(
            controller.last_sc_firing,
            controller.last_oracle_firing,
            atol=0.0,
        )

    def test_adaptive_profile_introduces_probabilistic_binary_margin(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        controller = NeuroSymbolicController(
            artifact=art,
            seed_base=101,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=64,
            sc_bitflip_rate=0.0,
            runtime_profile="adaptive",
        )
        obs: ControlObservation = {"R_axis_m": 6.15, "Z_axis_m": 0.0}
        controller.step(obs, 0)
        delta = np.max(
            np.abs(np.asarray(controller.last_sc_firing) - np.asarray(controller.last_oracle_firing))
        )
        assert float(delta) > 0.0

    def test_default_runtime_profile_is_adaptive_nonoracle_binary(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        controller = NeuroSymbolicController(
            artifact=art,
            seed_base=102,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=64,
            sc_bitflip_rate=0.0,
        )
        obs: ControlObservation = {"R_axis_m": 6.15, "Z_axis_m": 0.0}
        controller.step(obs, 0)
        delta = np.max(
            np.abs(
                np.asarray(controller.last_sc_firing)
                - np.asarray(controller.last_oracle_firing)
            )
        )
        assert float(delta) > 0.0

    def test_binary_probabilistic_margin_is_deterministic_and_nonoracle(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        kwargs = dict(
            artifact=art,
            seed_base=4242,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=64,
            sc_bitflip_rate=0.0,
            sc_binary_margin=0.2,
        )
        c1 = NeuroSymbolicController(**kwargs)
        c2 = NeuroSymbolicController(**kwargs)
        obs: ControlObservation = {"R_axis_m": 6.15, "Z_axis_m": 0.0}
        out1 = []
        out2 = []
        diffs = []
        for k in range(10):
            out1.append(c1.step(obs, k))
            diffs.append(
                float(np.max(np.abs(np.asarray(c1.last_sc_firing) - np.asarray(c1.last_oracle_firing))))
            )
            out2.append(c2.step(obs, k))
        assert out1 == out2
        assert any(d > 0.0 for d in diffs)

    def test_antithetic_stochastic_fractional_replay(
        self, artifact_path_fractional: str
    ) -> None:
        art = load_artifact(artifact_path_fractional)
        kwargs = dict(
            artifact=art,
            seed_base=222333444,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=15,
            sc_bitflip_rate=0.0,
            sc_antithetic=True,
        )
        c1 = NeuroSymbolicController(**kwargs)
        c2 = NeuroSymbolicController(**kwargs)
        obs: ControlObservation = {"R_axis_m": 6.145, "Z_axis_m": 0.0}
        out1 = [c1.step(obs, k) for k in range(20)]
        out2 = [c2.step(obs, k) for k in range(20)]
        assert out1 == out2

    def test_antithetic_sampling_reduces_stochastic_firing_error(
        self, artifact_path_fractional: str
    ) -> None:
        art = load_artifact(artifact_path_fractional)
        base_kwargs = dict(
            artifact=art,
            seed_base=5050,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=16,
            sc_bitflip_rate=0.0,
        )
        c_plain = NeuroSymbolicController(**base_kwargs, sc_antithetic=False)
        c_anti = NeuroSymbolicController(**base_kwargs, sc_antithetic=True)

        obs: ControlObservation = {"R_axis_m": 6.145, "Z_axis_m": 0.0}
        plain_gaps = []
        anti_gaps = []
        for k in range(80):
            c_plain.step(obs, k)
            c_anti.step(obs, k)
            plain_gaps.append(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(c_plain.last_sc_firing)
                            - np.asarray(c_plain.last_oracle_firing)
                        )
                    )
                )
            )
            anti_gaps.append(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(c_anti.last_sc_firing)
                            - np.asarray(c_anti.last_oracle_firing)
                        )
                    )
                )
            )

        assert float(np.mean(anti_gaps)) <= float(np.mean(plain_gaps)) + 1e-12


# ═════════════════════════════════════════════════════════════════════════════
# Level 2 — Primitive correctness (SC vs oracle)
# ═════════════════════════════════════════════════════════════════════════════


class TestLevel2Primitives:
    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_encode_mean_accuracy(self) -> None:
        """E[popcount(Encode(p))/L] ≈ p for a grid of probabilities."""
        from sc_neurocore import generate_bernoulli_bitstream, RNG
        from sc_neurocore.accel.vector_ops import pack_bitstream, vec_popcount

        L = 4096
        for p_target in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            rng = RNG(42)
            bits = generate_bernoulli_bitstream(p_target, L, rng=rng)
            packed = pack_bitstream(bits)
            p_est = int(vec_popcount(packed)) / L
            assert abs(p_est - p_target) < 3.0 / math.sqrt(L), (
                f"p={p_target}, est={p_est}"
            )

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_and_product_accuracy(self) -> None:
        """E[AND(w, p)] ≈ w*p."""
        from sc_neurocore import generate_bernoulli_bitstream, RNG
        from sc_neurocore.accel.vector_ops import (
            pack_bitstream,
            vec_and,
            vec_popcount,
        )

        L = 4096
        for w, p in [(0.5, 0.5), (0.8, 0.3), (1.0, 0.7), (0.0, 0.5)]:
            rng_w = RNG(100)
            rng_p = RNG(200)
            bits_w = generate_bernoulli_bitstream(w, L, rng=rng_w)
            bits_p = generate_bernoulli_bitstream(p, L, rng=rng_p)
            anded = vec_and(pack_bitstream(bits_w), pack_bitstream(bits_p))
            est = int(vec_popcount(anded)) / L
            assert abs(est - w * p) < 3.0 / math.sqrt(L), (
                f"w={w}, p={p}, est={est}, expected={w * p}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Level 3 — Petri semantics
# ═════════════════════════════════════════════════════════════════════════════


class TestLevel3PetriSemantics:
    def test_marking_bounds_200_steps(
        self, controller: NeuroSymbolicController
    ) -> None:
        """Marking stays in [0, 1] over 200 ticks with sinusoidal obs."""
        for k in range(200):
            obs: ControlObservation = {
                "R_axis_m": 6.2 + 0.2 * math.sin(0.1 * k),
                "Z_axis_m": 0.1 * math.cos(0.07 * k),
            }
            controller.step(obs, k)
            for v in controller.marking:
                assert 0.0 <= v <= 1.0, (
                    f"marking out of [0,1] at k={k}: {controller.marking}"
                )

    def test_fractional_firing_range(
        self, controller_fractional: NeuroSymbolicController
    ) -> None:
        """Fractional fire values stay in [0, 1]."""
        for k in range(100):
            obs: ControlObservation = {
                "R_axis_m": 6.2 + 0.3 * math.sin(0.15 * k),
                "Z_axis_m": 0.2 * math.cos(0.1 * k),
            }
            controller_fractional.step(obs, k)
            for v in controller_fractional.marking:
                assert 0.0 <= v <= 1.0

    def test_fractional_compiled_net(self) -> None:
        """CompiledNet.lif_fire fractional returns values in [0, 1]."""
        net = _build_controller_net()
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net, firing_mode="fractional", firing_margin=0.1)

        # Activations spanning below threshold, at threshold, above threshold
        currents = np.array([0.0, 0.1, 0.15, 0.3])  # threshold=0.1
        fired = compiled.lif_fire(currents)
        assert fired.shape == (4,)
        for v in fired:
            assert 0.0 <= v <= 1.0, f"fractional fire outside [0,1]: {v}"
        # Below threshold → 0
        assert fired[0] == 0.0
        # At threshold → 0 (since (0.1 - 0.1)/0.1 = 0.0)
        assert abs(fired[1]) < 1e-12
        # Above threshold → fractional
        assert fired[2] > 0.0
        assert fired[3] > 0.0

    def test_binary_compiled_net_unchanged(self) -> None:
        """Binary mode still returns {0.0, 1.0}."""
        net = _build_controller_net()
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net, firing_mode="binary")

        currents = np.array([0.0, 0.1, 0.15, 0.3])
        fired = compiled.lif_fire(currents)
        for v in fired:
            assert v in (0.0, 1.0), f"binary fire not in {{0,1}}: {v}"

    def test_timed_transition_defers_action_release(self) -> None:
        delayed_path = _build_artifact_path(
            firing_mode="binary", transition_delay_ticks=2
        )
        try:
            art = load_artifact(delayed_path)
            assert all(t.delay_ticks == 2 for t in art.topology.transitions)
            c = NeuroSymbolicController(
                artifact=art,
                seed_base=101,
                targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
                scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
                sc_n_passes=1,
                sc_bitflip_rate=0.0,
            )
            obs: ControlObservation = {"R_axis_m": 5.9, "Z_axis_m": 0.0}
            a0 = c.step(obs, 0)["dI_PF3_A"]
            a1 = c.step(obs, 1)["dI_PF3_A"]
            a2 = c.step(obs, 2)["dI_PF3_A"]
            assert abs(a0) < 1e-12
            assert abs(a1) < 1e-12
            assert a2 > 0.0
        finally:
            os.unlink(delayed_path)

    def test_sc_bitflip_path_stays_bounded(self, artifact_path_fractional: str) -> None:
        art = load_artifact(artifact_path_fractional)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=24680,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=8,
            sc_bitflip_rate=0.05,
        )
        for k in range(100):
            obs: ControlObservation = {
                "R_axis_m": 6.2 + 0.2 * math.sin(0.09 * k),
                "Z_axis_m": 0.1 * math.cos(0.11 * k),
            }
            c.step(obs, k)
            for v in c.marking:
                assert 0.0 <= v <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_sinusoidal_disturbance_bounded(
        self, controller: NeuroSymbolicController
    ) -> None:
        """Actions stay bounded under sinusoidal plant disturbance."""
        abs_max = controller.artifact.readout.abs_max
        for k in range(100):
            obs: ControlObservation = {
                "R_axis_m": 6.2 + 0.15 * math.sin(0.05 * k),
                "Z_axis_m": 0.1 * math.cos(0.03 * k),
            }
            act = controller.step(obs, k)
            assert abs(act["dI_PF3_A"]) <= abs_max[0] + 1e-10
            assert abs(act["dI_PF_topbot_A"]) <= abs_max[1] + 1e-10

    def test_step_disturbance_nonzero_response(
        self, controller: NeuroSymbolicController
    ) -> None:
        """A sudden offset in R or Z produces a non-zero action."""
        controller.reset()
        # Step disturbance: R axis shifted 0.3 m from target
        obs: ControlObservation = {"R_axis_m": 6.5, "Z_axis_m": 0.0}
        act = controller.step(obs, 0)
        # At least one action should be nonzero (controller reacts to error)
        assert act["dI_PF3_A"] != 0.0 or act["dI_PF_topbot_A"] != 0.0

    def test_slew_rate_limiting(self, artifact_path: str) -> None:
        """Consecutive actions are slew-rate limited."""
        art = load_artifact(artifact_path)
        # Override slew to a tight limit
        art.readout.slew_per_s = [100.0, 100.0]
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )

        dt = art.meta.dt_control_s
        max_delta = 100.0 * dt  # = 0.1 A per tick

        prev_act: ControlAction = {"dI_PF3_A": 0.0, "dI_PF_topbot_A": 0.0}
        for k in range(50):
            obs: ControlObservation = {"R_axis_m": 6.5, "Z_axis_m": 0.3}
            act = c.step(obs, k)
            for key in ("dI_PF3_A", "dI_PF_topbot_A"):
                delta = abs(act[key] - prev_act[key])
                assert delta <= max_delta + 1e-10, (
                    f"slew violation at k={k}: delta={delta}, max={max_delta}"
                )
            prev_act = act

    def test_jsonl_logging(self, controller: NeuroSymbolicController) -> None:
        """JSONL log file is written with expected keys."""
        fd, log_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            obs: ControlObservation = {"R_axis_m": 6.3, "Z_axis_m": 0.05}
            controller.step(obs, 0, log_path=log_path)
            controller.step(obs, 1, log_path=log_path)

            with open(log_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2

            rec = json.loads(lines[0])
            assert "k" in rec
            assert "obs" in rec
            assert "features" in rec
            assert "f_oracle" in rec
            assert "f_sc" in rec
            assert "marking" in rec
            assert "actions" in rec
            assert "timing_ms" in rec
        finally:
            os.unlink(log_path)

    def test_controller_supports_passthrough_injection_sources(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        art.initial_state.place_injections.append(
            PlaceInjection(
                place_id=0,
                source="density_norm",
                scale=1.0,
                offset=0.0,
                clamp_0_1=True,
            )
        )
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=77,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "density_norm": 0.73}
        act = c.step(obs, 0)
        assert "dI_PF3_A" in act
        assert "dI_PF_topbot_A" in act

    def test_controller_passthrough_injection_missing_key_raises(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        art.initial_state.place_injections.append(
            PlaceInjection(
                place_id=0,
                source="density_norm",
                scale=1.0,
                offset=0.0,
                clamp_0_1=True,
            )
        )
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=88,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )
        with pytest.raises(KeyError, match="density_norm"):
            c.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)

    def test_controller_supports_custom_feature_axes(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=333,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            feature_axes=[
                FeatureAxisSpec(
                    obs_key="beta_n",
                    target=1.8,
                    scale=0.6,
                    pos_key="x_R_pos",
                    neg_key="x_R_neg",
                ),
                FeatureAxisSpec(
                    obs_key="q95",
                    target=4.8,
                    scale=1.0,
                    pos_key="x_Z_pos",
                    neg_key="x_Z_neg",
                ),
            ],
        )
        act = c.step({"beta_n": 1.6, "q95": 5.2}, 0)
        assert "dI_PF3_A" in act
        assert "dI_PF_topbot_A" in act

    def test_controller_custom_feature_axes_missing_key_raises(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=334,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            feature_axes=[
                FeatureAxisSpec(
                    obs_key="beta_n",
                    target=1.8,
                    scale=0.6,
                    pos_key="x_R_pos",
                    neg_key="x_R_neg",
                ),
                FeatureAxisSpec(
                    obs_key="q95",
                    target=4.8,
                    scale=1.0,
                    pos_key="x_Z_pos",
                    neg_key="x_Z_neg",
                ),
            ],
        )
        with pytest.raises(KeyError, match="q95"):
            c.step({"beta_n": 1.7}, 0)

    def test_disable_oracle_diagnostics_skips_oracle_path(
        self, artifact_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=99,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            enable_oracle_diagnostics=False,
        )

        def _fail_oracle() -> tuple[list[float], list[float]]:
            raise AssertionError("_oracle_step must not run when diagnostics are disabled")

        monkeypatch.setattr(c, "_oracle_step", _fail_oracle)
        act = c.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)
        assert "dI_PF3_A" in act
        assert "dI_PF_topbot_A" in act
        assert c.last_oracle_firing == []
        assert c.last_oracle_marking == []

    def test_runtime_backend_rust_request_falls_back_when_unavailable(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=222,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_backend="rust",
        )
        if controller_mod._HAS_RUST_SCPN_RUNTIME:
            assert c.runtime_backend_name == "rust"
        else:
            assert c.runtime_backend_name == "numpy"

    def test_runtime_backend_auto_can_force_numpy_via_problem_threshold(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=223,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_backend="auto",
            rust_backend_min_problem_size=10**9,
        )
        assert c.runtime_backend_name == "numpy"

    def test_runtime_backend_auto_prefers_rust_when_available(
        self, artifact_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = load_artifact(artifact_path)
        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", True)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=228,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_backend="auto",
        )
        assert c.runtime_backend_name == "rust"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sc_n_passes": 0}, "sc_n_passes"),
            ({"sc_n_passes": 1.25}, "sc_n_passes"),
            ({"sc_bitflip_rate": -0.1}, "sc_bitflip_rate"),
            ({"sc_bitflip_rate": 1.1}, "sc_bitflip_rate"),
            ({"sc_bitflip_rate": float("nan")}, "sc_bitflip_rate"),
            ({"rust_backend_min_problem_size": 0}, "rust_backend_min_problem_size"),
            (
                {"rust_backend_min_problem_size": 2.5},
                "rust_backend_min_problem_size",
            ),
            ({"sc_binary_margin": -0.01}, "sc_binary_margin"),
            ({"sc_binary_margin": float("nan")}, "sc_binary_margin"),
            ({"sc_antithetic_chunk_size": 0}, "sc_antithetic_chunk_size"),
            ({"sc_antithetic_chunk_size": 3.5}, "sc_antithetic_chunk_size"),
        ],
    )
    def test_constructor_rejects_invalid_runtime_sampling_inputs(
        self, artifact_path: str, kwargs: dict[str, object], match: str
    ) -> None:
        art = load_artifact(artifact_path)
        with pytest.raises(ValueError, match=match):
            NeuroSymbolicController(
                artifact=art,
                seed_base=232,
                targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
                scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
                **kwargs,
            )

    def test_traceable_step_matches_mapping_step(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        kwargs = dict(
            artifact=art,
            seed_base=230,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_profile="traceable",
            runtime_backend="numpy",
            enable_oracle_diagnostics=False,
            sc_n_passes=1,
            sc_bitflip_rate=0.0,
        )
        c_map = NeuroSymbolicController(**kwargs)
        c_vec = NeuroSymbolicController(**kwargs)
        assert c_vec.runtime_profile_name == "traceable"

        obs = {"R_axis_m": 6.29, "Z_axis_m": -0.02}
        obs_vec = (obs["R_axis_m"], obs["Z_axis_m"])
        for k in range(10):
            a_map = c_map.step(obs, k)
            a_vec = c_vec.step_traceable(obs_vec, k)
            assert a_map["dI_PF3_A"] == pytest.approx(float(a_vec[0]), rel=0.0, abs=0.0)
            assert a_map["dI_PF_topbot_A"] == pytest.approx(
                float(a_vec[1]), rel=0.0, abs=0.0
            )

    def test_traceable_step_validates_vector_length(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=231,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_profile="traceable",
            enable_oracle_diagnostics=False,
            runtime_backend="numpy",
        )
        with pytest.raises(ValueError, match="obs_vector"):
            c.step_traceable((6.2,), 0)

    def test_antithetic_chunked_sampling_is_deterministic(
        self, artifact_path: str
    ) -> None:
        art = load_artifact(artifact_path)
        kwargs = dict(
            artifact=art,
            seed_base=229,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_profile="adaptive",
            sc_n_passes=8,
            sc_binary_margin=0.2,
            sc_antithetic=True,
            sc_antithetic_chunk_size=1,
            runtime_backend="numpy",
        )
        c1 = NeuroSymbolicController(**kwargs)
        c2 = NeuroSymbolicController(**kwargs)
        obs = {"R_axis_m": 6.29, "Z_axis_m": 0.03}
        for k in range(8):
            a1 = c1.step(obs, k)
            a2 = c2.step(obs, k)
            assert a1 == pytest.approx(a2, rel=0.0, abs=0.0)
            np.testing.assert_allclose(
                np.asarray(c1.last_sc_firing, dtype=np.float64),
                np.asarray(c2.last_sc_firing, dtype=np.float64),
                atol=0.0,
                rtol=0.0,
            )

    def test_runtime_backend_rust_path_executes_rust_kernels_when_available(
        self, artifact_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = load_artifact(artifact_path)
        calls = {"dense": 0, "update": 0}

        def _fake_dense(w_in: np.ndarray, marking: np.ndarray) -> np.ndarray:
            calls["dense"] += 1
            return np.asarray(w_in @ marking, dtype=np.float64)

        def _fake_update(
            marking: np.ndarray,
            w_in: np.ndarray,
            w_out: np.ndarray,
            firing: np.ndarray,
        ) -> np.ndarray:
            calls["update"] += 1
            cons = w_in.T @ firing
            prod = w_out @ firing
            return np.asarray(np.clip(marking - cons + prod, 0.0, 1.0), dtype=np.float64)

        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", True)
        monkeypatch.setattr(controller_mod, "_rust_dense_activations", _fake_dense)
        monkeypatch.setattr(controller_mod, "_rust_marking_update", _fake_update)

        c = NeuroSymbolicController(
            artifact=art,
            seed_base=224,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_backend="rust",
        )
        assert c.runtime_backend_name == "rust"

        act = c.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)
        assert "dI_PF3_A" in act
        assert "dI_PF_topbot_A" in act
        assert calls["dense"] >= 1
        assert calls["update"] >= 1

    def test_runtime_backend_rust_path_uses_rust_sampling_when_available(
        self, artifact_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = load_artifact(artifact_path)
        calls = {"dense": 0, "update": 0, "sample": 0}

        def _fake_dense(w_in: np.ndarray, marking: np.ndarray) -> np.ndarray:
            calls["dense"] += 1
            return np.asarray(w_in @ marking, dtype=np.float64)

        def _fake_update(
            marking: np.ndarray,
            w_in: np.ndarray,
            w_out: np.ndarray,
            firing: np.ndarray,
        ) -> np.ndarray:
            calls["update"] += 1
            cons = w_in.T @ firing
            prod = w_out @ firing
            return np.asarray(np.clip(marking - cons + prod, 0.0, 1.0), dtype=np.float64)

        def _fake_sample(
            p_fire: np.ndarray, n_passes: int, seed: int, antithetic: bool
        ) -> np.ndarray:
            calls["sample"] += 1
            assert n_passes > 1
            assert seed >= 0
            _ = antithetic
            return np.asarray(p_fire, dtype=np.float64)

        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", True)
        monkeypatch.setattr(controller_mod, "_rust_dense_activations", _fake_dense)
        monkeypatch.setattr(controller_mod, "_rust_marking_update", _fake_update)
        monkeypatch.setattr(controller_mod, "_rust_sample_firing", _fake_sample)

        c = NeuroSymbolicController(
            artifact=art,
            seed_base=227,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            runtime_backend="rust",
            sc_n_passes=8,
            sc_bitflip_rate=0.0,
            sc_binary_margin=0.2,
        )
        assert c.runtime_backend_name == "rust"

        act = c.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, 0)
        assert "dI_PF3_A" in act
        assert "dI_PF_topbot_A" in act
        assert calls["dense"] >= 1
        assert calls["update"] >= 1
        assert calls["sample"] >= 1

    def test_marking_property_returns_copy_not_alias(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=225,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )
        m = c.marking
        assert len(m) == art.nP
        m[0] = 1.0 - m[0]
        assert c.marking[0] != m[0]

    def test_marking_setter_validates_length(self, artifact_path: str) -> None:
        art = load_artifact(artifact_path)
        c = NeuroSymbolicController(
            artifact=art,
            seed_base=226,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        )
        with pytest.raises(ValueError, match="marking must have length"):
            c.marking = [0.0, 1.0]


# ═════════════════════════════════════════════════════════════════════════════
# Contract helpers
# ═════════════════════════════════════════════════════════════════════════════


class TestContracts:
    def test_extract_features_on_target(self) -> None:
        """When obs == target, all features should be 0."""
        obs: ControlObservation = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        feats = extract_features(
            obs, ControlTargets(), ControlScales()
        )
        assert feats["x_R_pos"] == 0.0
        assert feats["x_R_neg"] == 0.0
        assert feats["x_Z_pos"] == 0.0
        assert feats["x_Z_neg"] == 0.0

    def test_extract_features_positive_error(self) -> None:
        """R below target → positive R error → x_R_pos > 0."""
        obs: ControlObservation = {"R_axis_m": 6.0, "Z_axis_m": 0.0}
        feats = extract_features(
            obs, ControlTargets(), ControlScales()
        )
        assert feats["x_R_pos"] > 0.0
        assert feats["x_R_neg"] == 0.0

    def test_extract_features_negative_error(self) -> None:
        """R above target → negative R error → x_R_neg > 0."""
        obs: ControlObservation = {"R_axis_m": 6.5, "Z_axis_m": 0.0}
        feats = extract_features(
            obs, ControlTargets(), ControlScales()
        )
        assert feats["x_R_pos"] == 0.0
        assert feats["x_R_neg"] > 0.0

    def test_extract_features_clamped(self) -> None:
        """Extreme obs → features clamped to [0, 1]."""
        obs: ControlObservation = {"R_axis_m": 100.0, "Z_axis_m": -100.0}
        feats = extract_features(
            obs, ControlTargets(), ControlScales()
        )
        for v in feats.values():
            assert 0.0 <= v <= 1.0

    def test_extract_features_custom_axes(self) -> None:
        obs = {"beta_n": 2.2, "q95": 4.5}
        axes = [
            FeatureAxisSpec(
                obs_key="beta_n",
                target=1.8,
                scale=1.0,
                pos_key="x_beta_pos",
                neg_key="x_beta_neg",
            ),
            FeatureAxisSpec(
                obs_key="q95",
                target=5.0,
                scale=2.0,
                pos_key="x_q95_pos",
                neg_key="x_q95_neg",
            ),
        ]
        feats = extract_features(
            obs=obs,
            targets=ControlTargets(),
            scales=ControlScales(),
            feature_axes=axes,
        )
        assert set(feats.keys()) == {"x_beta_pos", "x_beta_neg", "x_q95_pos", "x_q95_neg"}
        assert feats["x_beta_pos"] == 0.0
        assert feats["x_beta_neg"] > 0.0
        assert feats["x_q95_pos"] > 0.0
        assert feats["x_q95_neg"] == 0.0

    def test_extract_features_passthrough_keys(self) -> None:
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "density_norm": 0.73}
        feats = extract_features(
            obs=obs,
            targets=ControlTargets(),
            scales=ControlScales(),
            passthrough_keys=["density_norm"],
        )
        assert "density_norm" in feats
        assert abs(feats["density_norm"] - 0.73) < 1e-12

    def test_extract_features_rejects_nonfinite_axis_observation(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            extract_features(
                obs={"R_axis_m": float("nan"), "Z_axis_m": 0.0},
                targets=ControlTargets(),
                scales=ControlScales(),
            )

    def test_extract_features_rejects_nonfinite_passthrough_observation(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            extract_features(
                obs={"R_axis_m": 6.2, "Z_axis_m": 0.0, "density_norm": float("inf")},
                targets=ControlTargets(),
                scales=ControlScales(),
                passthrough_keys=["density_norm"],
            )

    def test_extract_features_rejects_nonfinite_custom_axis_scale(self) -> None:
        with pytest.raises(ValueError, match="scale must be finite"):
            extract_features(
                obs={"beta_n": 1.8},
                targets=ControlTargets(),
                scales=ControlScales(),
                feature_axes=[
                    FeatureAxisSpec(
                        obs_key="beta_n",
                        target=2.0,
                        scale=float("nan"),
                        pos_key="x_beta_pos",
                        neg_key="x_beta_neg",
                    )
                ],
            )

    def test_decode_actions_basic(self) -> None:
        marking = [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.7, 0.1]
        specs = [
            ActionSpec(name="dI_PF3_A", pos_place=4, neg_place=5),
            ActionSpec(name="dI_PF_topbot_A", pos_place=6, neg_place=7),
        ]
        prev = [0.0, 0.0]
        result = decode_actions(
            marking, specs, gains=[100.0, 100.0],
            abs_max=[5000.0, 5000.0], slew_per_s=[1e6, 1e6],
            dt=0.001, prev=prev,
        )
        assert abs(result["dI_PF3_A"] - 60.0) < 1e-10
        assert abs(result["dI_PF_topbot_A"] - 60.0) < 1e-10

    def test_decode_actions_rejects_vector_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            decode_actions(
                marking=[0.0, 1.0],
                actions_spec=[ActionSpec(name="a", pos_place=0, neg_place=1)],
                gains=[1.0, 2.0],
                abs_max=[1.0],
                slew_per_s=[1.0],
                dt=0.01,
                prev=[0.0],
            )

    def test_decode_actions_rejects_nonpositive_or_nonfinite_dt(self) -> None:
        specs = [ActionSpec(name="a", pos_place=0, neg_place=1)]
        kwargs = dict(
            marking=[0.0, 1.0],
            actions_spec=specs,
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[1.0],
            prev=[0.0],
        )
        with pytest.raises(ValueError, match="dt must be finite and > 0"):
            decode_actions(dt=0.0, **kwargs)
        with pytest.raises(ValueError, match="dt must be finite and > 0"):
            decode_actions(dt=float("nan"), **kwargs)

    def test_decode_actions_rejects_negative_place_index(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            decode_actions(
                marking=[0.0, 1.0],
                actions_spec=[ActionSpec(name="a", pos_place=-1, neg_place=1)],
                gains=[1.0],
                abs_max=[1.0],
                slew_per_s=[1.0],
                dt=0.01,
                prev=[0.0],
            )

    def test_decode_actions_rejects_out_of_bounds_place_index(self) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            decode_actions(
                marking=[0.0, 1.0],
                actions_spec=[ActionSpec(name="a", pos_place=0, neg_place=2)],
                gains=[1.0],
                abs_max=[1.0],
                slew_per_s=[1.0],
                dt=0.01,
                prev=[0.0],
            )

    def test_clip01(self) -> None:
        assert _clip01(-0.5) == 0.0
        assert _clip01(0.5) == 0.5
        assert _clip01(1.5) == 1.0
