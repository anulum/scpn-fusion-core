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
    load_artifact,
    save_artifact,
)
from scpn_fusion.scpn.controller import NeuroSymbolicController


# ── Fixture: 8-place controller net ─────────────────────────────────────────
# 4 feature inputs (x_R_pos, x_R_neg, x_Z_pos, x_Z_neg)
# 4 action outputs (a_R_pos, a_R_neg, a_Z_pos, a_Z_neg)
# 4 pass-through transitions: T_Rp, T_Rn, T_Zp, T_Zn
# Each transition reads one input place and writes one output place.


def _build_controller_net() -> StochasticPetriNet:
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
    net.add_transition("T_Rp", threshold=0.1)
    net.add_transition("T_Rn", threshold=0.1)
    net.add_transition("T_Zp", threshold=0.1)
    net.add_transition("T_Zn", threshold=0.1)

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
) -> str:
    """Compile the 8-place net, export artifact, save to temp file."""
    net = _build_controller_net()
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

    def test_clip01(self) -> None:
        assert _clip01(-0.5) == 0.0
        assert _clip01(0.5) == 0.5
        assert _clip01(1.5) == 1.0
