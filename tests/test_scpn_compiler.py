# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Test Scpn Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Unit tests for Packets A & B — StochasticPetriNet + FusionCompiler.

Uses a 3-place, 3-transition traffic-light Petri Net:

    (Red) --1.0--> [T_r2g] --1.0--> (Green)
    (Green) --1.0--> [T_g2y] --1.0--> (Yellow)
    (Yellow) --1.0--> [T_y2r] --1.0--> (Red)

Initial marking: Red=1.0, Green=0.0, Yellow=0.0
All thresholds: 0.5
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn.compiler import FusionCompiler, CompiledNet, _HAS_SC_NEUROCORE


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _build_traffic_light() -> StochasticPetriNet:
    """Build the canonical 3-place cyclic traffic-light net."""
    net = StochasticPetriNet()

    net.add_place("Red", initial_tokens=1.0)
    net.add_place("Green", initial_tokens=0.0)
    net.add_place("Yellow", initial_tokens=0.0)

    net.add_transition("T_r2g", threshold=0.5)
    net.add_transition("T_g2y", threshold=0.5)
    net.add_transition("T_y2r", threshold=0.5)

    # Input arcs: Place -> Transition (consumption)
    net.add_arc("Red", "T_r2g", weight=1.0)
    net.add_arc("Green", "T_g2y", weight=1.0)
    net.add_arc("Yellow", "T_y2r", weight=1.0)

    # Output arcs: Transition -> Place (production)
    net.add_arc("T_r2g", "Green", weight=1.0)
    net.add_arc("T_g2y", "Yellow", weight=1.0)
    net.add_arc("T_y2r", "Red", weight=1.0)

    net.compile()
    return net


@pytest.fixture
def traffic_net() -> StochasticPetriNet:
    return _build_traffic_light()


@pytest.fixture
def compiled(traffic_net: StochasticPetriNet) -> CompiledNet:
    compiler = FusionCompiler(bitstream_length=1024, seed=42)
    return compiler.compile(traffic_net)


# ── Packet A: StochasticPetriNet tests ───────────────────────────────────────


class TestStochasticPetriNet:
    def test_counts(self, traffic_net: StochasticPetriNet) -> None:
        assert traffic_net.n_places == 3
        assert traffic_net.n_transitions == 3

    def test_names(self, traffic_net: StochasticPetriNet) -> None:
        assert traffic_net.place_names == ["Red", "Green", "Yellow"]
        assert traffic_net.transition_names == ["T_r2g", "T_g2y", "T_y2r"]

    def test_initial_marking(self, traffic_net: StochasticPetriNet) -> None:
        m = traffic_net.get_initial_marking()
        np.testing.assert_array_equal(m, [1.0, 0.0, 0.0])

    def test_thresholds(self, traffic_net: StochasticPetriNet) -> None:
        th = traffic_net.get_thresholds()
        np.testing.assert_array_equal(th, [0.5, 0.5, 0.5])

    def test_delay_ticks_default_zero(self, traffic_net: StochasticPetriNet) -> None:
        delays = traffic_net.get_delay_ticks()
        np.testing.assert_array_equal(delays, [0, 0, 0])

    def test_W_in_shape(self, traffic_net: StochasticPetriNet) -> None:
        assert traffic_net.W_in.shape == (3, 3)  # (nT, nP)

    def test_W_out_shape(self, traffic_net: StochasticPetriNet) -> None:
        assert traffic_net.W_out.shape == (3, 3)  # (nP, nT)

    def test_W_in_sparsity(self, traffic_net: StochasticPetriNet) -> None:
        """Each transition has exactly one input place."""
        assert traffic_net.W_in.nnz == 3
        W = traffic_net.W_in.toarray()
        # T_r2g reads Red (col 0), T_g2y reads Green (col 1), T_y2r reads Yellow (col 2)
        expected = np.array([
            [1.0, 0.0, 0.0],  # T_r2g <- Red
            [0.0, 1.0, 0.0],  # T_g2y <- Green
            [0.0, 0.0, 1.0],  # T_y2r <- Yellow
        ])
        np.testing.assert_array_equal(W, expected)

    def test_W_out_sparsity(self, traffic_net: StochasticPetriNet) -> None:
        """Each transition produces into exactly one place."""
        assert traffic_net.W_out.nnz == 3
        W = traffic_net.W_out.toarray()
        # T_r2g -> Green (row 1), T_g2y -> Yellow (row 2), T_y2r -> Red (row 0)
        expected = np.array([
            [0.0, 0.0, 1.0],  # Red   <- T_y2r
            [1.0, 0.0, 0.0],  # Green <- T_r2g
            [0.0, 1.0, 0.0],  # Yellow <- T_g2y
        ])
        np.testing.assert_array_equal(W, expected)

    def test_summary(self, traffic_net: StochasticPetriNet) -> None:
        s = traffic_net.summary()
        assert "P=3" in s
        assert "T=3" in s
        assert "compiled=True" in s

    def test_compiled_flag(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A")
        net.add_transition("T")
        assert not net.is_compiled
        net.add_arc("A", "T")
        net.compile()
        assert net.is_compiled
        # Adding an arc invalidates compilation
        net.add_place("B")
        assert not net.is_compiled

    # ── Validation tests ──

    def test_duplicate_node_rejected(self) -> None:
        net = StochasticPetriNet()
        net.add_place("X")
        with pytest.raises(ValueError, match="already exists"):
            net.add_place("X")

    def test_same_kind_arc_rejected(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A")
        net.add_place("B")
        with pytest.raises(ValueError, match="Place<->Transition"):
            net.add_arc("A", "B")

    def test_unknown_node_arc_rejected(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("A", "ghost")

    def test_empty_net_rejected(self) -> None:
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="at least one"):
            net.compile()

    def test_token_range_rejected(self) -> None:
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            net.add_place("bad", initial_tokens=1.5)

    def test_negative_weight_rejected(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A")
        net.add_transition("T")
        with pytest.raises(ValueError, match="weight must be > 0"):
            net.add_arc("A", "T", weight=-0.5)

    def test_negative_delay_ticks_rejected(self) -> None:
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="delay_ticks"):
            net.add_transition("T_bad", threshold=0.5, delay_ticks=-1)

    def test_transition_delay_ticks_persist(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=1.0)
        net.add_transition("T", threshold=0.5, delay_ticks=3)
        net.add_arc("P", "T", weight=1.0)
        net.add_arc("T", "P", weight=1.0)
        net.compile()
        np.testing.assert_array_equal(net.get_delay_ticks(), [3])
        assert "delay_ticks=3" in net.summary()

    def test_validate_topology_detects_dead_nodes(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A", initial_tokens=1.0)
        net.add_place("B_dead", initial_tokens=0.0)
        net.add_transition("T_live", threshold=0.5)
        net.add_transition("T_dead", threshold=0.5)
        net.add_arc("A", "T_live", weight=1.0)
        net.add_arc("T_live", "A", weight=1.0)

        report = net.validate_topology()
        assert report["dead_places"] == ["B_dead"]
        assert report["dead_transitions"] == ["T_dead"]
        assert report["unseeded_place_cycles"] == []

    def test_validate_topology_detects_unseeded_place_cycle(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P0", initial_tokens=0.0)
        net.add_place("P1", initial_tokens=0.0)
        net.add_transition("T0", threshold=0.1)
        net.add_transition("T1", threshold=0.1)
        net.add_arc("P0", "T0", weight=1.0)
        net.add_arc("T0", "P1", weight=1.0)
        net.add_arc("P1", "T1", weight=1.0)
        net.add_arc("T1", "P0", weight=1.0)

        report = net.validate_topology()
        assert report["dead_places"] == []
        assert report["dead_transitions"] == []
        assert report["unseeded_place_cycles"] == [["P0", "P1"]]

    def test_compile_validate_topology_populates_report(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A", initial_tokens=1.0)
        net.add_place("B_dead", initial_tokens=0.0)
        net.add_transition("T_live", threshold=0.5)
        net.add_transition("T_dead", threshold=0.5)
        net.add_arc("A", "T_live", weight=1.0)
        net.add_arc("T_live", "A", weight=1.0)

        net.compile(validate_topology=True)
        assert net.is_compiled
        assert net.last_validation_report is not None
        assert net.last_validation_report["dead_places"] == ["B_dead"]
        assert net.last_validation_report["dead_transitions"] == ["T_dead"]

    def test_compile_strict_validation_rejects_dead_nodes(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A", initial_tokens=1.0)
        net.add_place("B_dead", initial_tokens=0.0)
        net.add_transition("T_live", threshold=0.5)
        net.add_transition("T_dead", threshold=0.5)
        net.add_arc("A", "T_live", weight=1.0)
        net.add_arc("T_live", "A", weight=1.0)

        with pytest.raises(ValueError, match="Topology validation failed"):
            net.compile(strict_validation=True)

    def test_inhibitor_arc_requires_opt_in_during_compile(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.0)
        net.add_transition("T", threshold=0.5)
        net.add_arc("P", "T", weight=1.0, inhibitor=True)
        with pytest.raises(ValueError, match="allow_inhibitor=True"):
            net.compile()

    def test_inhibitor_arc_compiles_with_negative_weight_when_opted_in(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.0)
        net.add_transition("T", threshold=0.5)
        net.add_arc("P", "T", weight=2.0, inhibitor=True)
        net.compile(allow_inhibitor=True)
        assert net.W_in is not None
        np.testing.assert_array_equal(net.W_in.toarray(), [[-2.0]])

    def test_inhibitor_arc_rejects_transition_to_place_direction(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.0)
        net.add_transition("T", threshold=0.5)
        with pytest.raises(
            ValueError, match="only supported for Place->Transition"
        ):
            net.add_arc("T", "P", weight=1.0, inhibitor=True)


# ── Packet B: Compiler tests ────────────────────────────────────────────────


class TestFusionCompiler:
    def test_compile_strict_topology_rejects_dead_nodes(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A", initial_tokens=1.0)
        net.add_place("B_dead", initial_tokens=0.0)
        net.add_transition("T_live", threshold=0.5)
        net.add_transition("T_dead", threshold=0.5)
        net.add_arc("A", "T_live", weight=1.0)
        net.add_arc("T_live", "A", weight=1.0)

        compiler = FusionCompiler(bitstream_length=128, seed=7)
        with pytest.raises(ValueError, match="Topology validation failed"):
            compiler.compile(net, strict_topology=True)

    def test_compile_validate_topology_populates_report(self) -> None:
        net = StochasticPetriNet()
        net.add_place("A", initial_tokens=1.0)
        net.add_place("B_dead", initial_tokens=0.0)
        net.add_transition("T_live", threshold=0.5)
        net.add_transition("T_dead", threshold=0.5)
        net.add_arc("A", "T_live", weight=1.0)
        net.add_arc("T_live", "A", weight=1.0)

        compiler = FusionCompiler(bitstream_length=128, seed=7)
        compiled = compiler.compile(net, validate_topology=True)
        assert compiled.n_places == 2
        assert compiled.n_transitions == 2
        assert net.last_validation_report is not None
        assert net.last_validation_report["dead_places"] == ["B_dead"]
        assert net.last_validation_report["dead_transitions"] == ["T_dead"]

    def test_compile_allow_inhibitor_opt_in(self) -> None:
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.0)
        net.add_transition("T", threshold=0.5)
        net.add_arc("P", "T", weight=2.0, inhibitor=True)

        compiler = FusionCompiler(bitstream_length=128, seed=7)
        with pytest.raises(ValueError, match="allow_inhibitor=True"):
            compiler.compile(net)

        compiled = compiler.compile(net, allow_inhibitor=True)
        np.testing.assert_array_equal(compiled.W_in, [[-2.0]])

    def test_compiled_net_shapes(self, compiled: CompiledNet) -> None:
        assert compiled.W_in.shape == (3, 3)
        assert compiled.W_out.shape == (3, 3)
        assert compiled.n_places == 3
        assert compiled.n_transitions == 3

    def test_thresholds(self, compiled: CompiledNet) -> None:
        np.testing.assert_array_equal(compiled.thresholds, [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(compiled.transition_delay_ticks, [0, 0, 0])

    def test_initial_marking(self, compiled: CompiledNet) -> None:
        np.testing.assert_array_equal(
            compiled.initial_marking, [1.0, 0.0, 0.0]
        )

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_neurons_count(self, compiled: CompiledNet) -> None:
        assert len(compiled.neurons) == 3

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_neuron_thresholds(self, compiled: CompiledNet) -> None:
        for neuron in compiled.neurons:
            assert neuron.v_threshold == 0.5

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_neuron_no_leak(self, compiled: CompiledNet) -> None:
        """tau_mem=1e6 means effectively no leak."""
        for neuron in compiled.neurons:
            assert neuron.tau_mem == 1e6
            assert neuron.noise_std == 0.0

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_custom_lif_runtime_parameters(self, traffic_net: StochasticPetriNet) -> None:
        compiler = FusionCompiler(
            bitstream_length=128,
            seed=7,
            lif_tau_mem=10.0,
            lif_noise_std=0.1,
            lif_dt=0.5,
            lif_resistance=2.0,
            lif_refractory_period=3,
        )
        compiled = compiler.compile(traffic_net)
        assert compiled.lif_tau_mem == 10.0
        assert compiled.lif_noise_std == 0.1
        assert compiled.lif_dt == 0.5
        assert compiled.lif_resistance == 2.0
        assert compiled.lif_refractory_period == 3
        for neuron in compiled.neurons:
            assert neuron.tau_mem == 10.0
            assert neuron.noise_std == 0.1
            assert neuron.dt == 0.5
            assert neuron.resistance == 2.0
            assert neuron.refractory_period == 3

    def test_reactor_lif_factory_defaults(self, traffic_net: StochasticPetriNet) -> None:
        compiler = FusionCompiler.with_reactor_lif_defaults(
            bitstream_length=256,
            seed=11,
        )
        assert compiler.lif_tau_mem == 10.0
        assert compiler.lif_noise_std == 0.1
        assert compiler.lif_dt == 1.0
        assert compiler.lif_resistance == 1.0
        assert compiler.lif_refractory_period == 0

        compiled = compiler.compile(traffic_net)
        assert compiled.lif_tau_mem == 10.0
        assert compiled.lif_noise_std == 0.1

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_packed_weight_shapes(self, compiled: CompiledNet) -> None:
        n_words = int(np.ceil(1024 / 64))  # 16
        assert compiled.W_in_packed.shape == (3, 3, n_words)
        assert compiled.W_out_packed.shape == (3, 3, n_words)
        assert compiled.W_in_packed.dtype == np.uint64

    def test_float_forward_known_marking(self, compiled: CompiledNet) -> None:
        """Red=1.0 → only T_r2g should have activation=1.0."""
        marking = np.array([1.0, 0.0, 0.0])
        activations = compiled.dense_forward_float(compiled.W_in, marking)
        np.testing.assert_allclose(activations, [1.0, 0.0, 0.0])

    def test_lif_fire_above_threshold(self, compiled: CompiledNet) -> None:
        """Activation 1.0 > threshold 0.5 → fire."""
        currents = np.array([1.0, 0.0, 0.0])
        fired = compiled.lif_fire(currents)
        assert fired[0] == 1
        assert fired[1] == 0
        assert fired[2] == 0

    def test_lif_fire_below_threshold(self, compiled: CompiledNet) -> None:
        """Activation 0.3 < threshold 0.5 → don't fire."""
        currents = np.array([0.3, 0.3, 0.3])
        fired = compiled.lif_fire(currents)
        np.testing.assert_array_equal(fired, [0, 0, 0])

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_dense_forward_stochastic_high_activation(
        self, compiled: CompiledNet
    ) -> None:
        """Marking [1.0, 0.0, 0.0] through W_in → T_r2g should be ~1.0."""
        marking = np.array([1.0, 0.0, 0.0])
        activations = compiled.dense_forward(compiled.W_in_packed, marking)
        # Stochastic: expect ~1.0 for T_r2g, ~0.0 for others
        assert activations[0] > 0.8, f"Expected ~1.0, got {activations[0]}"
        assert activations[1] < 0.2, f"Expected ~0.0, got {activations[1]}"
        assert activations[2] < 0.2, f"Expected ~0.0, got {activations[2]}"

    @pytest.mark.skipif(
        not _HAS_SC_NEUROCORE, reason="sc_neurocore not installed"
    )
    def test_stochastic_matches_float(self, compiled: CompiledNet) -> None:
        """Stochastic path should approximate float path within tolerance."""
        marking = np.array([0.8, 0.3, 0.0])
        float_result = compiled.dense_forward_float(compiled.W_in, marking)
        stochastic_result = compiled.dense_forward(
            compiled.W_in_packed, marking
        )
        np.testing.assert_allclose(
            stochastic_result,
            float_result,
            atol=0.15,
            err_msg="Stochastic path deviates >0.15 from float path",
        )

    def test_export_artifact_prefers_env_git_sha(
        self,
        traffic_net: StochasticPetriNet,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        compiler = FusionCompiler(bitstream_length=128, seed=7)
        compiled = compiler.compile(traffic_net)
        monkeypatch.setenv("SCPN_GIT_SHA", "abcdef1234567890")

        artifact = compiled.export_artifact(name="sha_test")
        assert artifact.meta.compiler.git_sha == "abcdef1"
        assert all(t.delay_ticks == 0 for t in artifact.topology.transitions)


# ── Integration: 10-step cyclic flow ────────────────────────────────────────


class TestCyclicFlow:
    """Run 10 steps of the traffic-light using the float path.

    Expected cycle: Red → Green → Yellow → Red → ...
    """

    def test_ten_step_cycle(self, compiled: CompiledNet) -> None:
        marking = compiled.initial_marking.copy()
        history: list[np.ndarray] = [marking.copy()]

        for _ in range(10):
            # Phase 1: compute which transitions fire
            activations = compiled.dense_forward_float(
                compiled.W_in, marking
            )
            fired = compiled.lif_fire(activations)
            fired_float = fired.astype(np.float64)

            # Phase 2: update marking
            consumption = compiled.W_in.T @ fired_float
            production = compiled.dense_forward_float(
                compiled.W_out, fired_float
            )
            marking = marking - consumption + production
            marking = np.clip(marking, 0.0, 1.0)
            history.append(marking.copy())

        # The cycle is Red(1,0,0) → Green(0,1,0) → Yellow(0,0,1) → Red ...
        expected_cycle = [
            [1.0, 0.0, 0.0],  # step 0: Red
            [0.0, 1.0, 0.0],  # step 1: Green
            [0.0, 0.0, 1.0],  # step 2: Yellow
            [1.0, 0.0, 0.0],  # step 3: Red
            [0.0, 1.0, 0.0],  # step 4: Green
            [0.0, 0.0, 1.0],  # step 5: Yellow
            [1.0, 0.0, 0.0],  # step 6: Red
            [0.0, 1.0, 0.0],  # step 7: Green
            [0.0, 0.0, 1.0],  # step 8: Yellow
            [1.0, 0.0, 0.0],  # step 9: Red
            [0.0, 1.0, 0.0],  # step 10: Green
        ]

        for step, (actual, expected) in enumerate(
            zip(history, expected_cycle)
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-10,
                err_msg=f"Step {step}: expected {expected}, got {actual}",
            )

    def test_token_conservation(self, compiled: CompiledNet) -> None:
        """Total token mass should remain 1.0 across all steps."""
        marking = compiled.initial_marking.copy()

        for _ in range(30):
            activations = compiled.dense_forward_float(
                compiled.W_in, marking
            )
            fired = compiled.lif_fire(activations)
            fired_float = fired.astype(np.float64)

            consumption = compiled.W_in.T @ fired_float
            production = compiled.dense_forward_float(
                compiled.W_out, fired_float
            )
            marking = marking - consumption + production
            marking = np.clip(marking, 0.0, 1.0)

            assert abs(marking.sum() - 1.0) < 1e-10, (
                f"Token mass {marking.sum()} != 1.0"
            )

    def test_summary_string(self, compiled: CompiledNet) -> None:
        s = compiled.summary()
        assert "P=3" in s
        assert "T=3" in s

    def test_bitstream_length_minimum(self) -> None:
        with pytest.raises(ValueError, match="bitstream_length"):
            FusionCompiler(bitstream_length=32)

    def test_invalid_lif_parameter_rejected(self) -> None:
        with pytest.raises(ValueError, match="lif_tau_mem"):
            FusionCompiler(lif_tau_mem=0.0)
