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


# ── Packet B: Compiler tests ────────────────────────────────────────────────


class TestFusionCompiler:
    def test_compiled_net_shapes(self, compiled: CompiledNet) -> None:
        assert compiled.W_in.shape == (3, 3)
        assert compiled.W_out.shape == (3, 3)
        assert compiled.n_places == 3
        assert compiled.n_transitions == 3

    def test_thresholds(self, compiled: CompiledNet) -> None:
        np.testing.assert_array_equal(compiled.thresholds, [0.5, 0.5, 0.5])

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
