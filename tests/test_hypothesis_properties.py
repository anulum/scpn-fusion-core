# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Hypothesis Property-Based Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Property-based tests using Hypothesis for SCPN Fusion Core.

Covers:
  - StochasticPetriNet topology invariants
  - FusionCompiler matrix properties
  - Neuro-symbolic controller determinism
  - Physics module monotonicity / NaN-freedom
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn.compiler import FusionCompiler, CompiledNet


def _to_dense(arr):
    """Convert sparse or dense array to dense numpy array."""
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


# ── Strategies ────────────────────────────────────────────────────────

tokens_01 = st.floats(min_value=0.0, max_value=1.0)
threshold_pos = st.floats(min_value=0.01, max_value=10.0)
weight_pos = st.floats(min_value=0.01, max_value=10.0)
small_int = st.integers(min_value=1, max_value=8)


@st.composite
def petri_net(draw):
    """Generate a random StochasticPetriNet with at least 1 place, 1 transition, 1 arc."""
    n_places = draw(st.integers(min_value=1, max_value=8))
    n_transitions = draw(st.integers(min_value=1, max_value=8))

    net = StochasticPetriNet()

    for i in range(n_places):
        tok = draw(tokens_01)
        net.add_place(f"P{i}", initial_tokens=tok)

    for i in range(n_transitions):
        th = draw(threshold_pos)
        net.add_transition(f"T{i}", threshold=th)

    # At least one input arc and one output arc
    p0 = draw(st.integers(min_value=0, max_value=n_places - 1))
    t0 = draw(st.integers(min_value=0, max_value=n_transitions - 1))
    w = draw(weight_pos)
    net.add_arc(f"P{p0}", f"T{t0}", weight=w)

    p1 = draw(st.integers(min_value=0, max_value=n_places - 1))
    t1 = draw(st.integers(min_value=0, max_value=n_transitions - 1))
    w2 = draw(weight_pos)
    net.add_arc(f"T{t1}", f"P{p1}", weight=w2)

    # Optional extra arcs
    n_extra = draw(st.integers(min_value=0, max_value=5))
    for _ in range(n_extra):
        direction = draw(st.booleans())
        p = draw(st.integers(min_value=0, max_value=n_places - 1))
        t = draw(st.integers(min_value=0, max_value=n_transitions - 1))
        w_extra = draw(weight_pos)
        if direction:
            net.add_arc(f"P{p}", f"T{t}", weight=w_extra)
        else:
            net.add_arc(f"T{t}", f"P{p}", weight=w_extra)

    net.compile()
    return net


# ── Petri Net Topology Invariants ─────────────────────────────────────


class TestPetriNetProperties:
    """Property-based tests for StochasticPetriNet."""

    @given(petri_net())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_compiled_shapes(self, net: StochasticPetriNet):
        """W_in is (nT, nP) and W_out is (nP, nT)."""
        assert net.is_compiled
        assert net.W_in.shape == (net.n_transitions, net.n_places)
        assert net.W_out.shape == (net.n_places, net.n_transitions)

    @given(petri_net())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_initial_marking_bounds(self, net: StochasticPetriNet):
        """All initial token densities are in [0, 1]."""
        marking = net.get_initial_marking()
        assert marking.shape == (net.n_places,)
        assert np.all(marking >= 0.0)
        assert np.all(marking <= 1.0)

    @given(petri_net())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_thresholds_positive(self, net: StochasticPetriNet):
        """All firing thresholds are positive."""
        thresholds = net.get_thresholds()
        assert thresholds.shape == (net.n_transitions,)
        assert np.all(thresholds > 0.0)

    @given(petri_net())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_weight_matrices_nonneg(self, net: StochasticPetriNet):
        """Arc weight matrices are non-negative."""
        W_in = _to_dense(net.W_in)
        W_out = _to_dense(net.W_out)
        assert np.all(W_in >= 0.0)
        assert np.all(W_out >= 0.0)

    @given(petri_net())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_names_consistent(self, net: StochasticPetriNet):
        """Place / transition name lists match counts."""
        assert len(net.place_names) == net.n_places
        assert len(net.transition_names) == net.n_transitions
        # Names are unique
        assert len(set(net.place_names)) == net.n_places
        assert len(set(net.transition_names)) == net.n_transitions


# ── Compiler Properties ──────────────────────────────────────────────


class TestCompilerProperties:
    """Property-based tests for FusionCompiler."""

    @given(petri_net(), st.integers(min_value=64, max_value=2048))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compiled_net_preserves_topology(self, net, bitstream_len):
        """Compilation preserves place/transition counts."""
        compiler = FusionCompiler(bitstream_length=bitstream_len, seed=42)
        compiled = compiler.compile(net)

        assert compiled.n_places == net.n_places
        assert compiled.n_transitions == net.n_transitions
        assert compiled.W_in.shape == (net.n_transitions, net.n_places)
        assert compiled.W_out.shape == (net.n_places, net.n_transitions)

    @given(petri_net())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compiled_initial_marking_matches(self, net):
        """Compiled net has same initial marking as source net."""
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net)

        np.testing.assert_array_equal(
            compiled.initial_marking, net.get_initial_marking()
        )

    @given(petri_net())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_compiled_thresholds_positive(self, net):
        """Compiled thresholds are positive and match source net."""
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net)

        assert compiled.thresholds.shape == (net.n_transitions,)
        assert np.all(compiled.thresholds > 0.0)
        np.testing.assert_array_equal(compiled.thresholds, net.get_thresholds())

    @given(petri_net())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_dense_forward_no_nans(self, net):
        """Dense forward pass never produces NaN."""
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net)

        marking = net.get_initial_marking()
        output = compiled.dense_forward_float(_to_dense(compiled.W_in), marking)

        assert not np.any(np.isnan(output))
        assert output.shape == (net.n_transitions,)


# ── Compiler Seed Determinism ────────────────────────────────────────


class TestSeedDeterminism:
    """Same seed must produce identical compilations."""

    @given(petri_net(), st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_same_seed_same_result(self, net, seed):
        """Two compilations with the same seed produce identical weight matrices."""
        c1 = FusionCompiler(bitstream_length=1024, seed=seed)
        c2 = FusionCompiler(bitstream_length=1024, seed=seed)
        r1 = c1.compile(net)
        r2 = c2.compile(net)

        np.testing.assert_array_equal(_to_dense(r1.W_in), _to_dense(r2.W_in))
        np.testing.assert_array_equal(_to_dense(r1.W_out), _to_dense(r2.W_out))
        np.testing.assert_array_equal(r1.initial_marking, r2.initial_marking)
        np.testing.assert_array_equal(r1.thresholds, r2.thresholds)


# ── Validation Constraints ───────────────────────────────────────────


class TestValidationConstraints:
    """Test that invalid inputs are properly rejected."""

    @given(st.floats(min_value=-100.0, max_value=-0.01))
    def test_negative_tokens_rejected(self, tok):
        """Negative initial tokens must raise ValueError."""
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="initial_tokens"):
            net.add_place("P", initial_tokens=tok)

    @given(st.floats(min_value=1.01, max_value=100.0))
    def test_tokens_above_one_rejected(self, tok):
        """Tokens > 1 must raise ValueError."""
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="initial_tokens"):
            net.add_place("P", initial_tokens=tok)

    @given(st.floats(min_value=-100.0, max_value=-0.01))
    def test_negative_threshold_rejected(self, th):
        """Negative thresholds must raise ValueError."""
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="threshold"):
            net.add_transition("T", threshold=th)

    @given(st.floats(min_value=-100.0, max_value=0.0))
    def test_non_positive_weight_rejected(self, w):
        """Non-positive arc weight must raise ValueError."""
        net = StochasticPetriNet()
        net.add_place("P", initial_tokens=0.5)
        net.add_transition("T", threshold=0.5)
        with pytest.raises(ValueError, match="weight"):
            net.add_arc("P", "T", weight=w)

    def test_same_kind_arc_rejected(self):
        """Arc between two places or two transitions must fail."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_place("P2", initial_tokens=0.5)
        with pytest.raises(ValueError, match="Place.*Transition"):
            net.add_arc("P1", "P2", weight=1.0)

    def test_duplicate_name_rejected(self):
        """Adding a node with an existing name must fail."""
        net = StochasticPetriNet()
        net.add_place("X")
        with pytest.raises(ValueError, match="already exists"):
            net.add_place("X")
        with pytest.raises(ValueError, match="already exists"):
            net.add_transition("X")


# ── Multi-Step Token Evolution ───────────────────────────────────────


class TestTokenEvolution:
    """Property tests for stepping compiled nets."""

    @given(petri_net(), st.integers(min_value=1, max_value=50))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_marking_stays_finite(self, net, n_steps):
        """Marking never contains NaN or Inf after N steps."""
        compiler = FusionCompiler(bitstream_length=1024, seed=42)
        compiled = compiler.compile(net)

        marking = compiled.initial_marking.copy()
        W_in_dense = _to_dense(compiled.W_in)
        W_out_dense = _to_dense(compiled.W_out)

        for _ in range(n_steps):
            # Simple float-path step: fire → transfer tokens
            currents = W_in_dense @ marking
            fired = (currents >= compiled.thresholds).astype(float)
            consumed = W_in_dense.T @ fired
            produced = W_out_dense @ fired
            marking = np.clip(marking - consumed + produced, 0.0, 1e6)

        assert np.all(np.isfinite(marking)), f"Non-finite marking after {n_steps} steps"
