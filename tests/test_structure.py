# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stochastic Petri Net Structure Tests
"""Tests for the stochastic Petri-net structure, validation, and Monte-Carlo verification."""

from __future__ import annotations

import pytest

from scpn_fusion.scpn.structure import StochasticPetriNet


def _net() -> StochasticPetriNet:
    """Build and compile a minimal valid place/transition/place net."""
    net = StochasticPetriNet()
    net.add_place("p1", initial_tokens=1.0)
    net.add_place("p2", initial_tokens=0.0)
    net.add_transition("t1", threshold=0.9)
    net.add_arc("p1", "t1", weight=1.0)
    net.add_arc("t1", "p2", weight=1.0)
    net.compile()
    return net


class TestConstruction:
    """Place/transition/arc construction and its input contracts."""

    def test_counts_and_names(self) -> None:
        """A compiled net reports its place and transition counts and names."""
        net = _net()
        assert net.n_places == 2
        assert net.n_transitions == 1
        assert net.place_names == ["p1", "p2"]
        assert net.transition_names == ["t1"]
        assert net.is_compiled is True

    def test_duplicate_node_rejected(self) -> None:
        """Re-adding an existing node name is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        with pytest.raises(ValueError, match="already exists"):
            net.add_place("p1")

    def test_initial_tokens_range_enforced(self) -> None:
        """Initial tokens outside [0, 1] are rejected."""
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="initial_tokens"):
            net.add_place("p1", initial_tokens=2.0)

    def test_transition_parameter_contracts(self) -> None:
        """Negative threshold or delay is rejected."""
        net = StochasticPetriNet()
        with pytest.raises(ValueError, match="threshold"):
            net.add_transition("t1", threshold=-0.1)
        with pytest.raises(ValueError, match="delay_ticks"):
            net.add_transition("t2", delay_ticks=-1)

    def test_duplicate_transition_rejected(self) -> None:
        """Re-adding an existing transition name is rejected."""
        net = StochasticPetriNet()
        net.add_transition("t1")
        with pytest.raises(ValueError, match="already exists"):
            net.add_transition("t1")


class TestArcContracts:
    """Arc-direction and weight validation."""

    def test_unknown_node_rejected(self) -> None:
        """An arc referencing an unknown node is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("p1", "ghost")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("ghost", "p1")

    def test_same_kind_arc_rejected(self) -> None:
        """A place-to-place (same-kind) arc is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_place("p2")
        with pytest.raises(ValueError, match="Place<->Transition"):
            net.add_arc("p1", "p2")

    def test_nonpositive_weight_rejected(self) -> None:
        """A non-positive arc weight is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_transition("t1")
        with pytest.raises(ValueError, match="weight must be > 0"):
            net.add_arc("p1", "t1", weight=0.0)

    def test_inhibitor_must_be_place_to_transition(self) -> None:
        """An inhibitor arc on a Transition->Place edge is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_transition("t1")
        with pytest.raises(ValueError, match="inhibitor arcs are only"):
            net.add_arc("t1", "p1", inhibitor=True)

    def test_inhibitor_weight_magnitude_enforced(self) -> None:
        """An inhibitor arc with non-positive magnitude is rejected."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_transition("t1")
        with pytest.raises(ValueError, match="inhibitor arc weight"):
            net.add_arc("p1", "t1", weight=0.0, inhibitor=True)

    def test_inhibitor_arc_requires_explicit_compile_opt_in(self) -> None:
        """A valid inhibitor arc is negative and requires compile opt-in."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_place("p2")
        net.add_transition("t1")
        net.add_arc("p1", "t1", weight=0.25, inhibitor=True)
        net.add_arc("t1", "p2")

        with pytest.raises(ValueError, match="Negative input arc weights"):
            net.compile()

        net.compile(allow_inhibitor=True)
        assert net.W_in is not None
        assert net.W_in.toarray()[0, 0] == -0.25


class TestCompile:
    """Compilation and strict topology validation."""

    def test_empty_net_rejected(self) -> None:
        """Compiling a net with no places or transitions is rejected."""
        with pytest.raises(ValueError, match="at least one place"):
            StochasticPetriNet().compile()

    def test_strict_validation_rejects_dead_nodes(self) -> None:
        """Strict compilation rejects a topology with dead places."""
        net = StochasticPetriNet()
        net.add_place("p1", initial_tokens=1.0)
        net.add_place("dead", initial_tokens=0.0)
        net.add_transition("t1", threshold=0.5)
        net.add_arc("p1", "t1")
        net.add_arc("t1", "p1")
        with pytest.raises(ValueError, match="Topology validation failed"):
            net.compile(strict_validation=True)

    def test_strict_validation_reports_all_issue_classes(self) -> None:
        """Strict validation includes dead transitions, cycles, and overflow."""
        net = StochasticPetriNet()
        net.add_place("cycle_a", initial_tokens=0.0)
        net.add_place("cycle_b", initial_tokens=0.0)
        net.add_place("overflow_a", initial_tokens=1.0)
        net.add_place("overflow_b", initial_tokens=1.0)
        net.add_transition("a_to_b")
        net.add_transition("b_to_a")
        net.add_transition("overflow_t")
        net.add_transition("dead_t")
        net.add_arc("cycle_a", "a_to_b")
        net.add_arc("a_to_b", "cycle_b")
        net.add_arc("cycle_b", "b_to_a")
        net.add_arc("b_to_a", "cycle_a")
        net.add_arc("overflow_a", "overflow_t", weight=0.7)
        net.add_arc("overflow_b", "overflow_t", weight=0.4)
        net.add_arc("overflow_t", "overflow_a")

        with pytest.raises(ValueError) as exc_info:
            net.compile(strict_validation=True)

        message = str(exc_info.value)
        assert "dead_transitions=['dead_t']" in message
        assert "unseeded_place_cycles=[['cycle_a', 'cycle_b']]" in message
        assert "input_weight_overflow_transitions=['overflow_t']" in message
        assert net.last_validation_report is not None

    def test_compile_rejects_corrupt_inhibitor_output_arc(self) -> None:
        """Defensively reject impossible inhibitor metadata on output arcs."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_transition("t1")
        net.add_arc("p1", "t1")
        net.add_arc("t1", "p1")
        net._arcs[-1] = ("t1", "p1", -1.0, True)

        with pytest.raises(ValueError, match="only valid on Place->Transition"):
            net.compile(allow_inhibitor=True)


class TestAccessorsAndSummary:
    """Accessor and textual summary contracts."""

    def test_validation_report_and_delay_accessors(self) -> None:
        """Access validation reports and delay vectors through public accessors."""
        net = StochasticPetriNet()
        assert net.last_validation_report is None
        net.add_place("p1", initial_tokens=1.0)
        net.add_place("p2", initial_tokens=0.0)
        net.add_transition("t1", threshold=0.75, delay_ticks=4)
        net.add_arc("p1", "t1")
        net.add_arc("t1", "p2")

        net.compile(validate_topology=True)

        assert net.last_validation_report == {
            "dead_places": [],
            "dead_transitions": [],
            "unseeded_place_cycles": [],
            "input_weight_overflow_transitions": [],
        }
        assert net.get_initial_marking().tolist() == [1.0, 0.0]
        assert net.get_thresholds().tolist() == [0.75]
        assert net.get_delay_ticks().tolist() == [4]

    def test_summary_includes_nodes_arcs_and_sparse_shapes(self) -> None:
        """The human summary includes topology and compiled matrix metadata."""
        text = _net().summary()

        assert "StochasticPetriNet" in text
        assert "P=2  T=1" in text
        assert "p1" in text
        assert "t1" in text
        assert "W_in  (nT=1, nP=2)  nnz=1" in text
        assert "W_out (nP=2, nT=1)  nnz=1" in text

    def test_summary_rejects_corrupt_compiled_state(self) -> None:
        """Summary fails closed when compiled matrices are unexpectedly absent."""
        net = _net()
        net.W_in = None

        with pytest.raises(RuntimeError, match="missing sparse matrices"):
            net.summary()


class TestVerification:
    """Monte-Carlo boundedness and liveness verification."""

    def test_verify_requires_compiled_net(self) -> None:
        """Verification before compilation raises a runtime error."""
        net = StochasticPetriNet()
        net.add_place("p1")
        net.add_transition("t1")
        net.add_arc("p1", "t1")
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness(n_steps=2, n_trials=2)
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness(n_steps=2, n_trials=2)

    def test_verify_boundedness_reports_bounds(self) -> None:
        """Boundedness verification reports markings within the unit interval."""
        report = _net().verify_boundedness(n_steps=20, n_trials=5)
        assert report["bounded"] is True
        assert isinstance(report["max_marking"], float)
        assert isinstance(report["min_marking"], float)
        max_marking = report["max_marking"]
        assert isinstance(max_marking, float)
        assert max_marking <= 1.0 + 1e-9
        assert report["n_trials"] == 5
        assert report["n_steps"] == 20

    def test_verify_liveness_reports_fire_fractions(self) -> None:
        """Liveness verification reports a per-transition firing fraction."""
        report = _net().verify_liveness(n_steps=20, n_trials=10)
        assert isinstance(report["live"], bool)
        fire_pct = report["transition_fire_pct"]
        assert isinstance(fire_pct, dict)
        assert "t1" in fire_pct
        assert isinstance(report["min_fire_pct"], float)
        assert report["n_trials"] == 10
