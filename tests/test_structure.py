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
