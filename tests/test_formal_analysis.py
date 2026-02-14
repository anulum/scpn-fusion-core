# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Formal Analysis Tests
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit and integration tests for Petri net formal analysis.

Tests cover:
  - Discretization of continuous markings
  - Reachability graph enumeration
  - 1-boundedness verification
  - L1-liveness verification
  - Mutual exclusion verification
  - Home-state verification
  - Place liveness verification
  - Complete proof certificate generation
  - Edge cases (dead transitions, empty marking, self-loops)
  - Vertical control net property verification
"""

from __future__ import annotations

import pytest

from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn.vertical_control_net import (
    VerticalControlNet,
    PLACES,
    TRANSITIONS,
)
from scpn_fusion.scpn.formal_analysis import (
    discretize_marking,
    enumerate_reachable_markings,
    check_boundedness,
    check_liveness,
    check_mutual_exclusion,
    check_home_state,
    check_place_liveness,
    generate_proof_certificate,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_simple_net() -> StochasticPetriNet:
    """A minimal 2-place, 1-transition net: P_a --(1.0)--> T1 --(1.0)--> P_b.

    With P_a initially marked, T1 fires once moving the token to P_b,
    after which no transition is enabled.
    """
    net = StochasticPetriNet()
    net.add_place("P_a", initial_tokens=1.0)
    net.add_place("P_b", initial_tokens=0.0)
    net.add_transition("T1", threshold=0.5)
    net.add_arc("P_a", "T1", weight=1.0)
    net.add_arc("T1", "P_b", weight=1.0)
    net.compile()
    return net


def _build_cyclic_net() -> StochasticPetriNet:
    """A 2-place, 2-transition cyclic net:

        P_a --(1.0)--> T1 --(1.0)--> P_b
        P_b --(1.0)--> T2 --(1.0)--> P_a

    Token cycles between P_a and P_b.  Both transitions are live,
    and P_a (the initial marking) is a home state.
    """
    net = StochasticPetriNet()
    net.add_place("P_a", initial_tokens=1.0)
    net.add_place("P_b", initial_tokens=0.0)
    net.add_transition("T1", threshold=0.5)
    net.add_transition("T2", threshold=0.5)
    net.add_arc("P_a", "T1", weight=1.0)
    net.add_arc("T1", "P_b", weight=1.0)
    net.add_arc("P_b", "T2", weight=1.0)
    net.add_arc("T2", "P_a", weight=1.0)
    net.compile()
    return net


def _build_mutex_net() -> StochasticPetriNet:
    """A 3-place, 2-transition net with mutual exclusion.

        P_src --(1.0)--> T1 --(1.0)--> P_a
        P_src --(1.0)--> T2 --(1.0)--> P_b

    Both T1 and T2 consume from P_src, so only one can fire.
    P_a and P_b should be mutually exclusive.
    """
    net = StochasticPetriNet()
    net.add_place("P_src", initial_tokens=1.0)
    net.add_place("P_a", initial_tokens=0.0)
    net.add_place("P_b", initial_tokens=0.0)
    net.add_transition("T1", threshold=0.5)
    net.add_transition("T2", threshold=0.5)
    net.add_arc("P_src", "T1", weight=1.0)
    net.add_arc("T1", "P_a", weight=1.0)
    net.add_arc("P_src", "T2", weight=1.0)
    net.add_arc("T2", "P_b", weight=1.0)
    net.compile()
    return net


def _simple_initial() -> dict:
    return {"P_a": 1.0, "P_b": 0.0}


def _cyclic_initial() -> dict:
    return {"P_a": 1.0, "P_b": 0.0}


def _mutex_initial() -> dict:
    return {"P_src": 1.0, "P_a": 0.0, "P_b": 0.0}


def _vcn_initial() -> dict:
    """Initial marking for the vertical control net (P_idle = 1.0)."""
    return {p: (1.0 if p == "P_idle" else 0.0) for p in PLACES}


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def simple_net() -> StochasticPetriNet:
    return _build_simple_net()


@pytest.fixture
def cyclic_net() -> StochasticPetriNet:
    return _build_cyclic_net()


@pytest.fixture
def mutex_net() -> StochasticPetriNet:
    return _build_mutex_net()


@pytest.fixture
def vertical_net() -> StochasticPetriNet:
    vcn = VerticalControlNet()
    net = vcn.create_net()
    net.compile()
    return net


# ── Test: Discretization ─────────────────────────────────────────────────


class TestDiscretization:
    def test_above_threshold_marked(self):
        """Token > 0.5 is discretized to 1."""
        m = discretize_marking({"P_a": 0.8, "P_b": 0.2}, ["P_a", "P_b"])
        assert m == (1, 0)

    def test_at_threshold_unmarked(self):
        """Token == 0.5 (at threshold) is discretized to 0."""
        m = discretize_marking({"P_a": 0.5}, ["P_a"])
        assert m == (0,)

    def test_custom_threshold(self):
        """Custom threshold changes discretization boundary."""
        m = discretize_marking(
            {"P_a": 0.3, "P_b": 0.7}, ["P_a", "P_b"], threshold=0.2
        )
        assert m == (1, 1)

    def test_missing_place_defaults_zero(self):
        """Places not in the marking dict default to 0.0 (unmarked)."""
        m = discretize_marking({"P_a": 1.0}, ["P_a", "P_b"])
        assert m == (1, 0)


# ── Test: Simple Net Properties ──────────────────────────────────────────


class TestSimpleNet:
    def test_simple_net_bounded(self, simple_net):
        """Simple 2-place net is 1-bounded."""
        result = check_boundedness(simple_net, _simple_initial())
        assert result["bounded"] is True
        assert result["bound"] == 1
        assert result["violation"] is None

    def test_simple_net_live(self, simple_net):
        """Simple net: T1 is live (fireable from the initial marking)."""
        result = check_liveness(simple_net, _simple_initial())
        assert result["live"] is True
        assert result["dead_transitions"] == []

    def test_simple_net_reachable_markings(self, simple_net):
        """Simple net has exactly 2 reachable markings: (1,0) and (0,1)."""
        reachable = enumerate_reachable_markings(simple_net, _simple_initial())
        assert len(reachable) == 2
        assert (1, 0) in reachable
        assert (0, 1) in reachable


# ── Test: Cyclic Net Properties ──────────────────────────────────────────


class TestCyclicNet:
    def test_cyclic_net_bounded(self, cyclic_net):
        result = check_boundedness(cyclic_net, _cyclic_initial())
        assert result["bounded"] is True

    def test_cyclic_net_live(self, cyclic_net):
        result = check_liveness(cyclic_net, _cyclic_initial())
        assert result["live"] is True
        assert result["dead_transitions"] == []

    def test_cyclic_net_home_state(self, cyclic_net):
        """Initial marking (1,0) is a home state in the cyclic net."""
        result = check_home_state(cyclic_net, _cyclic_initial())
        assert result["home_state"] is True


# ── Test: Mutex Net Properties ───────────────────────────────────────────


class TestMutexNet:
    def test_mutex_verified(self, mutex_net):
        """P_a and P_b are mutually exclusive in the mutex net."""
        result = check_mutual_exclusion(
            mutex_net, _mutex_initial(), ("P_a", "P_b")
        )
        assert result["mutex"] is True
        assert result["violation_marking"] is None


# ── Test: Vertical Control Net Properties ────────────────────────────────


class TestVerticalControlNet:
    def test_vertical_net_bounded(self, vertical_net):
        """The vertical control net is 1-bounded."""
        result = check_boundedness(vertical_net, _vcn_initial())
        assert result["bounded"] is True
        assert result["bound"] == 1

    def test_vertical_net_live(self, vertical_net):
        """All 7 transitions in the vertical control net are L1-live."""
        result = check_liveness(vertical_net, _vcn_initial())
        assert result["live"] is True, (
            f"Dead transitions: {result['dead_transitions']}"
        )

    def test_vertical_net_mutex(self, vertical_net):
        """P_error_pos and P_error_neg are mutually exclusive."""
        result = check_mutual_exclusion(
            vertical_net, _vcn_initial(), ("P_error_pos", "P_error_neg")
        )
        assert result["mutex"] is True, (
            f"Mutex violation at marking: {result['violation_marking']}"
        )

    def test_vertical_net_home_state(self, vertical_net):
        """The exact initial marking (P_idle only) is NOT a home state.

        P_applied is a sink place: T_apply produces into it but no
        transition consumes from it.  After the first control cycle,
        P_applied retains a token permanently.  Therefore the pristine
        initial marking (1,0,0,0,0,0,0,0) is unreachable from post-cycle
        markings.

        This is a correct structural property of the net -- P_applied
        represents the "control signal has been sent" record.
        """
        result = check_home_state(vertical_net, _vcn_initial())
        assert result["home_state"] is False

    def test_vertical_net_p_idle_always_reachable(self, vertical_net):
        """P_idle is eventually marked from every reachable marking.

        Even though the exact initial marking is not a home state (due
        to P_applied accumulation), P_idle is always eventually reached,
        guaranteeing the controller never permanently stalls.
        """
        result = check_place_liveness(vertical_net, _vcn_initial(), "P_idle")
        assert result["place_live"] is True, (
            f"P_idle not reachable from: {result['stuck_marking']}"
        )

    def test_reachable_markings_finite(self, vertical_net):
        """Vertical net has a finite (small) reachability graph.

        With 8 places, the theoretical maximum is 2^8 = 256 markings.
        The actual count should be much smaller due to the structured
        topology (sequential flow with branching and merging).
        """
        reachable = enumerate_reachable_markings(vertical_net, _vcn_initial())
        assert len(reachable) > 0
        assert len(reachable) <= 256  # theoretical max
        # In practice, we expect far fewer markings (< 30).
        assert len(reachable) < 50


# ── Test: Certificate ────────────────────────────────────────────────────


class TestCertificate:
    def test_certificate_complete(self, vertical_net):
        """Certificate contains all required keys."""
        cert = generate_proof_certificate(
            vertical_net,
            _vcn_initial(),
            mutex_pairs=[("P_error_pos", "P_error_neg")],
            liveness_places=["P_idle"],
        )
        required_keys = {
            "bounded",
            "bound",
            "live",
            "dead_transitions",
            "mutex_pairs_verified",
            "mutex_violations",
            "home_state",
            "place_liveness",
            "reachable_markings_count",
            "place_names",
            "transition_names",
            "initial_marking_discrete",
            "analysis_time_ms",
        }
        assert required_keys.issubset(set(cert.keys())), (
            f"Missing keys: {required_keys - set(cert.keys())}"
        )

    def test_certificate_all_pass(self, vertical_net):
        """Vertical control net passes boundedness, liveness, mutex, and place-liveness."""
        cert = generate_proof_certificate(
            vertical_net,
            _vcn_initial(),
            mutex_pairs=[("P_error_pos", "P_error_neg")],
            liveness_places=["P_idle"],
        )
        assert cert["bounded"] is True
        assert cert["live"] is True
        assert len(cert["mutex_violations"]) == 0
        assert len(cert["mutex_pairs_verified"]) == 1
        assert cert["place_liveness"]["P_idle"] is True
        assert cert["reachable_markings_count"] > 0
        assert cert["analysis_time_ms"] >= 0.0
        # home_state is False for this net (P_applied is a sink), which
        # is a correct structural observation, not a failure.
        assert cert["home_state"] is False

    def test_certificate_place_names(self, vertical_net):
        """Certificate includes correct place and transition names."""
        cert = generate_proof_certificate(vertical_net, _vcn_initial())
        assert cert["place_names"] == PLACES
        assert cert["transition_names"] == TRANSITIONS

    def test_certificate_initial_marking(self, vertical_net):
        """Discretized initial marking has P_idle = 1, rest = 0."""
        cert = generate_proof_certificate(vertical_net, _vcn_initial())
        m0 = cert["initial_marking_discrete"]
        # P_idle is index 0 in PLACES.
        assert m0[0] == 1
        assert sum(m0) == 1


# ── Test: Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_transition_no_inputs(self):
        """A transition with no input arcs can never fire.

        T1 has threshold=0.5 and no input arcs, so activation=0.0 < 0.5.
        T1 should be dead (never enabled).
        """
        net = StochasticPetriNet()
        net.add_place("P_a", initial_tokens=0.0)
        net.add_place("P_b", initial_tokens=0.0)
        net.add_transition("T1", threshold=0.5)
        net.add_transition("T2", threshold=0.5)
        net.add_arc("P_a", "T2", weight=1.0)
        net.add_arc("T2", "P_b", weight=1.0)
        net.add_arc("T1", "P_a", weight=1.0)
        net.compile()

        result = check_liveness(net, {"P_a": 0.0, "P_b": 0.0})
        assert result["live"] is False
        assert "T1" in result["dead_transitions"]

    def test_all_places_initially_unmarked(self):
        """Net with no initial tokens: no transition can ever fire."""
        net = StochasticPetriNet()
        net.add_place("P_a", initial_tokens=0.0)
        net.add_place("P_b", initial_tokens=0.0)
        net.add_transition("T1", threshold=0.5)
        net.add_arc("P_a", "T1", weight=1.0)
        net.add_arc("T1", "P_b", weight=1.0)
        net.compile()

        reachable = enumerate_reachable_markings(
            net, {"P_a": 0.0, "P_b": 0.0}
        )
        # Only the initial marking (0, 0) is reachable.
        assert len(reachable) == 1
        assert (0, 0) in reachable

        result = check_liveness(net, {"P_a": 0.0, "P_b": 0.0})
        assert result["live"] is False
        assert "T1" in result["dead_transitions"]

    def test_self_loop_net(self):
        """Net with a transition that consumes and produces on the same place.

        P_a --(1.0)--> T1 --(1.0)--> P_a

        Token stays in P_a forever.  Net is live and P_a is a home state.
        """
        net = StochasticPetriNet()
        net.add_place("P_a", initial_tokens=1.0)
        net.add_transition("T1", threshold=0.5)
        net.add_arc("P_a", "T1", weight=1.0)
        net.add_arc("T1", "P_a", weight=1.0)
        net.compile()

        reachable = enumerate_reachable_markings(net, {"P_a": 1.0})
        assert len(reachable) == 1
        assert (1,) in reachable

        result = check_liveness(net, {"P_a": 1.0})
        assert result["live"] is True

        result = check_home_state(net, {"P_a": 1.0})
        assert result["home_state"] is True

    def test_zero_threshold_transition(self):
        """Transition with threshold=0.0 fires even with no marked inputs.

        With threshold=0.0, activation 0.0 >= 0.0, so the transition
        is always enabled.  The output place gets marked.
        """
        net = StochasticPetriNet()
        net.add_place("P_a", initial_tokens=0.0)
        net.add_place("P_b", initial_tokens=0.0)
        net.add_transition("T1", threshold=0.0)
        net.add_arc("P_a", "T1", weight=1.0)
        net.add_arc("T1", "P_b", weight=1.0)
        net.compile()

        reachable = enumerate_reachable_markings(
            net, {"P_a": 0.0, "P_b": 0.0}
        )
        # T1 fires because threshold=0.0: activation=0.0 >= 0.0
        assert (0, 1) in reachable
