# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Formal Petri Net Analysis
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Formal analysis of Petri nets: boundedness, liveness, mutual exclusion.

These properties are preserved by the compilation to SNN because the
SNN faithfully implements the Petri net firing semantics.

IMPORTANT: The existing StochasticPetriNet uses continuous tokens in [0,1].
For formal analysis, we discretize: token > 0.5 -> "marked", <= 0.5 -> "unmarked".
This gives us a finite reachability graph suitable for exhaustive analysis.

Transition enablement uses **weighted threshold** semantics matching the
continuous-token model: a transition *t* is enabled under discrete marking *m*
when ``sum(W_in[t, p] * m[p] for all input places p) >= threshold[t]``.
This correctly handles OR-style fan-in arcs where a single marked input
place can supply enough activation to exceed the threshold.

Firing semantics (discrete):
  - For each input arc ``(p -> t)``, consume from place *p* (set to 0)
    **only if** the arc weight is positive (normal arc) and place is marked.
  - For each output arc ``(t -> p)``, produce into place *p* (set to 1).
  - Inhibitor arcs (negative W_in entries) do NOT consume; they block
    firing when the inhibitor place IS marked.

The resulting reachability graph is finite (2^n_places states max) and
typically very small for control-oriented nets.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .structure import StochasticPetriNet


# ── Helpers ──────────────────────────────────────────────────────────────


def _ensure_compiled(net: StochasticPetriNet) -> None:
    """Compile the net if not already compiled."""
    if not net.is_compiled:
        net.compile(allow_inhibitor=True)


def _get_arc_maps(
    net: StochasticPetriNet,
) -> Tuple[
    Dict[str, List[Tuple[str, float]]],  # transition -> [(input_place, weight)]
    Dict[str, List[Tuple[str, float]]],  # transition -> [(output_place, weight)]
]:
    """Extract per-transition input and output arc maps from the raw arc list.

    Returns
    -------
    input_map : dict[str, list[tuple[str, float]]]
        Maps each transition name to a list of (place_name, arc_weight) for
        input arcs (Place -> Transition).  Weight is negative for inhibitor arcs.
    output_map : dict[str, list[tuple[str, float]]]
        Maps each transition name to a list of (place_name, arc_weight) for
        output arcs (Transition -> Place).
    """
    from .structure import _NodeKind

    input_map: Dict[str, List[Tuple[str, float]]] = {
        t: [] for t in net.transition_names
    }
    output_map: Dict[str, List[Tuple[str, float]]] = {
        t: [] for t in net.transition_names
    }

    for src, tgt, w, _is_inhibitor in net._arcs:
        if net._kind[src] is _NodeKind.PLACE:
            # Place -> Transition (input arc)
            input_map[tgt].append((src, w))
        else:
            # Transition -> Place (output arc)
            output_map[src].append((tgt, w))

    return input_map, output_map


# ── Discretization ──────────────────────────────────────────────────────


def discretize_marking(
    marking: Dict[str, float],
    place_names: List[str],
    threshold: float = 0.5,
) -> Tuple[int, ...]:
    """Convert a continuous marking dict to a binary tuple.

    Parameters
    ----------
    marking : dict[str, float]
        Place name -> token density in [0, 1].
    place_names : list[str]
        Ordered place names (defines tuple position).
    threshold : float
        Discretization threshold.  Token > threshold -> 1, else 0.

    Returns
    -------
    tuple[int, ...]
        Binary marking vector with one entry per place (0 or 1).
    """
    return tuple(
        1 if marking.get(p, 0.0) > threshold else 0
        for p in place_names
    )


def _marking_to_dict(
    marking_tuple: Tuple[int, ...],
    place_names: List[str],
) -> Dict[str, int]:
    """Convert a binary marking tuple back to a name-keyed dict."""
    return {p: m for p, m in zip(place_names, marking_tuple)}


# ── Transition Enablement & Firing ──────────────────────────────────────


def _is_enabled(
    marking: Tuple[int, ...],
    transition: str,
    place_names: List[str],
    input_map: Dict[str, List[Tuple[str, float]]],
    thresholds: Dict[str, float],
) -> bool:
    """Check if a transition is enabled under a discrete marking.

    Uses weighted threshold semantics: the transition is enabled when the
    weighted sum of marked input places meets or exceeds the transition
    threshold, AND no inhibitor-place is marked.

    Parameters
    ----------
    marking : tuple[int, ...]
        Binary marking vector.
    transition : str
        Transition name to test.
    place_names : list[str]
        Ordered place names for index mapping.
    input_map : dict[str, list[tuple[str, float]]]
        Per-transition input arc map from ``_get_arc_maps``.
    thresholds : dict[str, float]
        Transition name -> firing threshold.

    Returns
    -------
    bool
        True if transition is enabled under the given marking.
    """
    place_idx = {p: i for i, p in enumerate(place_names)}
    activation = 0.0

    for place, weight in input_map[transition]:
        pidx = place_idx[place]
        token = marking[pidx]

        if weight < 0.0:
            # Inhibitor arc: transition is blocked if this place IS marked.
            if token == 1:
                return False
        else:
            # Normal input arc: contribute weight if place is marked.
            activation += weight * token

    return activation >= thresholds[transition]


def _fire_transition(
    marking: Tuple[int, ...],
    transition: str,
    place_names: List[str],
    input_map: Dict[str, List[Tuple[str, float]]],
    output_map: Dict[str, List[Tuple[str, float]]],
) -> Tuple[int, ...]:
    """Fire a transition and return the new marking.

    Discrete firing semantics:
      - Consume: for each normal (positive-weight) input arc, set the
        source place to 0 (remove token).
      - Produce: for each output arc, set the target place to 1 (add token).
      - Inhibitor arcs do not consume.
      - Multiple tokens on the same place are clamped to 1 (1-bounded).

    Parameters
    ----------
    marking : tuple[int, ...]
        Current binary marking.
    transition : str
        Transition to fire (must be enabled; caller is responsible for checking).
    place_names : list[str]
        Ordered place names.
    input_map, output_map
        Arc maps from ``_get_arc_maps``.

    Returns
    -------
    tuple[int, ...]
        New binary marking after firing.
    """
    place_idx = {p: i for i, p in enumerate(place_names)}
    new_marking = list(marking)

    # Consume from input places (normal arcs only).
    for place, weight in input_map[transition]:
        if weight > 0.0:
            pidx = place_idx[place]
            if new_marking[pidx] == 1:
                new_marking[pidx] = 0

    # Produce into output places.
    for place, weight in output_map[transition]:
        pidx = place_idx[place]
        new_marking[pidx] = 1  # clamped to 1

    return tuple(new_marking)


# ── Reachability Graph ──────────────────────────────────────────────────


def enumerate_reachable_markings(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    threshold: float = 0.5,
    max_markings: int = 100_000,
) -> Set[Tuple[int, ...]]:
    """BFS over all reachable discrete markings.

    For each reachable marking, tries firing each transition individually.
    A transition is enabled if the weighted activation of its marked input
    places meets or exceeds its firing threshold (and no inhibitor place is
    marked).  Each enabled transition produces a successor marking via
    token consumption/production.

    Parameters
    ----------
    net : StochasticPetriNet
        The net to analyse (will be compiled if necessary).
    initial_marking : dict[str, float]
        Initial continuous marking (will be discretized).
    threshold : float
        Discretization threshold for tokens.
    max_markings : int
        Safety limit on reachability graph size to prevent runaway exploration.

    Returns
    -------
    set[tuple[int, ...]]
        Complete set of reachable discrete markings.

    Raises
    ------
    RuntimeError
        If the reachability graph exceeds ``max_markings``.
    """
    _ensure_compiled(net)

    place_names = net.place_names
    trans_names = net.transition_names
    input_map, output_map = _get_arc_maps(net)

    # Build threshold dict.
    thresholds = {
        t: th for t, th in zip(trans_names, net._transition_thresholds)
    }

    m0 = discretize_marking(initial_marking, place_names, threshold)

    visited: Set[Tuple[int, ...]] = {m0}
    queue: deque[Tuple[int, ...]] = deque([m0])

    while queue:
        current = queue.popleft()

        for t in trans_names:
            if _is_enabled(current, t, place_names, input_map, thresholds):
                successor = _fire_transition(
                    current, t, place_names, input_map, output_map
                )
                if successor not in visited:
                    if len(visited) >= max_markings:
                        raise RuntimeError(
                            f"Reachability graph exceeded {max_markings} "
                            f"markings — net may be unbounded or too large."
                        )
                    visited.add(successor)
                    queue.append(successor)

    return visited


# ── Property Checks ─────────────────────────────────────────────────────


def check_boundedness(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    bound: int = 1,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Check that no place ever holds more than *bound* tokens.

    Since the discrete model uses binary {0, 1} tokens, the net is
    trivially 1-bounded.  This function verifies the property exhaustively
    over the full reachability graph and checks against arbitrary bounds.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse.
    initial_marking : dict[str, float]
        Initial marking (continuous, will be discretized).
    bound : int
        Maximum token count per place.
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        ``{"bounded": bool, "bound": int, "violation": str | None}``
    """
    reachable = enumerate_reachable_markings(net, initial_marking, threshold)
    place_names = net.place_names

    for marking in reachable:
        for i, tok in enumerate(marking):
            if tok > bound:
                return {
                    "bounded": False,
                    "bound": bound,
                    "violation": place_names[i],
                }

    return {"bounded": True, "bound": bound, "violation": None}


def check_liveness(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Check that every transition can eventually fire from SOME reachable marking.

    A transition is *L1-live* (potentially fireable) if there exists at
    least one reachable marking where it is enabled.  Transitions that
    are never enabled in any reachable marking are reported as dead.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse.
    initial_marking : dict[str, float]
        Initial marking (continuous, will be discretized).
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        ``{"live": bool, "dead_transitions": list[str]}``
    """
    _ensure_compiled(net)

    reachable = enumerate_reachable_markings(net, initial_marking, threshold)
    place_names = net.place_names
    trans_names = net.transition_names
    input_map, _ = _get_arc_maps(net)

    thresholds_dict = {
        t: th for t, th in zip(trans_names, net._transition_thresholds)
    }

    fireable: Set[str] = set()

    for marking in reachable:
        for t in trans_names:
            if t not in fireable:
                if _is_enabled(marking, t, place_names, input_map, thresholds_dict):
                    fireable.add(t)

        # Early exit if all transitions found.
        if len(fireable) == len(trans_names):
            break

    dead = sorted(set(trans_names) - fireable)
    return {"live": len(dead) == 0, "dead_transitions": dead}


def check_mutual_exclusion(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    places: Tuple[str, str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Check that two places are never simultaneously marked.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse.
    initial_marking : dict[str, float]
        Initial marking.
    places : tuple[str, str]
        Pair of place names to check for mutual exclusion.
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        ``{"mutex": bool, "violation_marking": tuple | None}``
    """
    reachable = enumerate_reachable_markings(net, initial_marking, threshold)
    place_names = net.place_names
    idx_a = place_names.index(places[0])
    idx_b = place_names.index(places[1])

    for marking in reachable:
        if marking[idx_a] == 1 and marking[idx_b] == 1:
            return {"mutex": False, "violation_marking": marking}

    return {"mutex": True, "violation_marking": None}


def check_home_state(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Check that the initial marking is reachable from every reachable marking.

    A marking m0 is a *home state* if, from every reachable marking m,
    there exists a firing sequence that returns to m0.  This is verified
    by running BFS from each reachable marking and checking that m0
    appears in the resulting reachable set.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse.
    initial_marking : dict[str, float]
        Initial marking (continuous, will be discretized).  This marking
        is the candidate home state.
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        ``{"home_state": bool, "unreachable_from": tuple | None}``
    """
    _ensure_compiled(net)

    place_names = net.place_names
    trans_names = net.transition_names
    input_map, output_map = _get_arc_maps(net)
    thresholds_dict = {
        t: th for t, th in zip(trans_names, net._transition_thresholds)
    }

    m0 = discretize_marking(initial_marking, place_names, threshold)
    all_reachable = enumerate_reachable_markings(net, initial_marking, threshold)

    for start in all_reachable:
        if start == m0:
            continue

        # BFS from 'start' to see if m0 is reachable.
        visited: Set[Tuple[int, ...]] = {start}
        queue: deque[Tuple[int, ...]] = deque([start])
        found = False

        while queue:
            current = queue.popleft()
            if current == m0:
                found = True
                break

            for t in trans_names:
                if _is_enabled(current, t, place_names, input_map, thresholds_dict):
                    successor = _fire_transition(
                        current, t, place_names, input_map, output_map
                    )
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)

        if not found:
            return {"home_state": False, "unreachable_from": start}

    return {"home_state": True, "unreachable_from": None}


def check_place_liveness(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    place: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Check that a specific place is eventually marked from every reachable marking.

    This is a weaker property than home-state but often more practically
    relevant.  For the vertical control net, it verifies that P_idle is
    always eventually reached, guaranteeing the controller never permanently
    stalls.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse.
    initial_marking : dict[str, float]
        Initial marking (continuous, will be discretized).
    place : str
        Name of the place to check for eventual marking.
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        ``{"place_live": bool, "place": str, "stuck_marking": tuple | None}``
    """
    _ensure_compiled(net)

    place_names = net.place_names
    trans_names = net.transition_names
    input_map, output_map = _get_arc_maps(net)
    thresholds_dict = {
        t: th for t, th in zip(trans_names, net._transition_thresholds)
    }

    place_idx = place_names.index(place)
    all_reachable = enumerate_reachable_markings(net, initial_marking, threshold)

    for start in all_reachable:
        # If the place is already marked in this marking, skip.
        if start[place_idx] == 1:
            continue

        # BFS from 'start' to find any marking where 'place' is marked.
        visited: Set[Tuple[int, ...]] = {start}
        queue: deque[Tuple[int, ...]] = deque([start])
        found = False

        while queue:
            current = queue.popleft()
            if current[place_idx] == 1:
                found = True
                break

            for t in trans_names:
                if _is_enabled(current, t, place_names, input_map, thresholds_dict):
                    successor = _fire_transition(
                        current, t, place_names, input_map, output_map
                    )
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)

        if not found:
            return {"place_live": False, "place": place, "stuck_marking": start}

    return {"place_live": True, "place": place, "stuck_marking": None}


# ── Proof Certificate ───────────────────────────────────────────────────


def generate_proof_certificate(
    net: StochasticPetriNet,
    initial_marking: Dict[str, float],
    mutex_pairs: Optional[List[Tuple[str, str]]] = None,
    liveness_places: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Run ALL formal analyses and return a complete proof certificate.

    This is the top-level entry point for formal verification.  It
    executes boundedness, liveness, mutual exclusion, home-state, and
    place-liveness checks, and packages the results into a single
    certificate dict suitable for serialization to JSON.

    Parameters
    ----------
    net : StochasticPetriNet
        Net to analyse (will be compiled if necessary).
    initial_marking : dict[str, float]
        Initial marking (continuous, will be discretized).
    mutex_pairs : list[tuple[str, str]] | None
        Pairs of places to verify for mutual exclusion.
        If None, no mutex checks are performed.
    liveness_places : list[str] | None
        Places to verify for eventual marking (place liveness).
        If None, no place-liveness checks are performed.
    threshold : float
        Discretization threshold.

    Returns
    -------
    dict
        Complete proof certificate with keys:

        - ``bounded`` (bool): True if net is 1-bounded.
        - ``bound`` (int): The bound checked (always 1).
        - ``live`` (bool): True if all transitions are L1-live.
        - ``dead_transitions`` (list[str]): Names of dead transitions.
        - ``mutex_pairs_verified`` (list[tuple[str, str]]): Pairs verified mutex.
        - ``mutex_violations`` (list[dict]): Any mutex violations found.
        - ``home_state`` (bool): True if initial marking is a home state.
        - ``place_liveness`` (dict[str, bool]): Per-place eventual marking check.
        - ``reachable_markings_count`` (int): Size of reachability graph.
        - ``place_names`` (list[str]): Ordered place names.
        - ``transition_names`` (list[str]): Ordered transition names.
        - ``initial_marking_discrete`` (list[int]): Discretized initial marking.
        - ``analysis_time_ms`` (float): Wall-clock time in milliseconds.
    """
    t_start = time.perf_counter()

    _ensure_compiled(net)
    place_names = net.place_names

    # Reachable markings (computed once, reused).
    reachable = enumerate_reachable_markings(net, initial_marking, threshold)

    # Boundedness.
    bound_result = check_boundedness(net, initial_marking, bound=1, threshold=threshold)

    # Liveness.
    live_result = check_liveness(net, initial_marking, threshold=threshold)

    # Mutual exclusion.
    mutex_verified: List[Tuple[str, str]] = []
    mutex_violations: List[Dict[str, object]] = []
    if mutex_pairs:
        for pair in mutex_pairs:
            mx = check_mutual_exclusion(net, initial_marking, pair, threshold)
            if mx["mutex"]:
                mutex_verified.append(pair)
            else:
                mutex_violations.append(
                    {"pair": pair, "violation_marking": mx["violation_marking"]}
                )

    # Home state.
    home_result = check_home_state(net, initial_marking, threshold=threshold)

    # Place liveness.
    place_liveness_results: Dict[str, bool] = {}
    if liveness_places:
        for pl in liveness_places:
            pl_result = check_place_liveness(
                net, initial_marking, pl, threshold=threshold
            )
            place_liveness_results[pl] = pl_result["place_live"]

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0

    m0 = discretize_marking(initial_marking, place_names, threshold)

    return {
        "bounded": bound_result["bounded"],
        "bound": bound_result["bound"],
        "live": live_result["live"],
        "dead_transitions": live_result["dead_transitions"],
        "mutex_pairs_verified": [list(p) for p in mutex_verified],
        "mutex_violations": mutex_violations,
        "home_state": home_result["home_state"],
        "place_liveness": place_liveness_results,
        "reachable_markings_count": len(reachable),
        "place_names": place_names,
        "transition_names": net.transition_names,
        "initial_marking_discrete": list(m0),
        "analysis_time_ms": round(elapsed_ms, 3),
    }
