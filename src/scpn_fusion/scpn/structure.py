# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Packet A — Stochastic Petri Net structure definition.

Pure Python + numpy + scipy.  No sc_neurocore dependency.

Builds two sparse matrices that encode the net topology:
    W_in  : (n_transitions, n_places)  — input arc weights
    W_out : (n_places, n_transitions)  — output arc weights
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]


class _NodeKind(Enum):
    PLACE = auto()
    TRANSITION = auto()


class StochasticPetriNet:
    """Stochastic Petri Net with sparse matrix representation.

    Usage::

        net = StochasticPetriNet()
        net.add_place("P_red",   initial_tokens=1.0)
        net.add_place("P_green", initial_tokens=0.0)
        net.add_transition("T_r2g", threshold=0.5)
        net.add_arc("P_red", "T_r2g", weight=1.0)
        net.add_arc("T_r2g", "P_green", weight=1.0)
        net.compile()
    """

    def __init__(self) -> None:
        # Ordered registries -------------------------------------------------
        self._places: List[str] = []
        self._place_tokens: List[float] = []
        self._place_idx: Dict[str, int] = {}

        self._transitions: List[str] = []
        self._transition_thresholds: List[float] = []
        self._transition_delays: List[int] = []
        self._transition_idx: Dict[str, int] = {}

        # Node kind lookup (for arc validation) --------------------------------
        self._kind: Dict[str, _NodeKind] = {}

        # Arc storage (resolved at compile time) --------------------------------
        # Each arc: (source_name, target_name, weight, inhibitor_flag)
        self._arcs: List[Tuple[str, str, float, bool]] = []

        # Compiled products ----------------------------------------------------
        self.W_in: sparse.csr_matrix | None = None   # (nT, nP)
        self.W_out: sparse.csr_matrix | None = None   # (nP, nT)
        self._compiled: bool = False
        self._last_validation_report: Dict[str, object] | None = None

    # ── Builder API ──────────────────────────────────────────────────────────

    def add_place(self, name: str, initial_tokens: float = 0.0) -> None:
        """Add a place (state variable) with token density in [0, 1]."""
        if name in self._kind:
            raise ValueError(f"Node '{name}' already exists.")
        if not 0.0 <= initial_tokens <= 1.0:
            raise ValueError(
                f"initial_tokens must be in [0, 1], got {initial_tokens}"
            )
        idx = len(self._places)
        self._places.append(name)
        self._place_tokens.append(initial_tokens)
        self._place_idx[name] = idx
        self._kind[name] = _NodeKind.PLACE
        self._compiled = False

    def add_transition(
        self, name: str, threshold: float = 0.5, delay_ticks: int = 0
    ) -> None:
        """Add a transition (logic gate) with a firing threshold."""
        if name in self._kind:
            raise ValueError(f"Node '{name}' already exists.")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if delay_ticks < 0:
            raise ValueError(f"delay_ticks must be >= 0, got {delay_ticks}")
        idx = len(self._transitions)
        self._transitions.append(name)
        self._transition_thresholds.append(threshold)
        self._transition_delays.append(int(delay_ticks))
        self._transition_idx[name] = idx
        self._kind[name] = _NodeKind.TRANSITION
        self._compiled = False

    def add_arc(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        inhibitor: bool = False,
    ) -> None:
        """Add a directed arc between a Place and a Transition (either direction).

        Valid arcs:
            Place      -> Transition  (input arc,  stored in W_in)
            Transition -> Place       (output arc, stored in W_out)

        Parameters
        ----------
        inhibitor : bool, default False
            If True, arc is encoded as a negative weight inhibitor input arc.
            Only valid for ``Place -> Transition`` edges.

        Raises ``ValueError`` for same-kind connections or unknown nodes.
        """
        if source not in self._kind:
            raise ValueError(f"Unknown node '{source}'.")
        if target not in self._kind:
            raise ValueError(f"Unknown node '{target}'.")

        src_kind = self._kind[source]
        tgt_kind = self._kind[target]

        if src_kind == tgt_kind:
            raise ValueError(
                f"Arc must connect Place<->Transition, got "
                f"{src_kind.name}->{tgt_kind.name} ('{source}'->'{target}')."
            )
        if inhibitor:
            if not (
                src_kind is _NodeKind.PLACE and tgt_kind is _NodeKind.TRANSITION
            ):
                raise ValueError(
                    "inhibitor arcs are only supported for Place->Transition."
                )
            if weight <= 0.0:
                raise ValueError(
                    f"inhibitor arc weight must be > 0 (magnitude), got {weight}"
                )
            stored_weight = -abs(float(weight))
        else:
            if weight <= 0.0:
                raise ValueError(f"weight must be > 0, got {weight}")
            stored_weight = float(weight)

        self._arcs.append((source, target, stored_weight, bool(inhibitor)))
        self._compiled = False

    # ── Compile ──────────────────────────────────────────────────────────────

    def compile(
        self,
        validate_topology: bool = False,
        strict_validation: bool = False,
        allow_inhibitor: bool = False,
    ) -> None:
        """Build sparse W_in and W_out matrices from the arc list.

        Parameters
        ----------
        validate_topology : bool, default False
            Compute and store topology diagnostics in
            :attr:`last_validation_report`.
        strict_validation : bool, default False
            If True, raise ``ValueError`` when topology diagnostics contain
            dead nodes or unseeded place cycles.
        allow_inhibitor : bool, default False
            Allow negative ``Place -> Transition`` weights for inhibitor arcs.
            If False, compile fails when inhibitor arcs are present.
        """
        nP = len(self._places)
        nT = len(self._transitions)
        if nP == 0 or nT == 0:
            raise ValueError(
                "Net must have at least one place and one transition."
            )

        self._last_validation_report = None
        if validate_topology or strict_validation:
            report = self.validate_topology()
            self._last_validation_report = report
            has_issues = bool(
                report["dead_places"]
                or report["dead_transitions"]
                or report["unseeded_place_cycles"]
            )
            if strict_validation and has_issues:
                issues: List[str] = []
                if report["dead_places"]:
                    issues.append(f"dead_places={report['dead_places']}")
                if report["dead_transitions"]:
                    issues.append(f"dead_transitions={report['dead_transitions']}")
                if report["unseeded_place_cycles"]:
                    issues.append(
                        f"unseeded_place_cycles={report['unseeded_place_cycles']}"
                    )
                raise ValueError(
                    "Topology validation failed: " + "; ".join(issues)
                )

        # COO accumulators
        in_rows: List[int] = []
        in_cols: List[int] = []
        in_vals: List[float] = []

        out_rows: List[int] = []
        out_cols: List[int] = []
        out_vals: List[float] = []

        for src, tgt, w, is_inhibitor in self._arcs:
            if self._kind[src] is _NodeKind.PLACE:
                if w < 0.0 and not allow_inhibitor:
                    raise ValueError(
                        "Negative input arc weights detected; "
                        "re-run compile with allow_inhibitor=True."
                    )
                # Place -> Transition  => W_in[transition_idx, place_idx]
                t_idx = self._transition_idx[tgt]
                p_idx = self._place_idx[src]
                in_rows.append(t_idx)
                in_cols.append(p_idx)
                in_vals.append(w)
            else:
                if is_inhibitor or w < 0.0:
                    raise ValueError(
                        "Inhibitor arcs are only valid on Place->Transition edges."
                    )
                # Transition -> Place  => W_out[place_idx, transition_idx]
                t_idx = self._transition_idx[src]
                p_idx = self._place_idx[tgt]
                out_rows.append(p_idx)
                out_cols.append(t_idx)
                out_vals.append(w)

        self.W_in = sparse.csr_matrix(
            (in_vals, (in_rows, in_cols)), shape=(nT, nP), dtype=np.float64
        )
        self.W_out = sparse.csr_matrix(
            (out_vals, (out_rows, out_cols)), shape=(nP, nT), dtype=np.float64
        )
        self._compiled = True

    def validate_topology(self) -> Dict[str, object]:
        """Return topology diagnostics without mutating compiled matrices.

        Diagnostics:
        - ``dead_places``: places with zero total degree.
        - ``dead_transitions``: transitions with zero total degree.
        - ``unseeded_place_cycles``: place-level SCC cycles with no initial
          token mass.
        """
        place_in_deg = {p: 0 for p in self._places}
        place_out_deg = {p: 0 for p in self._places}
        trans_in_deg = {t: 0 for t in self._transitions}
        trans_out_deg = {t: 0 for t in self._transitions}

        transition_inputs: Dict[str, List[str]] = {t: [] for t in self._transitions}
        transition_outputs: Dict[str, List[str]] = {t: [] for t in self._transitions}

        for src, tgt, _w, _is_inhibitor in self._arcs:
            if self._kind[src] is _NodeKind.PLACE:
                place_out_deg[src] += 1
                trans_in_deg[tgt] += 1
                transition_inputs[tgt].append(src)
            else:
                trans_out_deg[src] += 1
                place_in_deg[tgt] += 1
                transition_outputs[src].append(tgt)

        dead_places = sorted(
            p
            for p in self._places
            if (place_in_deg[p] + place_out_deg[p]) == 0
        )
        dead_transitions = sorted(
            t
            for t in self._transitions
            if (trans_in_deg[t] + trans_out_deg[t]) == 0
        )

        place_adj: Dict[str, set[str]] = {p: set() for p in self._places}
        for t in self._transitions:
            inputs = transition_inputs[t]
            outputs = transition_outputs[t]
            for src_place in inputs:
                for dst_place in outputs:
                    place_adj[src_place].add(dst_place)

        unseeded_place_cycles: List[List[str]] = []
        for comp in self._strongly_connected_components(place_adj):
            has_cycle = len(comp) > 1
            if not has_cycle:
                node = comp[0]
                has_cycle = node in place_adj[node]
            if not has_cycle:
                continue

            if all(
                self._place_tokens[self._place_idx[p]] <= 0.0
                for p in comp
            ):
                unseeded_place_cycles.append(sorted(comp))

        unseeded_place_cycles.sort(key=lambda c: tuple(c))
        return {
            "dead_places": dead_places,
            "dead_transitions": dead_transitions,
            "unseeded_place_cycles": unseeded_place_cycles,
        }

    @staticmethod
    def _strongly_connected_components(
        graph: Dict[str, set[str]]
    ) -> List[List[str]]:
        """Tarjan SCC for small place graphs."""
        index = 0
        stack: List[str] = []
        on_stack: set[str] = set()
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        components: List[List[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                components.append(comp)

        for node in graph.keys():
            if node not in indices:
                strongconnect(node)
        return components

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def n_places(self) -> int:
        return len(self._places)

    @property
    def n_transitions(self) -> int:
        return len(self._transitions)

    @property
    def place_names(self) -> List[str]:
        return list(self._places)

    @property
    def transition_names(self) -> List[str]:
        return list(self._transitions)

    @property
    def is_compiled(self) -> bool:
        return self._compiled

    @property
    def last_validation_report(self) -> Dict[str, object] | None:
        return self._last_validation_report

    def get_initial_marking(self) -> FloatArray:
        """Return (n_places,) float64 vector of initial token densities."""
        return np.array(self._place_tokens, dtype=np.float64)

    def get_thresholds(self) -> FloatArray:
        """Return (n_transitions,) float64 vector of firing thresholds."""
        return np.array(self._transition_thresholds, dtype=np.float64)

    def get_delay_ticks(self) -> NDArray[np.int64]:
        """Return (n_transitions,) int64 vector of transition delay ticks."""
        return np.array(self._transition_delays, dtype=np.int64)

    def summary(self) -> str:
        """Human-readable summary of the net."""
        lines = [
            f"StochasticPetriNet  "
            f"P={self.n_places}  T={self.n_transitions}  "
            f"Arcs={len(self._arcs)}  compiled={self._compiled}",
            "",
            "Places:",
        ]
        for i, (name, tok) in enumerate(
            zip(self._places, self._place_tokens)
        ):
            lines.append(f"  [{i}] {name:20s}  tokens={tok:.3f}")

        lines.append("")
        lines.append("Transitions:")
        for i, (name, th) in enumerate(
            zip(self._transitions, self._transition_thresholds)
        ):
            delay = self._transition_delays[i]
            lines.append(
                f"  [{i}] {name:20s}  threshold={th:.3f}  delay_ticks={delay}"
            )

        lines.append("")
        lines.append("Arcs:")
        for src, tgt, w, is_inhibitor in self._arcs:
            extra = " [inhibitor]" if is_inhibitor else ""
            lines.append(f"  {src} --({w:.3f})--> {tgt}{extra}")

        if self._compiled:
            assert self.W_in is not None
            assert self.W_out is not None
            lines.append("")
            lines.append(
                f"W_in  (nT={self.W_in.shape[0]}, nP={self.W_in.shape[1]})  "
                f"nnz={self.W_in.nnz}"
            )
            lines.append(
                f"W_out (nP={self.W_out.shape[0]}, nT={self.W_out.shape[1]})  "
                f"nnz={self.W_out.nnz}"
            )
        return "\n".join(lines)
