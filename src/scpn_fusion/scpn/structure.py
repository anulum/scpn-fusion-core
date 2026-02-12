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
from scipy import sparse


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
        self._transition_idx: Dict[str, int] = {}

        # Node kind lookup (for arc validation) --------------------------------
        self._kind: Dict[str, _NodeKind] = {}

        # Arc storage (resolved at compile time) --------------------------------
        # Each arc: (source_name, target_name, weight)
        self._arcs: List[Tuple[str, str, float]] = []

        # Compiled products ----------------------------------------------------
        self.W_in: sparse.csr_matrix | None = None   # (nT, nP)
        self.W_out: sparse.csr_matrix | None = None   # (nP, nT)
        self._compiled: bool = False

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

    def add_transition(self, name: str, threshold: float = 0.5) -> None:
        """Add a transition (logic gate) with a firing threshold."""
        if name in self._kind:
            raise ValueError(f"Node '{name}' already exists.")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        idx = len(self._transitions)
        self._transitions.append(name)
        self._transition_thresholds.append(threshold)
        self._transition_idx[name] = idx
        self._kind[name] = _NodeKind.TRANSITION
        self._compiled = False

    def add_arc(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a directed arc between a Place and a Transition (either direction).

        Valid arcs:
            Place      -> Transition  (input arc,  stored in W_in)
            Transition -> Place       (output arc, stored in W_out)

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
        if weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {weight}")

        self._arcs.append((source, target, weight))
        self._compiled = False

    # ── Compile ──────────────────────────────────────────────────────────────

    def compile(self) -> None:
        """Build sparse W_in and W_out matrices from the arc list."""
        nP = len(self._places)
        nT = len(self._transitions)
        if nP == 0 or nT == 0:
            raise ValueError(
                "Net must have at least one place and one transition."
            )

        # COO accumulators
        in_rows: List[int] = []
        in_cols: List[int] = []
        in_vals: List[float] = []

        out_rows: List[int] = []
        out_cols: List[int] = []
        out_vals: List[float] = []

        for src, tgt, w in self._arcs:
            if self._kind[src] is _NodeKind.PLACE:
                # Place -> Transition  => W_in[transition_idx, place_idx]
                t_idx = self._transition_idx[tgt]
                p_idx = self._place_idx[src]
                in_rows.append(t_idx)
                in_cols.append(p_idx)
                in_vals.append(w)
            else:
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

    def get_initial_marking(self) -> np.ndarray:
        """Return (n_places,) float64 vector of initial token densities."""
        return np.array(self._place_tokens, dtype=np.float64)

    def get_thresholds(self) -> np.ndarray:
        """Return (n_transitions,) float64 vector of firing thresholds."""
        return np.array(self._transition_thresholds, dtype=np.float64)

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
            lines.append(f"  [{i}] {name:20s}  threshold={th:.3f}")

        lines.append("")
        lines.append("Arcs:")
        for src, tgt, w in self._arcs:
            lines.append(f"  {src} --({w:.3f})--> {tgt}")

        if self._compiled:
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
