# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Safety Interlocks
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Safety interlock Petri net utilities.

Inhibitor arcs provide symbolic hard-stop logic:
when a safety place receives tokens, the paired control transition is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .contracts import (
    DEFAULT_SAFETY_CONTRACTS,
    SafetyContract,
    verify_safety_contracts,
)
from .structure import StochasticPetriNet


SAFETY_CHANNELS: Dict[str, str] = {
    "thermal_limit": "heat_ramp",
    "density_limit": "density_ramp",
    "beta_limit": "power_ramp",
    "current_limit": "current_ramp",
    "vertical_limit": "position_move",
}

CONTROL_TRANSITIONS: tuple[str, ...] = (
    "heat_ramp",
    "density_ramp",
    "power_ramp",
    "current_ramp",
    "position_move",
)


@dataclass(frozen=True)
class SafetyLimits:
    """Default safety limits used to derive inhibitor-place tokens."""

    thermal_limit_keV: float = 25.0
    density_limit_1e19_m3: float = 12.0
    beta_limit: float = 2.8
    current_limit_MA: float = 15.0
    vertical_limit_m_s: float = 1.0


def build_safety_net() -> StochasticPetriNet:
    """Build canonical safety-interlock Petri net with inhibitor arcs."""
    net = StochasticPetriNet()

    # Requests (normal operation)
    net.add_place("heating_request", initial_tokens=1.0)
    net.add_place("density_request", initial_tokens=1.0)
    net.add_place("power_request", initial_tokens=1.0)
    net.add_place("current_request", initial_tokens=1.0)
    net.add_place("position_request", initial_tokens=1.0)

    # Safety flags (0: safe, >0: violated)
    net.add_place("thermal_limit", initial_tokens=0.0)
    net.add_place("density_limit", initial_tokens=0.0)
    net.add_place("beta_limit", initial_tokens=0.0)
    net.add_place("current_limit", initial_tokens=0.0)
    net.add_place("vertical_limit", initial_tokens=0.0)

    # Outputs
    net.add_place("heat_output", initial_tokens=0.0)
    net.add_place("density_output", initial_tokens=0.0)
    net.add_place("power_output", initial_tokens=0.0)
    net.add_place("current_output", initial_tokens=0.0)
    net.add_place("position_output", initial_tokens=0.0)

    for name, request, output, safety in [
        ("heat_ramp", "heating_request", "heat_output", "thermal_limit"),
        ("density_ramp", "density_request", "density_output", "density_limit"),
        ("power_ramp", "power_request", "power_output", "beta_limit"),
        ("current_ramp", "current_request", "current_output", "current_limit"),
        ("position_move", "position_request", "position_output", "vertical_limit"),
    ]:
        net.add_transition(name, threshold=0.5)
        net.add_arc(request, name, weight=1.0)
        net.add_arc(name, output, weight=1.0)
        net.add_arc(safety, name, weight=1.0, inhibitor=True)

    net.compile(allow_inhibitor=True)
    return net


def _safe_float(state: Mapping[str, float], key: str, fallback: float) -> float:
    value = float(state.get(key, fallback))
    return value if np.isfinite(value) else float(fallback)


def safety_tokens_from_state(
    state: Mapping[str, float],
    *,
    limits: SafetyLimits | None = None,
) -> Dict[str, float]:
    """Map plasma state values to binary safety-place tokens."""
    lim = limits or SafetyLimits()
    t_e = _safe_float(state, "T_e", 0.0)
    t_max = _safe_float(state, "T_max", lim.thermal_limit_keV)
    n_e = _safe_float(state, "n_e", 0.0)
    n_greenwald = _safe_float(state, "n_greenwald", lim.density_limit_1e19_m3)
    beta_n = _safe_float(state, "beta_N", 0.0)
    beta_max = _safe_float(state, "beta_no_wall", lim.beta_limit)
    i_p = _safe_float(state, "I_p", 0.0)
    i_max = _safe_float(state, "I_max", lim.current_limit_MA)
    dz_dt = _safe_float(state, "dZ_dt", 0.0)
    vde_max = _safe_float(state, "vde_threshold", lim.vertical_limit_m_s)

    return {
        "thermal_limit": 1.0 if t_e > t_max else 0.0,
        "density_limit": 1.0 if n_e > n_greenwald else 0.0,
        "beta_limit": 1.0 if beta_n > beta_max else 0.0,
        "current_limit": 1.0 if i_p > i_max else 0.0,
        "vertical_limit": 1.0 if abs(dz_dt) > vde_max else 0.0,
    }


def evaluate_transition_enablement(
    net: StochasticPetriNet,
    marking: np.ndarray,
) -> Dict[str, bool]:
    """Return deterministic transition enablement with inhibitor semantics."""
    if not net.is_compiled or net.W_in is None:
        raise RuntimeError("Safety net must be compiled before evaluation.")
    m = np.asarray(marking, dtype=np.float64)
    if m.shape != (net.n_places,):
        raise ValueError(
            f"marking must have shape ({net.n_places},), got {m.shape}"
        )

    w_in = net.W_in.toarray()
    thresholds = net.get_thresholds()
    out: Dict[str, bool] = {}
    for t_idx, t_name in enumerate(net.transition_names):
        row = w_in[t_idx]
        pos_idx = np.flatnonzero(row > 0.0)
        inh_idx = np.flatnonzero(row < 0.0)

        pos_ok = True
        if pos_idx.size:
            pos_ok = bool(np.all(m[pos_idx] >= row[pos_idx] - 1e-12))
        inh_ok = True
        if inh_idx.size:
            inh_ok = bool(np.all(m[inh_idx] < np.abs(row[inh_idx]) - 1e-12))
        activation = float(np.dot(np.maximum(row, 0.0), m))
        out[t_name] = bool(pos_ok and inh_ok and activation >= float(thresholds[t_idx]))
    return out


def default_safety_contracts() -> tuple[SafetyContract, ...]:
    """Canonical safety contracts for inhibitor channels."""
    return DEFAULT_SAFETY_CONTRACTS


class SafetyInterlockRuntime:
    """Stateful runtime evaluator for safety-interlock Petri nets."""

    def __init__(
        self,
        *,
        net: StochasticPetriNet | None = None,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.net = net if net is not None else build_safety_net()
        if not self.net.is_compiled:
            self.net.compile(allow_inhibitor=True)
        self.limits = limits or SafetyLimits()
        self._place_idx = {name: i for i, name in enumerate(self.net.place_names)}
        self._marking = self.net.get_initial_marking().copy()
        self.last_transition_enablement: Dict[str, bool] = {}
        self.last_contract_violations: list[str] = []
        self.last_tokens: Dict[str, float] = {k: 0.0 for k in SAFETY_CHANNELS}

    @property
    def marking(self) -> np.ndarray:
        return self._marking.copy()

    def set_safety_tokens(self, tokens: Mapping[str, float]) -> None:
        """Apply safety-place tokens (binary in practice)."""
        for place in SAFETY_CHANNELS:
            value = float(tokens.get(place, 0.0))
            self._marking[self._place_idx[place]] = 1.0 if value > 0.0 else 0.0
        self.last_tokens = {
            place: float(self._marking[self._place_idx[place]])
            for place in SAFETY_CHANNELS
        }

    def allowed_actions(self) -> Dict[str, bool]:
        """Compute transition allow/deny map for control actions."""
        enabled = evaluate_transition_enablement(self.net, self._marking)
        self.last_transition_enablement = {
            name: bool(enabled.get(name, False)) for name in CONTROL_TRANSITIONS
        }
        self.last_contract_violations = verify_safety_contracts(
            safety_tokens=self.last_tokens,
            transition_enabled=self.last_transition_enablement,
        )
        return dict(self.last_transition_enablement)

    def update_from_state(self, state: Mapping[str, float]) -> Dict[str, bool]:
        """Update safety tokens from state and return allowed control actions."""
        tokens = safety_tokens_from_state(state, limits=self.limits)
        self.set_safety_tokens(tokens)
        return self.allowed_actions()

