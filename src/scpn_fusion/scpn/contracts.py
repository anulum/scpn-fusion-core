# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Data contracts for the SCPN Fusion-Core Control API.

Observation / action TypedDicts, feature extraction (obs → unipolar [0,1]),
action decoding (marking → slew-limited actuator commands).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict


# ── Observation / Action TypedDicts ──────────────────────────────────────────


class ControlObservation(TypedDict):
    """Plant observation at a single control tick."""

    R_axis_m: float
    Z_axis_m: float


class ControlAction(TypedDict):
    """Actuator command output for a single control tick."""

    dI_PF3_A: float
    dI_PF_topbot_A: float


# ── Targets / Scales ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ControlTargets:
    """Setpoint targets for the control loop."""

    R_target_m: float = 6.2
    Z_target_m: float = 0.0


@dataclass(frozen=True)
class ControlScales:
    """Normalisation scales (error → [-1, 1] range)."""

    R_scale_m: float = 0.5
    Z_scale_m: float = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clip01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _seed64(seed_base: int, sid: str) -> int:
    """Deterministic seed derivation: sha256(seed_base || sid) → u64."""
    h = hashlib.sha256(f"{seed_base}:{sid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


# ── Feature extraction ───────────────────────────────────────────────────────


def extract_features(
    obs: ControlObservation,
    targets: ControlTargets,
    scales: ControlScales,
) -> Dict[str, float]:
    """Map observation → unipolar [0, 1] feature sources.

    Returns dict with keys ``x_R_pos``, ``x_R_neg``, ``x_Z_pos``, ``x_Z_neg``.
    """
    eR = (targets.R_target_m - obs["R_axis_m"]) / scales.R_scale_m
    eZ = (targets.Z_target_m - obs["Z_axis_m"]) / scales.Z_scale_m

    # Clamp signed error to [-1, 1]
    eR = max(-1.0, min(1.0, eR))
    eZ = max(-1.0, min(1.0, eZ))

    return {
        "x_R_pos": _clip01(max(0.0, eR)),
        "x_R_neg": _clip01(max(0.0, -eR)),
        "x_Z_pos": _clip01(max(0.0, eZ)),
        "x_Z_neg": _clip01(max(0.0, -eZ)),
    }


# ── Action decoding ─────────────────────────────────────────────────────────


@dataclass
class ActionSpec:
    """One action channel: positive/negative place differencing."""

    name: str
    pos_place: int
    neg_place: int


def decode_actions(
    marking: List[float],
    actions_spec: List[ActionSpec],
    gains: List[float],
    abs_max: List[float],
    slew_per_s: List[float],
    dt: float,
    prev: List[float],
) -> Dict[str, float]:
    """Decode marking → actuator commands with gain, slew-rate, and abs clamp.

    Parameters
    ----------
    marking : current marking vector (len >= max place index).
    actions_spec : per-action pos/neg place definitions.
    gains : per-action gain multiplier.
    abs_max : per-action absolute saturation.
    slew_per_s : per-action max change rate (units/s).
    dt : control tick period (s).
    prev : previous action outputs (same length as actions_spec).

    Returns
    -------
    dict mapping action name → clamped value.  Also mutates *prev* in-place.
    """
    result: Dict[str, float] = {}
    for i, spec in enumerate(actions_spec):
        pos = marking[spec.pos_place]
        neg = marking[spec.neg_place]
        raw = (pos - neg) * gains[i]

        # Slew-rate limiting
        max_delta = slew_per_s[i] * dt
        raw = max(prev[i] - max_delta, min(prev[i] + max_delta, raw))

        # Absolute saturation
        raw = max(-abs_max[i], min(abs_max[i], raw))

        prev[i] = raw
        result[spec.name] = raw

    return result
