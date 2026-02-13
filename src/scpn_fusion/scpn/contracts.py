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
from typing import Dict, List, Mapping, Optional, Sequence, TypedDict


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


@dataclass(frozen=True)
class FeatureAxisSpec:
    """Configurable feature axis mapping from observation -> unipolar features.

    Parameters
    ----------
    obs_key : observation key to read.
    target : target/setpoint value for this axis.
    scale : normalisation scale for signed error (target - obs) / scale.
    pos_key : output feature key for positive error component.
    neg_key : output feature key for negative error component.
    """

    obs_key: str
    target: float
    scale: float
    pos_key: str
    neg_key: str


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
    obs: Mapping[str, float],
    targets: ControlTargets,
    scales: ControlScales,
    feature_axes: Optional[Sequence[FeatureAxisSpec]] = None,
    passthrough_keys: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Map observation → unipolar [0, 1] feature sources.

    By default returns keys ``x_R_pos``, ``x_R_neg``, ``x_Z_pos``, ``x_Z_neg``.
    Custom feature mappings can be supplied via ``feature_axes`` for arbitrary
    observation dictionaries.
    """
    axes = (
        list(feature_axes)
        if feature_axes is not None
        else [
            FeatureAxisSpec(
                obs_key="R_axis_m",
                target=targets.R_target_m,
                scale=scales.R_scale_m,
                pos_key="x_R_pos",
                neg_key="x_R_neg",
            ),
            FeatureAxisSpec(
                obs_key="Z_axis_m",
                target=targets.Z_target_m,
                scale=scales.Z_scale_m,
                pos_key="x_Z_pos",
                neg_key="x_Z_neg",
            ),
        ]
    )

    out: Dict[str, float] = {}
    for axis in axes:
        if axis.obs_key not in obs:
            raise KeyError(f"Missing observation key for feature extraction: {axis.obs_key}")
        scale = axis.scale if abs(axis.scale) > 1e-12 else 1e-12
        err = (axis.target - float(obs[axis.obs_key])) / scale
        err = max(-1.0, min(1.0, err))
        out[axis.pos_key] = _clip01(max(0.0, err))
        out[axis.neg_key] = _clip01(max(0.0, -err))

    if passthrough_keys is not None:
        for key in passthrough_keys:
            if key not in obs:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            out[key] = _clip01(float(obs[key]))

    return out


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
