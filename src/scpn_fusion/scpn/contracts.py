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
import math
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
        obs_value = float(obs[axis.obs_key])
        if not math.isfinite(obs_value):
            raise ValueError(
                f"Observation value for feature extraction must be finite: {axis.obs_key}"
            )
        target = float(axis.target)
        if not math.isfinite(target):
            raise ValueError(f"Feature axis target must be finite: {axis.obs_key}")
        scale_raw = float(axis.scale)
        if not math.isfinite(scale_raw):
            raise ValueError(f"Feature axis scale must be finite: {axis.obs_key}")
        scale = scale_raw if abs(scale_raw) > 1e-12 else 1e-12
        err = (target - obs_value) / scale
        err = max(-1.0, min(1.0, err))
        out[axis.pos_key] = _clip01(max(0.0, err))
        out[axis.neg_key] = _clip01(max(0.0, -err))

    if passthrough_keys is not None:
        for key in passthrough_keys:
            if key not in obs:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            value = float(obs[key])
            if not math.isfinite(value):
                raise ValueError(f"Passthrough observation value must be finite: {key}")
            out[key] = _clip01(value)

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
    n_actions = len(actions_spec)
    if (
        len(gains) != n_actions
        or len(abs_max) != n_actions
        or len(slew_per_s) != n_actions
        or len(prev) != n_actions
    ):
        raise ValueError(
            "actions_spec, gains, abs_max, slew_per_s, and prev must have equal lengths."
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")

    n_places = len(marking)
    result: Dict[str, float] = {}
    for i, spec in enumerate(actions_spec):
        if spec.pos_place < 0 or spec.neg_place < 0:
            raise ValueError("Action place indices must be >= 0.")
        if spec.pos_place >= n_places or spec.neg_place >= n_places:
            raise ValueError(
                "Action place index out of bounds for marking vector."
            )
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


# ── Physics Invariants ──────────────────────────────────────────────────────
# Hard physics constraints that the controller loop must respect.
# Violations trigger disruption mitigation protocols.
# ────────────────────────────────────────────────────────────────────────────

_VALID_COMPARATORS = ("gt", "lt", "gte", "lte")


@dataclass(frozen=True)
class PhysicsInvariant:
    """A hard physics constraint the controller loop must respect.

    Parameters
    ----------
    name : str
        Short identifier for the invariant (e.g. ``"q_min"``, ``"beta_N"``).
    description : str
        Human-readable description including the physics origin.
    threshold : float
        Threshold value for the invariant condition.
    comparator : str
        One of ``"gt"``, ``"lt"``, ``"gte"``, ``"lte"`` — the relationship
        that the *measured value* must satisfy with respect to ``threshold``
        for the invariant to hold.
    """

    name: str
    description: str
    threshold: float
    comparator: str  # "gt" | "lt" | "gte" | "lte"

    def __post_init__(self) -> None:
        if self.comparator not in _VALID_COMPARATORS:
            raise ValueError(
                f"Invalid comparator {self.comparator!r}; "
                f"must be one of {_VALID_COMPARATORS}"
            )
        if not math.isfinite(self.threshold):
            raise ValueError("PhysicsInvariant threshold must be finite.")


@dataclass(frozen=True)
class PhysicsInvariantViolation:
    """Record of a physics invariant violation.

    Parameters
    ----------
    invariant : PhysicsInvariant
        The invariant that was violated.
    actual_value : float
        The measured/computed value that violated the invariant.
    margin : float
        Absolute distance between ``actual_value`` and the invariant
        ``threshold`` (always >= 0).
    severity : str
        ``"warning"`` if margin <= 20 % of |threshold|,
        ``"critical"`` otherwise.
    """

    invariant: PhysicsInvariant
    actual_value: float
    margin: float
    severity: str  # "warning" | "critical"


# ── Default tokamak physics invariants ──────────────────────────────────────

DEFAULT_PHYSICS_INVARIANTS: List[PhysicsInvariant] = [
    PhysicsInvariant(
        name="q_min",
        description=(
            "Kruskal-Shafranov MHD stability limit: the edge safety factor "
            "q must exceed 1.0 to avoid the m=1/n=1 external kink mode.  "
            "Ref: Kruskal & Schwarzschild (1954); Shafranov (1970)."
        ),
        threshold=1.0,
        comparator="gt",
    ),
    PhysicsInvariant(
        name="beta_N",
        description=(
            "Troyon no-wall beta limit: normalised beta β_N = β(%) · a(m) · B_T(T) / I_P(MA) "
            "must remain below ~2.8 to avoid resistive wall modes without a conducting wall.  "
            "Ref: Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209."
        ),
        threshold=2.8,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="greenwald",
        description=(
            "Greenwald density limit: the line-averaged density normalised to "
            "n_GW = I_P / (π a²) must stay below ~1.2 to avoid radiative collapse "
            "and density-limit disruptions.  "
            "Ref: Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27."
        ),
        threshold=1.2,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="T_i",
        description=(
            "Ion temperature cap: T_i must remain below 25 keV to stay within "
            "the operating window of current first-wall / divertor materials and "
            "avoid excessive neutron wall-loading.  "
            "Ref: ITER Physics Basis, Nucl. Fusion 39 (1999) 2137."
        ),
        threshold=25.0,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="energy_conservation_error",
        description=(
            "Energy bookkeeping: the fractional mismatch between injected, "
            "radiated, and stored energy must stay below 1 % to trust the "
            "simulation state.  Tolerance: |ΔW/W| < 0.01."
        ),
        threshold=0.01,
        comparator="lt",
    ),
]


# ── Invariant checking ──────────────────────────────────────────────────────


def _is_satisfied(comparator: str, value: float, threshold: float) -> bool:
    """Return True when *value* satisfies the *comparator* w.r.t. *threshold*."""
    if comparator == "gt":
        return value > threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "gte":
        return value >= threshold
    if comparator == "lte":
        return value <= threshold
    raise ValueError(f"Unknown comparator: {comparator!r}")  # pragma: no cover


def check_physics_invariant(
    invariant: PhysicsInvariant,
    value: float,
) -> Optional[PhysicsInvariantViolation]:
    """Check a single physics invariant against a measured *value*.

    Returns ``None`` if the invariant is satisfied, otherwise returns a
    :class:`PhysicsInvariantViolation` with computed margin and severity.

    Severity classification
    -----------------------
    * ``"critical"`` — margin exceeds 20 % of ``|threshold|``
      (or 20 % of 1.0 when threshold == 0).
    * ``"warning"``  — violated but within the 20 % band.

    Parameters
    ----------
    invariant : PhysicsInvariant
        The invariant to check.
    value : float
        Current measured / computed value for the quantity.
    """
    if not math.isfinite(value):
        # Non-finite values always violate; treat as critical.
        return PhysicsInvariantViolation(
            invariant=invariant,
            actual_value=value,
            margin=float("inf"),
            severity="critical",
        )

    if _is_satisfied(invariant.comparator, value, invariant.threshold):
        return None

    margin = abs(value - invariant.threshold)
    ref = abs(invariant.threshold) if invariant.threshold != 0.0 else 1.0
    severity = "critical" if margin > 0.20 * ref else "warning"

    return PhysicsInvariantViolation(
        invariant=invariant,
        actual_value=value,
        margin=margin,
        severity=severity,
    )


def check_all_invariants(
    values: Dict[str, float],
    invariants: Optional[List[PhysicsInvariant]] = None,
) -> List[PhysicsInvariantViolation]:
    """Check every invariant whose name appears in *values*.

    Parameters
    ----------
    values : dict
        Mapping from invariant ``name`` to the current measured value.
        Names not present in the invariant list are silently ignored.
    invariants : list, optional
        The invariant set to check.  Defaults to
        :data:`DEFAULT_PHYSICS_INVARIANTS`.

    Returns
    -------
    list[PhysicsInvariantViolation]
        All detected violations (empty list when everything is nominal).
    """
    if invariants is None:
        invariants = DEFAULT_PHYSICS_INVARIANTS

    violations: List[PhysicsInvariantViolation] = []
    for inv in invariants:
        if inv.name in values:
            v = check_physics_invariant(inv, values[inv.name])
            if v is not None:
                violations.append(v)
    return violations


def should_trigger_mitigation(
    violations: List[PhysicsInvariantViolation],
) -> bool:
    """Return ``True`` if any violation has ``severity == "critical"``.

    This is the top-level disruption-mitigation gate: a single critical
    violation means the controller must engage protective actions (e.g.
    massive gas injection, current quench, or safe ramp-down).
    """
    return any(v.severity == "critical" for v in violations)
