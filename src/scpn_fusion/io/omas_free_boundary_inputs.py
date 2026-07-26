# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — OMAS Free-Boundary Input Contract
"""Strict OMAS extraction for predictive free-boundary input evidence.

The adapter deliberately does not interpolate channels, invent uncertainties,
or infer provenance.  Development payloads can be inspected with
``require_ingestion_ready=False``; strict mode fails closed until every
declared channel-ingestion requirement is present.  This bounded contract can
never admit the wider Tier-0 scientific claim by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import operator
import re
from typing import Any, Literal, Protocol, cast

import numpy as np


OMAS_FREE_BOUNDARY_INPUT_SCHEMA = "scpn-fusion.omas-free-boundary-inputs.v1"
CANONICAL_COCOS = frozenset((*range(1, 9), *range(11, 19)))
TIER0_OUT_OF_SCOPE_BLOCKERS = (
    "equilibrium_and_control_targets_out_of_scope",
    "shot_split_and_heldout_evaluation_out_of_scope",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAS_VERSION_RE = re.compile(r"[1-9][0-9]*\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
_MISSING = object()
_MAX_COLLECTION_ITEMS = 10_000


class ODSLike(Protocol):
    """Minimal dotted-path read surface required from an OMAS ODS."""

    def get(self, key: str, default: Any = None) -> Any:
        """Return a dotted-path value or ``default`` when absent."""


@dataclass(frozen=True)
class OmasSourceProvenance:
    """External binding that an ODS alone cannot prove reliably."""

    machine: str
    shot_id: int
    run_id: int
    source_uri: str
    source_sha256: str
    license_id: str

    def __post_init__(self) -> None:
        """Reject incomplete or malformed source bindings."""
        for field_name in ("machine", "source_uri", "license_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"provenance {field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.source_sha256, str):
            raise ValueError("provenance source_sha256 must be a hexadecimal string")
        digest = self.source_sha256.lower()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("provenance source_sha256 must be 64 hexadecimal characters")
        object.__setattr__(self, "source_sha256", digest)
        try:
            shot_id = operator.index(self.shot_id)
            run_id = operator.index(self.run_id)
        except TypeError as exc:
            raise ValueError("provenance shot_id and run_id must be integers") from exc
        if isinstance(self.shot_id, bool) or isinstance(self.run_id, bool):
            raise ValueError("provenance shot_id and run_id must be integers")
        if shot_id < 0 or run_id < 0:
            raise ValueError("provenance shot_id and run_id must be non-negative")
        object.__setattr__(self, "shot_id", shot_id)
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True)
class TimeSeriesSI:
    """One finite, strictly ordered SI-unit time series."""

    time_s: tuple[float, ...]
    values: tuple[float, ...]
    error_lower: tuple[float, ...] | None
    error_upper: tuple[float, ...] | None
    validity: int | None


@dataclass(frozen=True)
class PfElementGeometry:
    """One signed PF element with an explicit IMAS geometry representation."""

    identifier: str
    turns_with_sign: float
    geometry_type: Literal[1, 2]
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    width_m: float | None
    height_m: float | None


@dataclass(frozen=True)
class PfCoilInput:
    """PF-coil current history and matching signed geometry."""

    identifier: str
    elements: tuple[PfElementGeometry, ...]
    current_a: TimeSeriesSI


@dataclass(frozen=True)
class PoloidalFieldProbeInput:
    """Poloidal-field probe position, orientation, and field history."""

    identifier: str
    r_m: float
    z_m: float
    poloidal_angle_rad: float
    length_m: float
    field_t: TimeSeriesSI


@dataclass(frozen=True)
class FluxLoopInput:
    """Flux-loop geometry and poloidal-flux history."""

    identifier: str
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    flux_wb: TimeSeriesSI


@dataclass(frozen=True)
class OmasFreeBoundaryInputs:
    """Validated OMAS channels with distinct ingestion and Tier-0 states."""

    schema: str
    cocos: int
    imas_version: str | None
    provenance: OmasSourceProvenance | None
    time_alignment: Literal["native_unaligned", "exact_common_axis"]
    pf_coils: tuple[PfCoilInput, ...]
    bpol_probes: tuple[PoloidalFieldProbeInput, ...]
    flux_loops: tuple[FluxLoopInput, ...]
    ingestion_blockers: tuple[str, ...]
    ingestion_ready: bool
    tier0_claim_blockers: tuple[str, ...]
    tier0_claim_admission_ready: Literal[False]
    payload_sha256: str


def _required(ods: ODSLike, path: str) -> Any:
    value = ods.get(path, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"missing required OMAS path: {path}")
    return value


def _finite_scalar(value: Any, path: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{path} must be a real scalar, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _finite_vector(value: Any, path: str, *, nonempty: bool = True) -> tuple[float, ...]:
    raw_array = np.asarray(value)
    if np.issubdtype(raw_array.dtype, np.bool_):
        raise ValueError(f"{path} must be a real array, not boolean")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a one-dimensional real array") from exc
    if array.ndim != 1 or (nonempty and array.size == 0):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{path} must be a {qualifier}one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} must contain only finite values")
    return tuple(float(item) for item in array)


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, (str, np.str_)):
        raise ValueError(f"{path} must be a string identifier")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{path} must be a non-empty identifier")
    return result


def _series(ods: ODSLike, prefix: str, units_name: str) -> TimeSeriesSI:
    time = _finite_vector(_required(ods, f"{prefix}.time"), f"{prefix}.time")
    values = _finite_vector(_required(ods, f"{prefix}.data"), f"{prefix}.data")
    if len(time) != len(values):
        raise ValueError(f"{prefix} time/data lengths differ")
    if any(right <= left for left, right in zip(time, time[1:])):
        raise ValueError(f"{prefix}.time must be strictly increasing")

    errors: dict[str, tuple[float, ...] | None] = {}
    for suffix in ("error_lower", "error_upper"):
        path = f"{prefix}.data_{suffix}"
        raw = ods.get(path, _MISSING)
        if raw is _MISSING or raw is None:
            errors[suffix] = None
            continue
        error = _finite_vector(raw, path)
        if len(error) != len(values):
            raise ValueError(f"{path} length differs from {prefix}.data")
        if any(item < 0.0 for item in error):
            raise ValueError(f"{path} must contain non-negative {units_name} uncertainties")
        errors[suffix] = error

    raw_validity = ods.get(f"{prefix}.validity", _MISSING)
    validity: int | None = None
    if raw_validity is not _MISSING and raw_validity is not None:
        numeric = _finite_scalar(raw_validity, f"{prefix}.validity")
        validity = int(numeric)
        if numeric != validity or validity > 1:
            raise ValueError(f"{prefix}.validity must be an integer no greater than 1")

    return TimeSeriesSI(
        time_s=time,
        values=values,
        error_lower=errors["error_lower"],
        error_upper=errors["error_upper"],
        validity=validity,
    )


def _collection_indices(ods: ODSLike, path: str) -> tuple[int, ...]:
    raw_collection = ods.get(path, _MISSING)
    indices: list[int] = []
    if raw_collection is not _MISSING:
        keys_method = getattr(raw_collection, "keys", None)
        if not callable(keys_method):
            raise ValueError(f"{path} must be an enumerable OMAS array of structures")
        for key in keys_method():
            try:
                index = operator.index(key)
            except TypeError as exc:
                raise ValueError(f"{path} contains a non-integer array index") from exc
            indices.append(index)
    elif isinstance(ods, Mapping):
        prefix = f"{path}."
        for key in ods:
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            index_text = key[len(prefix) :].partition(".")[0]
            if not index_text.isdecimal():
                raise ValueError(f"{path} contains a non-integer dotted array index")
            indices.append(int(index_text))
    else:
        raise ValueError(f"{path} cannot be enumerated without an OMAS array or dotted mapping")

    unique_indices = tuple(sorted(set(indices)))
    if any(index < 0 for index in unique_indices):
        raise ValueError(f"{path} contains a negative array index")
    if len(unique_indices) > _MAX_COLLECTION_ITEMS:
        raise ValueError(f"{path} exceeds the {_MAX_COLLECTION_ITEMS}-item safety limit")
    if unique_indices != tuple(range(len(unique_indices))):
        raise ValueError(f"{path} contains sparse or gapped array indices: {unique_indices}")
    return unique_indices


def _pf_element(ods: ODSLike, coil_index: int, element_index: int) -> PfElementGeometry:
    prefix = f"pf_active.coil.{coil_index}.element.{element_index}"
    identifier = _identifier(_required(ods, f"{prefix}.identifier"), f"{prefix}.identifier")
    turns = _finite_scalar(_required(ods, f"{prefix}.turns_with_sign"), f"{prefix}.turns_with_sign")
    if turns == 0.0:
        raise ValueError(f"{prefix}.turns_with_sign must be non-zero")
    raw_geometry_type = _finite_scalar(
        _required(ods, f"{prefix}.geometry.geometry_type"),
        f"{prefix}.geometry.geometry_type",
    )
    geometry_type = int(raw_geometry_type)
    if raw_geometry_type != geometry_type or geometry_type not in (1, 2):
        raise ValueError(f"{prefix}.geometry.geometry_type must be 1 (outline) or 2 (rectangle)")

    if geometry_type == 1:
        r_m = _finite_vector(
            _required(ods, f"{prefix}.geometry.outline.r"), f"{prefix}.geometry.outline.r"
        )
        z_m = _finite_vector(
            _required(ods, f"{prefix}.geometry.outline.z"), f"{prefix}.geometry.outline.z"
        )
        if len(r_m) != len(z_m) or len(r_m) < 3:
            raise ValueError(
                f"{prefix}.geometry.outline must have matching r/z arrays of length >= 3"
            )
        if any(r <= 0.0 for r in r_m):
            raise ValueError(f"{prefix}.geometry.outline.r must be positive")
        return PfElementGeometry(identifier, turns, 1, r_m, z_m, None, None)

    r = _finite_scalar(
        _required(ods, f"{prefix}.geometry.rectangle.r"), f"{prefix}.geometry.rectangle.r"
    )
    z = _finite_scalar(
        _required(ods, f"{prefix}.geometry.rectangle.z"), f"{prefix}.geometry.rectangle.z"
    )
    width = _finite_scalar(
        _required(ods, f"{prefix}.geometry.rectangle.width"),
        f"{prefix}.geometry.rectangle.width",
    )
    height = _finite_scalar(
        _required(ods, f"{prefix}.geometry.rectangle.height"),
        f"{prefix}.geometry.rectangle.height",
    )
    if r <= 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError(f"{prefix}.geometry.rectangle requires positive r, width, and height")
    return PfElementGeometry(identifier, turns, 2, (r,), (z,), width, height)


def _extract_pf_coils(ods: ODSLike) -> tuple[PfCoilInput, ...]:
    coils: list[PfCoilInput] = []
    for coil_index in _collection_indices(ods, "pf_active.coil"):
        prefix = f"pf_active.coil.{coil_index}"
        identifier = _identifier(_required(ods, f"{prefix}.identifier"), f"{prefix}.identifier")
        elements: list[PfElementGeometry] = []
        for element_index in _collection_indices(ods, f"{prefix}.element"):
            elements.append(_pf_element(ods, coil_index, element_index))
        if not elements:
            raise ValueError(f"{prefix} must contain at least one geometry element")
        coils.append(
            PfCoilInput(identifier, tuple(elements), _series(ods, f"{prefix}.current", "A"))
        )
    return tuple(coils)


def _extract_bpol_probes(ods: ODSLike) -> tuple[PoloidalFieldProbeInput, ...]:
    probes: list[PoloidalFieldProbeInput] = []
    for index in _collection_indices(ods, "magnetics.b_field_pol_probe"):
        prefix = f"magnetics.b_field_pol_probe.{index}"
        identifier = _identifier(_required(ods, f"{prefix}.identifier"), f"{prefix}.identifier")
        r = _finite_scalar(_required(ods, f"{prefix}.position.r"), f"{prefix}.position.r")
        z = _finite_scalar(_required(ods, f"{prefix}.position.z"), f"{prefix}.position.z")
        angle = _finite_scalar(
            _required(ods, f"{prefix}.poloidal_angle"), f"{prefix}.poloidal_angle"
        )
        length = _finite_scalar(_required(ods, f"{prefix}.length"), f"{prefix}.length")
        if r <= 0.0 or length <= 0.0:
            raise ValueError(f"{prefix} requires positive position.r and length")
        probes.append(
            PoloidalFieldProbeInput(
                identifier, r, z, angle, length, _series(ods, f"{prefix}.field", "T")
            )
        )
    return tuple(probes)


def _extract_flux_loops(ods: ODSLike) -> tuple[FluxLoopInput, ...]:
    loops: list[FluxLoopInput] = []
    for index in _collection_indices(ods, "magnetics.flux_loop"):
        prefix = f"magnetics.flux_loop.{index}"
        identifier = _identifier(_required(ods, f"{prefix}.identifier"), f"{prefix}.identifier")
        r_values: list[float] = []
        z_values: list[float] = []
        for position_index in _collection_indices(ods, f"{prefix}.position"):
            position = f"{prefix}.position.{position_index}"
            r = _finite_scalar(_required(ods, f"{position}.r"), f"{position}.r")
            z = _finite_scalar(_required(ods, f"{position}.z"), f"{position}.z")
            if r <= 0.0:
                raise ValueError(f"{position}.r must be positive")
            r_values.append(r)
            z_values.append(z)
        if not r_values:
            raise ValueError(f"{prefix} must contain at least one position")
        loops.append(
            FluxLoopInput(
                identifier,
                tuple(r_values),
                tuple(z_values),
                _series(ods, f"{prefix}.flux", "Wb"),
            )
        )
    return tuple(loops)


def _resolve_cocos(ods: ODSLike, cocos: int | None) -> int:
    candidate = cocos
    if candidate is None:
        candidate = cast(int | None, getattr(ods, "cocos", None))
    if candidate is None:
        raise ValueError("COCOS must be supplied explicitly or available as ods.cocos")
    if isinstance(candidate, bool):
        raise ValueError("COCOS must be one of 1-8 or 11-18")
    try:
        resolved = operator.index(candidate)
    except TypeError as exc:
        raise ValueError("COCOS must be an integer in 1-8 or 11-18") from exc
    if resolved not in CANONICAL_COCOS:
        raise ValueError("COCOS must be one of 1-8 or 11-18")
    return resolved


def _require_unique_identifiers(items: tuple[Any, ...], collection: str) -> None:
    identifiers = [str(item.identifier) for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{collection} identifiers must be unique")


def _all_series(
    coils: tuple[PfCoilInput, ...],
    probes: tuple[PoloidalFieldProbeInput, ...],
    loops: tuple[FluxLoopInput, ...],
) -> tuple[TimeSeriesSI, ...]:
    return (
        *(coil.current_a for coil in coils),
        *(probe.field_t for probe in probes),
        *(loop.flux_wb for loop in loops),
    )


def _ingestion_blockers(
    provenance: OmasSourceProvenance | None,
    imas_version: str | None,
    time_alignment: Literal["native_unaligned", "exact_common_axis"],
    coils: tuple[PfCoilInput, ...],
    probes: tuple[PoloidalFieldProbeInput, ...],
    loops: tuple[FluxLoopInput, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if provenance is None:
        blockers.append("missing_source_provenance")
    if imas_version is None:
        blockers.append("missing_imas_version")
    if not coils:
        blockers.append("missing_pf_active_coils")
    if not probes:
        blockers.append("missing_bpol_probes")
    if not loops:
        blockers.append("missing_flux_loops")

    labelled_series = (
        *((f"pf_current_{index}", coil.current_a, False) for index, coil in enumerate(coils)),
        *((f"bpol_probe_{index}", probe.field_t, True) for index, probe in enumerate(probes)),
        *((f"flux_loop_{index}", loop.flux_wb, True) for index, loop in enumerate(loops)),
    )
    for label, channel, requires_validity in labelled_series:
        if channel.error_lower is None or channel.error_upper is None:
            blockers.append(f"{label}_missing_uncertainty")
        if requires_validity:
            if channel.validity is None:
                blockers.append(f"{label}_missing_validity")
            elif channel.validity < 0:
                blockers.append(f"{label}_invalid_or_uncertified")

    series = _all_series(coils, probes, loops)
    if time_alignment != "exact_common_axis":
        blockers.append("time_alignment_not_exact_common_axis")
    elif series and any(channel.time_s != series[0].time_s for channel in series[1:]):
        blockers.append("declared_common_axis_mismatch")
    return tuple(blockers)


def _payload_digest(
    cocos: int,
    imas_version: str | None,
    provenance: OmasSourceProvenance | None,
    time_alignment: str,
    coils: tuple[PfCoilInput, ...],
    probes: tuple[PoloidalFieldProbeInput, ...],
    loops: tuple[FluxLoopInput, ...],
    blockers: tuple[str, ...],
) -> str:
    payload = {
        "schema": OMAS_FREE_BOUNDARY_INPUT_SCHEMA,
        "cocos": cocos,
        "imas_version": imas_version,
        "provenance": asdict(provenance) if provenance is not None else None,
        "time_alignment": time_alignment,
        "pf_coils": [asdict(coil) for coil in coils],
        "bpol_probes": [asdict(probe) for probe in probes],
        "flux_loops": [asdict(loop) for loop in loops],
        "ingestion_blockers": blockers,
        "tier0_claim_blockers": TIER0_OUT_OF_SCOPE_BLOCKERS,
        "tier0_claim_admission_ready": False,
    }
    canonical = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def extract_omas_free_boundary_inputs(
    ods: ODSLike,
    *,
    provenance: OmasSourceProvenance | None = None,
    cocos: int | None = None,
    time_alignment: Literal["native_unaligned", "exact_common_axis"] = "native_unaligned",
    require_ingestion_ready: bool = True,
) -> OmasFreeBoundaryInputs:
    """Extract strict SI-unit PF and magnetics channels from an OMAS ODS.

    Parameters
    ----------
    ods:
        OMAS ODS (or compatible dotted-path mapping).  IMAS schema units are
        preserved: seconds, amperes, metres, radians, tesla, and webers.
    provenance:
        Immutable source binding supplied by the acquisition layer.
    cocos:
        Canonical COCOS index.  When omitted, ``ods.cocos`` is used.
    time_alignment:
        Declare whether every channel was acquired on one exact time axis.
        The adapter verifies an ``exact_common_axis`` declaration byte-for-byte.
    require_ingestion_ready:
        Raise when any provenance, uncertainty, validity, channel, or alignment
        gate is missing.  Set false only for explicit development inspection.

    Returns
    -------
    OmasFreeBoundaryInputs
        Immutable extracted inputs, ingestion blockers, readiness state, and
        digest.  Full Tier-0 claim admission remains explicitly false.

    Raises
    ------
    ValueError
        If the ODS is structurally malformed or strict ingestion is blocked.
    """
    if time_alignment not in ("native_unaligned", "exact_common_axis"):
        raise ValueError("time_alignment must be native_unaligned or exact_common_axis")
    resolved_cocos = _resolve_cocos(ods, cocos)
    raw_imas_version = getattr(ods, "imas_version", None)
    imas_version: str | None = None
    if raw_imas_version is not None:
        if not isinstance(raw_imas_version, str):
            raise ValueError("ods.imas_version must be a version string when present")
        imas_version = raw_imas_version.strip()
        if _IMAS_VERSION_RE.fullmatch(imas_version) is None:
            raise ValueError("ods.imas_version must use a supported semantic version form")
    coils = _extract_pf_coils(ods)
    probes = _extract_bpol_probes(ods)
    loops = _extract_flux_loops(ods)
    _require_unique_identifiers(coils, "pf_active.coil")
    _require_unique_identifiers(probes, "magnetics.b_field_pol_probe")
    _require_unique_identifiers(loops, "magnetics.flux_loop")
    for coil in coils:
        _require_unique_identifiers(coil.elements, f"pf_active coil {coil.identifier} elements")
    blockers = _ingestion_blockers(provenance, imas_version, time_alignment, coils, probes, loops)
    digest = _payload_digest(
        resolved_cocos,
        imas_version,
        provenance,
        time_alignment,
        coils,
        probes,
        loops,
        blockers,
    )
    result = OmasFreeBoundaryInputs(
        schema=OMAS_FREE_BOUNDARY_INPUT_SCHEMA,
        cocos=resolved_cocos,
        imas_version=imas_version,
        provenance=provenance,
        time_alignment=time_alignment,
        pf_coils=coils,
        bpol_probes=probes,
        flux_loops=loops,
        ingestion_blockers=blockers,
        ingestion_ready=not blockers,
        tier0_claim_blockers=TIER0_OUT_OF_SCOPE_BLOCKERS,
        tier0_claim_admission_ready=False,
        payload_sha256=digest,
    )
    if require_ingestion_ready and blockers:
        raise ValueError(
            "OMAS free-boundary inputs are not ingestion-ready: " + ", ".join(blockers)
        )
    return result


__all__ = [
    "CANONICAL_COCOS",
    "OMAS_FREE_BOUNDARY_INPUT_SCHEMA",
    "TIER0_OUT_OF_SCOPE_BLOCKERS",
    "FluxLoopInput",
    "OmasFreeBoundaryInputs",
    "OmasSourceProvenance",
    "PfCoilInput",
    "PfElementGeometry",
    "PoloidalFieldProbeInput",
    "TimeSeriesSI",
    "extract_omas_free_boundary_inputs",
]
