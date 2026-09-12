# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — explicit RustBCA sampling request
"""Validate bounded ion/material sampling requests before starting RustBCA."""

from __future__ import annotations

import math
from typing import Any, cast


def _number(value: object, name: str, *, positive: bool = False) -> float:
    """Normalize a finite nonnegative scalar, optionally requiring strict positivity."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number, not a boolean or string")
    try:
        number = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    """Require an exact nonboolean integer within the inclusive request bounds."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _record(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    """Copy a record only when its keys exactly match the request schema."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value.copy()


def _material(value: object, name: str, *, target: bool) -> dict[str, Any]:
    """Validate species fields and provenance, including target-only density and binding energy."""
    keys = {"Z", "m", "Ec", "Es", "label", "provenance"}
    if target:
        keys |= {"Eb", "n"}
    record = _record(value, name, keys)
    for field in ("label", "provenance"):
        text = record[field]
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError(f"{name}.{field} must be nonempty text of at most 4096 characters")
    record["Z"] = _integer(record["Z"], name + ".Z", 1, 118)
    for field in keys - {"Z", "label", "provenance"}:
        record[field] = _number(
            record[field], name + "." + field, positive=field in {"m", "Ec", "n"}
        )
    return record


def _grid(value: object, name: str, upper: float, *, positive: bool) -> list[float]:
    """Normalize one to eight unique scalars below the exclusive physical limit."""
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError(f"{name} must be a list with one to eight values")
    numbers = [_number(item, name, positive=positive) for item in value]
    if len(set(numbers)) != len(numbers) or any(number >= upper for number in numbers):
        raise ValueError(f"{name} must be unique and below {upper}")
    return numbers


def validate_rustbca_request(request: object) -> dict[str, Any]:
    """Return an independent normalized request for the ergonomic RustBCA APIs.

    Parameters
    ----------
    request : object
        JSON-shaped mapping with schema, ion, target, energies_eV, angles_deg,
        samples_per_seed, seeds and threads. Species use Z, mass m in amu,
        cutoff Ec and surface binding Es in eV; target also requires bulk binding
        Eb in eV and number density n in m^-3. Both require label and provenance.

    Returns
    -------
    dict[str, Any]
        Detached request with finite floating-point material/grid values and
        integer atomic numbers, sample counts, seeds and thread count.

    Raises
    ------
    ValueError
        For missing/unknown keys, invalid numeric/text values, duplicate grid or
        seed entries, or requests exceeding the computational sampling contract.

    Notes
    -----
    The contract allows at most eight energies below 100 keV and eight angles
    from zero up to but excluding 90 degrees, two to sixteen unique unsigned
    64-bit seeds, one to eight threads and up to one million incident histories
    per observable over the grid. These are workload bounds, not physics validity
    limits. Each energy must exceed the incident-ion cutoff. At least two seeds
    permit empirical batch dispersion; they do not guarantee a confidence bound.
    Provenance is caller-supplied attribution, not authenticated material evidence.
    No defaults, clipping, species inference or source/physical admission occurs.
    """
    record = _record(
        request,
        "request",
        {
            "schema",
            "ion",
            "target",
            "energies_eV",
            "angles_deg",
            "samples_per_seed",
            "seeds",
            "threads",
        },
    )
    if record["schema"] != "scpn-fusion.rustbca-request.v1":
        raise ValueError("Unsupported RustBCA request schema")
    record["ion"] = _material(record["ion"], "ion", target=False)
    record["target"] = _material(record["target"], "target", target=True)
    record["energies_eV"] = _grid(record["energies_eV"], "energies_eV", 100_000.0, positive=True)
    record["angles_deg"] = _grid(record["angles_deg"], "angles_deg", 90.0, positive=False)
    if any(energy <= record["ion"]["Ec"] for energy in record["energies_eV"]):
        raise ValueError("Incident energies must exceed the ion cutoff Ec")
    record["samples_per_seed"] = _integer(
        record["samples_per_seed"], "samples_per_seed", 1, 100_000
    )
    record["threads"] = _integer(record["threads"], "threads", 1, 8)
    seeds = record["seeds"]
    if not isinstance(seeds, list) or not 2 <= len(seeds) <= 16:
        raise ValueError("seeds must contain two to sixteen explicit values")
    record["seeds"] = [_integer(seed, "seed", 0, 2**64 - 1) for seed in seeds]
    if len(set(record["seeds"])) != len(record["seeds"]):
        raise ValueError("seeds must be unique")
    histories = (
        len(record["energies_eV"])
        * len(record["angles_deg"])
        * len(seeds)
        * record["samples_per_seed"]
    )
    if histories > 1_000_000:
        raise ValueError("Request exceeds one million histories per observable")
    return record
