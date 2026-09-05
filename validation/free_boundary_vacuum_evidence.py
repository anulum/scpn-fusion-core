# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vacuum evidence consistency
"""Check stored vacuum samples and coil records without granting scientific admission."""

from __future__ import annotations

import math
from typing import Any, TypeGuard

import numpy as np


def _finite_number(value: object) -> TypeGuard[int | float]:
    """Reject booleans and nonfinite or unrepresentable JSON numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _coil_records_match(coils: list[dict[str, Any]], filament_count: object) -> bool:
    """Validate named coil records and the producer's current/turns/count identities."""
    names: set[str] = set()
    total = 0
    for coil in coils:
        name = coil.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            return False
        names.add(name)
        current, turns = coil.get("current_a"), coil.get("turns")
        effective = coil.get("effective_current_a_turns")
        if (
            not _finite_number(current)
            or not _finite_number(turns)
            or turns <= 0
            or not _finite_number(effective)
        ):
            return False
        product = float(current) * float(turns)
        if not math.isfinite(product) or effective != product:
            return False
        count = coil.get("filament_count")
        if type(count) is not int or count < 1:
            return False
        total += count
    return type(filament_count) is int and filament_count == total


def _vacuum_sidecar_structure_matches(sidecar: object) -> bool:
    """Check reported sidecar membership without qualifying a vacuum solution."""
    if not isinstance(sidecar, dict):
        return False
    coils = sidecar.get("coils")
    if (
        not isinstance(coils, list)
        or not coils
        or any(not isinstance(coil, dict) for coil in coils)
        or type(sidecar.get("coil_count")) is not int
        or sidecar["coil_count"] != len(coils)
    ):
        return False
    if not _coil_records_match(coils, sidecar.get("filament_count")):
        return False
    count = sidecar.get("sample_point_count")
    if type(count) is not int or count <= 0:
        return False
    for name in ("native_vacuum_psi_sample", "freegs_vacuum_psi_sample"):
        samples = sidecar.get(name)
        if (
            not isinstance(samples, list)
            or len(samples) != count
            or not all(_finite_number(value) for value in samples)
        ):
            return False
    return True


def _vacuum_numerics_consistent(sidecar: object) -> bool:
    """Replay producer residual metrics and fixed limits on stored psi samples."""
    if not isinstance(sidecar, dict) or not _vacuum_sidecar_structure_matches(sidecar):
        return False
    native = np.asarray(sidecar["native_vacuum_psi_sample"], dtype=np.float64)
    reference = np.asarray(sidecar["freegs_vacuum_psi_sample"], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = native - reference
        scale = max(float(np.max(reference) - np.min(reference)), 1e-30)
        rmse = float(np.sqrt(np.mean(residual * residual)))
        nrmse = float(rmse / scale)
        max_abs = float(np.max(np.abs(residual)))
    computed = {"rmse_wb": rmse, "nrmse": nrmse, "max_abs_error_wb": max_abs}
    for name, value in computed.items():
        reported = sidecar.get(name)
        if not math.isfinite(value) or not _finite_number(reported) or reported != value:
            return False
    thresholds = sidecar.get("thresholds")
    if not isinstance(thresholds, dict) or any(
        not _finite_number(thresholds.get(name)) or thresholds[name] != 1e-12
        for name in ("nrmse", "max_abs_error_wb")
    ):
        return False
    passed = nrmse <= 1e-12 and max_abs <= 1e-12
    return sidecar.get("pass") is passed


def evaluate_vacuum_evidence(sidecar: object) -> dict[str, bool]:
    """Check sidecar structure and numerical self-consistency.

    Parameters
    ----------
    sidecar : object
        Stored vacuum comparison from the public FreeGS reconstruction report.

    Returns
    -------
    dict[str, bool]
        Structure and metric consistency only, not field provenance or physics
        admission. Consistently reported failed comparisons remain failures.
    """
    return {
        "structure_consistent": _vacuum_sidecar_structure_matches(sidecar),
        "metrics_consistent": _vacuum_numerics_consistent(sidecar),
    }
