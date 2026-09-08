# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — PROCESS diagnostic artifact reader
"""Read actual PROCESS power records without turning convergence into admission."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from pathlib import Path
from typing import Any


_FIELDS = (
    "ifail",
    "error_status",
    "sqsumsq",
    "p_plasma_imbalance_mw",
    "p_reactor_imbalance_mw",
    "p_electric_imbalance",
    "p_plant_imbalance_mw",
    "p_plant_electric_gross_mw",
    "p_plant_electric_recirc_mw",
    "p_plant_electric_net_mw",
)


def read_process_power_report(
    path: Path,
    *,
    expected_sha256: str,
    expected_constraints: tuple[int, ...],
    equality_tolerance: float,
) -> dict[str, Any]:
    """Inspect every scan in a caller-identified PROCESS MFILE artifact.

    Parameters
    ----------
    path : Path
        Actual PROCESS MFILE.DAT output, read through its public MFile parser.
    expected_sha256 : str
        Digest recorded by the producing run; mismatched artifacts are rejected.
    expected_constraints : tuple[int, ...]
        Exact active constraint IDs from the input case, including equalities.
    equality_tolerance : float
        Explicit finite positive bound for the recorded equality residual norm.

    Returns
    -------
    dict[str, Any]
        All scan values, inequality violations and separate recorded-check results.
        These are diagnostics, not source-provenance or physical admission proof.
        Plasma/reactor/plant imbalance limits use upstream's 0.1 MW diagnostic
        threshold. Electrical arithmetic is checked with the same bound.

    Raises
    ------
    ValueError
        For bad custody, missing constraints/fields, inconsistent scan counts,
        nonfinite values, conflicting repeated diagnostics, or invalid caller configuration.
    ModuleNotFoundError
        If the optional PROCESS runtime is unavailable.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA256 digest")
    if not math.isfinite(equality_tolerance) or equality_tolerance <= 0:
        raise ValueError("equality_tolerance must be finite and positive")
    if (
        not expected_constraints
        or len(set(expected_constraints)) != len(expected_constraints)
        or any(type(k) is not int or not 1 <= k <= 999 for k in expected_constraints)
    ):
        raise ValueError("expected_constraints must contain unique positive IDs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("PROCESS artifact digest mismatch")
    parser_module = importlib.import_module("process.core.io.mfile")
    parsed = parser_module.MFile(str(path))
    # MFile's missing-key sentinel can manufacture defaults: require membership.
    constraint_keys = sorted(k for k in parsed.data if re.fullmatch(r"(?:eq|ineq)_con[0-9]{3}", k))
    ids = [int(k[-3:]) for k in constraint_keys]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_constraints):
        raise ValueError("PROCESS active constraints do not match the declared case")
    keys = list(_FIELDS) + constraint_keys
    if any(k not in parsed.data for k in keys):
        raise ValueError("PROCESS power record is missing required diagnostics")
    # The upstream parser numbers each variable independently and ignores some
    # duplicates. Check physical scan boundaries before trusting its arrays.
    segments: list[list[tuple[str, str]]] = [[]]
    for line in raw.decode("utf-8").splitlines():
        words = line.split()
        if len(words) < 3 or words[0].startswith("#"):
            continue
        key = words[1].strip("_()").lower()
        if key == "iscan":
            segments.append([])
        elif key in keys:
            segments[-1].append((key, words[2]))
    if "iscan" in parsed.data:
        scan_count = parsed.data["iscan"].get_number_of_scans()
        labels = [parsed.data["iscan"].get_scan(i) for i in range(1, scan_count + 1)]
        if labels != list(range(1, scan_count + 1)) or len(segments) != scan_count + 1:
            raise ValueError("PROCESS scan labels must be consecutive from one")
        if segments[0]:
            raise ValueError("PROCESS diagnostics precede the first scan header")
        segments = segments[1:]
    else:
        scan_count = 1
    if any({key for key, _ in segment} != set(keys) for segment in segments):
        raise ValueError("PROCESS scan is missing required diagnostics")
    counts = {parsed.data[k].get_number_of_scans() for k in keys}
    if counts != {scan_count} or scan_count < 1:
        raise ValueError("PROCESS diagnostic scan counts are inconsistent")
    rows = []
    for scan in range(1, scan_count + 1):
        values = {k: float(parsed.data[k].get_scan(scan)) for k in keys}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("PROCESS diagnostic values must be finite")
        for key, token in segments[scan - 1]:
            raw_value = float(token)
            if not math.isfinite(raw_value):
                raise ValueError("PROCESS diagnostic values must be finite")
            if raw_value != values[key]:
                raise ValueError("PROCESS scan contains conflicting diagnostic values")
        if any(values[k] != int(values[k]) for k in ("ifail", "error_status")):
            raise ValueError("PROCESS status codes must be integers")
        violations = [k for k in constraint_keys if k.startswith("ineq_") and values[k] < 0]
        balances = {k: values[k] for k in _FIELDS if "imbalance" in k}
        electrical_residual = (
            values["p_plant_electric_gross_mw"]
            - values["p_plant_electric_recirc_mw"]
            - values["p_plant_electric_net_mw"]
        )
        if not math.isfinite(electrical_residual):
            raise ValueError("PROCESS electrical accounting residual must be finite")
        checks = {
            "solver_converged": values["ifail"] == 1,
            "upstream_has_no_errors": values["error_status"] == 0,
            "equality_norm_within_tolerance": 0 <= values["sqsumsq"] <= equality_tolerance,
            "all_declared_inequalities_satisfied": not violations,
            "recorded_power_balances_within_tolerance": all(
                abs(v) <= 0.1 for v in balances.values()
            ),
            "electrical_accounting_closes": abs(electrical_residual) <= 0.1,
        }
        rows.append(
            {
                "scan": scan,
                "values": values,
                "checks": checks,
                "violated_constraints": violations,
                "electrical_accounting_residual_mw": electrical_residual,
            }
        )
    # Detect concurrent mutation between custody check and upstream parser read.
    if path.read_bytes() != raw:
        raise ValueError("PROCESS artifact changed during parsing")
    return {
        "schema": "scpn-fusion.process-power-diagnostics.v1",
        "artifact_sha256": expected_sha256,
        "expected_constraints": list(expected_constraints),
        "equality_tolerance": equality_tolerance,
        "power_tolerance_mw": 0.1,
        "scans": rows,
        "actionable": False,
        "federated": False,
        "evidence_claimed": False,
        "provenance_verified": False,
        "scope": "recorded diagnostics only; source execution and physical admission not established",
    }
