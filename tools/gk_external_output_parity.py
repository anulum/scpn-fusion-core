#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed nonlinear GK external output conversion and parity checks.

This tool accepts only redistribution-permitted same-deck GENE, CGYRO, and GS2
nonlinear output payloads. Missing outputs produce blocked benchmark rows; input
decks, web pages, native synthetic payloads, and reduced-order surrogates are
never promoted to full-fidelity parity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "external" / "full_fidelity_public_sources"
EXTERNAL_OUTPUT_ROOT = CACHE_ROOT / "gk_external_outputs"
ARTIFACT_DIR = ROOT / "validation" / "reference_data" / "full_fidelity_public_artifacts"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "gk_external_nonlinear_parity.json"
MD_REPORT = REPORT_DIR / "gk_external_nonlinear_parity.md"
REQUIRED_SOLVER_FAMILIES = ("GENE", "CGYRO", "GS2")
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "gk-nonlinear-external-output-manifest.v1"
OUTPUT_SCHEMA = "gk-nonlinear-external-output.v1"
REQUIRED_MANIFEST_CASE_FIELDS = (
    "case_id",
    "deck_id",
    "benchmark_case_id",
    "deck_physics_sha256",
    "output_path",
    "provenance_url",
    "redistribution_license",
    "sha256",
)
ALLOWED_REDISTRIBUTION_LICENSES = {
    "agpl-3.0-or-later",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "gpl-2.0-or-later",
    "gpl-3.0-or-later",
    "mit",
}
BLOCKED_LICENSE_TOKENS = (
    "all-rights-reserved",
    "internal",
    "non-redistributable",
    "private",
    "proprietary",
    "restricted",
    "unknown",
)


def _rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _public_path(path: Path, artifact_dir: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    if root_resolved in resolved.parents or resolved == root_resolved:
        return _rel(resolved, root_resolved)
    return _rel(path, artifact_dir.parent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_case() -> dict[str, Any]:
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    cases = manifest["surfaces"]["native_nonlinear_gyrokinetics"]["required_cases"]
    if not cases:
        raise ValueError("native nonlinear GK reference case is missing")
    return dict(cases[0])


def _missing_requirements() -> list[str]:
    return [
        "same_deck_external_nonlinear_output",
        "nonlinear_distribution_output",
        "heat_flux_spectra_time_kx_ky_species",
        "field_energy_history_phi_apar_bpar",
        "zonal_flow_and_saturation_metrics",
        "native_same_case_solver_output_comparison",
        "grid_convergence_evidence",
        "production_scale_scaling_evidence",
    ]


def _blocked_row(family: str, status: str) -> dict[str, Any]:
    return {
        "available_observables": [],
        "benchmark_case_id": None,
        "case_id": None,
        "complete_required_observables": False,
        "converted_artifact_path": None,
        "deck_physics_sha256": None,
        "deck_id": None,
        "missing_requirements": _missing_requirements(),
        "native_same_case_comparison_passed": False,
        "native_same_case_comparison_ready": False,
        "provenance_url": None,
        "redistribution_license": None,
        "reference_output_ready": False,
        "sha256": None,
        "solver_family": family,
        "status": status,
        "threshold_evaluation": {"ready": False, "passed": False, "checks": []},
    }


def _load_manifest(source_root: Path) -> dict[str, Any] | None:
    path = source_root / MANIFEST_NAME
    if not path.exists():
        return None
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("GK external output manifest schema mismatch")
    return payload


def _looks_like_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _validate_provenance_license(
    provenance_url: Any, redistribution_license: Any
) -> tuple[bool, str]:
    license_text = str(redistribution_license).strip().lower()
    if (
        not license_text
        or license_text not in ALLOWED_REDISTRIBUTION_LICENSES
        or any(token in license_text for token in BLOCKED_LICENSE_TOKENS)
    ):
        return False, "non_redistributable_license"
    url = str(provenance_url).strip()
    if not (url.startswith("https://") or url.startswith("http://")):
        return False, "non_public_provenance_url"
    return True, "redistribution_and_provenance_valid"


def _resolve_under(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError(f"path escapes GK external output root: {raw_path}")
    return candidate


def _load_payload(
    path: Path,
    *,
    coordinate_names: Collection[str] | None = None,
    observable_names: Collection[str] | None = None,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != OUTPUT_SCHEMA:
            raise ValueError("GK nonlinear external JSON output schema mismatch")
        coordinates = payload.get("coordinates", {})
        observables = payload.get("observables", {})
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        coordinate_set = {str(name) for name in coordinate_names or ()}
        observable_set = {str(name) for name in observable_names or ()}
        if coordinate_set or observable_set:
            coordinates = {name: arrays[name] for name in coordinate_set if name in arrays}
            observables = {
                name: arrays[name]
                for name in observable_set
                if name in arrays and name not in coordinate_set
            }
            observables.update(
                {name: array for name, array in arrays.items() if name not in coordinate_set}
            )
        else:
            coordinates = arrays
            observables = arrays
    else:
        raise ValueError(f"unsupported GK external output format: {path.suffix}")

    if not isinstance(coordinates, dict) or not isinstance(observables, dict):
        raise ValueError("GK nonlinear external output must define coordinate and observable maps")
    return {
        "coordinates": {
            str(key): np.asarray(value, dtype=np.float64) for key, value in coordinates.items()
        },
        "observables": {
            str(key): np.asarray(value, dtype=np.float64) for key, value in observables.items()
        },
    }


def _validate_coordinates(
    coordinates: dict[str, NDArray[np.float64]], contracts: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for name, raw_contract in contracts.items():
        contract = raw_contract if isinstance(raw_contract, dict) else {}
        if name not in coordinates:
            failures.append({"coordinate": name, "reason": "missing"})
            continue
        array = np.asarray(coordinates[name], dtype=float)
        if array.ndim != 1:
            failures.append({"coordinate": name, "reason": "not_one_dimensional"})
            continue
        if array.size < int(contract.get("min_length", 1)):
            failures.append({"coordinate": name, "reason": "below_min_length"})
            continue
        if not bool(np.all(np.isfinite(array))):
            failures.append({"coordinate": name, "reason": "non_finite"})
            continue
        if bool(contract.get("strictly_increasing", False)) and not bool(
            np.all(np.diff(array) > 0.0)
        ):
            failures.append({"coordinate": name, "reason": "not_strictly_increasing"})
    return not failures, failures


def _validate_observables(
    observables: dict[str, NDArray[np.float64]],
    coordinates: dict[str, NDArray[np.float64]],
    required: list[str],
    contracts: dict[str, Any],
    coordinate_contracts: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for name in required:
        contract = contracts.get(name, {}) if isinstance(contracts.get(name, {}), dict) else {}
        if name not in observables:
            failures.append({"observable": name, "reason": "missing"})
            continue
        array = np.asarray(observables[name], dtype=float)
        axes = contract.get("axes", [])
        if not bool(contract.get("numeric", True)):
            failures.append({"observable": name, "reason": "non_numeric_contract"})
            continue
        if bool(contract.get("non_empty", True)) and array.size == 0:
            failures.append({"observable": name, "reason": "empty"})
            continue
        if bool(contract.get("finite", True)) and not bool(np.all(np.isfinite(array))):
            failures.append({"observable": name, "reason": "non_finite"})
            continue
        if array.ndim < int(contract.get("min_rank", 0)):
            failures.append({"observable": name, "reason": "rank_below_minimum"})
            continue
        if isinstance(axes, list):
            missing_axes = [str(axis) for axis in axes if str(axis) not in coordinate_contracts]
            if missing_axes:
                failures.append(
                    {
                        "observable": name,
                        "reason": "axis_contract_missing",
                        "missing_axes": missing_axes,
                    }
                )
                continue
            if array.ndim < len(axes):
                failures.append({"observable": name, "reason": "rank_below_axis_count"})
                continue
            axis_length_mismatch: list[dict[str, Any]] = []
            for axis_index, axis in enumerate(axes):
                coordinate = coordinates.get(str(axis))
                if coordinate is None:
                    axis_length_mismatch.append(
                        {
                            "axis": str(axis),
                            "reason": "coordinate_payload_missing",
                        }
                    )
                    continue
                if array.shape[axis_index] != coordinate.size:
                    axis_length_mismatch.append(
                        {
                            "axis": str(axis),
                            "coordinate_length": int(coordinate.size),
                            "observable_length": int(array.shape[axis_index]),
                        }
                    )
            if axis_length_mismatch:
                failures.append(
                    {
                        "axis_length_mismatch": axis_length_mismatch,
                        "observable": name,
                        "reason": "axis_length_mismatch",
                    }
                )
    return not failures, failures


def _write_npz(path: Path, payload: dict[str, dict[str, NDArray[np.float64]]]) -> None:
    arrays: dict[str, NDArray[np.float64]] = {}
    arrays.update(payload["coordinates"])
    arrays.update(payload["observables"])
    kwargs: dict[str, Any] = {key: value for key, value in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_evidence(
    rows: Any,
    required_keys: tuple[str, ...],
    required_families: tuple[str, ...],
    expected_case_ids: dict[str, str] | None = None,
) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    seen_families: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in required_keys):
            return False
        family = str(row.get("solver_family", "")).upper()
        if family not in required_families:
            return False
        if expected_case_ids is not None and str(row.get("case_id", "")) != expected_case_ids.get(
            family, ""
        ):
            return False
        seen_families.add(family)
        numeric_values: list[float] = []
        for key, value in row.items():
            if key in {"case_id", "observable", "device", "solver_family"}:
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
                continue
            if isinstance(value, list):
                try:
                    array = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError):
                    return False
                if array.size == 0 or not bool(np.all(np.isfinite(array))):
                    return False
                numeric_values.extend(float(item) for item in array.ravel())
        if not numeric_values or not all(np.isfinite(value) for value in numeric_values):
            return False
    return seen_families == set(required_families)


def _same_deck_group_status(
    cases: dict[str, dict[str, Any]], required_families: tuple[str, ...]
) -> dict[str, Any]:
    """Return fail-closed shared physics/deck identity status for solver rows."""
    missing_families = [family for family in required_families if family not in cases]
    if missing_families:
        return {
            "benchmark_case_id": None,
            "deck_physics_sha256": None,
            "missing_families": missing_families,
            "ready": False,
            "reason": "missing_solver_family_same_deck_rows",
        }

    benchmark_case_ids = {
        str(cases[family].get("benchmark_case_id", "")) for family in required_families
    }
    deck_physics_hashes = {
        str(cases[family].get("deck_physics_sha256", "")) for family in required_families
    }
    if "" in benchmark_case_ids or not all(
        _looks_like_sha256(value) for value in deck_physics_hashes
    ):
        return {
            "benchmark_case_id": None,
            "deck_physics_sha256": None,
            "missing_families": [],
            "ready": False,
            "reason": "missing_or_invalid_same_deck_identity",
        }
    if len(benchmark_case_ids) != 1 or len(deck_physics_hashes) != 1:
        return {
            "benchmark_case_id": sorted(benchmark_case_ids),
            "deck_physics_sha256": sorted(deck_physics_hashes),
            "missing_families": [],
            "ready": False,
            "reason": "same_deck_identity_mismatch",
        }
    return {
        "benchmark_case_id": next(iter(benchmark_case_ids)),
        "deck_physics_sha256": next(iter(deck_physics_hashes)),
        "missing_families": [],
        "ready": True,
        "reason": "same_deck_identity_valid",
    }


def _solver_family_completeness(
    rows: list[dict[str, Any]], required_observables: list[str]
) -> dict[str, Any]:
    """Return per-family observable/comparison completeness without promotion."""
    matrix: list[dict[str, Any]] = []
    for row in rows:
        available_observables = {
            str(observable) for observable in row.get("available_observables", [])
        }
        observable_presence = {
            observable: observable in available_observables for observable in required_observables
        }
        complete_required_observables = bool(observable_presence) and all(
            observable_presence.values()
        )
        native_ready = bool(row.get("native_same_case_comparison_ready", False))
        native_passed = bool(row.get("native_same_case_comparison_passed", False))
        reference_ready = bool(row.get("reference_output_ready", False))
        matrix.append(
            {
                "case_id": row.get("case_id"),
                "complete_required_observables": complete_required_observables,
                "native_same_case_comparison_passed": native_passed,
                "native_same_case_comparison_ready": native_ready,
                "observable_presence": observable_presence,
                "same_deck_reference_output_ready": reference_ready,
                "solver_family": row["solver_family"],
                "status": row["status"],
            }
        )
    ready = bool(matrix) and all(
        bool(row["same_deck_reference_output_ready"])
        and bool(row["complete_required_observables"])
        and bool(row["native_same_case_comparison_ready"])
        and bool(row["native_same_case_comparison_passed"])
        for row in matrix
    )
    return {"ready": ready, "rows": matrix}


def _evidence_family_ready_map(
    rows: Any,
    required_keys: tuple[str, ...],
    expected_case_ids: dict[str, str],
) -> dict[str, bool]:
    """Return per-solver evidence readiness for linked convergence/scaling rows."""
    readiness = {family: False for family in REQUIRED_SOLVER_FAMILIES}
    if not isinstance(rows, list):
        return readiness
    for family in REQUIRED_SOLVER_FAMILIES:
        readiness[family] = _validate_evidence(
            [row for row in rows if isinstance(row, dict) and row.get("solver_family") == family],
            required_keys,
            (family,),
            expected_case_ids,
        )
    return readiness


def _threshold_contract_matrix(reference_case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the published threshold contract in report-friendly row form."""
    threshold_contracts = reference_case["threshold_contracts"]
    thresholds = reference_case["thresholds"]
    matrix: list[dict[str, Any]] = []
    for threshold_name in sorted(thresholds):
        contract = threshold_contracts.get(threshold_name, {})
        matrix.append(
            {
                "comparator": str(contract.get("comparator", "")),
                "limit": float(thresholds[threshold_name]),
                "metric": str(contract.get("metric", "")),
                "observable": str(contract.get("observable", "")),
                "threshold": threshold_name,
            }
        )
    return matrix


def _evidence_package_contract(reference_case: dict[str, Any]) -> dict[str, Any]:
    """Return the enterprise acceptance contract for nonlinear GK parity evidence."""
    return {
        "contract_id": "gk_external_nonlinear_full_fidelity_evidence_package_v1",
        "fail_closed": True,
        "required_artifact_schema": OUTPUT_SCHEMA,
        "required_manifest_schema": MANIFEST_SCHEMA,
        "required_metadata_schema": "gk-external-nonlinear-output-metadata.v1",
        "required_manifest_case_fields": list(REQUIRED_MANIFEST_CASE_FIELDS),
        "required_solver_families": list(REQUIRED_SOLVER_FAMILIES),
        "required_observables": list(reference_case["required_observables"]),
        "required_thresholds": sorted(reference_case["thresholds"]),
        "required_evidence_surfaces": [
            "same_deck_external_outputs",
            "converted_reference_artifacts",
            "converted_metadata",
            "native_same_case_comparison",
            "grid_convergence_evidence",
            "production_scale_scaling_evidence",
        ],
    }


def _evidence_package_matrix(
    rows: list[dict[str, Any]],
    *,
    grid_ready_by_family: dict[str, bool],
    scaling_ready_by_family: dict[str, bool],
) -> list[dict[str, Any]]:
    """Return per-family acceptance rows for the complete parity evidence package."""
    matrix: list[dict[str, Any]] = []
    for row in rows:
        family = str(row["solver_family"])
        provenance_ok, _ = _validate_provenance_license(
            row.get("provenance_url"), row.get("redistribution_license")
        )
        manifest_row_ready = bool(
            row.get("case_id")
            and row.get("deck_id")
            and row.get("benchmark_case_id")
            and _looks_like_sha256(row.get("deck_physics_sha256"))
        )
        converted_artifact_sha256 = str(
            row.get("converted_artifact_sha256") or row.get("sha256") or ""
        )
        converted_metadata_sha256 = str(row.get("converted_metadata_sha256") or "")
        converted_artifact_ready = bool(
            row.get("reference_output_ready") is True
            and row.get("converted_artifact_path")
            and _looks_like_sha256(converted_artifact_sha256)
        )
        converted_metadata_ready = bool(
            row.get("metadata_path") and _looks_like_sha256(converted_metadata_sha256)
        )
        required_observables_ready = bool(row.get("complete_required_observables") is True)
        native_ready = bool(row.get("native_same_case_comparison_ready") is True)
        native_passed = bool(row.get("native_same_case_comparison_passed") is True)
        grid_ready = bool(grid_ready_by_family.get(family, False))
        scaling_ready = bool(scaling_ready_by_family.get(family, False))
        checks = {
            "manifest_row_ready": manifest_row_ready,
            "public_provenance_ready": provenance_ok,
            "redistribution_license_ready": provenance_ok,
            "source_checksum_ready": _looks_like_sha256(row.get("source_output_sha256")),
            "converted_artifact_ready": converted_artifact_ready,
            "converted_metadata_ready": converted_metadata_ready,
            "required_observables_ready": required_observables_ready,
            "native_same_case_comparison_ready": native_ready,
            "native_same_case_thresholds_passed": native_passed,
            "grid_convergence_evidence_ready": grid_ready,
            "production_scale_scaling_evidence_ready": scaling_ready,
        }
        matrix.append(
            {
                **checks,
                "case_id": row.get("case_id"),
                "converted_artifact_sha256": converted_artifact_sha256,
                "converted_metadata_sha256": converted_metadata_sha256,
                "ready": all(checks.values()),
                "solver_family": family,
            }
        )
    return matrix


def _observable_array(
    payload: dict[str, dict[str, NDArray[np.float64]]], observable: str
) -> NDArray[np.float64] | None:
    array = payload["observables"].get(observable)
    if array is None or array.size == 0 or not bool(np.all(np.isfinite(array))):
        return None
    return array


def _metric_value(
    candidate: NDArray[np.float64], reference: NDArray[np.float64], metric: str
) -> float:
    delta = candidate - reference
    if metric == "absolute_error":
        return float(np.max(np.abs(delta)))
    if metric == "relative_error":
        return float(np.max(np.abs(delta)) / max(float(np.max(np.abs(reference))), 1.0e-30))
    if metric == "relative_l2":
        return float(
            np.linalg.norm(delta.ravel()) / max(float(np.linalg.norm(reference.ravel())), 1.0e-30)
        )
    raise ValueError(f"unsupported artifact metric: {metric}")


def _evaluate_thresholds(
    native_payload: dict[str, dict[str, NDArray[np.float64]]],
    reference_payload: dict[str, dict[str, NDArray[np.float64]]],
    thresholds: dict[str, Any],
    threshold_contracts: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for threshold_name, raw_limit in thresholds.items():
        contract = threshold_contracts.get(threshold_name, {})
        check: dict[str, Any] = {
            "passed": False,
            "threshold": str(threshold_name),
            "valid": False,
        }
        if not isinstance(contract, dict):
            check["reason"] = "missing_threshold_contract"
            checks.append(check)
            continue
        try:
            limit = float(raw_limit)
        except (TypeError, ValueError):
            check["reason"] = "invalid_threshold_value"
            checks.append(check)
            continue
        metric = contract.get("metric")
        observable = contract.get("observable")
        comparator = contract.get("comparator")
        check.update(
            {
                "comparator": comparator,
                "limit": limit,
                "metric": metric,
                "observable": observable,
            }
        )
        if comparator not in {"<=", ">="}:
            check["reason"] = "unsupported_comparator"
            checks.append(check)
            continue
        if metric not in {"absolute_error", "relative_error", "relative_l2"}:
            check["reason"] = "unsupported_metric"
            checks.append(check)
            continue
        if not isinstance(observable, str):
            check["reason"] = "missing_observable_contract"
            checks.append(check)
            continue
        native = _observable_array(native_payload, observable)
        reference = _observable_array(reference_payload, observable)
        if native is None:
            check["reason"] = "invalid_native_observable"
            checks.append(check)
            continue
        if reference is None:
            check["reason"] = "invalid_reference_observable"
            checks.append(check)
            continue
        if native.shape != reference.shape:
            check.update(
                {
                    "candidate_shape": list(native.shape),
                    "reason": "observable_shape_mismatch",
                    "reference_shape": list(reference.shape),
                }
            )
            checks.append(check)
            continue
        value = _metric_value(native, reference, str(metric))
        passed = value <= limit if comparator == "<=" else value >= limit
        check.update({"passed": bool(passed), "valid": True, "value": value})
        checks.append(check)
    ready = bool(checks) and all(bool(check["valid"]) for check in checks)
    return {
        "checks": checks,
        "passed": ready and all(bool(check["passed"]) for check in checks),
        "ready": ready,
    }


def _case_map(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("GK external output manifest cases must be a list")
    mapped: dict[str, dict[str, Any]] = {}
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        family = str(raw_case.get("solver_family", "")).upper()
        if family in REQUIRED_SOLVER_FAMILIES and family not in mapped:
            mapped[family] = raw_case
    return mapped


def _convert_case(
    raw_case: dict[str, Any],
    family: str,
    source_root: Path,
    artifact_dir: Path,
    reference_case: dict[str, Any],
    write: bool,
) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_MANIFEST_CASE_FIELDS if not raw_case.get(field)]
    if missing_fields:
        row = _blocked_row(family, "blocked_external_output_manifest_incomplete")
        row["missing_fields"] = missing_fields
        return row
    if not _looks_like_sha256(raw_case["deck_physics_sha256"]):
        row = _blocked_row(family, "blocked_external_output_manifest_incomplete")
        row["missing_fields"] = ["deck_physics_sha256"]
        return row
    provenance_ok, provenance_reason = _validate_provenance_license(
        raw_case["provenance_url"], raw_case["redistribution_license"]
    )
    if not provenance_ok:
        row = _blocked_row(family, "blocked_external_output_provenance_or_license_invalid")
        row["case_id"] = str(raw_case["case_id"])
        row["reason"] = provenance_reason
        return row

    case_id = str(raw_case["case_id"])
    try:
        output_path = _resolve_under(source_root, str(raw_case["output_path"]))
    except ValueError as exc:
        row = _blocked_row(family, "blocked_external_output_path_invalid")
        row["case_id"] = case_id
        row["reason"] = str(exc)
        return row
    if not output_path.exists():
        row = _blocked_row(family, "blocked_external_output_file_missing")
        row["case_id"] = case_id
        return row
    if _sha256(output_path) != str(raw_case["sha256"]):
        row = _blocked_row(family, "blocked_external_output_checksum_mismatch")
        row["case_id"] = case_id
        return row

    required_observables = [str(name) for name in reference_case["required_observables"]]
    coordinate_contracts = reference_case["coordinate_contracts"]
    observable_contracts = reference_case["observable_contracts"]
    try:
        reference_payload = _load_payload(
            output_path,
            coordinate_names=coordinate_contracts,
            observable_names=required_observables,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        row = _blocked_row(family, "blocked_external_output_payload_invalid")
        row["case_id"] = case_id
        row["reason"] = str(exc)
        return row

    coordinates_ok, coordinate_failures = _validate_coordinates(
        reference_payload["coordinates"], coordinate_contracts
    )
    observables_ok, observable_failures = _validate_observables(
        reference_payload["observables"],
        reference_payload["coordinates"],
        required_observables,
        observable_contracts,
        coordinate_contracts,
    )
    if not coordinates_ok or not observables_ok:
        row = _blocked_row(family, "blocked_external_output_contract_invalid")
        row.update(
            {
                "case_id": case_id,
                "coordinate_failures": coordinate_failures,
                "observable_failures": observable_failures,
            }
        )
        return row

    artifact_path = artifact_dir / "gk_nonlinear_external_outputs" / f"{case_id}.npz"
    metadata_path = artifact_dir / "gk_nonlinear_external_outputs" / f"{case_id}.metadata.json"
    if write:
        _write_npz(artifact_path, reference_payload)
    artifact_sha = _sha256(artifact_path) if artifact_path.exists() else ""

    metadata = {
        "accepted_full_fidelity": False,
        "artifact_id": case_id,
        "artifact_path": _public_path(artifact_path, artifact_dir),
        "artifact_role": "same_deck_external_nonlinear_gk_reference_output",
        "available_coordinates": sorted(reference_payload["coordinates"]),
        "available_observables": sorted(reference_payload["observables"]),
        "benchmark_case_id": str(raw_case["benchmark_case_id"]),
        "deck_id": str(raw_case["deck_id"]),
        "deck_physics_sha256": str(raw_case["deck_physics_sha256"]),
        "finite_numeric_payload": True,
        "metadata_schema": "gk-external-nonlinear-output-metadata.v1",
        "missing_required_observables": [],
        "provenance_url": str(raw_case["provenance_url"]),
        "redistribution_license": str(raw_case["redistribution_license"]),
        "reference_family": family,
        "sha256": artifact_sha,
        "solver_output_comparison_ready": False,
        "solver_output_comparison_status": "blocked_missing_native_same_case_output_comparison",
        "source_output_path": _rel(output_path, source_root),
        "source_sha256": str(raw_case["sha256"]),
        "surface": "native_nonlinear_gyrokinetics",
    }

    threshold_evaluation = {"ready": False, "passed": False, "checks": []}
    comparison_ready = False
    comparison_passed = False
    status = "blocked_missing_native_same_case_output_comparison"
    native_path_value = raw_case.get("native_output_path")
    if native_path_value:
        try:
            native_path = _resolve_under(source_root, str(native_path_value))
            native_sha256 = raw_case.get("native_output_sha256")
            if not native_sha256:
                raise ValueError("native_output_sha256_missing")
            if not _looks_like_sha256(native_sha256):
                raise ValueError("native_output_sha256_invalid")
            if _sha256(native_path) != str(native_sha256):
                raise ValueError("native_output_checksum_mismatch")
            native_payload = _load_payload(
                native_path,
                coordinate_names=coordinate_contracts,
                observable_names=required_observables,
            )
            native_coordinates_ok, native_coordinate_failures = _validate_coordinates(
                native_payload["coordinates"], coordinate_contracts
            )
            native_observables_ok, native_observable_failures = _validate_observables(
                native_payload["observables"],
                native_payload["coordinates"],
                required_observables,
                observable_contracts,
                coordinate_contracts,
            )
            if native_coordinates_ok and native_observables_ok:
                threshold_evaluation = _evaluate_thresholds(
                    native_payload,
                    reference_payload,
                    reference_case["thresholds"],
                    reference_case["threshold_contracts"],
                )
                comparison_ready = bool(threshold_evaluation["ready"])
                comparison_passed = bool(threshold_evaluation["passed"])
                status = (
                    "native_same_case_comparison_passed"
                    if comparison_passed
                    else "blocked_native_same_case_comparison_failed"
                )
            else:
                threshold_evaluation = {
                    "ready": False,
                    "passed": False,
                    "checks": [],
                    "coordinate_failures": native_coordinate_failures,
                    "observable_failures": native_observable_failures,
                }
                status = "blocked_native_same_case_output_contract_invalid"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc)
            checksum_status = {
                "native_output_checksum_mismatch": "blocked_native_same_case_output_checksum_mismatch",
                "native_output_sha256_invalid": "blocked_native_same_case_output_checksum_invalid",
                "native_output_sha256_missing": "blocked_native_same_case_output_checksum_missing",
            }
            threshold_evaluation = {
                "ready": False,
                "passed": False,
                "checks": [],
                "reason": reason,
            }
            status = checksum_status.get(
                reason, "blocked_native_same_case_output_missing_or_invalid"
            )

    metadata["solver_output_comparison_ready"] = comparison_ready
    metadata["solver_output_comparison_status"] = status
    if write:
        _write_metadata(metadata_path, metadata)
    metadata_sha = _sha256(metadata_path) if metadata_path.exists() else ""

    missing_requirements = []
    if not comparison_ready:
        missing_requirements.append("native_same_case_solver_output_comparison")
    if not comparison_passed:
        missing_requirements.append("native_same_case_threshold_pass")
    return {
        "available_observables": sorted(reference_payload["observables"]),
        "benchmark_case_id": str(raw_case["benchmark_case_id"]),
        "case_id": case_id,
        "complete_required_observables": True,
        "converted_artifact_sha256": artifact_sha,
        "converted_artifact_path": _public_path(artifact_path, artifact_dir),
        "converted_metadata_sha256": metadata_sha,
        "deck_physics_sha256": str(raw_case["deck_physics_sha256"]),
        "deck_id": str(raw_case["deck_id"]),
        "metadata_path": _public_path(metadata_path, artifact_dir),
        "missing_requirements": missing_requirements,
        "native_same_case_comparison_passed": comparison_passed,
        "native_same_case_comparison_ready": comparison_ready,
        "provenance_url": str(raw_case["provenance_url"]),
        "redistribution_license": str(raw_case["redistribution_license"]),
        "reference_output_ready": True,
        "sha256": artifact_sha,
        "solver_family": family,
        "source_output_sha256": str(raw_case["sha256"]),
        "status": status,
        "threshold_evaluation": threshold_evaluation,
    }


def _report_status(
    manifest: dict[str, Any] | None,
    reference_ready: bool,
    same_deck_ready: bool,
    native_ready: bool,
    grid_ready: bool,
    scaling_ready: bool,
) -> str:
    if manifest is None:
        return "blocked_missing_external_output_manifest"
    if not reference_ready:
        return "blocked_missing_same_deck_external_outputs"
    if not same_deck_ready:
        return "blocked_same_deck_identity_mismatch"
    if not native_ready:
        return "blocked_missing_native_same_case_output_comparison"
    if not grid_ready:
        return "blocked_missing_grid_convergence_evidence"
    if not scaling_ready:
        return "blocked_missing_production_scale_scaling_evidence"
    return "accepted_full_fidelity_ready"


def build_gk_external_output_parity_report(
    *,
    source_root: Path = EXTERNAL_OUTPUT_ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
    report_dir: Path = REPORT_DIR,
    write: bool = True,
) -> dict[str, Any]:
    """Build the strict GENE/CGYRO/GS2 nonlinear output parity report."""
    manifest = _load_manifest(source_root)
    reference_case = _load_reference_case()
    cases = _case_map(manifest)
    same_deck_group = _same_deck_group_status(cases, REQUIRED_SOLVER_FAMILIES)
    rows = [
        _convert_case(cases[family], family, source_root, artifact_dir, reference_case, write)
        if family in cases
        else _blocked_row(
            family,
            "blocked_missing_same_deck_external_output"
            if manifest is not None
            else "blocked_missing_external_output_manifest",
        )
        for family in REQUIRED_SOLVER_FAMILIES
    ]
    required_observables = [str(name) for name in reference_case["required_observables"]]
    completeness = _solver_family_completeness(rows, required_observables)

    grid_rows = manifest.get("grid_convergence_evidence", []) if manifest else []
    scaling_rows = manifest.get("production_scaling_evidence", []) if manifest else []
    expected_case_ids = {
        str(row["solver_family"]): str(row["case_id"])
        for row in rows
        if row.get("reference_output_ready") and row.get("case_id")
    }
    grid_ready = _validate_evidence(
        grid_rows,
        ("case_id", "solver_family", "observable", "coarse_grid", "fine_grid", "relative_l2"),
        REQUIRED_SOLVER_FAMILIES,
        expected_case_ids,
    )
    scaling_ready = _validate_evidence(
        scaling_rows,
        ("case_id", "solver_family", "device", "grid", "ranks", "wall_time_s"),
        REQUIRED_SOLVER_FAMILIES,
        expected_case_ids,
    )
    grid_ready_by_family = _evidence_family_ready_map(
        grid_rows,
        ("case_id", "solver_family", "observable", "coarse_grid", "fine_grid", "relative_l2"),
        expected_case_ids,
    )
    scaling_ready_by_family = _evidence_family_ready_map(
        scaling_rows,
        ("case_id", "solver_family", "device", "grid", "ranks", "wall_time_s"),
        expected_case_ids,
    )
    reference_ready = all(bool(row["reference_output_ready"]) for row in rows)
    same_deck_ready = bool(same_deck_group["ready"])
    native_ready = all(bool(row["native_same_case_comparison_ready"]) for row in rows)
    native_passed = all(bool(row["native_same_case_comparison_passed"]) for row in rows)
    evidence_package_matrix = _evidence_package_matrix(
        rows,
        grid_ready_by_family=grid_ready_by_family,
        scaling_ready_by_family=scaling_ready_by_family,
    )
    evidence_package_ready = same_deck_ready and bool(evidence_package_matrix) and all(
        bool(row["ready"]) for row in evidence_package_matrix
    )
    accepted = (
        reference_ready
        and same_deck_ready
        and native_ready
        and native_passed
        and grid_ready
        and scaling_ready
        and evidence_package_ready
    )
    status = _report_status(
        manifest, reference_ready, same_deck_ready, native_ready, grid_ready, scaling_ready
    )
    if status == "accepted_full_fidelity_ready" and not evidence_package_ready:
        status = "blocked_incomplete_evidence_package"
    missing = []
    if not reference_ready:
        missing.extend(
            [
                "same-deck external nonlinear distribution output for GENE, CGYRO, and GS2",
                "heat_flux_spectra_time_kx_ky_species for all required solver families",
                "field_energy_history_phi_apar_bpar for all required solver families",
                "zonal_flow_and_saturation_metrics for all required solver families",
            ]
        )
    if not same_deck_ready:
        missing.append(
            "shared benchmark_case_id and deck_physics_sha256 across GENE, CGYRO, and GS2"
        )
    if not native_ready:
        missing.append("native same-case nonlinear GK solver-output comparison")
    if not grid_ready:
        missing.append("grid-convergence evidence for converted public nonlinear GK outputs")
    if not scaling_ready:
        missing.append(
            "production-scale scaling evidence for converted public nonlinear GK outputs"
        )
    if not evidence_package_ready:
        missing.append("complete checksum/provenance/threshold evidence package")

    report = {
        "accepted_full_fidelity_ready": accepted,
        "converted_reference_artifacts": sum(bool(row["reference_output_ready"]) for row in rows),
        "description": (
            "Strict fail-closed GENE/CGYRO/GS2 nonlinear GK external-output "
            "conversion and native parity report."
        ),
        "external_output_manifest": _rel(source_root / MANIFEST_NAME),
        "external_output_rows": rows,
        "evidence_package_contract": _evidence_package_contract(reference_case),
        "evidence_package_matrix": evidence_package_matrix,
        "evidence_package_ready": evidence_package_ready,
        "grid_convergence_ready": grid_ready,
        "grid_convergence_ready_by_family": grid_ready_by_family,
        "grid_convergence_rows": grid_rows if isinstance(grid_rows, list) else [],
        "missing_full_fidelity_requirements": missing,
        "native_same_case_comparison_ready": native_ready,
        "production_scale_scaling_ready": scaling_ready,
        "production_scale_scaling_ready_by_family": scaling_ready_by_family,
        "production_scaling_rows": scaling_rows if isinstance(scaling_rows, list) else [],
        "reference_output_ready": reference_ready,
        "required_observables": reference_case["required_observables"],
        "required_solver_families": list(REQUIRED_SOLVER_FAMILIES),
        "schema": "gk-external-nonlinear-output-parity-report.v1",
        "same_deck_group": same_deck_group,
        "same_deck_group_ready": same_deck_ready,
        "solver_family_completeness_matrix": completeness["rows"],
        "solver_family_completeness_ready": completeness["ready"],
        "status": status,
        "threshold_contract_matrix": _threshold_contract_matrix(reference_case),
    }
    if write:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / JSON_REPORT.name).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (report_dir / MD_REPORT.name).write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GK External Nonlinear Output Parity",
        "",
        str(report["description"]),
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted full-fidelity ready: `{report['accepted_full_fidelity_ready']}`",
        f"- Reference output ready: `{report['reference_output_ready']}`",
        f"- Same-deck group ready: `{report['same_deck_group_ready']}`",
        f"- Native same-case comparison ready: `{report['native_same_case_comparison_ready']}`",
        f"- Grid convergence ready: `{report['grid_convergence_ready']}`",
        f"- Production-scale scaling ready: `{report['production_scale_scaling_ready']}`",
        f"- Evidence package ready: `{report['evidence_package_ready']}`",
        f"- Solver-family completeness ready: `{report['solver_family_completeness_ready']}`",
        f"- Converted reference artefacts: `{report['converted_reference_artifacts']}`",
        f"- Same-deck group reason: `{report['same_deck_group']['reason']}`",
        "",
        "## Solver-family rows",
        "",
        "| Solver | Status | Reference output | Native comparison | Missing requirements |",
        "|---|---|:---:|:---:|---|",
    ]
    for row in report["external_output_rows"]:
        missing = ", ".join(row["missing_requirements"]) if row["missing_requirements"] else "-"
        lines.append(
            "| {solver} | `{status}` | `{reference}` | `{native}` | {missing} |".format(
                solver=row["solver_family"],
                status=row["status"],
                reference=row["reference_output_ready"],
                native=row["native_same_case_comparison_ready"],
                missing=missing,
            )
        )
    lines.extend(
        [
            "",
            "## Solver-family completeness matrix",
            "",
            "| Solver | Reference output | Required observables | Native comparison | Native thresholds |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
    )
    for row in report["solver_family_completeness_matrix"]:
        lines.append(
            "| {solver} | `{reference}` | `{observables}` | `{native}` | `{thresholds}` |".format(
                solver=row["solver_family"],
                reference=row["same_deck_reference_output_ready"],
                observables=row["complete_required_observables"],
                native=row["native_same_case_comparison_ready"],
                thresholds=row["native_same_case_comparison_passed"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence package matrix",
            "",
            "| Solver | Manifest | Provenance/license | Artefact | Metadata | Native thresholds | Grid | Scaling | Ready |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for row in report["evidence_package_matrix"]:
        lines.append(
            "| {solver} | `{manifest}` | `{provenance}` | `{artifact}` | `{metadata}` | `{thresholds}` | `{grid}` | `{scaling}` | `{ready}` |".format(
                solver=row["solver_family"],
                manifest=row["manifest_row_ready"],
                provenance=(
                    row["public_provenance_ready"] and row["redistribution_license_ready"]
                ),
                artifact=row["converted_artifact_ready"],
                metadata=row["converted_metadata_ready"],
                thresholds=row["native_same_case_thresholds_passed"],
                grid=row["grid_convergence_evidence_ready"],
                scaling=row["production_scale_scaling_evidence_ready"],
                ready=row["ready"],
            )
        )
    lines.extend(
        [
            "",
            "## Published threshold contract",
            "",
            "| Threshold | Observable | Metric | Comparator | Limit |",
            "|---|---|---|:---:|---:|",
        ]
    )
    for row in report["threshold_contract_matrix"]:
        lines.append(
            "| {threshold} | `{observable}` | `{metric}` | `{comparator}` | {limit:.6g} |".format(
                threshold=row["threshold"],
                observable=row["observable"],
                metric=row["metric"],
                comparator=row["comparator"],
                limit=float(row["limit"]),
            )
        )
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for requirement in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {requirement}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if full fidelity is blocked"
    )
    args = parser.parse_args()
    report = build_gk_external_output_parity_report(write=True)
    if args.check and not report["accepted_full_fidelity_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
