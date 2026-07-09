#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Validate independent MAST disruption/non-disruption label manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "reference_data" / "mast" / "independent_labels.json"
DEFAULT_REPORT = REPO_ROOT / "validation" / "reports" / "mast_label_readiness.json"
MANIFEST_VERSION = "mast-independent-disruption-labels-v1"
LABELS = {"disruptive", "non_disruptive"}
INDEPENDENT_SOURCE_TYPES = {
    "curated_database",
    "facility_log",
    "operator_log",
    "published_table",
}
DISALLOWED_SOURCE_TYPES = {
    "current_collapse_proxy",
    "neighboring_control_heuristic",
    "model_prediction",
    "manual_guess",
}
PLACEHOLDER_VALUES = {"N/A", "TBD", "TODO", "UNKNOWN"}
PLACEHOLDER_PREFIX = "REPLACE_WITH_"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_SHOTS = 500_000


def _load_json(path: Path) -> dict[str, Any]:
    size = int(path.stat().st_size)
    if size > _MAX_JSON_BYTES:
        raise ValueError(f"{path} exceeds max JSON size ({_MAX_JSON_BYTES} bytes).")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level object.")
    return payload


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _evidence_str(value: Any) -> bool:
    if not _nonempty_str(value):
        return False
    assert isinstance(value, str)
    normalized = value.strip().upper()
    return normalized not in PLACEHOLDER_VALUES and not normalized.startswith(PLACEHOLDER_PREFIX)


def _valid_iso8601(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for an independent MAST label manifest."""
    errors: list[str] = []
    if payload.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION!r}.")
    if not _nonempty_str(payload.get("dataset")):
        errors.append("dataset must be a non-empty string.")
    if not _evidence_str(payload.get("label_authority")):
        errors.append("label_authority must identify the independent source authority.")

    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        errors.append("shots must be a non-empty list.")
        return errors
    if len(shots) > _MAX_SHOTS:
        errors.append(f"shots exceeds max {_MAX_SHOTS}.")
        return errors

    seen: set[int] = set()
    label_counts = {"disruptive": 0, "non_disruptive": 0}
    for index, item in enumerate(shots):
        prefix = f"shots[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        shot_id = item.get("shot_id")
        if not _positive_int(shot_id):
            errors.append(f"{prefix}.shot_id must be a positive integer.")
            continue
        assert isinstance(shot_id, int)
        if shot_id in seen:
            errors.append(f"{prefix}.shot_id duplicates shot {shot_id}.")
        seen.add(shot_id)

        label = item.get("label")
        if label not in LABELS:
            errors.append(f"{prefix}.label must be one of {sorted(LABELS)}.")
        else:
            label_counts[str(label)] += 1

        source_type = item.get("source_type")
        if source_type in DISALLOWED_SOURCE_TYPES:
            errors.append(f"{prefix}.source_type {source_type!r} is not independent evidence.")
        elif source_type not in INDEPENDENT_SOURCE_TYPES:
            errors.append(
                f"{prefix}.source_type must be one of {sorted(INDEPENDENT_SOURCE_TYPES)}."
            )
        if not _evidence_str(item.get("source_reference")):
            errors.append(f"{prefix}.source_reference must cite a table, log, DOI, URL, or file.")
        if not _evidence_str(item.get("labeled_by")):
            errors.append(f"{prefix}.labeled_by must identify the label curator/source.")
        labeled_at = item.get("labeled_at_utc")
        if not _nonempty_str(labeled_at) or not _valid_iso8601(str(labeled_at)):
            errors.append(f"{prefix}.labeled_at_utc must be an ISO-8601 timestamp.")
        if item.get("review_status") != "accepted":
            errors.append(f"{prefix}.review_status must be 'accepted'.")

        if label == "disruptive":
            disruption_time_s = item.get("disruption_time_s")
            if not isinstance(disruption_time_s, (int, float)) or disruption_time_s <= 0.0:
                errors.append(f"{prefix}.disruption_time_s must be positive for disruptive shots.")
        elif "disruption_time_s" in item:
            errors.append(f"{prefix}.disruption_time_s must be absent for non_disruptive shots.")

    if label_counts["disruptive"] == 0:
        errors.append("manifest must include at least one disruptive shot.")
    if label_counts["non_disruptive"] == 0:
        errors.append("manifest must include at least one non_disruptive shot.")
    return errors


def build_report(
    manifest_path: Path, errors: list[str], payload: dict[str, Any] | None
) -> dict[str, Any]:
    """Build a machine-readable readiness report."""
    shots = payload.get("shots", []) if payload else []
    disruptive = sum(
        1 for item in shots if isinstance(item, dict) and item.get("label") == "disruptive"
    )
    non_disruptive = sum(
        1 for item in shots if isinstance(item, dict) and item.get("label") == "non_disruptive"
    )
    return {
        "status": "accepted" if not errors else "blocked_invalid_or_missing_independent_labels",
        "manifest_path": str(manifest_path),
        "manifest_version": payload.get("manifest_version") if payload else None,
        "shot_count": len(shots) if isinstance(shots, list) else 0,
        "disruptive_count": disruptive,
        "non_disruptive_count": non_disruptive,
        "errors": errors,
        "claim_boundary": (
            "MAST disruption detector FPR/AUC claims require independently sourced "
            "disruptive and non_disruptive labels. Current-collapse proxies, neighboring "
            "shot heuristics, and model predictions are not accepted labels."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the configured label manifest and optionally write a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    payload: dict[str, Any] | None = None
    if not manifest_path.exists():
        errors = [f"label manifest not found: {manifest_path}"]
    else:
        try:
            payload = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
        else:
            errors = validate_manifest(payload)

    report = build_report(manifest_path, errors, payload)
    if args.write_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if errors:
        print(f"MAST label manifest validation FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1
    print(
        "MAST label manifest validation passed: "
        f"disruptive={report['disruptive_count']}, "
        f"non_disruptive={report['non_disruptive_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
