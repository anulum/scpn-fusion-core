#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Validate the external SAS dataset manifest for full-fidelity parity work.

This benchmark is deliberately a readiness gate, not a parity shortcut. Public
source trees, web snapshots, and locally authorised reference inputs are useful
only as acquisition evidence. Same-deck solver outputs remain blocked until
they are present, licensed for redistribution, checksummed, converted to the
tracked JSON/NPZ artifact contracts, and compared against native output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def _default_dataset_root() -> Path:
    """Return the configured or auto-discovered external dataset root."""
    configured = os.environ.get("SCPN_FUSION_DATASET_ROOT")
    if configured:
        return Path(configured)
    for parent in ROOT.parents:
        candidate = parent / "DATASETS" / "SCPN-FUSION-CORE"
        if candidate.exists():
            return candidate
    return Path("DATASETS") / "SCPN-FUSION-CORE"


DEFAULT_DATASET_ROOT = _default_dataset_root()
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "manifests" / "dataset_manifest.json"
DEFAULT_CHECKSUMS = DEFAULT_DATASET_ROOT / "manifests" / "checksums.json"
DATASET_ROOT_LABEL = "DATASETS/SCPN-FUSION-CORE"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "sas_dataset_readiness.json"
MD_REPORT = REPORT_DIR / "sas_dataset_readiness.md"
REQUIRED_AVAILABLE_LANES = {
    "aurora_bundled_atomic_data",
    "free_boundary",
    "public_git_snapshot",
    "public_web_snapshot",
    "repo_curated_reference_data",
}
REQUIRED_BLOCKED_LANES = {
    "facility_raw_data",
    "free_boundary_coil_current_sidecars",
    "gyrokinetics_em_same_deck_outputs",
    "gyrokinetics_same_deck_outputs",
    "impurity_transport_same_deck_outputs",
    "runaway_same_deck_outputs",
}


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _display_path(path: Path) -> str:
    """Return a public-safe path label for the default external dataset root."""
    try:
        relative = path.resolve().relative_to(DEFAULT_DATASET_ROOT.resolve())
    except ValueError:
        return str(path)
    return str(Path(DATASET_ROOT_LABEL) / relative)


def _as_rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a list of mapping rows from a manifest payload."""
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _missing_lanes(rows: list[dict[str, Any]], required: set[str]) -> list[str]:
    """Return required lanes not represented in rows."""
    present = {str(row.get("lane", "")) for row in rows}
    return sorted(required - present)


def _invalid_available_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return invalid available dataset rows."""
    invalid: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        destination = str(row.get("destination_path", ""))
        if not name:
            invalid.append({"name": name, "reason": "missing_name"})
        if not destination:
            invalid.append({"name": name, "reason": "missing_destination_path"})
        if not status.startswith("available_"):
            invalid.append({"name": name, "reason": "non_available_status"})
        if "licence_or_terms" not in row:
            invalid.append({"name": name, "reason": "missing_licence_or_terms"})
        if "file_count" not in row:
            invalid.append({"name": name, "reason": "missing_file_count"})
    return invalid


def _invalid_blocked_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return invalid blocked dataset rows."""
    invalid: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        if not name:
            invalid.append({"name": name, "reason": "missing_name"})
        if not status.startswith("blocked_"):
            invalid.append({"name": name, "reason": "non_blocked_status"})
        if "notes" not in row:
            invalid.append({"name": name, "reason": "missing_notes"})
    return invalid


def _checksum_count(checksums_path: Path) -> int:
    """Return the number of checksum rows from the sidecar if present."""
    if not checksums_path.exists():
        return 0
    payload = _load_json(checksums_path)
    rows = payload.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def evaluate_manifest(manifest_path: Path, checksums_path: Path | None = None) -> dict[str, Any]:
    """Evaluate a SAS dataset manifest without promoting blocked rows to parity."""
    if checksums_path is None:
        checksums_path = manifest_path.with_name("checksums.json")
    if not manifest_path.exists():
        return {
            "schema": "sas-dataset-readiness-benchmark.v1",
            "manifest_path": _display_path(manifest_path),
            "checksums_path": _display_path(checksums_path),
            "status": "blocked_missing_sas_dataset_manifest",
            "manifest_present": False,
            "available_entries": 0,
            "blocked_entries": 0,
            "checksum_rows": 0,
            "missing_available_lanes": sorted(REQUIRED_AVAILABLE_LANES),
            "missing_blocked_lanes": sorted(REQUIRED_BLOCKED_LANES),
            "invalid_available_rows": [],
            "invalid_blocked_rows": [],
            "external_parity_outputs_ready": False,
            "accepted_full_fidelity_dataset_ready": False,
            "next_required_evidence": sorted(REQUIRED_BLOCKED_LANES),
        }

    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "scpn_fusion_core_sas_dataset_manifest_v1":
        raise ValueError("SAS dataset manifest schema mismatch")
    available = _as_rows(manifest, "available")
    blocked = _as_rows(manifest, "blocked_or_required")
    missing_available = _missing_lanes(available, REQUIRED_AVAILABLE_LANES)
    missing_blocked = _missing_lanes(blocked, REQUIRED_BLOCKED_LANES)
    invalid_available = _invalid_available_rows(available)
    invalid_blocked = _invalid_blocked_rows(blocked)
    checksum_rows = _checksum_count(checksums_path)
    external_parity_ready = not blocked and not missing_blocked
    structurally_ready = (
        bool(available)
        and checksum_rows > 0
        and not missing_available
        and not invalid_available
        and not invalid_blocked
    )
    status = (
        "accepted_full_fidelity_dataset_ready"
        if structurally_ready and external_parity_ready
        else "blocked_missing_required_external_parity_datasets"
        if structurally_ready
        else "blocked_incomplete_sas_dataset_manifest"
    )
    next_required = sorted(
        {
            str(row.get("lane"))
            for row in blocked
            if str(row.get("lane", "")).startswith(
                (
                    "facility_",
                    "free_boundary_",
                    "gyrokinetics_",
                    "impurity_",
                    "runaway_",
                )
            )
        }
        | set(missing_blocked)
    )
    return {
        "schema": "sas-dataset-readiness-benchmark.v1",
        "manifest_path": _display_path(manifest_path),
        "checksums_path": _display_path(checksums_path),
        "dataset_root": DATASET_ROOT_LABEL,
        "status": status,
        "manifest_present": True,
        "available_entries": len(available),
        "blocked_entries": len(blocked),
        "checksum_rows": checksum_rows,
        "missing_available_lanes": missing_available,
        "missing_blocked_lanes": missing_blocked,
        "invalid_available_rows": invalid_available,
        "invalid_blocked_rows": invalid_blocked,
        "external_parity_outputs_ready": external_parity_ready,
        "accepted_full_fidelity_dataset_ready": status == "accepted_full_fidelity_dataset_ready",
        "next_required_evidence": next_required,
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write machine-readable and Markdown dataset readiness reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SAS Dataset Readiness",
        "",
        "This report validates acquisition readiness only. It does not mark source snapshots or partial data as full-fidelity parity.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Manifest present: `{report['manifest_present']}`",
        f"- Manifest path: `{report['manifest_path']}`",
        f"- Checksums path: `{report['checksums_path']}`",
        f"- Available entries: `{report['available_entries']}`",
        f"- Blocked entries: `{report['blocked_entries']}`",
        f"- Checksum rows: `{report['checksum_rows']}`",
        f"- External parity outputs ready: `{report['external_parity_outputs_ready']}`",
        (
            "- Accepted full-fidelity dataset ready: "
            f"`{report['accepted_full_fidelity_dataset_ready']}`"
        ),
        "",
        "## Missing or blocked evidence",
        "",
    ]
    for item in report["next_required_evidence"]:
        lines.append(f"- `{item}`")
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _tracked_report_fallback() -> dict[str, Any] | None:
    """Return the tracked SAS readiness report when the external manifest is absent."""
    if not JSON_REPORT.exists():
        return None
    try:
        report = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if report.get("schema") != "sas-dataset-readiness-benchmark.v1":
        return None
    if report.get("manifest_present") is not True:
        return None
    return dict(report)


def run_benchmark(
    manifest_path: Path = DEFAULT_MANIFEST,
    checksums_path: Path = DEFAULT_CHECKSUMS,
) -> dict[str, Any]:
    """Run the SAS dataset readiness benchmark and persist reports."""
    if not manifest_path.exists():
        fallback = _tracked_report_fallback()
        if fallback is not None:
            write_reports(fallback)
            return fallback
    report = evaluate_manifest(manifest_path, checksums_path)
    write_reports(report)
    return report


def main() -> int:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    args = parser.parse_args()
    report = run_benchmark(args.manifest, args.checksums)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
