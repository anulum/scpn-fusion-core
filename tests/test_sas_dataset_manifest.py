"""Tests for SAS dataset readiness gating."""

from __future__ import annotations

import json
from pathlib import Path

from validation.benchmark_sas_dataset_manifest import evaluate_manifest


def _write_manifest(path: Path, *, blocked: list[dict[str, object]]) -> Path:
    payload = {
        "schema": "scpn_fusion_core_sas_dataset_manifest_v1",
        "dataset_root": str(path.parent.parent),
        "available": [
            {
                "destination_path": str(path.parent.parent / "reference_data"),
                "file_count": 1,
                "lane": "repo_curated_reference_data",
                "licence_or_terms": "repository policy",
                "name": "reference_data",
                "status": "available_local_hardlink_snapshot",
            },
            {
                "destination_path": str(path.parent.parent / "source"),
                "file_count": 1,
                "lane": "public_git_snapshot",
                "licence_or_terms": "upstream licence",
                "name": "source",
                "status": "available_git_snapshot",
            },
            {
                "destination_path": str(path.parent.parent / "web"),
                "file_count": 1,
                "lane": "public_web_snapshot",
                "licence_or_terms": "upstream documentation terms",
                "name": "web",
                "status": "available_local_hardlink_snapshot",
            },
            {
                "destination_path": str(path.parent.parent / "free_boundary"),
                "file_count": 1,
                "lane": "free_boundary",
                "licence_or_terms": "source manifest",
                "name": "free_boundary",
                "status": "available_local_hardlink_snapshot",
            },
            {
                "destination_path": str(path.parent.parent / "adas"),
                "file_count": 1,
                "lane": "aurora_bundled_atomic_data",
                "licence_or_terms": "Aurora bundled terms",
                "name": "adas",
                "status": "available_local_hardlink_snapshot",
            },
        ],
        "blocked_or_required": blocked,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    checksums = path.with_name("checksums.json")
    checksums.write_text(
        json.dumps({"rows": [{"path": "reference_data/example.json", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )
    return checksums


def test_sas_dataset_manifest_blocks_missing_external_parity_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "dataset_manifest.json"
    manifest.parent.mkdir()
    blocked: list[dict[str, object]] = [
        {"lane": lane, "name": lane, "notes": "missing same-case output", "status": "blocked_external_execution_required"}
        for lane in [
            "facility_raw_data",
            "free_boundary_coil_current_sidecars",
            "gyrokinetics_em_same_deck_outputs",
            "gyrokinetics_same_deck_outputs",
            "impurity_transport_same_deck_outputs",
            "runaway_same_deck_outputs",
        ]
    ]
    checksums = _write_manifest(manifest, blocked=blocked)

    report = evaluate_manifest(manifest, checksums)

    assert report["schema"] == "sas-dataset-readiness-benchmark.v1"
    assert report["status"] == "blocked_missing_required_external_parity_datasets"
    assert report["manifest_present"] is True
    assert report["available_entries"] == 5
    assert report["blocked_entries"] == 6
    assert report["checksum_rows"] == 1
    assert report["external_parity_outputs_ready"] is False
    assert report["accepted_full_fidelity_dataset_ready"] is False
    assert report["missing_available_lanes"] == []
    assert report["missing_blocked_lanes"] == []
    assert "gyrokinetics_same_deck_outputs" in report["next_required_evidence"]


def test_sas_dataset_manifest_fails_closed_when_manifest_is_missing(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "manifests" / "dataset_manifest.json"

    report = evaluate_manifest(missing_manifest, missing_manifest.with_name("checksums.json"))

    assert report["status"] == "blocked_missing_sas_dataset_manifest"
    assert report["manifest_present"] is False
    assert report["accepted_full_fidelity_dataset_ready"] is False
    assert report["missing_available_lanes"]
    assert report["missing_blocked_lanes"]
