"""Tests for public free-boundary machine metadata inventory reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import inventory_free_boundary_public_machine_metadata as machine_inventory
from tools.inventory_free_boundary_public_machine_metadata import (
    build_free_boundary_machine_metadata_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_free_boundary_machine_metadata_inventory_is_fail_closed() -> None:
    report = build_free_boundary_machine_metadata_inventory(write=True)

    assert report["schema"] == "free-boundary-public-machine-metadata-inventory-report.v1"
    assert report["status"].startswith("blocked_")
    assert report["accepted_full_fidelity_ready"] is False
    assert report["reference_output_ready"] is False
    assert report["missing_full_fidelity_requirements"]

    if not report["machine_metadata_ready"]:
        assert report["machine_config_count"] == 0
        assert report["status"] == "blocked_missing_free_boundary_machine_metadata_cache"
        return

    assert report["machine_config_count"] >= 3
    assert {"ITER", "MAST-U", "SPARC"}.issubset(set(report["machines"]))
    assert len(report["sha256"]) == 64

    artifact_path = ROOT / report["artifact_path"]
    metadata_path = ROOT / report["metadata_path"]
    assert artifact_path.exists()
    assert metadata_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    roles = {record["role"] for record in artifact["machine_configs"]}
    assert {"active_coils", "passive_coils", "limiter", "wall"}.issubset(roles)
    assert any(record["probe_counts"] for record in artifact["machine_configs"])
    assert all(record["sha256"] for record in artifact["machine_configs"])
    assert all(
        record["deserialisation_guard"] == "restricted_numpy_builtin_unpickler"
        for record in artifact["machine_configs"]
    )


def test_free_boundary_machine_metadata_preserves_tracked_inventory_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(machine_inventory, "_machine_config_paths", lambda: [])
    monkeypatch.setattr(machine_inventory, "_freegs_example_records", lambda _commit: [])

    report = build_free_boundary_machine_metadata_inventory(write=False)

    assert report["report_generation_mode"] == "tracked_report_fallback"
    assert report["machine_config_count"] > 0
    assert report["machine_metadata_ready"] is True
