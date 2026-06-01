"""Tests for public nonlinear GK deck inventory reporting."""

from __future__ import annotations

import json
from pathlib import Path

from tools.inventory_gk_public_reference_decks import build_gk_public_deck_inventory

ROOT = Path(__file__).resolve().parents[1]


def test_gk_public_deck_inventory_is_fail_closed() -> None:
    report = build_gk_public_deck_inventory(write=True)

    assert report["schema"] == "gk-public-reference-deck-inventory-report.v1"
    assert report["status"].startswith("blocked_")
    assert report["accepted_full_fidelity_ready"] is False
    assert report["missing_full_fidelity_requirements"]
    assert report["reference_output_ready"] is False
    assert report["public_output_candidate_ready"] is False
    assert {row["solver_family"] for row in report["public_output_candidate_matrix"]} == {
        "GENE",
        "CGYRO",
        "GS2",
    }
    for row in report["public_output_candidate_matrix"]:
        assert row["ready"] is False
        assert row["same_deck_output_ready"] is False
        assert "nonlinear_distribution_function" in row["required_observables"]
        assert "electromagnetic_apar_energy" in row["required_observables"]
        assert row["required_output_schema"] == "gk-nonlinear-external-output.v1"

    if report["deck_count"] == 0:
        assert report["status"] == "blocked_missing_gk_public_source_cache"
        return

    assert report["deck_count"] >= 2
    assert report["output_summary_count"] >= 1
    assert len(report["sha256"]) == 64

    artifact_path = ROOT / report["artifact_path"]
    metadata_path = ROOT / report["metadata_path"]
    assert artifact_path.exists()
    assert metadata_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["output_acceptance_contract"]["required_solver_families"] == [
        "GENE",
        "CGYRO",
        "GS2",
    ]
    assert artifact["output_acceptance_contract"]["required_manifest_schema"] == (
        "gk-nonlinear-external-output-manifest.v1"
    )
    solvers = {deck["solver_family"] for deck in artifact["decks"]}
    assert {"GS2", "CGYRO"}.issubset(solvers)
    assert any(deck["nonlinear"] for deck in artifact["decks"] if deck["solver_family"] == "GS2")
    assert any(deck["nonlinear"] for deck in artifact["decks"] if deck["solver_family"] == "CGYRO")
    assert all(output["finite_numeric_payload"] for output in artifact["output_summaries"])
