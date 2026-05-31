"""Tests for DREAM public reference execution request reporting."""

from __future__ import annotations

from tools.run_dream_reference_artifact import build_dream_reference_execution_report


def test_dream_reference_execution_report_is_fail_closed_without_backend() -> None:
    report = build_dream_reference_execution_report(write=True)

    assert report["schema"] == "dream-reference-execution-request.v1"
    assert report["source_family"] == "DREAM"
    assert report["settings_deck_generated"] is True
    assert report["settings_deck_sha256"]
    assert report["reference_output_ready"] is False
    assert report["accepted_full_fidelity_ready"] is False
    assert report["required_backend"]["dreami_available"] in {True, False}

    if not report["required_backend"]["dreami_available"]:
        assert report["status"] == "blocked_missing_dream_backend"
        assert "PETSc" in report["next_action"]
