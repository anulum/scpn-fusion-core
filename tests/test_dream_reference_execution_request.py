"""Tests for DREAM public reference execution request reporting."""

from __future__ import annotations

import pytest

from tools import run_dream_reference_artifact as dream_reference
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


def test_dream_reference_execution_uses_tracked_deck_evidence_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_repo = dream_reference.ROOT / "data" / "external" / "missing_dream_cache"
    missing_example = missing_repo / "examples" / "2kinetic"
    monkeypatch.setattr(dream_reference, "DREAM_REPO", missing_repo)
    monkeypatch.setattr(dream_reference, "DREAM_EXAMPLE", missing_example)
    monkeypatch.setattr(
        dream_reference,
        "DREAM_SETTINGS",
        missing_example / "dream_settings.h5",
    )
    monkeypatch.setattr(
        dream_reference,
        "DREAM_OUTPUT",
        missing_example / "output.h5",
    )

    report = build_dream_reference_execution_report(write=False)

    assert report["source_cache_available"] is False
    assert report["settings_deck_generated"] is True
    assert report["settings_generation"]["mode"] == "tracked_report_fallback"
    assert report["reference_output_ready"] is False
