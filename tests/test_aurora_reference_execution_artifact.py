"""Tests for Aurora/Open-ADAS public reference artifact generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.run_aurora_reference_artifact import build_aurora_reference_execution_report

ROOT = Path(__file__).resolve().parents[1]


def test_aurora_reference_execution_artifact_is_fail_closed() -> None:
    report = build_aurora_reference_execution_report(write=True)

    assert report["schema"] == "aurora-reference-execution-artifact.v1"
    assert report["source_family"] == "Aurora"
    assert report["accepted_full_fidelity_ready"] is False
    assert report["status"].startswith("blocked_")
    assert report["missing_full_fidelity_requirements"]

    if not report["artifact_generated"]:
        assert report["reference_output_ready"] is False
        return

    assert report["reference_output_ready"] is True
    artifact = report["artifact"]
    artifact_path = ROOT / artifact["artifact_path"]
    metadata_path = ROOT / artifact["metadata_path"]
    assert artifact_path.exists()
    assert metadata_path.exists()
    assert len(artifact["sha256"]) == 64
    assert artifact["finite_numeric_payload"] is True
    assert artifact["solver_output_comparison_ready"] is False

    with np.load(artifact_path, allow_pickle=False) as payload:
        assert payload["fz_no_cx"].shape == (64, 19)
        assert payload["fz_with_cx"].shape == (64, 19)
        np.testing.assert_allclose(payload["fraction_sum_no_cx"], 1.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(payload["fraction_sum_with_cx"], 1.0, rtol=0.0, atol=1e-12)
        assert np.all(np.isfinite(payload["Te_eV"]))
        assert np.all(np.isfinite(payload["ne_cm3"]))
