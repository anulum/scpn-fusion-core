# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Honest validated-coverage tool tests (WS-6)
"""Tests for tools/honest_validated_coverage.py (fleet WS-6 coverage measurement)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "honest_validated_coverage.py"
_SPEC = importlib.util.spec_from_file_location("honest_validated_coverage", _TOOL_PATH)
assert _SPEC and _SPEC.loader
hvc = importlib.util.module_from_spec(_SPEC)
# Register before exec so the module's frozen dataclasses can resolve their own __module__.
sys.modules[_SPEC.name] = hvc
_SPEC.loader.exec_module(hvc)


def test_scope_boundary_claim_is_boundary_band() -> None:
    """A scope-boundary declaration counts as boundary, never as a gap or validated."""
    claim = {"id": "readme_native_gk_scope_boundary", "evidence_files": ["README.md"]}
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.BOUNDARY


def test_scope_in_pattern_is_boundary_band() -> None:
    """A claim whose source pattern declares a scope is a boundary band."""
    claim = {"id": "x", "source_pattern": "metric-scoped clarification", "evidence_files": ["a"]}
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.BOUNDARY


def test_administrative_claim_is_producer_asserted() -> None:
    """A security/supported-release claim is administrative, not a physics claim."""
    claim = {"id": "security_current_supported_release", "evidence_files": ["pyproject.toml"]}
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.PRODUCER_ASSERTED


def test_real_shot_evidence_is_reference_validated() -> None:
    """Real-shot validation evidence earns reference-validated."""
    claim = {
        "id": "results_real_shot_overall_pass",
        "evidence_files": ["artifacts/real_shot_validation.json"],
    }
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.REFERENCE_VALIDATED


def test_freegs_parity_is_reference_validated() -> None:
    """External-reference-code (FreeGS) parity earns reference-validated."""
    claim = {
        "id": "results_freegs_equilibrium_benchmark",
        "evidence_files": ["artifacts/freegs_benchmark.json"],
    }
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.REFERENCE_VALIDATED


def test_internal_metric_is_bounded_model() -> None:
    """A surrogate's own test metric is bounded-model, not reference-validated."""
    claim = {
        "id": "readme_qlknn_transport_claim",
        "evidence_files": ["weights/neural_transport_qlknn.metrics.json"],
    }
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.BOUNDED_MODEL


def test_no_evidence_is_validation_gap() -> None:
    """A claim with no backing evidence is a validation gap."""
    claim = {"id": "some_unbacked_claim", "evidence_files": []}
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.VALIDATION_GAP


def test_unrecognised_evidence_floors_to_bounded_model() -> None:
    """Evidence present but neither reference-grade nor a known metric floors to bounded-model."""
    claim = {"id": "novel_claim", "evidence_files": ["docs/SOME_NOTE.md"]}
    result = hvc.classify_claim(claim)
    assert result.band is hvc.CoverageBand.BOUNDED_MODEL
    assert "not reference-grade" in result.rationale


def test_evidence_patterns_are_read_when_files_absent() -> None:
    """A claim using evidence_patterns (not evidence_files) is still classified."""
    claim = {"id": "p", "evidence_patterns": ["artifacts/real_shot_validation.json"]}
    assert hvc.classify_claim(claim).band is hvc.CoverageBand.REFERENCE_VALIDATED


def test_measure_coverage_on_real_manifest_is_conservative() -> None:
    """The real ledger yields a measured, conservative coverage with the gap flagged."""
    result = hvc.measure_coverage()
    assert result["total"] == sum(result["distribution"].values())
    assert 0.0 <= result["honest_validated_coverage"] <= 1.0
    # Conservative floor: reference-validated must not exceed the real-shot/freegs claim count.
    assert result["distribution"].get("reference-validated", 0) >= 1
    assert "ClaimStatus" in result["ledger_gap"]
    assert len(result["per_claim"]) == result["total"]


def test_measure_coverage_on_synthetic_manifest(tmp_path: Path) -> None:
    """A synthetic manifest exercises the full distribution deterministically."""
    manifest = {
        "version": 1,
        "claims": [
            {"id": "a_scope_boundary", "evidence_files": ["README.md"]},
            {"id": "security_current_supported_release", "evidence_files": ["pyproject.toml"]},
            {"id": "ref", "evidence_files": ["artifacts/real_shot_validation.json"]},
            {"id": "metric", "evidence_files": ["x.metrics.json"]},
            {"id": "gap", "evidence_files": []},
        ],
    }
    path = tmp_path / "claims_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = hvc.measure_coverage(path)
    assert result["total"] == 5
    assert result["honest_validated_coverage"] == 1 / 5
    dist = result["distribution"]
    assert dist["boundary"] == 1
    assert dist["producer-asserted"] == 1
    assert dist["reference-validated"] == 1
    assert dist["bounded-model"] == 1
    assert dist["validation-gap"] == 1


def test_measure_coverage_empty_manifest(tmp_path: Path) -> None:
    """An empty claims list yields zero coverage without dividing by zero."""
    path = tmp_path / "claims_manifest.json"
    path.write_text(json.dumps({"version": 1, "claims": []}), encoding="utf-8")
    result = hvc.measure_coverage(path)
    assert result["total"] == 0
    assert result["honest_validated_coverage"] == 0.0


def test_main_prints_coverage_report(capsys) -> None:
    """The CLI prints the band distribution, the coverage percent, and the ledger gap."""
    assert hvc.main() == 0
    out = capsys.readouterr().out
    assert "honest validated-coverage" in out
    assert "reference-validated" in out
    assert "ledger gap" in out
