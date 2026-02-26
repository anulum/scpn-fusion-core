# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claims Evidence Map Generator Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_claims_evidence_map.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_claims_evidence_map.py"
SPEC = importlib.util.spec_from_file_location("generate_claims_evidence_map", MODULE_PATH)
assert SPEC and SPEC.loader
claims_map = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims_map
SPEC.loader.exec_module(claims_map)


def test_render_markdown_includes_summary_and_claim_ids() -> None:
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_map.load_manifest(manifest)
    rendered = claims_map.render_markdown(claims, manifest_path="validation/claims_manifest.json")
    assert "# Claims Evidence Map" in rendered
    assert "## Summary" in rendered
    assert "readme_pretrained_coverage_claim" in rendered
    assert "results_fno_validated_status" in rendered


def test_check_mode_reports_stale_output(tmp_path: Path) -> None:
    manifest_data = {
        "claims": [
            {
                "id": "claim-a",
                "source_file": "README.md",
                "source_pattern": "SCPN",
                "evidence_files": ["RESULTS.md"],
                "evidence_patterns": [],
            }
        ]
    }
    manifest = tmp_path / "claims_manifest.json"
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    output.write_text("# stale\n", encoding="utf-8")

    rc = claims_map.main(
        ["--manifest", str(manifest), "--output", str(output), "--check"]
    )
    assert rc == 1


def test_check_mode_passes_when_output_matches(tmp_path: Path) -> None:
    manifest_data = {
        "claims": [
            {
                "id": "claim-a",
                "source_file": "README.md",
                "source_pattern": "SCPN",
                "evidence_files": ["RESULTS.md"],
                "evidence_patterns": [],
            }
        ]
    }
    manifest = tmp_path / "claims_manifest.json"
    output = tmp_path / "CLAIMS_EVIDENCE_MAP.md"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    claims = claims_map.load_manifest(manifest)
    rendered = claims_map.render_markdown(claims, manifest_path=manifest.as_posix())
    output.write_text(rendered, encoding="utf-8")

    rc = claims_map.main(
        ["--manifest", str(manifest), "--output", str(output), "--check"]
    )
    assert rc == 0
