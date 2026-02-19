# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claims Audit Tests
# ----------------------------------------------------------------------
"""Tests for tools/claims_audit.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "claims_audit.py"
SPEC = importlib.util.spec_from_file_location("claims_audit", MODULE_PATH)
assert SPEC and SPEC.loader
claims_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims_audit
SPEC.loader.exec_module(claims_audit)


def test_claims_manifest_passes_against_repo() -> None:
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_audit.load_manifest(manifest)
    errors = claims_audit.run_audit(claims, ROOT)
    assert errors == []


def test_claims_audit_reports_missing_evidence_file(tmp_path: Path) -> None:
    manifest = {
        "claims": [
            {
                "id": "missing-evidence",
                "source_file": "README.md",
                "source_pattern": "SCPN Fusion Core",
                "evidence_files": ["does/not/exist.json"],
                "evidence_patterns": [],
            }
        ]
    }
    path = tmp_path / "claims_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    claims = claims_audit.load_manifest(path)
    errors = claims_audit.run_audit(claims, ROOT)
    assert any("evidence file missing" in err for err in errors)


def test_claims_audit_reports_missing_source_pattern(tmp_path: Path) -> None:
    manifest = {
        "claims": [
            {
                "id": "missing-source-pattern",
                "source_file": "README.md",
                "source_pattern": "THIS_PATTERN_SHOULD_NOT_EXIST_123456",
                "evidence_files": [],
                "evidence_patterns": [],
            }
        ]
    }
    path = tmp_path / "claims_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    claims = claims_audit.load_manifest(path)
    errors = claims_audit.run_audit(claims, ROOT)
    assert any("source pattern not found" in err for err in errors)


def test_claims_audit_reports_untracked_evidence_file() -> None:
    manifest = ROOT / "validation" / "claims_manifest.json"
    claims = claims_audit.load_manifest(manifest)
    errors = claims_audit.run_audit(claims, ROOT, tracked_files=set())
    assert any("evidence file not tracked by git" in err for err in errors)
