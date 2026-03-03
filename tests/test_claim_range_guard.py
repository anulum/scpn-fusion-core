# ----------------------------------------------------------------------
# SCPN Fusion Core -- Claim Range Guard Tests
# ----------------------------------------------------------------------
"""Tests for tools/claim_range_guard.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "claim_range_guard.py"
SPEC = importlib.util.spec_from_file_location("claim_range_guard", MODULE_PATH)
assert SPEC and SPEC.loader
claim_range_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claim_range_guard
SPEC.loader.exec_module(claim_range_guard)


def test_claim_range_guard_passes_with_repo_config() -> None:
    config = ROOT / "validation" / "claim_range_thresholds.json"
    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=ROOT)
    assert errors == []
    assert summary["failed_checks"] == 0


def test_claim_range_guard_reports_threshold_violation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"metric": 5.0}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "id": "max-check",
                        "file": artifact.name,
                        "path": ["metric"],
                        "max": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert len(errors) == 1
    assert "expected <= 1.0" in errors[0]
    assert summary["failed_checks"] == 1


def test_claim_range_guard_reports_missing_path(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"present": 1.0}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "id": "missing-path",
                        "file": artifact.name,
                        "path": ["missing", "field"],
                        "min": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checks = claim_range_guard.load_checks(config)
    errors, summary = claim_range_guard.run_checks(checks, repo_root=tmp_path)
    assert len(errors) == 1
    assert "missing key" in errors[0]
    assert summary["failed_checks"] == 1
