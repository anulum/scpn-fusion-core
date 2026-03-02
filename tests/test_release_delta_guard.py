"""Tests for tools/release_delta_guard.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "release_delta_guard.py"
SPEC = importlib.util.spec_from_file_location("release_delta_guard", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _baseline() -> dict:
    return {
        "source_total": 4,
        "source_p0p1": 4,
        "docs_claims_total": 154,
        "docs_claims_p0p1": 15,
        "claims_tracked": 2,
    }


def _underdev_summary(*, source_p0p1: int, docs_p0p1: int) -> dict:
    return {
        "snapshots": [
            {
                "scope": "source",
                "total_entries": 4,
                "p0_p1_entries": source_p0p1,
                "marker_counts": {"MONOLITH": 4},
            },
            {
                "scope": "docs_claims",
                "total_entries": 154,
                "p0_p1_entries": docs_p0p1,
                "marker_counts": {"FALLBACK": 80},
            },
        ]
    }


def _claims_manifest(n: int) -> dict:
    return {"claims": [{"id": f"c{i}", "source_file": "x", "source_pattern": "y", "evidence_files": []} for i in range(n)]}


def test_main_passes_on_non_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    underdev = tmp_path / "underdev.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(underdev, _underdev_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--underdeveloped-summary",
            str(underdev),
            "--claims-manifest",
            str(claims),
            "--summary-json",
            str(summary),
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True


def test_main_requires_positive_delta_when_requested(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    underdev = tmp_path / "underdev.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(underdev, _underdev_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--underdeveloped-summary",
            str(underdev),
            "--claims-manifest",
            str(claims),
            "--summary-json",
            str(summary),
            "--require-positive-delta",
        ]
    )
    assert rc == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is False


def test_main_positive_delta_passes_when_reduction_exists(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    underdev = tmp_path / "underdev.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(underdev, _underdev_summary(source_p0p1=3, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--underdeveloped-summary",
            str(underdev),
            "--claims-manifest",
            str(claims),
            "--summary-json",
            str(summary),
            "--require-positive-delta",
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
