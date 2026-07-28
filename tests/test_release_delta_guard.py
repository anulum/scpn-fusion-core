# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/release_delta_guard.py."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "release_delta_guard.py"
SPEC = importlib.util.spec_from_file_location("release_delta_guard", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _baseline() -> dict[str, object]:
    return {
        "source_total": 4,
        "source_p0p1": 4,
        "docs_claims_total": 154,
        "docs_claims_p0p1": 15,
        "claims_tracked": 2,
    }


def _readiness_summary(*, source_p0p1: int, docs_p0p1: int) -> dict[str, object]:
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


def _claims_manifest(n: int) -> dict[str, object]:
    return {
        "claims": [
            {"id": f"c{i}", "source_file": "x", "source_pattern": "y", "evidence_files": []}
            for i in range(n)
        ]
    }


def test_main_passes_on_non_regression(tmp_path: Path) -> None:
    """Accept current release metrics that do not regress the pinned baseline."""
    baseline = tmp_path / "baseline.json"
    readiness = tmp_path / "readiness.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(readiness, _readiness_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--readiness-summary",
            str(readiness),
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
    """Reject an unchanged snapshot when positive improvement is required."""
    baseline = tmp_path / "baseline.json"
    readiness = tmp_path / "readiness.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(readiness, _readiness_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--readiness-summary",
            str(readiness),
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
    """Accept a lower source P0/P1 count as a positive release delta."""
    baseline = tmp_path / "baseline.json"
    readiness = tmp_path / "readiness.json"
    claims = tmp_path / "claims.json"
    summary = tmp_path / "summary.json"
    _write_json(baseline, _baseline())
    _write_json(readiness, _readiness_summary(source_p0p1=3, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    rc = mod.main(
        [
            "--baseline",
            str(baseline),
            "--readiness-summary",
            str(readiness),
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


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_baseline", "release delta baseline not found"),
        ("non_object_baseline", "release delta baseline must be a JSON object"),
        ("non_integer_baseline", "baseline[source_total] must be an integer"),
        ("negative_baseline", "baseline[source_total] must be >= 0"),
        ("non_list_snapshots", "readiness summary must contain snapshots list"),
        ("missing_scope", "must include source and docs_claims snapshots"),
        ("non_list_claims", "claims manifest must contain claims list"),
    ],
)
def test_main_rejects_malformed_inputs(tmp_path: Path, case: str, message: str) -> None:
    """Reject malformed release inputs through the production CLI boundary."""
    baseline = tmp_path / "baseline.json"
    readiness = tmp_path / "readiness.json"
    claims = tmp_path / "claims.json"
    _write_json(baseline, _baseline())
    _write_json(readiness, _readiness_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(claims, _claims_manifest(2))

    if case == "missing_baseline":
        baseline = tmp_path / "missing.json"
    elif case == "non_object_baseline":
        baseline.write_text("[]\n", encoding="utf-8")
    elif case == "non_integer_baseline":
        payload = _baseline()
        payload["source_total"] = "4"
        _write_json(baseline, payload)
    elif case == "negative_baseline":
        payload = _baseline()
        payload["source_total"] = -1
        _write_json(baseline, payload)
    elif case == "non_list_snapshots":
        _write_json(readiness, {"snapshots": {}})
    elif case == "missing_scope":
        _write_json(readiness, {"snapshots": [None, {"scope": 3}, {"scope": "source"}]})
    else:
        _write_json(claims, {"claims": {}})

    with pytest.raises((FileNotFoundError, ValueError), match=re.escape(message)):
        mod.main(
            [
                "--baseline",
                str(baseline),
                "--readiness-summary",
                str(readiness),
                "--claims-manifest",
                str(claims),
                "--summary-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_main_resolves_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve every CLI path relative to the canonical repository root."""
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    _write_json(tmp_path / "baseline.json", _baseline())
    _write_json(tmp_path / "readiness.json", _readiness_summary(source_p0p1=4, docs_p0p1=15))
    _write_json(tmp_path / "claims.json", _claims_manifest(2))

    assert (
        mod.main(
            [
                "--baseline",
                "baseline.json",
                "--readiness-summary",
                "readiness.json",
                "--claims-manifest",
                "claims.json",
                "--summary-json",
                "reports/summary.json",
            ]
        )
        == 0
    )
    assert (tmp_path / "reports" / "summary.json").is_file()
