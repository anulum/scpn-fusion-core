# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Label Manifest Tests

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_mast_label_manifest.py"
SPEC = importlib.util.spec_from_file_location("tools.check_mast_label_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
manifest_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_check
SPEC.loader.exec_module(manifest_check)


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_manifest() -> dict[str, Any]:
    """Return a minimal independent MAST label manifest."""
    return {
        "manifest_version": "mast-independent-disruption-labels-v1",
        "dataset": "FAIR-MAST Level-2 bounded disruption panel",
        "label_authority": "facility log export",
        "shots": [
            {
                "shot_id": 30456,
                "label": "disruptive",
                "disruption_time_s": 0.2218,
                "source_type": "facility_log",
                "source_reference": "facility-log://mast/30456",
                "labeled_by": "UKAEA facility log",
                "labeled_at_utc": "2026-06-16T00:00:00Z",
                "review_status": "accepted",
            },
            {
                "shot_id": 30420,
                "label": "non_disruptive",
                "source_type": "published_table",
                "source_reference": "doi:10.example/mast-controls",
                "labeled_by": "published table",
                "labeled_at_utc": "2026-06-16T00:00:00Z",
                "review_status": "accepted",
            },
        ],
    }


def test_valid_independent_manifest_passes() -> None:
    """Valid independent labels pass manifest validation."""
    assert manifest_check.validate_manifest(_valid_manifest()) == []


def test_rejects_current_collapse_proxy_as_independent_label() -> None:
    """Current-collapse proxies are rejected as independent labels."""
    payload = _valid_manifest()
    shots = payload["shots"]
    assert isinstance(shots, list)
    shot = dict(shots[0])
    shot["source_type"] = "current_collapse_proxy"
    shots[0] = shot

    errors = manifest_check.validate_manifest(payload)

    assert any("not independent evidence" in error for error in errors)


def test_requires_both_disruptive_and_non_disruptive_labels() -> None:
    """A manifest must include disruptive and non-disruptive labels."""
    payload = _valid_manifest()
    shots = payload["shots"]
    assert isinstance(shots, list)
    payload["shots"] = [shots[0]]

    errors = manifest_check.validate_manifest(payload)

    assert any("non_disruptive" in error for error in errors)


def test_requires_at_least_one_disruptive_label() -> None:
    """A manifest must include at least one disruptive label."""
    payload = _valid_manifest()
    shots = payload["shots"]
    assert isinstance(shots, list)
    payload["shots"] = [shots[1]]

    errors = manifest_check.validate_manifest(payload)

    assert any("disruptive shot" in error for error in errors)


def test_rejects_invalid_top_level_manifest_fields() -> None:
    """Top-level manifest identity fields and shot containers are required."""
    errors = manifest_check.validate_manifest(
        {
            "manifest_version": "wrong",
            "dataset": " ",
            "label_authority": "",
            "shots": [],
        }
    )

    assert "manifest_version must be 'mast-independent-disruption-labels-v1'." in errors
    assert "dataset must be a non-empty string." in errors
    assert "label_authority must identify the independent source authority." in errors
    assert "shots must be a non-empty list." in errors


def test_rejects_template_placeholder_evidence_fields() -> None:
    """Unreplaced template placeholders are not independent evidence."""
    payload = _valid_manifest()
    payload["label_authority"] = "REPLACE_WITH_INDEPENDENT_SOURCE_AUTHORITY"
    shots = payload["shots"]
    assert isinstance(shots, list)
    first = dict(shots[0])
    first["source_reference"] = "REPLACE_WITH_FACILITY_LOG_TABLE_DOI_URL_OR_FILE"
    first["labeled_by"] = "TBD"
    shots[0] = first

    errors = manifest_check.validate_manifest(payload)

    expected_fragments = [
        "label_authority must identify",
        "shots[0].source_reference must cite",
        "shots[0].labeled_by must identify",
    ]
    for fragment in expected_fragments:
        assert any(fragment in error for error in errors), fragment


def test_rejects_oversized_shot_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shot lists larger than the configured guardrail are rejected."""
    monkeypatch.setattr(manifest_check, "_MAX_SHOTS", 1)

    errors = manifest_check.validate_manifest(_valid_manifest())

    assert errors == ["shots exceeds max 1."]


def test_rejects_malformed_shot_records() -> None:
    """Malformed shot rows report all independent-label contract failures."""
    payload = _valid_manifest()
    payload["shots"] = [
        "not-an-object",
        {"shot_id": True},
        {
            "shot_id": 30456,
            "label": "disruptive",
            "disruption_time_s": 0.0,
            "source_type": "facility_log",
            "source_reference": "facility-log://mast/30456",
            "labeled_by": "UKAEA facility log",
            "labeled_at_utc": "2026-06-16T00:00:00Z",
            "review_status": "accepted",
        },
        {
            "shot_id": 30456,
            "label": "unknown",
            "source_type": "unreviewed_spreadsheet",
            "source_reference": "",
            "labeled_by": "",
            "labeled_at_utc": "not-a-date",
            "review_status": "draft",
        },
        {
            "shot_id": 30457,
            "label": "non_disruptive",
            "disruption_time_s": 0.2,
            "source_type": "operator_log",
            "source_reference": "operator-log://mast/30457",
            "labeled_by": "operator log",
            "labeled_at_utc": "2026-06-16T00:00:00Z",
            "review_status": "accepted",
        },
    ]

    errors = manifest_check.validate_manifest(payload)

    expected_fragments = [
        "shots[0] must be an object.",
        "shots[1].shot_id must be a positive integer.",
        "shots[2].disruption_time_s must be positive for disruptive shots.",
        "shots[3].shot_id duplicates shot 30456.",
        "shots[3].label must be one of",
        "shots[3].source_type must be one of",
        "shots[3].source_reference must cite",
        "shots[3].labeled_by must identify",
        "shots[3].labeled_at_utc must be an ISO-8601 timestamp.",
        "shots[3].review_status must be 'accepted'.",
        "shots[4].disruption_time_s must be absent for non_disruptive shots.",
    ]
    for fragment in expected_fragments:
        assert any(fragment in error for error in errors), fragment


def test_build_report_counts_only_list_payloads() -> None:
    """Report building handles invalid non-list shot payloads."""
    report = manifest_check.build_report(
        Path("mast.json"),
        ["bad shots"],
        {
            "manifest_version": "mast-independent-disruption-labels-v1",
            "shots": "not-a-list",
        },
    )

    assert report["shot_count"] == 0
    assert report["disruptive_count"] == 0
    assert report["non_disruptive_count"] == 0


def test_load_json_rejects_non_object_and_oversized_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON loading rejects non-object payloads and oversized files."""
    non_object = tmp_path / "labels.json"
    _write_json(non_object, [])

    with pytest.raises(ValueError, match="top-level object"):
        manifest_check._load_json(non_object)

    oversized = tmp_path / "oversized.json"
    oversized.write_text('{"manifest_version":"x"}', encoding="utf-8")
    monkeypatch.setattr(manifest_check, "_MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds max JSON size"):
        manifest_check._load_json(oversized)


def test_cli_writes_accepted_report_for_valid_manifest(tmp_path: Path) -> None:
    """The CLI writes an accepted report for a valid manifest."""
    manifest_path = tmp_path / "labels.json"
    report_path = tmp_path / "readiness.json"
    _write_json(manifest_path, _valid_manifest())

    rc = manifest_check.main(
        [
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--write-report",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "accepted"
    assert report["disruptive_count"] == 1
    assert report["non_disruptive_count"] == 1


def test_cli_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI resolves manifest and report paths relative to ``REPO_ROOT``."""
    data_dir = tmp_path / "validation" / "reference_data" / "mast"
    data_dir.mkdir(parents=True)
    _write_json(data_dir / "labels.json", _valid_manifest())
    monkeypatch.setattr(manifest_check, "REPO_ROOT", tmp_path)

    rc = manifest_check.main(
        [
            "--manifest",
            "validation/reference_data/mast/labels.json",
            "--report",
            "validation/reports/mast_label_readiness.json",
            "--write-report",
        ]
    )

    assert rc == 0
    assert (tmp_path / "validation" / "reports" / "mast_label_readiness.json").exists()


def test_cli_writes_blocked_report_when_manifest_missing(tmp_path: Path) -> None:
    """The CLI writes a blocked report when the manifest is missing."""
    report_path = tmp_path / "readiness.json"
    rc = manifest_check.main(
        [
            "--manifest",
            str(tmp_path / "missing.json"),
            "--report",
            str(report_path),
            "--write-report",
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_invalid_or_missing_independent_labels"


def test_cli_writes_blocked_report_for_non_object_json(tmp_path: Path) -> None:
    """The CLI writes a blocked report when JSON is not an object."""
    manifest_path = tmp_path / "labels.json"
    report_path = tmp_path / "readiness.json"
    _write_json(manifest_path, [])

    rc = manifest_check.main(
        [
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--write-report",
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_invalid_or_missing_independent_labels"
    assert "top-level object" in report["errors"][0]


def test_script_entrypoint_delegates_to_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the script entrypoint delegates to ``main``."""
    manifest_path = tmp_path / "labels.json"
    _write_json(manifest_path, _valid_manifest())
    monkeypatch.setattr(sys, "argv", ["check_mast_label_manifest.py", "--manifest", str(manifest_path)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert excinfo.value.code == 0
