# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MAST Labelled Shot Download Tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "download_mast_labelled_shots.py"
SPEC = importlib.util.spec_from_file_location("download_mast_labelled_shots", MODULE_PATH)
assert SPEC and SPEC.loader
download_mast = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_mast
SPEC.loader.exec_module(download_mast)


def _valid_manifest() -> dict[str, object]:
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


def test_missing_manifest_writes_blocked_download_report(tmp_path: Path) -> None:
    report_path = tmp_path / "download_report.json"

    rc = download_mast.main(
        [
            "--manifest",
            str(tmp_path / "missing.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_invalid_or_missing_independent_labels"
    assert report["target_count"] == 0
    assert report["downloaded_count"] == 0


def test_valid_manifest_dry_run_targets_only_labelled_shots(tmp_path: Path) -> None:
    manifest_path = tmp_path / "independent_labels.json"
    report_path = tmp_path / "download_report.json"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    rc = download_mast.main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["target_shots"] == [30456, 30420]
    assert report["target_count"] == 2
