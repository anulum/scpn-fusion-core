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
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_mast_label_manifest.py"
SPEC = importlib.util.spec_from_file_location("check_mast_label_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
manifest_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_check
SPEC.loader.exec_module(manifest_check)


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


def test_valid_independent_manifest_passes() -> None:
    assert manifest_check.validate_manifest(_valid_manifest()) == []


def test_rejects_current_collapse_proxy_as_independent_label() -> None:
    payload = _valid_manifest()
    shots = payload["shots"]
    assert isinstance(shots, list)
    shot = dict(shots[0])
    shot["source_type"] = "current_collapse_proxy"
    shots[0] = shot

    errors = manifest_check.validate_manifest(payload)

    assert any("not independent evidence" in error for error in errors)


def test_requires_both_disruptive_and_non_disruptive_labels() -> None:
    payload = _valid_manifest()
    payload["shots"] = [payload["shots"][0]]  # type: ignore[index]

    errors = manifest_check.validate_manifest(payload)

    assert any("non_disruptive" in error for error in errors)


def test_cli_writes_blocked_report_when_manifest_missing(tmp_path: Path) -> None:
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
