# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Shot Manifest Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_disruption_shot_manifest.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_disruption_shot_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_disruption_shot_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
shot_manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shot_manifest
SPEC.loader.exec_module(shot_manifest)


def test_build_manifest_contains_expected_fields() -> None:
    shot_dir = ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
    manifest = shot_manifest.build_manifest(shot_dir)
    assert manifest["manifest_version"] == "diiid-disruption-shots-v2"
    assert manifest["shot_count"] > 0
    assert len(manifest["shots"]) == manifest["shot_count"]
    assert all(len(item["sha256"]) == 64 for item in manifest["shots"])


def test_repo_manifest_check_passes() -> None:
    rc = shot_manifest.main(["--check"])
    assert rc == 0


def test_manifest_check_detects_stale_output(tmp_path: Path) -> None:
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    (shot_dir / "shot_123456_demo.npz").write_bytes(b"synthetic-content")

    manifest_path = tmp_path / "manifest.json"
    rc_write = shot_manifest.main(["--shot-dir", str(shot_dir), "--manifest", str(manifest_path)])
    assert rc_write == 0

    manifest_path.write_text('{"stale": true}\n', encoding="utf-8")
    rc_check = shot_manifest.main(
        ["--shot-dir", str(shot_dir), "--manifest", str(manifest_path), "--check"]
    )
    assert rc_check == 1


def test_build_manifest_applies_metadata_overrides(tmp_path: Path) -> None:
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    (shot_dir / "shot_123456_raw_hmode.npz").write_bytes(b"raw-content")
    metadata_path = tmp_path / "disruption_shot_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "manifest_overrides": {
                    "data_license": "mixed-v1",
                    "real_data_notice": "raw-enabled",
                },
                "shot_overrides": {
                    "shot_123456_raw_hmode.npz": {
                        "source_type": "raw_diiid_mdsplus_proxy",
                        "generator": "tools/onboard_diiid_raw_disruption_shots.py",
                        "license": "facility-restricted-not-redistributable",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = shot_manifest.build_manifest(shot_dir, metadata_path=metadata_path)
    assert manifest["data_license"] == "mixed-v1"
    assert manifest["real_data_notice"] == "raw-enabled"
    shot = manifest["shots"][0]
    assert shot["source_type"] == "raw_diiid_mdsplus_proxy"
    assert shot["generator"] == "tools/onboard_diiid_raw_disruption_shots.py"
