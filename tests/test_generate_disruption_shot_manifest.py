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

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_disruption_shot_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_disruption_shot_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
shot_manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shot_manifest
SPEC.loader.exec_module(shot_manifest)


def test_build_manifest_contains_expected_fields() -> None:
    """Build the real repository manifest with hashes and normalized fields."""
    shot_dir = ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
    manifest = shot_manifest.build_manifest(shot_dir)
    assert manifest["manifest_version"] == "diiid-disruption-shots-v2"
    assert manifest["shot_count"] > 0
    assert len(manifest["shots"]) == manifest["shot_count"]
    assert all(len(item["sha256"]) == 64 for item in manifest["shots"])


def test_repo_manifest_check_passes() -> None:
    """Exercise the real tracked-manifest drift check."""
    rc = shot_manifest.main(["--check"])
    assert rc == 0


def test_manifest_check_detects_stale_output(tmp_path: Path) -> None:
    """Reject stale manifest content through the public CLI boundary."""
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
    """Apply admitted manifest and per-shot provenance overrides."""
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


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "must be a JSON object"),
        ({"manifest_overrides": []}, "manifest_overrides must be an object"),
        ({"shot_overrides": []}, "shot_overrides must be an object"),
        ({"manifest_overrides": {"unknown": "value"}}, "Unsupported manifest override"),
        ({"manifest_overrides": {"dataset": 1}}, "must be a non-empty string"),
        ({"manifest_overrides": {"dataset": ""}}, "must be a non-empty string"),
        ({"shot_overrides": {"shot.npz": "invalid"}}, "must be an object"),
        ({"shot_overrides": {"shot.npz": {"unknown": "value"}}}, "Unsupported shot"),
        ({"shot_overrides": {"shot.npz": {"shot": True}}}, "positive integer"),
        ({"shot_overrides": {"shot.npz": {"shot": "1"}}}, "positive integer"),
        ({"shot_overrides": {"shot.npz": {"shot": 0}}}, "positive integer"),
        ({"shot_overrides": {"shot.npz": {"label": 1}}}, "non-empty string"),
        ({"shot_overrides": {"shot.npz": {"label": ""}}}, "non-empty string"),
    ],
)
def test_metadata_override_validation(tmp_path: Path, payload: object, match: str) -> None:
    """Reject malformed metadata containers, keys, and typed values."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        shot_manifest._load_metadata_overrides(metadata_path)


@pytest.mark.parametrize("filename", [1, ""])
def test_metadata_override_rejects_invalid_filename_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: object,
) -> None:
    """Reject non-string or empty shot-override filenames after JSON loading."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        shot_manifest.json,
        "loads",
        lambda _text: {"shot_overrides": {filename: {}}},
    )
    with pytest.raises(ValueError, match="non-empty filenames"):
        shot_manifest._load_metadata_overrides(metadata_path)


def test_manifest_input_and_override_failures(tmp_path: Path) -> None:
    """Reject missing/empty directories, invalid filenames, and stale overrides."""
    with pytest.raises(FileNotFoundError, match="Shot directory not found"):
        shot_manifest.build_manifest(tmp_path / "missing")

    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    with pytest.raises(ValueError, match="No .npz shot files"):
        shot_manifest.build_manifest(shot_dir)

    (shot_dir / "invalid.npz").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Unexpected shot filename"):
        shot_manifest.build_manifest(shot_dir)
    (shot_dir / "invalid.npz").unlink()

    (shot_dir / "shot_123_demo_safe.npz").write_bytes(b"safe")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "shot_overrides": {
                    "shot_123_demo_safe.npz": {"shot": 456},
                    "shot_999_missing.npz": {"label": "safe"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference missing shot files"):
        shot_manifest.build_manifest(shot_dir, metadata_path=metadata_path)

    metadata_path.write_text(
        json.dumps({"shot_overrides": {"shot_123_demo_safe.npz": {"shot": 456}}}),
        encoding="utf-8",
    )
    manifest = shot_manifest.build_manifest(shot_dir, metadata_path=metadata_path)
    assert manifest["shots"][0]["shot"] == 456
    assert manifest["shots"][0]["label"] == "safe"


def test_main_resolves_relative_paths_and_rejects_missing_check_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve all relative CLI paths and fail when check output is absent."""
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    (shot_dir / "shot_123_demo.npz").write_bytes(b"demo")
    monkeypatch.setattr(shot_manifest, "REPO_ROOT", tmp_path)

    assert (
        shot_manifest.main(
            [
                "--shot-dir",
                "shots",
                "--manifest",
                "out/manifest.json",
                "--metadata",
                "missing-metadata.json",
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (
        shot_manifest.main(
            [
                "--shot-dir",
                "shots",
                "--manifest",
                "out/missing.json",
                "--metadata",
                "missing-metadata.json",
                "--check",
            ]
        )
        == 1
    )
