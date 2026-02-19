# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Shot Split Tests
# ----------------------------------------------------------------------
"""Tests for tools/check_disruption_shot_splits.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_disruption_shot_splits.py"
SPEC = importlib.util.spec_from_file_location("check_disruption_shot_splits", MODULE_PATH)
assert SPEC and SPEC.loader
split_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = split_check
SPEC.loader.exec_module(split_check)


def test_repo_split_file_passes() -> None:
    rc = split_check.main([])
    assert rc == 0


def test_detects_split_overlap() -> None:
    split_data = {
        "train": [1, 2],
        "val": [2, 3],
        "test": [4],
    }
    manifest = {
        "shots": [
            {"shot": 1},
            {"shot": 2},
            {"shot": 3},
            {"shot": 4},
        ]
    }
    errors = split_check.validate_splits(split_data, manifest)
    assert any("overlap" in err for err in errors)


def test_detects_missing_manifest_ids(tmp_path: Path) -> None:
    split_file = tmp_path / "splits.json"
    manifest_file = tmp_path / "manifest.json"

    split_file.write_text(
        json.dumps({"train": [10], "val": [11], "test": [12]}),
        encoding="utf-8",
    )
    manifest_file.write_text(
        json.dumps({"shots": [{"shot": 10}, {"shot": 11}, {"shot": 12}, {"shot": 13}]}),
        encoding="utf-8",
    )
    rc = split_check.main(
        ["--splits", str(split_file), "--manifest", str(manifest_file)]
    )
    assert rc == 1
