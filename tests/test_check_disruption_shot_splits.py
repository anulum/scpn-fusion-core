# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Shot Split Tests
# ----------------------------------------------------------------------
"""Tests for tools/check_disruption_shot_splits.py."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_disruption_shot_splits.py"
SPEC = importlib.util.spec_from_file_location("tools.check_disruption_shot_splits", MODULE_PATH)
assert SPEC and SPEC.loader
split_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = split_check
SPEC.loader.exec_module(split_check)


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload to ``path``."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_split_payload() -> dict[str, list[int]]:
    """Return a minimal valid split payload."""
    return {"train": [10], "val": [11], "test": [12]}


def _valid_manifest_payload(extra: list[int] | None = None) -> dict[str, list[dict[str, int]]]:
    """Return a minimal valid manifest payload."""
    shots = [10, 11, 12, *(extra or [])]
    return {"shots": [{"shot": shot} for shot in shots]}


def _run_with_payloads(
    tmp_path: Path,
    split_payload: object,
    manifest_payload: object,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Write split and manifest payloads, then run the CLI path."""
    split_file = tmp_path / "splits.json"
    manifest_file = tmp_path / "manifest.json"
    _write_json(split_file, split_payload)
    _write_json(manifest_file, manifest_payload)
    return int(split_check.main(["--splits", str(split_file), "--manifest", str(manifest_file)]))


def test_repo_split_file_passes() -> None:
    """The repository disruption split files satisfy the guard."""
    rc = split_check.main([])

    assert rc == 0


def test_detects_split_overlap() -> None:
    """Split validation reports overlaps between buckets."""
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


def test_main_detects_missing_manifest_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main reports manifest shots missing from the split buckets."""
    rc = _run_with_payloads(
        tmp_path,
        _valid_split_payload(),
        _valid_manifest_payload(extra=[13]),
        monkeypatch,
    )

    assert rc == 1


def test_main_detects_unknown_split_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main reports split shots absent from the manifest."""
    rc = _run_with_payloads(
        tmp_path,
        {"train": [10], "val": [11], "test": [99]},
        _valid_manifest_payload(),
        monkeypatch,
    )

    assert rc == 1


def test_main_detects_duplicate_split_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main reports duplicate shot ids inside a split bucket."""
    rc = _run_with_payloads(
        tmp_path,
        {"train": [10, 10], "val": [11], "test": [12]},
        _valid_manifest_payload(),
        monkeypatch,
    )

    assert rc == 1


def test_main_rejects_invalid_split_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main rejects empty, non-list, non-integer, boolean, and non-positive split ids."""
    invalid_splits: list[object] = [
        {"train": [], "val": [11], "test": [12]},
        {"train": "bad", "val": [11], "test": [12]},
        {"train": [True], "val": [11], "test": [12]},
        {"train": ["10"], "val": [11], "test": [12]},
        {"train": [0], "val": [11], "test": [12]},
    ]

    for payload in invalid_splits:
        assert _run_with_payloads(tmp_path, payload, _valid_manifest_payload(), monkeypatch) == 1


def test_main_rejects_oversized_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main rejects split buckets that exceed the configured id-count limit."""
    monkeypatch.setattr(split_check, "_MAX_SPLIT_IDS_PER_BUCKET", 2)
    rc = _run_with_payloads(
        tmp_path,
        {"train": [10, 13, 14], "val": [11], "test": [12]},
        {"shots": [{"shot": 10}, {"shot": 11}, {"shot": 12}, {"shot": 13}, {"shot": 14}]},
        monkeypatch,
    )

    assert rc == 1


def test_main_rejects_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main rejects malformed manifest shapes and shot ids."""
    invalid_manifests: list[object] = [
        {"shots": []},
        {"shots": "bad"},
        {"shots": ["bad"]},
        {"shots": [{"shot": True}]},
        {"shots": [{"shot": "10"}]},
        {"shots": [{"shot": 0}]},
    ]

    for payload in invalid_manifests:
        assert _run_with_payloads(tmp_path, _valid_split_payload(), payload, monkeypatch) == 1


def test_main_rejects_oversized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main rejects manifests that exceed the configured shot-count limit."""
    monkeypatch.setattr(split_check, "_MAX_MANIFEST_SHOTS", 2)
    rc = _run_with_payloads(tmp_path, _valid_split_payload(), _valid_manifest_payload(), monkeypatch)

    assert rc == 1


def test_main_rejects_oversized_json_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main rejects JSON payloads that exceed the configured byte limit."""
    payload_path = tmp_path / "oversized.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(payload_path, _valid_split_payload())
    _write_json(manifest_path, _valid_manifest_payload())
    monkeypatch.setattr(split_check, "_MAX_JSON_BYTES", 1)

    assert split_check.main(["--splits", str(payload_path), "--manifest", str(manifest_path)]) == 1


def test_main_rejects_non_object_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main rejects non-object JSON split payloads."""
    rc = _run_with_payloads(tmp_path, [], _valid_manifest_payload(), monkeypatch)

    assert rc == 1


def test_main_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main resolves relative split and manifest paths from ``REPO_ROOT``."""
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    _write_json(validation_dir / "splits.json", _valid_split_payload())
    _write_json(validation_dir / "manifest.json", _valid_manifest_payload())
    monkeypatch.setattr(split_check, "REPO_ROOT", tmp_path)

    rc = split_check.main(["--splits", "validation/splits.json", "--manifest", "validation/manifest.json"])

    assert rc == 0


def test_main_reports_missing_input_files(tmp_path: Path) -> None:
    """Main raises a clear error when an input file is missing."""
    split_file = tmp_path / "missing_splits.json"
    manifest_file = tmp_path / "manifest.json"
    _write_json(manifest_file, _valid_manifest_payload())

    with pytest.raises(FileNotFoundError, match="Split file not found"):
        split_check.main(["--splits", str(split_file), "--manifest", str(manifest_file)])

    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        split_check.main(["--splits", str(manifest_file), "--manifest", str(tmp_path / "missing.json")])


def test_script_entrypoint_exits_with_main_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executable entrypoint delegates through ``main`` and exits with its code."""
    split_file = tmp_path / "splits.json"
    manifest_file = tmp_path / "manifest.json"
    _write_json(split_file, _valid_split_payload())
    _write_json(manifest_file, _valid_manifest_payload())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--splits", str(split_file), "--manifest", str(manifest_file)],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
