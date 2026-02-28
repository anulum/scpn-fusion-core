#!/usr/bin/env python
"""Validate train/val/test split hygiene for DIII-D disruption shots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shot_splits.json"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots_manifest.json"
)
SPLIT_KEYS = ("train", "val", "test")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_SPLIT_IDS_PER_BUCKET = 200_000
_MAX_MANIFEST_SHOTS = 200_000


def _load_json_text(path: Path) -> str:
    size = int(path.stat().st_size)
    if size > _MAX_JSON_BYTES:
        raise ValueError(
            f"{path} exceeds max JSON size "
            f"({_MAX_JSON_BYTES} bytes)."
        )
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(_load_json_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level object.")
    return data


def _parse_split_ids(name: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Split '{name}' must be a non-empty list.")
    if len(value) > _MAX_SPLIT_IDS_PER_BUCKET:
        raise ValueError(
            f"Split '{name}' has {len(value)} ids, exceeding max "
            f"{_MAX_SPLIT_IDS_PER_BUCKET}."
        )
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Split '{name}[{i}]' must be an integer shot id.")
        if item <= 0:
            raise ValueError(f"Split '{name}[{i}]' must be positive.")
        out.append(item)
    return out


def _manifest_shot_ids(manifest: dict[str, Any]) -> set[int]:
    shots = manifest.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Manifest must contain non-empty 'shots' list.")
    if len(shots) > _MAX_MANIFEST_SHOTS:
        raise ValueError(
            "Manifest shot count exceeds max "
            f"{_MAX_MANIFEST_SHOTS}."
        )
    out: set[int] = set()
    for i, item in enumerate(shots):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest shot[{i}] must be an object.")
        shot = item.get("shot")
        if isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0:
            raise ValueError(f"Manifest shot[{i}].shot must be a positive integer.")
        out.add(shot)
    return out


def validate_splits(split_data: dict[str, Any], manifest_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    parsed: dict[str, list[int]] = {}
    try:
        for key in SPLIT_KEYS:
            parsed[key] = _parse_split_ids(key, split_data.get(key))
    except ValueError as exc:
        return [str(exc)]

    for key in SPLIT_KEYS:
        values = parsed[key]
        if len(values) != len(set(values)):
            errors.append(f"Split '{key}' contains duplicate shot ids.")

    for i, left in enumerate(SPLIT_KEYS):
        left_set = set(parsed[left])
        for right in SPLIT_KEYS[i + 1 :]:
            overlap = sorted(left_set & set(parsed[right]))
            if overlap:
                errors.append(
                    f"Split overlap between '{left}' and '{right}': {overlap}"
                )

    try:
        manifest_ids = _manifest_shot_ids(manifest_data)
    except ValueError as exc:
        return [str(exc)]

    split_ids = set(parsed["train"]) | set(parsed["val"]) | set(parsed["test"])
    missing = sorted(manifest_ids - split_ids)
    extra = sorted(split_ids - manifest_ids)
    if missing:
        errors.append(f"Shots missing from splits: {missing}")
    if extra:
        errors.append(f"Unknown shot ids in splits: {extra}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default=str(DEFAULT_SPLITS),
        help="Path to disruption shot split JSON.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to disruption shot manifest JSON.",
    )
    args = parser.parse_args(argv)

    split_path = Path(args.splits)
    manifest_path = Path(args.manifest)
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    errors = validate_splits(
        _load_json(split_path),
        _load_json(manifest_path),
    )
    if errors:
        print(f"Disruption shot split validation FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1

    split_data = _load_json(split_path)
    print(
        "Disruption shot split validation passed: "
        f"train={len(split_data['train'])}, "
        f"val={len(split_data['val'])}, "
        f"test={len(split_data['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
