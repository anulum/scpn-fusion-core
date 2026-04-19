#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate and validate DIII-D disruption shot provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOT_DIR = REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
DEFAULT_MANIFEST = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots_manifest.json"
)
DEFAULT_METADATA = (
    REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shot_metadata.json"
)
SHOT_FILE_RE = re.compile(r"^shot_(?P<shot>\d+)_(?P<scenario>[a-z0-9_]+)\.npz$")
_MANIFEST_OVERRIDE_KEYS = {
    "manifest_version",
    "dataset",
    "dataset_root",
    "data_license",
    "real_data_notice",
    "generator_reference",
}
_SHOT_OVERRIDE_KEYS = {
    "shot",
    "scenario",
    "label",
    "source_type",
    "generator",
    "license",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_scenario(scenario: str) -> str:
    return "safe" if scenario.endswith("_safe") else "disruptive"


def _load_metadata_overrides(
    metadata_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not metadata_path.exists():
        return {}, {}

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata overrides must be a JSON object: {metadata_path}")

    manifest_raw = payload.get("manifest_overrides", {})
    shot_raw = payload.get("shot_overrides", {})

    if not isinstance(manifest_raw, dict):
        raise ValueError("metadata.manifest_overrides must be an object")
    if not isinstance(shot_raw, dict):
        raise ValueError("metadata.shot_overrides must be an object")

    manifest_overrides: dict[str, Any] = {}
    for key, value in manifest_raw.items():
        if key not in _MANIFEST_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported manifest override key: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest override '{key}' must be a non-empty string")
        manifest_overrides[key] = value

    shot_overrides: dict[str, dict[str, Any]] = {}
    for filename, override_raw in shot_raw.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("shot_overrides keys must be non-empty filenames")
        if not isinstance(override_raw, dict):
            raise ValueError(f"shot_overrides['{filename}'] must be an object")
        override: dict[str, Any] = {}
        for key, value in override_raw.items():
            if key not in _SHOT_OVERRIDE_KEYS:
                raise ValueError(f"Unsupported shot override key for {filename}: {key}")
            if key == "shot":
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"shot_overrides['{filename}'].shot must be a positive integer"
                    )
                override[key] = int(value)
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"shot_overrides['{filename}'].{key} must be a non-empty string")
            override[key] = value
        shot_overrides[filename] = override

    return manifest_overrides, shot_overrides


def build_manifest(shot_dir: Path, *, metadata_path: Path | None = None) -> dict[str, Any]:
    if not shot_dir.exists():
        raise FileNotFoundError(f"Shot directory not found: {shot_dir}")
    if metadata_path is None:
        metadata_path = shot_dir.parent / "disruption_shot_metadata.json"

    manifest_overrides, shot_overrides = _load_metadata_overrides(metadata_path)

    entries: list[dict[str, Any]] = []
    for path in sorted(shot_dir.glob("*.npz")):
        match = SHOT_FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"Unexpected shot filename format: {path.name}")
        shot = int(match.group("shot"))
        scenario = match.group("scenario")
        entry = {
            "file": path.name,
            "shot": shot,
            "scenario": scenario,
            "label": _classify_scenario(scenario),
            "source_type": "synthetic_diiid_like",
            "generator": "tools/generate_disruption_profiles.py",
            "license": "synthetic-v1",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.name in shot_overrides:
            entry.update(shot_overrides[path.name])
        entries.append(entry)

    if not entries:
        raise ValueError(f"No .npz shot files found in {shot_dir}")

    seen_files = {entry["file"] for entry in entries}
    stale_overrides = sorted(set(shot_overrides.keys()) - seen_files)
    if stale_overrides:
        raise ValueError(
            "Metadata overrides reference missing shot files: " + ", ".join(stale_overrides)
        )

    manifest = {
        "manifest_version": "diiid-disruption-shots-v2",
        "dataset": "diiid_disruption_shots",
        "dataset_root": "validation/reference_data/diiid/disruption_shots",
        "data_license": "synthetic-v1",
        "real_data_notice": (
            "Bundled files are synthetic DIII-D-like profiles. "
            "Real DIII-D data access/reuse follows DOE/GA facility terms."
        ),
        "generator_reference": "tools/generate_disruption_profiles.py",
        "shot_count": len(entries),
        "shots": entries,
    }
    manifest.update(manifest_overrides)
    return manifest


def render_manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shot-dir",
        default=str(DEFAULT_SHOT_DIR),
        help="Directory containing disruption shot .npz files.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Output/input manifest JSON path.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help=(
            "Optional metadata override JSON "
            "(default: validation/reference_data/diiid/disruption_shot_metadata.json)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if manifest is stale instead of writing it.",
    )
    args = parser.parse_args(argv)

    shot_dir = Path(args.shot_dir)
    manifest_path = Path(args.manifest)
    metadata_path = Path(args.metadata)
    if not shot_dir.is_absolute():
        shot_dir = REPO_ROOT / shot_dir
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not metadata_path.is_absolute():
        metadata_path = REPO_ROOT / metadata_path

    rendered = render_manifest_json(build_manifest(shot_dir, metadata_path=metadata_path))

    if args.check:
        if not manifest_path.exists():
            print(f"Shot manifest missing: {manifest_path}")
            return 1
        existing = manifest_path.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                "Shot manifest is stale. Run tools/generate_disruption_shot_manifest.py to refresh."
            )
            return 1
        print(f"Shot manifest is up to date: {manifest_path}")
        return 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote disruption shot manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
