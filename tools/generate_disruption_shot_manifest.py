#!/usr/bin/env python
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
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots_manifest.json"
SHOT_FILE_RE = re.compile(r"^shot_(?P<shot>\d+)_(?P<scenario>[a-z0-9_]+)\.npz$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_scenario(scenario: str) -> str:
    return "safe" if scenario.endswith("_safe") else "disruptive"


def build_manifest(shot_dir: Path) -> dict[str, Any]:
    if not shot_dir.exists():
        raise FileNotFoundError(f"Shot directory not found: {shot_dir}")

    entries: list[dict[str, Any]] = []
    for path in sorted(shot_dir.glob("*.npz")):
        match = SHOT_FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"Unexpected shot filename format: {path.name}")
        shot = int(match.group("shot"))
        scenario = match.group("scenario")
        entries.append(
            {
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
        )

    if not entries:
        raise ValueError(f"No .npz shot files found in {shot_dir}")

    return {
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
        "--check",
        action="store_true",
        help="Fail if manifest is stale instead of writing it.",
    )
    args = parser.parse_args(argv)

    shot_dir = Path(args.shot_dir)
    manifest_path = Path(args.manifest)
    if not shot_dir.is_absolute():
        shot_dir = REPO_ROOT / shot_dir
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    rendered = render_manifest_json(build_manifest(shot_dir))

    if args.check:
        if not manifest_path.exists():
            print(f"Shot manifest missing: {manifest_path}")
            return 1
        existing = manifest_path.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                "Shot manifest is stale. "
                "Run tools/generate_disruption_shot_manifest.py to refresh."
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
