#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate SHA256SUMS manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = REPO_ROOT / "dist"
DEFAULT_OUTPUT = DEFAULT_DIST / "SHA256SUMS"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "release_checksums_summary.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_release_files(dist_dir: Path, manifest_name: str) -> list[Path]:
    files = [
        path for path in sorted(dist_dir.iterdir()) if path.is_file() and path.name != manifest_name
    ]
    return files


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generate_manifest(dist_dir: Path, output_file: Path) -> dict[str, Any]:
    files = _iter_release_files(dist_dir, output_file.name)
    if not files:
        raise ValueError(f"No release artifacts found in {dist_dir}")

    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    for path in files:
        digest = _sha256(path)
        lines.append(f"{digest}  {path.name}")
        entries.append({"file": path.name, "sha256": digest, "size_bytes": path.stat().st_size})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest": _display_path(output_file),
        "dist_dir": _display_path(dist_dir),
        "count": len(entries),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    dist_dir = _resolve(args.dist_dir)
    output_file = _resolve(args.output_file)
    summary_json = _resolve(args.summary_json)
    summary = generate_manifest(dist_dir=dist_dir, output_file=output_file)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Release checksums generated: count={summary['count']} manifest={summary['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
