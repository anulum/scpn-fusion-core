#!/usr/bin/env python
"""Verify SHA256SUMS manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = REPO_ROOT / "dist"
DEFAULT_MANIFEST = DEFAULT_DIST / "SHA256SUMS"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "release_checksums_verify_summary.json"


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


def _parse_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum manifest line: {line!r}")
        digest, file_name = parts
        entries.append((digest.strip(), file_name.strip()))
    if not entries:
        raise ValueError(f"Checksum manifest is empty: {path}")
    return entries


def _iter_dist_file_names(dist_dir: Path, manifest_name: str) -> set[str]:
    return {
        path.name for path in dist_dir.iterdir() if path.is_file() and path.name != manifest_name
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def verify_manifest(*, dist_dir: Path, manifest: Path) -> dict[str, Any]:
    entries = _parse_manifest(manifest)
    failures: list[str] = []
    verified: list[dict[str, Any]] = []

    for declared_digest, file_name in entries:
        artifact = dist_dir / file_name
        if not artifact.exists():
            failures.append(f"missing artifact: {file_name}")
            continue
        actual_digest = _sha256(artifact)
        if actual_digest != declared_digest:
            failures.append(f"digest mismatch: {file_name}")
            continue
        verified.append(
            {
                "file": file_name,
                "sha256": actual_digest,
                "size_bytes": artifact.stat().st_size,
            }
        )

    expected = {name for _, name in entries}
    observed = _iter_dist_file_names(dist_dir, manifest.name)
    unexpected = sorted(observed - expected)
    missing_from_manifest = sorted(expected - observed)
    if unexpected:
        failures.extend(f"untracked artifact: {name}" for name in unexpected)
    if missing_from_manifest:
        failures.extend(f"missing artifact: {name}" for name in missing_from_manifest)

    return {
        "manifest": _display_path(manifest),
        "dist_dir": _display_path(dist_dir),
        "checked_entries": len(entries),
        "verified_entries": len(verified),
        "failures": failures,
        "verified": verified,
        "overall_pass": len(failures) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    dist_dir = _resolve(args.dist_dir)
    manifest = _resolve(args.manifest)
    summary_json = _resolve(args.summary_json)

    summary = verify_manifest(dist_dir=dist_dir, manifest=manifest)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Release checksum verification: "
        f"pass={summary['overall_pass']} failures={len(summary['failures'])}"
    )
    if not bool(summary["overall_pass"]):
        print("Failures:")
        for row in summary["failures"]:
            print(f"- {row}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
