#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — LFS Hygiene Guard
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Guard repository binary hygiene for large model/data artifacts.

Checks:
1. Tracked ``.npz/.npy`` artifacts under model/data directories must resolve to
   ``filter=lfs`` via git attributes.
2. Any tracked binary file above the size threshold must use ``filter=lfs``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
_BINARY_EXTENSIONS = (".npz", ".npy", ".pt", ".pth", ".onnx", ".bin", ".pkl")
_LFS_REQUIRED_GLOBS = (
    "weights/*.npz",
    "weights/*.npy",
    "weights/*.pt",
    "weights/*.pth",
    "weights/*.onnx",
    "validation/reference_data/**/*.npz",
    "validation/reference_data/**/*.npy",
    "training_logs/**/*.npz",
    "training_logs/**/*.npy",
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tracked_files() -> list[str]:
    raw = _git("ls-files", "-z")
    return [item for item in raw.split("\0") if item]


def _matches_required_glob(path: str) -> bool:
    posix = PurePosixPath(path)
    return any(posix.match(pattern) for pattern in _LFS_REQUIRED_GLOBS)


def _check_attr_filter(paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    result = subprocess.run(
        ["git", "check-attr", "filter", "--", *paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        # Format: "<path>: filter: <value>"
        parts = line.split(": ", 2)
        if len(parts) == 3:
            out[parts[0]] = parts[2].strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Git LFS repository hygiene.")
    parser.add_argument(
        "--max-nonlfs-bytes",
        type=int,
        default=5 * 1024 * 1024,
        help="Fail if tracked binary file exceeds this size and is not LFS-managed.",
    )
    args = parser.parse_args(argv)

    gitattributes = REPO_ROOT / ".gitattributes"
    if not gitattributes.exists():
        print("ERROR: .gitattributes missing; LFS policy not defined.", file=sys.stderr)
        return 1

    tracked = _tracked_files()
    required = [p for p in tracked if _matches_required_glob(p)]
    attr_values = _check_attr_filter(required)
    missing_required = [p for p in required if attr_values.get(p) != "lfs"]

    oversize_non_lfs: list[tuple[str, int, str]] = []
    candidate_binary = [p for p in tracked if Path(p).suffix.lower() in _BINARY_EXTENSIONS]
    binary_attr_values = _check_attr_filter(candidate_binary)
    for rel in candidate_binary:
        full = REPO_ROOT / rel
        if not full.exists():
            continue
        size = full.stat().st_size
        if size <= args.max_nonlfs_bytes:
            continue
        attr = binary_attr_values.get(rel, "unspecified")
        if attr != "lfs":
            oversize_non_lfs.append((rel, size, attr))

    if missing_required or oversize_non_lfs:
        if missing_required:
            print(
                "ERROR: Required model/data artifacts are not tracked via Git LFS:",
                file=sys.stderr,
            )
            for rel in missing_required:
                print(f"  - {rel}", file=sys.stderr)
        if oversize_non_lfs:
            threshold_mb = args.max_nonlfs_bytes / (1024 * 1024)
            print(
                (
                    f"ERROR: Oversize tracked binary files (> {threshold_mb:.1f} MiB) "
                    "must use Git LFS:"
                ),
                file=sys.stderr,
            )
            for rel, size, attr in oversize_non_lfs:
                print(
                    f"  - {rel} ({size / (1024 * 1024):.2f} MiB), filter={attr}",
                    file=sys.stderr,
                )
        return 1

    print(
        (
            "LFS hygiene passed: "
            f"required_files={len(required)} "
            f"binary_candidates={len(candidate_binary)} "
            "oversize_non_lfs=0"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
