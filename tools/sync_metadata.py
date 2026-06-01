# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Metadata Synchronization Utility
"""
Orchestrates project metadata across all manifest and documentation files.

Usage:
  - Apply updates: ``python tools/sync_metadata.py``
  - Drift check only: ``python tools/sync_metadata.py --check``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "src" / "scpn_fusion" / "VERSION"


def get_version() -> str:
    """Read and return the current package version from ``VERSION`` file.

    Returns
    -------
    str
        Normalized version string from ``src/scpn_fusion/VERSION``.

    Raises
    ------
    OSError
        If the version file cannot be read.
    """
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def update_file(
    path: Path,
    pattern: str,
    replacement: str,
    *,
    check_only: bool,
    count: int = 0,
) -> bool:
    """Update one file by replacing pattern matches with a replacement string.

    Parameters
    ----------
    path:
        Target file path.
    pattern:
        Regular expression pattern accepted by :func:`re.sub`.
    replacement:
        Replacement text to apply.
    check_only:
        When ``True``, do not write updates; only report whether any change is
        required.
    count:
        Max number of matches to replace. ``0`` means replace all.

    Returns
    -------
    bool
        ``True`` if the replacement changed file contents or would change the
        content in check-only mode.
    """
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return False
    content = path.read_text(encoding="utf-8")
    new_content = re.sub(pattern, replacement, content, count=count)
    changed = new_content != content
    rel = path.relative_to(REPO_ROOT)

    if check_only:
        if changed:
            print(f"Out of sync: {rel}")
        else:
            print(f"Up to date: {rel}")
        return changed

    if changed:
        path.write_text(new_content, encoding="utf-8")
        print(f"Updated {rel}")
    else:
        print(f"No change needed for {rel}")
    return changed


def sync(*, check_only: bool = False) -> int:
    """Synchronize or validate version metadata across repository manifests.

    Parameters
    ----------
    check_only:
        Run in dry-run mode and do not write files.

    Returns
    -------
    int
        ``0`` when metadata is fully synchronized, ``1`` when drift is detected
        in check-only mode.
    """
    version = get_version()
    mode = "Checking" if check_only else "Synchronizing"
    print(f"--- {mode} Metadata for v{version} ---")

    changed_count = 0

    # 1. pyproject.toml (single source of truth for Python packaging metadata)
    changed_count += int(
        update_file(
            REPO_ROOT / "pyproject.toml",
            r'(?m)^version = ".*?"',
            f'version = "{version}"',
            check_only=check_only,
        )
    )

    # 2. __init__.py
    changed_count += int(
        update_file(
            REPO_ROOT / "src" / "scpn_fusion" / "__init__.py",
            r'__version__ = ".*?"',
            f'__version__ = "{version}"',
            check_only=check_only,
        )
    )

    # 3. CITATION.cff
    changed_count += int(
        update_file(
            REPO_ROOT / "CITATION.cff",
            r'(?m)^version: ".*?"',
            f'version: "{version}"',
            check_only=check_only,
        )
    )

    # 3b. validation/__init__.py
    changed_count += int(
        update_file(
            REPO_ROOT / "validation" / "__init__.py",
            r'__version__ = ".*?"',
            f'__version__ = "{version}"',
            check_only=check_only,
        )
    )

    # 3c. scpn-fusion-rs/crates/fusion-python/pyproject.toml
    changed_count += int(
        update_file(
            REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-python" / "pyproject.toml",
            r'(?m)^version = ".*?"',
            f'version = "{version}"',
            check_only=check_only,
        )
    )

    # 3d. docs/RELEASE_READINESS.md
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "RELEASE_READINESS.md",
            r"Release Version: `v\d+\.\d+\.\d+`",
            f"Release Version: `v{version}`",
            check_only=check_only,
        )
    )

    # 3e. docs/competitive_analysis.md table rows
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "competitive_analysis.md",
            r"SCPN v\d+\.\d+\.\d+",
            f"SCPN v{version}",
            check_only=check_only,
        )
    )

    # 5. README.md badges and release text
    changed_count += int(
        update_file(
            REPO_ROOT / "README.md",
            r"Version-\d+\.\d+\.\d+-brightgreen",
            f"Version-{version}-brightgreen",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "README.md",
            r"v\d+\.\d+\.\d+ Breakthrough",
            f"v{version} Breakthrough",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "README.md",
            r"v\d+\.\d+\.\d+ Performance Breakthrough",
            f"v{version} Performance Breakthrough",
            check_only=check_only,
        )
    )

    # 6. RESULTS.md / VALIDATION.md headers
    for doc in ["RESULTS.md", "VALIDATION.md"]:
        changed_count += int(
            update_file(
                REPO_ROOT / doc,
                r"\(v\d+\.\d+\.\d+\)",
                f"(v{version})",
                check_only=check_only,
            )
        )
        changed_count += int(
            update_file(
                REPO_ROOT / doc,
                r"\*\*Version:\*\* \d+\.\d+\.\d+",
                f"**Version:** {version}",
                check_only=check_only,
            )
        )

    # 7. docs/ version references
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "competitive_analysis.md",
            r"SCPN Fusion Core v\d+\.\d+\.\d+",
            f"SCPN Fusion Core v{version}",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "STREAMLIT_DEMO_PLAYBOOK.md",
            r"SCPN Fusion Core v\d+\.\d+\.\d+",
            f"SCPN Fusion Core v{version}",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "internal" / "VALIDATION_GATE_MATRIX.md",
            r"As of v\d+\.\d+\.x",
            f"As of v{'.'.join(version.split('.')[:2])}.x",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "README.md",
            r"pending upload for v\d+\.\d+\.\d+ release notes",
            f"pending upload for v{version} release notes",
            check_only=check_only,
        )
    )
    changed_count += int(
        update_file(
            REPO_ROOT / "docs" / "RELEASE_READINESS.md",
            r"v\d+\.\d+\.\d+ Release Acceptance",
            f"v{version} Release Acceptance",
            check_only=check_only,
        )
    )

    # 8. CHANGELOG.md — only update the first version header (latest release)
    changed_count += int(
        update_file(
            REPO_ROOT / "CHANGELOG.md",
            r"## \[\d+\.\d+\.\d+\]",
            f"## [{version}]",
            check_only=check_only,
            count=1,
        )
    )

    if check_only:
        if changed_count:
            print(f"Metadata drift detected in {changed_count} location(s).")
            print("Run: python tools/sync_metadata.py")
            return 1
        print("Metadata is synchronized.")
        return 0

    print("--- Synchronization Complete ---")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for metadata synchronization and drift checks.

    Parameters
    ----------
    argv:
        Optional CLI argument list for testability.

    Returns
    -------
    int
        Exit status from :func:`sync`.
    """
    parser = argparse.ArgumentParser(description="Synchronize repository metadata.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check metadata drift without modifying files.",
    )
    args = parser.parse_args(argv)
    return sync(check_only=bool(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
