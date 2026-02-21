# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Metadata Synchronization Utility
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Orchestrates project metadata across all manifest and documentation files.
Usage: python tools/sync_metadata.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "src" / "scpn_fusion" / "VERSION"

def get_version():
    return VERSION_FILE.read_text().strip()

def update_file(path, pattern, replacement):
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return
    content = path.read_text(encoding="utf-8")
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
        print(f"Updated {path.relative_to(REPO_ROOT)}")
    else:
        print(f"No change needed for {path.relative_to(REPO_ROOT)}")

def sync():
    version = get_version()
    print(f"--- Synchronizing Metadata to v{version} ---")

    # 1. pyproject.toml
    update_file(
        REPO_ROOT / "pyproject.toml",
        r'^version = ".*?"',
        f'version = "{version}"'
    )

    # 2. setup.py
    update_file(
        REPO_ROOT / "setup.py",
        r'version=".*?"',
        f'version="{version}"'
    )

    # 3. __init__.py
    update_file(
        REPO_ROOT / "src" / "scpn_fusion" / "__init__.py",
        r'__version__ = ".*?"',
        f'__version__ = "{version}"'
    )

    # 4. CITATION.cff
    update_file(
        REPO_ROOT / "CITATION.cff",
        r'^version: ".*?"',
        f'version: "{version}"'
    )

    # 5. README.md badges
    # Matches ![Version](...version-3.8.3...)
    update_file(
        REPO_ROOT / "README.md",
        r'Version-\d+\.\d+\.\d+-brightgreen',
        f'Version-{version}-brightgreen'
    )
    # Matches v3.x.x Breakthrough
    update_file(
        REPO_ROOT / "README.md",
        r'v\d+\.\d+\.\d+ Breakthrough',
        f'v{version} Breakthrough'
    )
    # Matches v3.x.x Performance Breakthrough
    update_file(
        REPO_ROOT / "README.md",
        r'v\d+\.\d+\.\d+ Performance Breakthrough',
        f'v{version} Performance Breakthrough'
    )

    # 6. RESULTS.md / VALIDATION.md headers
    for doc in ["RESULTS.md", "VALIDATION.md"]:
        update_file(
            REPO_ROOT / doc,
            r'\(v\d+\.\d+\.\d+\)',
            f'(v{version})'
        )
        update_file(
            REPO_ROOT / doc,
            r'\*\*Version:\*\* \d+\.\d+\.\d+',
            f'**Version:** {version}'
        )

    # 7. CHANGELOG.md (Update [Unreleased] or top header)
    # This is trickier, but let's at least update the top version if it's there
    update_file(
        REPO_ROOT / "CHANGELOG.md",
        r'## \[\d+\.\d+\.\d+\]',
        f'## [{version}]'
    )

    print("--- Synchronization Complete ---")

if __name__ == "__main__":
    sync()
