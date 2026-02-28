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

    # 1. pyproject.toml (single source of truth for Python packaging metadata)
    update_file(
        REPO_ROOT / "pyproject.toml",
        r'(?m)^version = ".*?"',
        f'version = "{version}"'
    )

    # 2. __init__.py
    update_file(
        REPO_ROOT / "src" / "scpn_fusion" / "__init__.py",
        r'__version__ = ".*?"',
        f'__version__ = "{version}"'
    )

    # 3. CITATION.cff
    update_file(
        REPO_ROOT / "CITATION.cff",
        r'(?m)^version: ".*?"',
        f'version: "{version}"'
    )

    # 3b. validation/__init__.py
    update_file(
        REPO_ROOT / "validation" / "__init__.py",
        r'__version__ = ".*?"',
        f'__version__ = "{version}"'
    )

    # 3c. scpn-fusion-rs/crates/fusion-python/pyproject.toml
    update_file(
        REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-python" / "pyproject.toml",
        r'(?m)^version = ".*?"',
        f'version = "{version}"'
    )

    # 3d. docs/RELEASE_ACCEPTANCE_CHECKLIST.md
    update_file(
        REPO_ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md",
        r'Release Version: `v\d+\.\d+\.\d+`',
        f'Release Version: `v{version}`'
    )

    # 3e. docs/competitive_analysis.md table rows
    update_file(
        REPO_ROOT / "docs" / "competitive_analysis.md",
        r'SCPN v\d+\.\d+\.\d+',
        f'SCPN v{version}'
    )

    # 5. README.md badges
    # Matches ![Version](...version-X.Y.Z...)
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

    # 7. docs/ version references
    # competitive_analysis.md header
    update_file(
        REPO_ROOT / "docs" / "competitive_analysis.md",
        r'SCPN Fusion Core v\d+\.\d+\.\d+',
        f'SCPN Fusion Core v{version}'
    )
    # STREAMLIT_DEMO_PLAYBOOK.md
    update_file(
        REPO_ROOT / "docs" / "STREAMLIT_DEMO_PLAYBOOK.md",
        r'SCPN Fusion Core v\d+\.\d+\.\d+',
        f'SCPN Fusion Core v{version}'
    )
    # VALIDATION_GATE_MATRIX.md
    update_file(
        REPO_ROOT / "docs" / "VALIDATION_GATE_MATRIX.md",
        r'As of v\d+\.\d+\.x',
        f'As of v{".".join(version.split(".")[:2])}.x'
    )
    # README.md YouTube reference
    update_file(
        REPO_ROOT / "README.md",
        r'pending upload for v\d+\.\d+\.\d+ release notes',
        f'pending upload for v{version} release notes'
    )
    # RELEASE_ACCEPTANCE_CHECKLIST.md
    update_file(
        REPO_ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md",
        r'v\d+\.\d+\.\d+ Release Acceptance',
        f'v{version} Release Acceptance'
    )

    # 8. CHANGELOG.md — only update the FIRST version header (latest release)
    # NOTE: Do NOT use update_file() here because re.sub replaces ALL matches,
    # which would corrupt historical version headers.
    changelog = REPO_ROOT / "CHANGELOG.md"
    if changelog.exists():
        text = changelog.read_text(encoding="utf-8")
        new_text = re.sub(r'## \[\d+\.\d+\.\d+\]', f'## [{version}]', text, count=1)
        if new_text != text:
            changelog.write_text(new_text, encoding="utf-8")
            print(f"Updated {changelog.relative_to(REPO_ROOT)} (first header only)")
        else:
            print(f"No change needed for {changelog.relative_to(REPO_ROOT)}")
    else:
        print(f"Warning: {changelog} not found, skipping.")

    print("--- Synchronization Complete ---")

if __name__ == "__main__":
    sync()
