# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Version Metadata Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Version consistency checks across release metadata files."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / "src" / "scpn_fusion" / "__init__.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"
SETUP_PATH = ROOT / "setup.py"
CITATION_PATH = ROOT / "CITATION.cff"
SPHINX_CONF_PATH = ROOT / "docs" / "sphinx" / "conf.py"


def _extract_version(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"Failed to parse version from {label}")
    return match.group(1).strip()


def test_release_metadata_versions_are_consistent() -> None:
    init_text = INIT_PATH.read_text(encoding="utf-8")
    package_version = _extract_version(
        r'__version__\s*=\s*"([^"]+)"',
        init_text,
        "__init__.py",
    )

    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    pyproject_version = _extract_version(
        r'(?m)^version\s*=\s*"([^"]+)"',
        pyproject_text,
        "pyproject.toml",
    )

    setup_text = SETUP_PATH.read_text(encoding="utf-8")
    setup_version = _extract_version(
        r'version\s*=\s*"([^"]+)"',
        setup_text,
        "setup.py",
    )

    citation_text = CITATION_PATH.read_text(encoding="utf-8")
    citation_version = _extract_version(
        r'(?m)^version:\s*"([^"]+)"',
        citation_text,
        "CITATION.cff",
    )

    spec = importlib.util.spec_from_file_location("sphinx_conf", SPHINX_CONF_PATH)
    assert spec and spec.loader
    sphinx_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sphinx_conf)
    sphinx_release = str(getattr(sphinx_conf, "release"))

    assert pyproject_version == package_version
    assert setup_version == package_version
    assert citation_version == package_version
    assert sphinx_release == package_version
