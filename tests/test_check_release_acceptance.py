# ----------------------------------------------------------------------
# SCPN Fusion Core -- Release Acceptance Checklist Tests
# ----------------------------------------------------------------------
"""Tests for tools/check_release_acceptance.py."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_release_acceptance.py"
SPEC = importlib.util.spec_from_file_location("check_release_acceptance", MODULE_PATH)
assert SPEC and SPEC.loader
release_acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_acceptance
SPEC.loader.exec_module(release_acceptance)


def test_repo_release_checklist_passes() -> None:
    checklist = ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md"
    errors = release_acceptance.check_release_acceptance(
        checklist,
        expected_version=None,
        require_ready_state=True,
    )
    assert errors == []


def test_release_checklist_detects_unchecked_required_item(tmp_path: Path) -> None:
    src = (ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md").read_text(
        encoding="utf-8"
    )
    modified = re.sub(
        r"- \[x\] Release preflight",
        "- [ ] Release preflight",
        src,
        count=1,
    )
    checklist = tmp_path / "RELEASE_ACCEPTANCE_CHECKLIST.md"
    checklist.write_text(modified, encoding="utf-8")

    errors = release_acceptance.check_release_acceptance(
        checklist,
        expected_version=None,
        require_ready_state=True,
    )
    assert any("not checked" in err for err in errors)


def test_release_checklist_detects_version_mismatch() -> None:
    checklist = ROOT / "docs" / "RELEASE_ACCEPTANCE_CHECKLIST.md"
    errors = release_acceptance.check_release_acceptance(
        checklist,
        expected_version="v9.9.9",
        require_ready_state=True,
    )
    assert any("version mismatch" in err for err in errors)
