# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/github_token_format_guard.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "github_token_format_guard.py"
SPEC = importlib.util.spec_from_file_location("github_token_format_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def test_flags_len_40_token_assumption() -> None:
    line = "if len(github_token) == 40:"
    for pattern, _ in guard.SUSPECT_PATTERNS:
        if pattern.search(line):
            break
    else:  # pragma: no cover
        raise AssertionError("Expected token-length hardcoding pattern to match.")


def test_flags_legacy_ghs_regex_assumption() -> None:
    line = r'pattern = r"ghs_[A-Za-z0-9]{40}"'
    for pattern, _ in guard.SUSPECT_PATTERNS:
        if pattern.search(line):
            break
    else:  # pragma: no cover
        raise AssertionError("Expected legacy ghs_ regex pattern to match.")


def test_does_not_flag_opaque_token_pass_through() -> None:
    line = 'headers = {"Authorization": f"Bearer {token}"}'
    assert all(not pattern.search(line) for pattern, _ in guard.SUSPECT_PATTERNS)
