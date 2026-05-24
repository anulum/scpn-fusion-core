# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safe Loader Tests
"""Tests for bounded JSON and NumPy archive loader contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_fusion.io.safe_loaders import (
    MAX_JSON_BYTES,
    MAX_NPZ_BYTES,
    checked_json_load,
    checked_np_load,
)


def test_checked_json_load_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    with path.open("wb") as handle:
        handle.truncate(MAX_JSON_BYTES + 1)

    with pytest.raises(ValueError, match="JSON file too large"):
        checked_json_load(path)


def test_checked_np_load_rejects_oversized_archive(tmp_path: Path) -> None:
    path = tmp_path / "huge.npz"
    with path.open("wb") as handle:
        handle.truncate(MAX_NPZ_BYTES + 1)

    with pytest.raises(ValueError, match="NumPy archive file too large"):
        checked_np_load(path)
