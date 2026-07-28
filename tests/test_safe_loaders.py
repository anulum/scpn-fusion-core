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

import numpy as np
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


def test_checked_np_load_rejects_oversized_expanded_member(tmp_path: Path) -> None:
    path = tmp_path / "compressed_member.npz"
    np.savez_compressed(path, payload=np.zeros(512, dtype=np.uint8))

    with pytest.raises(ValueError, match="member too large"):
        checked_np_load(path, max_member_bytes=512)


def test_checked_np_load_rejects_oversized_expanded_total(tmp_path: Path) -> None:
    path = tmp_path / "compressed_total.npz"
    np.savez_compressed(
        path,
        first=np.zeros(128, dtype=np.uint8),
        second=np.zeros(128, dtype=np.uint8),
        third=np.zeros(128, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="expands too large"):
        checked_np_load(path, max_member_bytes=512, max_total_bytes=700)


def test_checked_np_load_accepts_bounded_npz_and_plain_npy(tmp_path: Path) -> None:
    archive_path = tmp_path / "bounded.npz"
    array_path = tmp_path / "bounded.npy"
    expected = np.arange(16, dtype=np.float64)
    np.savez_compressed(archive_path, payload=expected)
    np.save(array_path, expected)

    with checked_np_load(
        archive_path,
        max_member_bytes=512,
        max_total_bytes=512,
    ) as archive:
        np.testing.assert_array_equal(archive["payload"], expected)
    np.testing.assert_array_equal(
        checked_np_load(array_path, max_member_bytes=1, max_total_bytes=1),
        expected,
    )
