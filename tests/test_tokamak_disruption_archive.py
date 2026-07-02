# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Disruption Archive Tests
"""Tests for tokamak disruption-shot archive loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import scpn_fusion.io.tokamak_archive as archive
from scpn_fusion.io.tokamak_disruption_archive import (
    list_disruption_shots,
    load_disruption_shot,
)

FloatArray: TypeAlias = NDArray[np.float64]
PayloadValue: TypeAlias = (
    NDArray[np.float64] | NDArray[np.bool_] | NDArray[np.int64] | NDArray[np.str_]
)

ARRAY_KEYS: tuple[str, ...] = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
)


def _write_disruption_shot(
    path: Path,
    *,
    is_disruption: bool = True,
    disruption_time_idx: int = 2,
    disruption_type: str = "locked_mode",
    omit_key: str | None = None,
) -> None:
    """Write a deterministic disruption NPZ payload through NumPy's archive API."""
    time_axis: FloatArray = np.array([0.0, 0.05, 0.1], dtype=np.float64)
    payload: dict[str, PayloadValue] = {}
    for index, key in enumerate(ARRAY_KEYS):
        if key != omit_key:
            payload[key] = time_axis + float(index)
    if omit_key != "is_disruption":
        payload["is_disruption"] = np.array(is_disruption, dtype=np.bool_)
    if omit_key != "disruption_time_idx":
        payload["disruption_time_idx"] = np.array(disruption_time_idx, dtype=np.int64)
    if omit_key != "disruption_type":
        payload["disruption_type"] = np.array(disruption_type, dtype=np.str_)
    np.savez(path, **payload)


def test_list_disruption_shots_returns_empty_list_for_missing_directory(
    tmp_path: Path,
) -> None:
    """Directory discovery fails closed when the archive directory is absent."""
    assert list_disruption_shots(disruption_dir=tmp_path / "missing") == []


def test_list_disruption_shots_returns_sorted_npz_stems(tmp_path: Path) -> None:
    """Directory discovery returns sorted NPZ stems and ignores other files."""
    _write_disruption_shot(tmp_path / "shot_z.npz")
    _write_disruption_shot(tmp_path / "shot_a.npz")
    (tmp_path / "notes.txt").write_text("not an archive", encoding="utf-8")

    assert list_disruption_shots(disruption_dir=tmp_path) == ["shot_a", "shot_z"]


def test_load_disruption_shot_resolves_stem_through_archive_facade(
    tmp_path: Path,
) -> None:
    """The public tokamak archive facade resolves shot stems into NPZ payloads."""
    _write_disruption_shot(
        tmp_path / "shot_170001.npz",
        is_disruption=False,
        disruption_time_idx=1,
        disruption_type="vertical",
    )

    loaded = archive.load_disruption_shot("shot_170001", disruption_dir=tmp_path)

    np.testing.assert_array_equal(
        cast(FloatArray, loaded["time_s"]),
        np.array([0.0, 0.05, 0.1], dtype=np.float64),
    )
    assert cast(FloatArray, loaded["Ip_MA"]).dtype == np.float64
    assert cast(bool, loaded["is_disruption"]) is False
    assert cast(int, loaded["disruption_time_idx"]) == 1
    assert cast(str, loaded["disruption_type"]) == "vertical"


def test_load_disruption_shot_accepts_uppercase_npz_path(tmp_path: Path) -> None:
    """The loader accepts explicit NPZ paths case-insensitively."""
    source = tmp_path / "shot_upper.npz"
    target = tmp_path / "shot_upper.NPZ"
    _write_disruption_shot(source)
    source.rename(target)

    loaded = load_disruption_shot(target, disruption_dir=tmp_path)

    np.testing.assert_array_equal(
        cast(FloatArray, loaded["vertical_position_m"]),
        np.array([10.0, 10.05, 10.1], dtype=np.float64),
    )


def test_load_disruption_shot_rejects_non_npz_extension(tmp_path: Path) -> None:
    """The loader rejects non-NPZ file names before reading from disk."""
    with pytest.raises(ValueError, match="must be .npz"):
        load_disruption_shot(tmp_path / "shot.txt", disruption_dir=tmp_path)


def test_load_disruption_shot_rejects_missing_npz_file(tmp_path: Path) -> None:
    """The loader reports a missing NPZ shot after path resolution."""
    with pytest.raises(FileNotFoundError, match="shot_missing.npz"):
        load_disruption_shot("shot_missing", disruption_dir=tmp_path)


def test_load_disruption_shot_rejects_missing_required_key(tmp_path: Path) -> None:
    """The loader fails closed when a required disruption payload key is absent."""
    _write_disruption_shot(tmp_path / "shot_incomplete.npz", omit_key="locked_mode_amp")

    with pytest.raises(ValueError, match="locked_mode_amp"):
        load_disruption_shot("shot_incomplete", disruption_dir=tmp_path)
