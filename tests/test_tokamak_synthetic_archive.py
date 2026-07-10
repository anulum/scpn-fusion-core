# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for the synthetic tokamak shot archive listing and loading.

Covers the shot-directory listing (present, empty, and missing directories)
and the single-shot NPZ loader's happy path plus its suffix, existence, and
required-key guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.io.tokamak_synthetic_archive import (
    list_synthetic_shots,
    load_synthetic_shot,
)

_ARRAY_KEYS = ("time_s", "Ip_MA", "BT_T", "ne_1e19", "Te_keV", "Ti_keV", "q95", "beta_N")


def _write_shot(path: Path, *, machine: str = "ITER", disruption: bool = False) -> None:
    """Write a schema-valid synthetic shot NPZ to ``path``."""
    arrays = {key: np.linspace(0.0, 1.0, 8, dtype=np.float64) for key in _ARRAY_KEYS}
    np.savez_compressed(
        str(path),
        disruption_label=np.bool_(disruption),
        machine=np.array(machine, dtype="U16"),
        **arrays,
    )


class TestListSyntheticShots:
    """Directory listing of synthetic shot files."""

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        """A non-existent directory yields an empty list."""
        assert list_synthetic_shots(synthetic_dir=tmp_path / "nope") == []

    def test_lists_sorted_stems(self, tmp_path: Path) -> None:
        """Present NPZ files are returned as sorted names without extension."""
        _write_shot(tmp_path / "shot_B.npz")
        _write_shot(tmp_path / "shot_A.npz")
        assert list_synthetic_shots(synthetic_dir=tmp_path) == ["shot_A", "shot_B"]


class TestLoadSyntheticShot:
    """Single-shot NPZ loading and validation."""

    def test_load_by_name_round_trips(self, tmp_path: Path) -> None:
        """Loading by bare name resolves to the directory and returns all fields."""
        _write_shot(tmp_path / "shot_X.npz", machine="SPARC", disruption=True)
        shot = load_synthetic_shot("shot_X", synthetic_dir=tmp_path)
        assert shot["machine"] == "SPARC"
        assert shot["disruption_label"] is True
        for key in _ARRAY_KEYS:
            assert shot[key].shape == (8,)

    def test_load_by_explicit_path(self, tmp_path: Path) -> None:
        """Loading by an explicit .npz path returns the parsed shot."""
        path = tmp_path / "shot_Y.npz"
        _write_shot(path)
        shot = load_synthetic_shot(path, synthetic_dir=tmp_path)
        assert shot["disruption_label"] is False

    def test_non_npz_suffix_raises(self, tmp_path: Path) -> None:
        """A path with a non-.npz suffix is rejected."""
        with pytest.raises(ValueError, match="must be .npz"):
            load_synthetic_shot(tmp_path / "shot.bin", synthetic_dir=tmp_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing shot file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_synthetic_shot("absent", synthetic_dir=tmp_path)

    def test_missing_keys_raise(self, tmp_path: Path) -> None:
        """An NPZ missing required keys is rejected."""
        path = tmp_path / "shot_bad.npz"
        np.savez_compressed(str(path), time_s=np.zeros(4))
        with pytest.raises(ValueError, match="missing keys"):
            load_synthetic_shot(path, synthetic_dir=tmp_path)
