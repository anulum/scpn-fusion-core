# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Dataset Generator Tests
"""Contract tests for the TGLF-like turbulence dataset generator.

Covers input validation (non-positive / non-integer sample counts and an
empty reference-case table), both spectral-slope branches (ITG vs non-ITG
regime names), and the on-disk NPZ contract for a real-reference run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import generate_tglf_dataset as tool
from tools.generate_tglf_dataset import generate_tglf_like_dataset

_GRID = 64


def _regime() -> dict[str, list[float]]:
    """Build a minimal single-rho reference-case payload."""
    return {"rho_points": [0.5], "gamma_max": [0.1]}


class TestValidation:
    """Argument and reference-table validation."""

    @pytest.mark.parametrize("n_samples", [0, -1, -100])
    def test_non_positive_sample_count_raises(self, n_samples: int) -> None:
        """A non-positive sample count is rejected."""
        with pytest.raises(ValueError, match="positive integer"):
            generate_tglf_like_dataset(n_samples=n_samples)

    def test_non_integer_sample_count_raises(self, tmp_path: Path) -> None:
        """A non-integer sample count is rejected."""
        bad: Any = 1.5
        with pytest.raises(ValueError, match="positive integer"):
            generate_tglf_like_dataset(output_path=str(tmp_path / "x.npz"), n_samples=bad)

    def test_empty_reference_cases_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty reference-case table is rejected before generation."""
        monkeypatch.setattr(tool, "REFERENCE_CASES", {})
        with pytest.raises(ValueError, match="REFERENCE_CASES must be populated"):
            generate_tglf_like_dataset(output_path=str(tmp_path / "x.npz"), n_samples=1)


class TestGeneration:
    """Field generation and the NPZ output contract."""

    @pytest.mark.parametrize(
        "regime_name",
        ["ITG-dominated", "TEM-dominated"],
    )
    def test_slope_branches_write_dataset(
        self, regime_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both spectral-slope branches (ITG and non-ITG) produce a dataset.

        A single-regime table forces the branch deterministically: ITG names
        take the alpha=3.0 slope, everything else takes alpha=2.5.
        """
        monkeypatch.setattr(tool, "REFERENCE_CASES", {regime_name: _regime()})
        out = tmp_path / "data.npz"
        generate_tglf_like_dataset(output_path=str(out), n_samples=1)
        with np.load(out) as data:
            assert data["X"].shape == (1, _GRID, _GRID, 1)
            assert data["Y"].shape == (1,)
            assert data["X"].dtype == np.float32
            assert np.all(np.isfinite(data["X"]))

    def test_real_reference_cases_multi_sample(self, tmp_path: Path) -> None:
        """A multi-sample run over the bundled reference cases writes shapes."""
        out = tmp_path / "real.npz"
        generate_tglf_like_dataset(output_path=str(out), n_samples=4)
        with np.load(out) as data:
            assert data["X"].shape == (4, _GRID, _GRID, 1)
            assert data["Y"].shape == (4,)
            assert np.all(data["Y"] > 0.0)
