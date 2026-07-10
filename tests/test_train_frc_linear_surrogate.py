# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Linear Surrogate Tool Tests
"""Contract tests for the FRC linear (Ridge/PCA) surrogate trainer.

The Rust FRC oracle is stubbed so these tests drive the tool's own logic —
the rejection-sampling data loop (converged/finite filter, exception skip,
attempt exhaustion, progress logging), the insufficient-sample abort, and the
PCA + Ridge normal-equation fit with its NPZ output — using the real
``MinimalPCA``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tools import train_frc_linear_surrogate as tool
from tools.train_frc_linear_surrogate import generate_frc_data, main


def _converged(rho_grid: NDArray[np.float64], inputs: Any) -> SimpleNamespace:
    """Build a converged state with a B_z profile that varies with inputs."""
    base = inputs.n0 / 1e20 + inputs.T_i_eV / 1000.0 + inputs.R_s + inputs.B_ext
    profile = base * (1.0 + 0.1 * np.cos(np.arange(len(rho_grid))))
    return SimpleNamespace(converged=True, B_z=profile.astype(float))


def _stub(
    func: Callable[[NDArray[np.float64], Any], SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a solver stub of signature (inputs, rho_grid, **kwargs)."""

    def _solve(inputs: Any, rho_grid: NDArray[np.float64], **_kw: Any) -> SimpleNamespace:
        return func(rho_grid, inputs)

    monkeypatch.setattr(tool, "solve_frc_equilibrium", _solve)


class TestGenerateFrcData:
    """Rejection-sampling data generation loop."""

    def test_collects_valid_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Converged, finite samples are collected with the right shapes."""
        _stub(_converged, monkeypatch)
        x, y = generate_frc_data(n_samples=3, grid_size=8)
        assert x.shape == (3, 7)
        assert y.shape == (3, 8)

    def test_progress_logging_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Crossing the 50-sample mark exercises the progress log branch."""
        _stub(_converged, monkeypatch)
        x, _ = generate_frc_data(n_samples=51, grid_size=4)
        assert x.shape == (51, 7)

    def test_non_converged_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-converged states are rejected until attempts are exhausted."""
        _stub(
            lambda g, i: SimpleNamespace(converged=False, B_z=np.ones(len(g))),
            monkeypatch,
        )
        x, y = generate_frc_data(n_samples=2, grid_size=4)
        assert x.shape == (0,)
        assert y.shape == (0,)

    def test_non_finite_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Converged-but-non-finite fields are rejected."""
        _stub(
            lambda g, i: SimpleNamespace(converged=True, B_z=np.full(len(g), np.nan)),
            monkeypatch,
        )
        x, _ = generate_frc_data(n_samples=2, grid_size=4)
        assert x.shape == (0,)

    def test_solver_exception_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A solver exception is swallowed and the sample skipped."""

        def _boom(g: NDArray[np.float64], i: Any) -> SimpleNamespace:
            raise RuntimeError("oracle failure")

        _stub(_boom, monkeypatch)
        x, _ = generate_frc_data(n_samples=2, grid_size=4)
        assert x.shape == (0,)


class TestMain:
    """CLI training entry point."""

    def test_full_training_writes_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sufficient run fits PCA+Ridge and saves the weight NPZ."""
        _stub(_converged, monkeypatch)
        out = tmp_path / "w" / "linear.npz"
        monkeypatch.setattr(
            "sys.argv",
            [
                "train_frc_linear_surrogate",
                "--samples",
                "15",
                "--grid",
                "12",
                "--out",
                str(out),
            ],
        )
        main()
        assert out.exists()
        with np.load(out) as data:
            assert int(data["grid_size"][0]) == 12
            assert data["w_linear"].shape[0] == 7
            assert data["pca_components"].shape[1] == 12
            assert data["input_mean"].shape == (7,)

    def test_aborts_on_insufficient_samples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fewer than 10 valid samples aborts before fitting."""
        _stub(_converged, monkeypatch)
        out = tmp_path / "linear.npz"
        monkeypatch.setattr(
            "sys.argv",
            ["train_frc_linear_surrogate", "--samples", "5", "--grid", "8", "--out", str(out)],
        )
        main()
        assert not out.exists()
