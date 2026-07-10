# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Jacobian Surrogate Tool Tests
"""Contract tests for the FRC local-Jacobian surrogate extractor.

The FRC equilibrium oracle (``solve_frc_equilibrium``, Rust-backed) is
stubbed so these tests exercise the tool's own logic — feature unpacking,
the fixed zero ``theta_dot`` column, the forward-difference assembly, and the
NPZ save — rather than re-testing the solver, which has its own suite.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tools import train_frc_jacobian_surrogate as tool
from tools.train_frc_jacobian_surrogate import compute_local_jacobian, main

# Feature order: n0, T_i, T_e, theta_dot, R_s, B_ext, delta
_X_NOM = np.array([3.0, 10.0, 5.0, 0.0, 0.2, 5.0, 0.02])


def _fake_solve(
    inputs: Any, rho_grid: NDArray[np.float64], *, solver: str = "numpy", tolerance: float = 1e-10
) -> SimpleNamespace:
    """Return a B_z field linear in every input except T_e and theta_dot.

    This makes the n0 partial derivative non-zero and the T_e partial zero,
    so the assembled Jacobian is verifiable without the real oracle.
    """
    value = inputs.n0 / 1e20 + inputs.T_i_eV / 1000.0 + inputs.R_s + inputs.B_ext + inputs.delta
    return SimpleNamespace(B_z=np.full(len(rho_grid), value, dtype=float))


@pytest.fixture(autouse=True)
def _stub_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Rust FRC oracle with the deterministic linear stub."""
    monkeypatch.setattr(tool, "solve_frc_equilibrium", _fake_solve)


class TestComputeLocalJacobian:
    """Finite-difference Jacobian assembly."""

    def test_shapes(self) -> None:
        """Y_nom and J carry the grid/feature shapes."""
        grid = 8
        y_nom, jac = compute_local_jacobian(_X_NOM, grid)
        assert y_nom.shape == (grid,)
        assert jac.shape == (grid, len(_X_NOM))

    def test_theta_dot_column_is_zero(self) -> None:
        """The theta_dot column (index 3) is fail-closed to zero."""
        _, jac = compute_local_jacobian(_X_NOM, 4)
        assert np.all(jac[:, 3] == 0.0)

    def test_sensitive_and_insensitive_columns(self) -> None:
        """n0 has a non-zero partial; T_e (absent from B_z) has a zero one."""
        _, jac = compute_local_jacobian(_X_NOM, 4)
        assert np.all(jac[:, 0] != 0.0)  # d/dn0
        assert np.allclose(jac[:, 2], 0.0)  # d/dT_e

    def test_zero_valued_feature_uses_floor_perturbation(self) -> None:
        """A zero-valued sensitive feature falls back to the 1e-8 floor step."""
        # delta at index 6 set to 0 → X_perturbed[i]*eps == 0 → floor 1e-8 used.
        features = _X_NOM.copy()
        features[6] = 0.0
        _, jac = compute_local_jacobian(features, 4)
        assert np.all(np.isfinite(jac))


class TestMain:
    """CLI entry point."""

    def test_writes_surrogate_npz(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() extracts the surrogate and saves the expected NPZ keys."""
        out = tmp_path / "nested" / "surrogate.npz"
        monkeypatch.setattr(
            "sys.argv",
            ["train_frc_jacobian_surrogate", "--grid", "6", "--out", str(out)],
        )
        main()
        assert out.exists()
        with np.load(out) as data:
            assert int(data["grid_size"][0]) == 6
            assert data["x_nom"].shape == (7,)
            assert data["y_nom"].shape == (6,)
            assert data["jacobian"].shape == (6, 7)
