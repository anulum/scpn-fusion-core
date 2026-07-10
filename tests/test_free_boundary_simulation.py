# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for free-boundary supervisory-simulation edge paths.

Covers the disturbance-recovery finiteness guard and the optional-path branches
(default ITER config, verbose step logging, plot saving) that the primary
simulation tests skip by always passing an explicit config and disabling output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.control._free_boundary_simulation import (
    run_free_boundary_supervisory_simulation,
)
from scpn_fusion.control._free_boundary_supervisory_types import FreeBoundaryTarget


class _DivertedDummyKernel:
    """Deterministic diverted equilibrium response for simulation tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
        }
        self.R = np.linspace(5.75, 6.25, 41)
        self.Z = np.linspace(-0.45, 0.45, 41)
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self._x_point = (5.02, -3.48)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Recompute a deterministic diverted flux surface from coil currents."""
        coils = cast(list[dict[str, Any]], self.cfg["coils"])
        i = np.asarray([float(c["current"]) for c in coils], dtype=np.float64)
        physics = cast(dict[str, Any], self.cfg["physics"])
        ip = float(physics["plasma_current_target"])
        radial_drive = 0.95 * i[2] - 0.42 * i[1] + 0.16 * i[3]
        vertical_drive = 0.82 * i[3] - 0.68 * i[0] + 0.18 * i[2]
        divertor_drive_r = 0.74 * i[1] - 0.38 * i[0] + 0.12 * i[2]
        divertor_drive_z = 0.88 * i[3] - 0.52 * i[2] + 0.10 * i[0]

        current_shift = ip - 7.0
        center_r = 6.0 + 0.07 * np.tanh(radial_drive / 0.75) + 0.010 * current_shift
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 0.75) - 0.006 * current_shift

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.005 * ip
        x_r = 5.02 + 0.05 * np.tanh(divertor_drive_r / 0.70) + 0.006 * current_shift
        x_z = -3.48 + 0.06 * np.tanh(divertor_drive_z / 0.72) - 0.010 * current_shift
        self._x_point = (float(x_r), float(x_z))

    def find_x_point(self, _psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        """Return the deterministic diverted X-point and its flux value."""
        return self._x_point, 0.0


def test_rejects_non_finite_disturbance_recovery() -> None:
    """A non-finite disturbance-recovery rate is rejected before the run starts."""
    with pytest.raises(ValueError, match="disturbance_recovery_per_step_ma must be finite"):
        run_free_boundary_supervisory_simulation(
            config_file="dummy.json",
            shot_length=32,
            disturbance_recovery_per_step_ma=float("nan"),
            save_plot=False,
            verbose=False,
            kernel_factory=_DivertedDummyKernel,
        )


def test_default_config_verbose_and_plot_paths(tmp_path: Path) -> None:
    """Exercise the default-config, verbose-logging, and plot-saving branches.

    The primary simulation tests always pass an explicit config and disable
    output; this run leaves ``config_file`` unset (default ITER config path),
    enables verbose step logging, and requests a saved plot to a temporary path.
    """
    output_path = str(tmp_path / "free_boundary_control.png")
    summary = run_free_boundary_supervisory_simulation(
        config_file=None,
        shot_length=24,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        save_plot=True,
        output_path=output_path,
        verbose=True,
        rng_seed=42,
        kernel_factory=_DivertedDummyKernel,
    )
    assert "plot_saved" in summary
    assert summary["steps"] == 24
