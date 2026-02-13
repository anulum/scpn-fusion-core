# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Kernel Divergence Guards
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel


def _write_cfg(path: Path, fail_on_diverge: bool) -> Path:
    cfg = {
        "reactor_name": "Unit-Test",
        "grid_resolution": [8, 8],
        "dimensions": {"R_min": 1.0, "R_max": 2.0, "Z_min": -1.0, "Z_max": 1.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [],
        "solver": {
            "max_iterations": 3,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 0.1,
            "fail_on_diverge": fail_on_diverge,
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _force_divergence(kernel: FusionKernel) -> None:
    zeros = np.zeros_like(kernel.Psi)
    nans = np.full_like(kernel.Psi, np.nan)
    kernel.calculate_vacuum_field = lambda: zeros.copy()  # type: ignore[assignment]
    kernel._seed_plasma = lambda _mu0: None  # type: ignore[assignment]
    kernel._find_magnetic_axis = (  # type: ignore[assignment]
        lambda: (0.0, 0.0, 1.0)
    )
    kernel.find_x_point = lambda _psi: ((0.0, 0.0), 0.0)  # type: ignore[assignment]
    kernel.update_plasma_source_nonlinear = (  # type: ignore[assignment]
        lambda _axis, _boundary: zeros.copy()
    )
    kernel._elliptic_solve = lambda _source, _vac: nans.copy()  # type: ignore[assignment]
    kernel.compute_b_field = lambda: None  # type: ignore[assignment]


def test_solve_equilibrium_divergence_reverts_when_fail_disabled(
    tmp_path: Path,
) -> None:
    cfg_path = _write_cfg(tmp_path / "cfg_no_fail.json", fail_on_diverge=False)
    kernel = FusionKernel(cfg_path)
    _force_divergence(kernel)

    kernel.solve_equilibrium()
    assert np.all(np.isfinite(kernel.Psi))


def test_solve_equilibrium_divergence_raises_when_fail_enabled(
    tmp_path: Path,
) -> None:
    cfg_path = _write_cfg(tmp_path / "cfg_fail.json", fail_on_diverge=True)
    kernel = FusionKernel(cfg_path)
    _force_divergence(kernel)

    with pytest.raises(RuntimeError, match="diverged"):
        kernel.solve_equilibrium()
