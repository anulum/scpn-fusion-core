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
import warnings

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


def test_residual_rms_avoids_overflow_for_extreme_fields(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path / "cfg_rms.json", fail_on_diverge=False)
    kernel = FusionKernel(cfg_path)
    psi = np.zeros_like(kernel.Psi)
    psi[1:-1, 1:-1] = np.where(
        (np.indices(psi[1:-1, 1:-1].shape).sum(axis=0) % 2) == 0,
        1e300,
        -1e300,
    )
    kernel.Psi = psi
    source = np.zeros_like(kernel.Psi)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rms = kernel._compute_gs_residual_rms(source)

    assert np.isfinite(rms)
    assert rms > 0.0


def test_find_x_point_avoids_overflow_with_extreme_gradients(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path / "cfg_xpt.json", fail_on_diverge=False)
    kernel = FusionKernel(cfg_path)
    psi = np.outer(np.linspace(-1e300, 1e300, kernel.NZ), np.ones(kernel.NR))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x_point, psi_x = kernel.find_x_point(psi)

    assert np.isfinite(x_point[0])
    assert np.isfinite(x_point[1])
    assert np.isfinite(psi_x)


def test_sor_step_sanitizes_non_finite_inputs(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path / "cfg_sor.json", fail_on_diverge=False)
    kernel = FusionKernel(cfg_path)
    psi = np.zeros_like(kernel.Psi)
    psi[2, 2] = np.inf
    psi[3, 3] = np.nan
    source = np.zeros_like(kernel.Psi)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        updated = kernel._sor_step(psi, source, omega=1.2)

    assert np.all(np.isfinite(updated))
