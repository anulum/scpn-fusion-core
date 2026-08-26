# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Solver Tests
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scpn_fusion.core.gk_interface import GKLocalParams
from scpn_fusion.core.gk_tglf import (
    TGLFSolver,
    _classify_dominant_mode,
    generate_tglf_input,
    parse_tglf_output,
)


@pytest.fixture
def cyclone_params():
    """Cyclone Base Case parameters (Dimits et al. 2000)."""
    return GKLocalParams(
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.0,
        nu_star=0.01,
        beta_e=0.0,
        epsilon=0.18,
        kappa=1.0,
        delta=0.0,
        rho=0.5,
        R0=2.78,
        a=1.0,
        B0=2.0,
        n_e=5.0,
        T_e_keV=2.0,
        T_i_keV=2.0,
    )


def _write_current_outputs(run_dir: Path) -> None:
    (run_dir / "out.tglf.gbflux").write_text(
        "-0.4 -0.4 3.2 5.1 0 0 0 0\n",
        encoding="utf-8",
    )
    (run_dir / "out.tglf.eigenvalue_spectrum").write_text(
        "# gamma/frequency pairs\n# mode 1 mode 2\n0.2 -0.5 0.1 0.2\n0.4 -0.8 0.3 0.4\n0.6 1.2 0.5 -0.2\n",
        encoding="utf-8",
    )
    (run_dir / "out.tglf.ky_spectrum").write_text(
        "# ky rho_s\n# values\n0.1\n0.4\n1.5\n",
        encoding="utf-8",
    )


def test_generate_tglf_input_contains_current_keys(cyclone_params):
    text = generate_tglf_input(cyclone_params)
    assert "&tglf_namelist" not in text
    assert "RLTS_1 = 2.48201438849" in text
    assert "RLTS_2 = 2.48201438849" in text
    assert "Q_LOC = 1.4" in text
    assert "Q_PRIME_LOC" in text
    assert "BETAE = 0" in text
    assert "MASS_1 = 2.723000e-4" in text
    assert "ZS_1 = -1.0" in text


def test_generate_tglf_input_geometry(cyclone_params):
    cyclone_params.kappa = 1.7
    cyclone_params.delta = 0.33
    text = generate_tglf_input(cyclone_params)
    assert "KAPPA_LOC = 1.7" in text
    assert "DELTA_LOC = 0.33" in text
    assert "RMAJ_LOC = 2.78" in text
    assert "RMIN_LOC = 0.5" in text


def test_generate_tglf_input_rejects_nonlinear_request(cyclone_params):
    cyclone_params.physics_model = "nonlinear_electrostatic"
    with pytest.raises(ValueError, match="quasilinear"):
        generate_tglf_input(cyclone_params)


def test_parse_tglf_output_requires_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_tglf_output(tmp_path)


def test_parse_tglf_output_current_gacode_files(tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    _write_current_outputs(tmp_path)
    result = parse_tglf_output(tmp_path)
    assert result.converged is True
    assert result.heat_flux_i_gb == pytest.approx(5.1)
    assert result.heat_flux_e_gb == pytest.approx(3.2)
    assert result.particle_flux_e_gb == pytest.approx(-0.4)
    assert result.particle_flux_i_gb == pytest.approx(-0.4)
    assert np.isfinite([result.chi_i, result.chi_e, result.D_e, result.D_i]).all()
    np.testing.assert_allclose(result.k_y, [0.1, 0.4, 1.5])
    np.testing.assert_allclose(result.gamma, [0.2, 0.4, 0.6])
    assert result.dominant_mode == "ETG"


def test_classify_dominant_modes():
    assert _classify_dominant_mode(np.array([0.0, -0.1]), np.zeros(2)) == "stable"
    assert _classify_dominant_mode(np.array([0.1, 0.3]), np.array([0.5, -0.8])) == "ITG"
    assert _classify_dominant_mode(np.array([0.1, 0.3]), np.array([-0.5, 0.8])) == "TEM"
    assert (
        _classify_dominant_mode(
            np.array([0.1, 0.3]),
            np.array([-0.5, 0.8]),
            np.array([0.2, 1.5]),
        )
        == "ETG"
    )


def test_tglf_solver_not_available():
    solver = TGLFSolver(binary="nonexistent_tglf_binary_xyz")
    assert solver.is_available() is False


def test_tglf_solver_rejects_binary_paths(tmp_path):
    solver = TGLFSolver(binary="/opt/gacode/tglf", work_dir=tmp_path)
    assert solver.is_available() is False


def test_tglf_solver_prepare_input(tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    run_dir = solver.prepare_input(cyclone_params)
    assert (run_dir / "input.tglf").exists()
    assert (run_dir / "scpn_fusion_tglf_deck.json").exists()


def test_tglf_solver_run_binary_missing(tmp_path, cyclone_params):
    solver = TGLFSolver(binary="nonexistent_tglf_binary_xyz", work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    with pytest.raises(RuntimeError, match="unavailable through PATH"):
        solver.run(tmp_path)


@patch("scpn_fusion.core.gk_tglf._resolve_tglf_command", return_value="/usr/bin/tglf")
@patch("scpn_fusion.core.gk_tglf.subprocess.run")
def test_tglf_solver_run_mocked_success(mock_run, mock_resolve, tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    _write_current_outputs(tmp_path)
    mock_run.return_value = subprocess.CompletedProcess([], 0)
    result = solver.run(tmp_path)
    assert result.converged is True
    assert result.heat_flux_i_gb == pytest.approx(5.1)
    mock_run.assert_called_once_with(
        ["/usr/bin/tglf", "-e", "."],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=True,
    )


@patch("scpn_fusion.core.gk_tglf._resolve_tglf_command", return_value="/usr/bin/tglf")
@patch(
    "scpn_fusion.core.gk_tglf.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="tglf", timeout=1.0),
)
def test_tglf_solver_timeout_fails_closed(mock_run, mock_resolve, tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    with pytest.raises(RuntimeError, match="TGLF execution failed"):
        solver.run(tmp_path, timeout_s=1.0)
