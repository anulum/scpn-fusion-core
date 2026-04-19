# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Solver Tests
from __future__ import annotations

import subprocess
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


def test_generate_tglf_input_contains_keys(cyclone_params):
    text = generate_tglf_input(cyclone_params)
    assert "&tglf_namelist" in text
    assert "RLTS_1 = 6.900000" in text
    assert "Q_LOC = 1.400000" in text
    assert "SHAT = 0.780000" in text
    assert "BETAE = 0.000000e+00" in text


def test_generate_tglf_input_geometry(cyclone_params):
    cyclone_params.kappa = 1.7
    cyclone_params.delta = 0.33
    text = generate_tglf_input(cyclone_params)
    assert "KAPPA_LOC = 1.700000" in text
    assert "DELTA_LOC = 0.330000" in text


def test_generate_tglf_input_aspect_ratio(cyclone_params):
    text = generate_tglf_input(cyclone_params)
    # R0/a = 2.78 / 1.0
    assert "RMAJ_LOC = 2.780000" in text


def test_parse_tglf_output_empty(tmp_path):
    result = parse_tglf_output(tmp_path)
    assert result.chi_i == 0.0
    assert result.converged is False


def test_parse_tglf_output_transport_file(tmp_path):
    transport = tmp_path / "out.tglf.transport"
    transport.write_text("chi_i 2.5\nchi_e 1.8\nd_e 0.4\n")
    result = parse_tglf_output(tmp_path)
    assert result.chi_i == pytest.approx(2.5)
    assert result.chi_e == pytest.approx(1.8)
    assert result.D_e == pytest.approx(0.4)
    assert result.converged is True


def test_parse_tglf_output_eigenvalue_file(tmp_path):
    transport = tmp_path / "out.tglf.transport"
    transport.write_text("chi_i 1.0\nchi_e 0.8\n")
    eigen = tmp_path / "out.tglf.eigenvalue_spectrum"
    data = np.column_stack(
        [
            np.linspace(0.1, 2.0, 12),
            np.random.default_rng(0).random(12) * 0.3,
            -np.random.default_rng(1).random(12) * 0.5,
        ]
    )
    np.savetxt(eigen, data, header="ky gamma omega_r")
    result = parse_tglf_output(tmp_path)
    assert len(result.k_y) == 12
    assert len(result.gamma) == 12
    assert result.dominant_mode == "ITG"


def test_classify_dominant_mode_stable():
    gamma = np.array([0.0, -0.1, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    assert _classify_dominant_mode(gamma, omega) == "stable"


def test_classify_dominant_mode_itg():
    gamma = np.array([0.1, 0.3, 0.05])
    omega = np.array([-0.5, -0.8, 0.2])
    assert _classify_dominant_mode(gamma, omega) == "ITG"


def test_classify_dominant_mode_tem():
    gamma = np.array([0.1, 0.05, 0.3])
    omega = np.array([0.5, -0.2, 0.8])
    assert _classify_dominant_mode(gamma, omega) == "TEM"


def test_tglf_solver_not_available():
    solver = TGLFSolver(binary="nonexistent_tglf_binary_xyz")
    assert solver.is_available() is False


def test_tglf_solver_prepare_input(tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    run_dir = solver.prepare_input(cyclone_params)
    assert (run_dir / "input.tglf").exists()
    content = (run_dir / "input.tglf").read_text()
    assert "RLTS_1" in content


def test_tglf_solver_run_binary_missing(tmp_path, cyclone_params):
    solver = TGLFSolver(binary="nonexistent_tglf_binary_xyz", work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    result = solver.run(tmp_path)
    assert result.converged is False
    assert result.chi_i == 0.0


@patch("shutil.which", return_value="/usr/bin/tglf")
@patch("subprocess.run")
def test_tglf_solver_run_mocked_success(mock_run, mock_which, tmp_path, cyclone_params):
    """Mock a successful TGLF execution by writing output files."""
    solver = TGLFSolver(work_dir=tmp_path)
    solver.prepare_input(cyclone_params)

    # Simulate TGLF writing output
    (tmp_path / "out.tglf.transport").write_text("chi_i 3.2\nchi_e 2.1\nd_e 0.6\n")
    mock_run.return_value = None

    result = solver.run(tmp_path)
    assert result.converged is True
    assert result.chi_i == pytest.approx(3.2)
    mock_run.assert_called_once()


@patch("shutil.which", return_value="/usr/bin/tglf")
@patch(
    "subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="tglf", timeout=1.0),
)
def test_tglf_solver_timeout_fallback(mock_run, mock_which, tmp_path, cyclone_params):
    solver = TGLFSolver(work_dir=tmp_path)
    solver.prepare_input(cyclone_params)
    result = solver.run(tmp_path, timeout_s=1.0)
    assert result.converged is False


def test_tglf_solver_run_from_params(tmp_path, cyclone_params):
    solver = TGLFSolver(binary="nonexistent_tglf_binary_xyz", work_dir=tmp_path)
    result = solver.run_from_params(cyclone_params)
    assert result.converged is False
