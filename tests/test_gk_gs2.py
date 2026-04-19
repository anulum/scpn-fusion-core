# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS2 Solver Tests
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from scpn_fusion.core.gk_gs2 import (
    GS2Solver,
    generate_gs2_input,
    parse_gs2_output,
)
from scpn_fusion.core.gk_interface import GKLocalParams


@pytest.fixture()
def default_params():
    return GKLocalParams(R_L_Ti=6.0, R_L_Te=6.0, R_L_ne=2.0, q=1.4, s_hat=0.8)


# ── generate_gs2_input ──────────────────────────────────────────────


class TestGenerateGS2Input:
    def test_namelist_sections(self, default_params):
        text = generate_gs2_input(default_params)
        for section in (
            "&theta_grid_eik_knobs",
            "&theta_grid_parameters",
            "&species_knobs",
            "&species_parameters_1",
            "&species_parameters_2",
            "&kt_grids_knobs",
            "&kt_grids_single_parameters",
        ):
            assert section in text

    def test_geometry_params_embedded(self, default_params):
        text = generate_gs2_input(default_params)
        assert f"rhoc = {default_params.rho:.6f}" in text
        assert f"qinp = {default_params.q:.6f}" in text
        assert f"shat = {default_params.s_hat:.6f}" in text

    def test_shaping_params(self, default_params):
        text = generate_gs2_input(default_params)
        assert f"akappa = {default_params.kappa:.6f}" in text
        assert f"tri = {default_params.delta:.6f}" in text

    def test_species_gradients(self, default_params):
        text = generate_gs2_input(default_params)
        assert f"tprim = {default_params.R_L_Ti:.6f}" in text
        assert f"tprim = {default_params.R_L_Te:.6f}" in text
        assert f"fprim = {default_params.R_L_ne:.6f}" in text

    def test_ky_value(self, default_params):
        text = generate_gs2_input(default_params)
        assert "aky = 0.3" in text

    def test_small_a_safe(self):
        params = GKLocalParams(R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, a=0.0)
        text = generate_gs2_input(params)
        assert "rmaj" in text

    def test_electron_species_temp(self):
        params = GKLocalParams(R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, Te_Ti=1.5)
        text = generate_gs2_input(params)
        assert "1.500000" in text


# ── parse_gs2_output ────────────────────────────────────────────────


class TestParseGS2Output:
    def test_missing_file_unconverged(self, tmp_path):
        out = parse_gs2_output(tmp_path)
        assert not out.converged
        assert out.chi_i == 0.0

    def test_valid_omega_itg(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        # ky, gamma, omega_r
        omega_file.write_text("0.3 0.20 -0.40\n")
        out = parse_gs2_output(tmp_path)
        assert out.converged
        assert out.gamma[0] == pytest.approx(0.20)
        assert out.omega_r[0] == pytest.approx(-0.40)
        assert out.k_y[0] == pytest.approx(0.3)
        assert out.dominant_mode == "ITG"

    def test_positive_omega_gives_tem(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        omega_file.write_text("0.3 0.15 0.35\n")
        out = parse_gs2_output(tmp_path)
        assert out.dominant_mode == "TEM"

    def test_negative_gamma_clamped(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        omega_file.write_text("0.3 -0.10 0.20\n")
        out = parse_gs2_output(tmp_path)
        assert out.chi_i == 0.0
        assert out.chi_e == 0.0

    def test_chi_e_ratio(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        omega_file.write_text("0.3 0.50 -0.10\n")
        out = parse_gs2_output(tmp_path)
        assert out.chi_e == pytest.approx(0.50 * 0.8)

    def test_corrupt_file_unconverged(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        omega_file.write_text("corrupt\n")
        out = parse_gs2_output(tmp_path)
        assert not out.converged

    def test_insufficient_columns_unconverged(self, tmp_path):
        omega_file = tmp_path / "gs2.omega"
        omega_file.write_text("0.3 0.2\n")  # only 2 values
        out = parse_gs2_output(tmp_path)
        assert not out.converged


# ── GS2Solver ────────────────────────────────────────────────────────


class TestGS2Solver:
    def test_is_available_false(self):
        solver = GS2Solver(binary="nonexistent_gs2_xyz")
        assert not solver.is_available()

    def test_prepare_input_creates_gs2_in(self, tmp_path, default_params):
        solver = GS2Solver(work_dir=tmp_path)
        result = solver.prepare_input(default_params)
        assert (result / "gs2.in").exists()
        content = (result / "gs2.in").read_text()
        assert "&theta_grid_eik_knobs" in content

    def test_prepare_input_auto_tmpdir(self, default_params):
        solver = GS2Solver()
        result = solver.prepare_input(default_params)
        assert (result / "gs2.in").exists()

    def test_run_unavailable_unconverged(self, tmp_path, default_params):
        solver = GS2Solver(binary="nonexistent_gs2_xyz", work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged

    @patch("scpn_fusion.core.gk_gs2.shutil.which", return_value="/usr/bin/gs2")
    @patch("scpn_fusion.core.gk_gs2.subprocess.run")
    def test_run_parses_output(self, mock_run, mock_which, tmp_path, default_params):
        solver = GS2Solver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        (inp / "gs2.omega").write_text("0.3 0.18 -0.35\n")
        out = solver.run(inp)
        mock_run.assert_called_once()
        assert out.converged
        assert out.chi_i == pytest.approx(0.18)

    @patch("scpn_fusion.core.gk_gs2.shutil.which", return_value="/usr/bin/gs2")
    @patch(
        "scpn_fusion.core.gk_gs2.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="gs2", timeout=30),
    )
    def test_run_timeout(self, mock_run, mock_which, tmp_path, default_params):
        solver = GS2Solver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged

    def test_run_from_params(self, default_params):
        solver = GS2Solver(binary="nonexistent_gs2_xyz")
        out = solver.run_from_params(default_params)
        assert not out.converged
