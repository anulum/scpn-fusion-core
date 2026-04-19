# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GENE Solver Tests
from __future__ import annotations

from unittest.mock import patch

import pytest

from scpn_fusion.core.gk_gene import (
    GENESolver,
    generate_gene_input,
    parse_gene_output,
)
from scpn_fusion.core.gk_interface import GKLocalParams


@pytest.fixture()
def default_params():
    return GKLocalParams(R_L_Ti=6.0, R_L_Te=6.0, R_L_ne=2.0, q=1.4, s_hat=0.8)


# ── generate_gene_input ─────────────────────────────────────────────


class TestGenerateGeneInput:
    def test_contains_namelist_sections(self, default_params):
        text = generate_gene_input(default_params)
        for section in ("&parallelization", "&box", "&general", "&geometry", "&species"):
            assert section in text

    def test_geometry_values(self, default_params):
        text = generate_gene_input(default_params)
        assert f"q0 = {default_params.q:.6f}" in text
        assert f"shat = {default_params.s_hat:.6f}" in text
        assert f"trpeps = {default_params.epsilon:.6f}" in text

    def test_species_gradients(self, default_params):
        text = generate_gene_input(default_params)
        assert f"omt = {default_params.R_L_Ti:.6f}" in text
        assert f"omt = {default_params.R_L_Te:.6f}" in text
        assert f"omn = {default_params.R_L_ne:.6f}" in text

    def test_small_a_clamped(self):
        params = GKLocalParams(R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, a=0.0)
        text = generate_gene_input(params)
        assert "major_R" in text

    def test_nonlinear_false(self, default_params):
        text = generate_gene_input(default_params)
        assert "nonlinear = .false." in text

    def test_miller_geometry(self, default_params):
        text = generate_gene_input(default_params)
        assert "magn_geometry = 'miller'" in text
        assert f"kappa = {default_params.kappa:.6f}" in text
        assert f"delta = {default_params.delta:.6f}" in text


# ── parse_gene_output ────────────────────────────────────────────────


class TestParseGeneOutput:
    def test_no_nrg_files_unconverged(self, tmp_path):
        out = parse_gene_output(tmp_path)
        assert not out.converged

    def test_valid_nrg_itg(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        # time, gamma, omega_r
        nrg.write_text("0.0 0.20 -0.40\n1.0 0.18 -0.38\n")
        out = parse_gene_output(tmp_path)
        assert out.converged
        assert out.gamma[0] == pytest.approx(0.18)  # last row
        assert out.omega_r[0] == pytest.approx(-0.38)
        assert out.dominant_mode == "ITG"

    def test_valid_nrg_tem(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("0.0 0.10 0.30\n")
        out = parse_gene_output(tmp_path)
        assert out.dominant_mode == "TEM"

    def test_uses_last_nrg_file(self, tmp_path):
        (tmp_path / "nrg_0001").write_text("0.0 0.10 -0.20\n")
        (tmp_path / "nrg_0002").write_text("0.0 0.30 -0.60\n")
        out = parse_gene_output(tmp_path)
        assert out.gamma[0] == pytest.approx(0.30)

    def test_single_row_nrg(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("0.5 0.12 -0.25\n")
        out = parse_gene_output(tmp_path)
        assert out.converged
        assert out.chi_i == pytest.approx(0.12)

    def test_negative_gamma_clamped(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("0.0 -0.05 0.10\n")
        out = parse_gene_output(tmp_path)
        assert out.chi_i == 0.0
        assert out.chi_e == 0.0
        assert out.D_e == 0.0

    def test_corrupt_nrg_unconverged(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("garbage data here\n")
        out = parse_gene_output(tmp_path)
        assert not out.converged

    def test_d_e_computed(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("0.0 1.0 -0.5\n")
        out = parse_gene_output(tmp_path)
        assert out.D_e == pytest.approx(0.1)  # gamma * 0.1

    def test_narrow_nrg_few_columns(self, tmp_path):
        nrg = tmp_path / "nrg_0001"
        nrg.write_text("0.0\n")  # only 1 column -> IndexError -> unconverged
        out = parse_gene_output(tmp_path)
        assert not out.converged


# ── GENESolver ───────────────────────────────────────────────────────


class TestGENESolver:
    def test_is_available_false(self):
        solver = GENESolver(binary="nonexistent_gene_xyz")
        assert not solver.is_available()

    def test_prepare_input_writes_parameters_file(self, tmp_path, default_params):
        solver = GENESolver(work_dir=tmp_path)
        result_path = solver.prepare_input(default_params)
        assert (result_path / "parameters").exists()
        content = (result_path / "parameters").read_text()
        assert "&geometry" in content

    def test_prepare_input_auto_tmpdir(self, default_params):
        solver = GENESolver()
        result_path = solver.prepare_input(default_params)
        assert (result_path / "parameters").exists()

    def test_run_unavailable_returns_unconverged(self, tmp_path, default_params):
        solver = GENESolver(binary="nonexistent_gene_xyz", work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged

    @patch("scpn_fusion.core.gk_gene.shutil.which", return_value="/usr/bin/gene")
    @patch("scpn_fusion.core.gk_gene.subprocess.run")
    def test_run_success(self, mock_run, mock_which, tmp_path, default_params):
        solver = GENESolver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        (inp / "nrg_0001").write_text("0.0 0.22 -0.44\n")
        out = solver.run(inp)
        mock_run.assert_called_once()
        assert out.converged
        assert out.chi_i == pytest.approx(0.22)

    @patch("scpn_fusion.core.gk_gene.shutil.which", return_value="/usr/bin/gene")
    @patch("scpn_fusion.core.gk_gene.subprocess.run", side_effect=FileNotFoundError)
    def test_run_file_not_found(self, mock_run, mock_which, tmp_path, default_params):
        solver = GENESolver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged
