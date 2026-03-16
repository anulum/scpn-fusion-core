# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CGYRO Solver Tests
from __future__ import annotations

import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from scpn_fusion.core.gk_cgyro import (
    CGYROSolver,
    generate_cgyro_input,
    parse_cgyro_output,
)
from scpn_fusion.core.gk_interface import GKLocalParams


@pytest.fixture()
def default_params():
    return GKLocalParams(R_L_Ti=6.0, R_L_Te=6.0, R_L_ne=2.0, q=1.4, s_hat=0.8)


# ── generate_cgyro_input ────────────────────────────────────────────


class TestGenerateCgyroInput:
    def test_contains_required_tokens(self, default_params):
        text = generate_cgyro_input(default_params)
        for token in ("EQUILIBRIUM_MODEL", "RMAJ", "RMIN", "Q=", "S=", "N_SPECIES"):
            assert token in text

    def test_rmin_matches_rho(self, default_params):
        text = generate_cgyro_input(default_params)
        for line in text.splitlines():
            if line.startswith("RMIN="):
                assert float(line.split("=")[1]) == pytest.approx(default_params.rho)

    def test_rmaj_uses_R0_over_a(self, default_params):
        text = generate_cgyro_input(default_params)
        expected = default_params.R0 / default_params.a
        for line in text.splitlines():
            if line.startswith("RMAJ="):
                assert float(line.split("=")[1]) == pytest.approx(expected, rel=1e-4)

    def test_small_a_clamped(self):
        params = GKLocalParams(R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, a=0.0)
        text = generate_cgyro_input(params)
        for line in text.splitlines():
            if line.startswith("RMAJ="):
                val = float(line.split("=")[1])
                assert np.isfinite(val)

    def test_temperature_ratio_written(self, default_params):
        default_params = GKLocalParams(
            R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, Te_Ti=1.3
        )
        text = generate_cgyro_input(default_params)
        assert "1.300000" in text

    def test_gradients_appear(self, default_params):
        text = generate_cgyro_input(default_params)
        assert "DLNTDR_1=6.000000" in text
        assert "DLNTDR_2=6.000000" in text


# ── parse_cgyro_output ──────────────────────────────────────────────


class TestParseCgyroOutput:
    def test_missing_file_returns_unconverged(self, tmp_path):
        out = parse_cgyro_output(tmp_path)
        assert not out.converged
        assert out.chi_i == 0.0

    def test_valid_itg_file(self, tmp_path):
        freq = tmp_path / "out.cgyro.freq"
        freq.write_text("0.25 -0.5\n")
        out = parse_cgyro_output(tmp_path)
        assert out.converged
        assert out.gamma[0] == pytest.approx(0.25)
        assert out.omega_r[0] == pytest.approx(-0.5)
        assert out.dominant_mode == "ITG"  # omega_r < 0
        assert out.chi_i == pytest.approx(0.25)
        assert out.chi_e == pytest.approx(0.25 * 0.8)

    def test_positive_omega_gives_tem(self, tmp_path):
        freq = tmp_path / "out.cgyro.freq"
        freq.write_text("0.3 0.4\n")
        out = parse_cgyro_output(tmp_path)
        assert out.dominant_mode == "TEM"

    def test_negative_gamma_clamped_to_zero(self, tmp_path):
        freq = tmp_path / "out.cgyro.freq"
        freq.write_text("-0.1 0.2\n")
        out = parse_cgyro_output(tmp_path)
        assert out.chi_i == 0.0
        assert out.chi_e == 0.0

    def test_corrupt_file_returns_unconverged(self, tmp_path):
        freq = tmp_path / "out.cgyro.freq"
        freq.write_text("not a number\n")
        out = parse_cgyro_output(tmp_path)
        assert not out.converged

    def test_single_value_row_unconverged(self, tmp_path):
        freq = tmp_path / "out.cgyro.freq"
        freq.write_text("0.5\n")
        out = parse_cgyro_output(tmp_path)
        assert not out.converged


# ── CGYROSolver ─────────────────────────────────────────────────────


class TestCGYROSolver:
    def test_is_available_false_when_no_binary(self):
        solver = CGYROSolver(binary="nonexistent_cgyro_xyz")
        assert not solver.is_available()

    def test_prepare_input_creates_file(self, tmp_path, default_params):
        solver = CGYROSolver(work_dir=tmp_path)
        result_path = solver.prepare_input(default_params)
        assert (result_path / "input.cgyro").exists()
        content = (result_path / "input.cgyro").read_text()
        assert "EQUILIBRIUM_MODEL" in content

    def test_prepare_input_auto_tmpdir(self, default_params):
        solver = CGYROSolver()
        result_path = solver.prepare_input(default_params)
        assert (result_path / "input.cgyro").exists()

    def test_run_returns_unconverged_when_unavailable(self, tmp_path, default_params):
        solver = CGYROSolver(binary="nonexistent_cgyro_xyz", work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged

    @patch("scpn_fusion.core.gk_cgyro.shutil.which", return_value="/usr/bin/cgyro")
    @patch("scpn_fusion.core.gk_cgyro.subprocess.run")
    def test_run_calls_binary_and_parses(self, mock_run, mock_which, tmp_path, default_params):
        solver = CGYROSolver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        (inp / "out.cgyro.freq").write_text("0.15 -0.3\n")
        out = solver.run(inp)
        mock_run.assert_called_once()
        assert out.converged
        assert out.chi_i == pytest.approx(0.15)

    @patch("scpn_fusion.core.gk_cgyro.shutil.which", return_value="/usr/bin/cgyro")
    @patch(
        "scpn_fusion.core.gk_cgyro.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="cgyro", timeout=30),
    )
    def test_run_timeout_returns_unconverged(self, mock_run, mock_which, tmp_path, default_params):
        solver = CGYROSolver(work_dir=tmp_path)
        inp = solver.prepare_input(default_params)
        out = solver.run(inp)
        assert not out.converged

    def test_run_from_params_convenience(self, default_params):
        solver = CGYROSolver(binary="nonexistent_cgyro_xyz")
        out = solver.run_from_params(default_params)
        assert not out.converged
