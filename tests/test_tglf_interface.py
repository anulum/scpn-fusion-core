# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TGLF Interface Tests
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the TGLF interface: dataclasses, input deck generation,
output parsing, and benchmark comparison utilities.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import scpn_fusion.core.tglf_interface as tglf_mod
from scpn_fusion.core.tglf_interface import (
    TGLFInputDeck,
    TGLFOutput,
    TGLFComparisonResult,
    TGLFBenchmark,
    REFERENCE_CASES,
    generate_input_deck,
    parse_tglf_output,
    write_tglf_input_file,
    write_reference_data,
)


# ── 1. TGLFInputDeck Dataclass ───────────────────────────────────────

class TestTGLFInputDeck:

    def test_input_deck_defaults(self) -> None:
        """TGLFInputDeck() should have sensible default values."""
        deck = TGLFInputDeck()
        assert deck.rho == 0.5
        assert deck.s_hat == 1.0
        assert deck.q == 1.5
        assert deck.kappa == 1.7
        assert deck.delta == 0.3
        assert deck.R_LTi == 6.0
        assert deck.R_LTe == 6.0
        assert deck.R_Lne == 2.0
        assert deck.Z_eff == 1.5
        assert deck.T_e_keV == 10.0
        assert deck.T_i_keV == 10.0
        assert deck.n_e_19 == 8.0
        assert deck.R_major == 6.2
        assert deck.a_minor == 2.0
        assert deck.B_toroidal == 5.3

    def test_input_deck_custom(self) -> None:
        """TGLFInputDeck accepts custom ITER-like parameters."""
        deck = TGLFInputDeck(
            rho=0.4,
            R_major=6.2,
            a_minor=2.0,
            B_toroidal=5.3,
            T_e_keV=15.0,
            T_i_keV=14.0,
            n_e_19=10.0,
            R_LTi=8.0,
            R_LTe=7.5,
            kappa=1.85,
            delta=0.35,
        )
        assert deck.rho == 0.4
        assert deck.R_major == 6.2
        assert deck.T_e_keV == 15.0
        assert deck.T_i_keV == 14.0
        assert deck.n_e_19 == 10.0
        assert deck.kappa == 1.85

    def test_input_deck_iter_params_reasonable(self) -> None:
        """ITER-specific parameter ranges should be within physical bounds."""
        deck = TGLFInputDeck()
        # ITER: R_major ~ 6.2 m, B ~ 5.3 T, a ~ 2.0 m
        assert 5.0 < deck.R_major < 8.0
        assert 4.0 < deck.B_toroidal < 7.0
        assert 1.0 < deck.a_minor < 3.0
        # ITER-like parameters
        assert deck.kappa > 1.0  # elongation must be > 1
        assert 0.0 <= deck.delta < 1.0  # triangularity
        assert deck.Z_eff >= 1.0  # Z_eff cannot be < 1
        assert deck.q > 0  # safety factor positive
        assert deck.beta_e > 0  # finite beta

    def test_input_deck_has_all_tglf_fields(self) -> None:
        """The dataclass should expose all fields needed for TGLF input."""
        deck = TGLFInputDeck()
        field_names = {f.name for f in fields(deck)}
        expected = {
            "rho", "s_hat", "q", "alpha_mhd", "kappa", "delta",
            "R_LTi", "R_LTe", "R_Lne", "R_Lni", "beta_e", "Z_eff",
            "T_e_keV", "T_i_keV", "n_e_19", "R_major", "a_minor", "B_toroidal",
        }
        assert expected.issubset(field_names)


# ── 2. TGLFOutput Dataclass ──────────────────────────────────────────

class TestTGLFOutput:

    def test_output_dataclass_defaults(self) -> None:
        """TGLFOutput has zero defaults for chi and flux values."""
        out = TGLFOutput()
        assert out.rho == 0.5
        assert out.chi_i == 0.0
        assert out.chi_e == 0.0
        assert out.gamma_max == 0.0
        assert out.q_i == 0.0
        assert out.q_e == 0.0

    def test_output_custom_values(self) -> None:
        """TGLFOutput stores known transport values correctly."""
        out = TGLFOutput(
            rho=0.5, chi_i=2.5, chi_e=1.8, gamma_max=0.15, q_i=3.0, q_e=2.1
        )
        assert out.chi_i == 2.5
        assert out.chi_e == 1.8
        assert out.gamma_max == 0.15
        assert out.q_i == 3.0
        assert out.q_e == 2.1

    def test_output_fields(self) -> None:
        """Output dataclass has the expected field set."""
        field_names = {f.name for f in fields(TGLFOutput)}
        assert {"rho", "chi_i", "chi_e", "gamma_max", "q_i", "q_e"} == field_names


# ── 3. TGLFComparisonResult ──────────────────────────────────────────

class TestTGLFComparisonResult:

    def test_comparison_result_defaults(self) -> None:
        """TGLFComparisonResult defaults to empty lists and zero errors."""
        result = TGLFComparisonResult()
        assert result.case_name == ""
        assert result.rho_points == []
        assert result.our_chi_i == []
        assert result.tglf_chi_i == []
        assert result.rms_error_chi_i == 0.0
        assert result.correlation_chi_i == 0.0

    def test_comparison_result_custom(self) -> None:
        """TGLFComparisonResult stores comparison data."""
        result = TGLFComparisonResult(
            case_name="ITG test",
            rho_points=[0.3, 0.5, 0.7],
            our_chi_i=[1.0, 2.0, 3.0],
            tglf_chi_i=[1.1, 2.1, 2.9],
            rms_error_chi_i=0.12,
            correlation_chi_i=0.98,
        )
        assert result.case_name == "ITG test"
        assert len(result.rho_points) == 3
        assert result.rms_error_chi_i == pytest.approx(0.12)
        assert result.correlation_chi_i == pytest.approx(0.98)


# ── 4. Input Deck Generation from TransportSolver ────────────────────

class TestGenerateInputDeck:

    @pytest.fixture
    def mock_solver(self, tmp_path: Path):
        """Create a minimal TransportSolver for testing generate_input_deck."""
        config = {
            "reactor_name": "TGLF-Test",
            "grid_resolution": [20, 20],
            "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
            "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
            "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
            "solver": {
                "max_iterations": 10,
                "convergence_threshold": 1e-4,
                "relaxation_factor": 0.1,
            },
        }
        cfg_path = tmp_path / "tglf_test.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")

        from scpn_fusion.core.integrated_transport_solver import TransportSolver

        ts = TransportSolver(str(cfg_path))
        ts.Ti = 10.0 * (1 - ts.rho ** 2)
        ts.Te = 10.0 * (1 - ts.rho ** 2)
        ts.ne = 8.0 * (1 - ts.rho ** 2) ** 0.5
        return ts

    def test_generate_input_deck_returns_deck(self, mock_solver) -> None:
        """generate_input_deck returns a TGLFInputDeck instance."""
        deck = generate_input_deck(mock_solver, rho_idx=25)
        assert isinstance(deck, TGLFInputDeck)

    def test_generate_input_deck_rho_matches(self, mock_solver) -> None:
        """Generated deck rho should match the solver grid point."""
        idx = 25
        deck = generate_input_deck(mock_solver, rho_idx=idx)
        assert deck.rho == pytest.approx(mock_solver.rho[idx])

    def test_generate_input_deck_temperatures_match(self, mock_solver) -> None:
        """Generated deck temperatures should come from the solver profiles."""
        idx = 10
        deck = generate_input_deck(mock_solver, idx)
        assert deck.T_i_keV == pytest.approx(mock_solver.Ti[idx])
        assert deck.T_e_keV == pytest.approx(mock_solver.Te[idx])
        assert deck.n_e_19 == pytest.approx(mock_solver.ne[idx])

    def test_generate_input_deck_gradients_finite(self, mock_solver) -> None:
        """Gradient scale lengths (R/L_T, R/L_n) should be finite."""
        deck = generate_input_deck(mock_solver, rho_idx=25)
        assert np.isfinite(deck.R_LTi)
        assert np.isfinite(deck.R_LTe)
        assert np.isfinite(deck.R_Lne)

    def test_generate_input_deck_midradius(self, mock_solver) -> None:
        """At mid-radius, gradients should be non-zero for peaked profiles."""
        deck = generate_input_deck(mock_solver, rho_idx=25)
        # Parabolic profiles have non-zero gradients at mid-radius
        assert deck.R_LTi != 0.0 or deck.rho < 0.01


# ── 5. Output Parsing ────────────────────────────────────────────────

class TestOutputParsing:

    def test_parse_tglf_output_from_json(self, tmp_path: Path) -> None:
        """parse_tglf_output reads JSON files and returns TGLFOutput list."""
        data = {
            "rho_points": [0.3, 0.5, 0.7],
            "chi_i": [1.0, 2.5, 4.0],
            "chi_e": [0.8, 1.5, 2.2],
            "gamma_max": [0.05, 0.12, 0.20],
            "q_i": [1.0, 2.0, 3.0],
            "q_e": [0.5, 1.0, 1.5],
        }
        (tmp_path / "test_output.json").write_text(json.dumps(data), encoding="utf-8")
        outputs = parse_tglf_output(tmp_path)
        assert len(outputs) == 3
        assert outputs[0].rho == 0.3
        assert outputs[1].chi_i == 2.5
        assert outputs[2].chi_e == 2.2

    def test_parse_empty_directory(self, tmp_path: Path) -> None:
        """parse_tglf_output on empty directory returns empty list."""
        outputs = parse_tglf_output(tmp_path)
        assert outputs == []

    def test_parse_tglf_output_accepts_scalar_numeric_payload(self, tmp_path: Path) -> None:
        """Scalar rho/chi payload should parse as a single-point output."""
        data = {
            "rho": 0.55,
            "chi_i": 1.8,
            "chi_e": 1.1,
            "gamma_max": 0.14,
            "q_i": 2.3,
            "q_e": 1.7,
        }
        (tmp_path / "scalar_output.json").write_text(json.dumps(data), encoding="utf-8")
        outputs = parse_tglf_output(tmp_path)
        assert len(outputs) == 1
        assert outputs[0].rho == pytest.approx(0.55)
        assert outputs[0].chi_i == pytest.approx(1.8)
        assert outputs[0].chi_e == pytest.approx(1.1)

    def test_parse_tglf_output_skips_non_object_payloads(self, tmp_path: Path) -> None:
        """Top-level JSON arrays are invalid for this parser and should be skipped."""
        (tmp_path / "bad_payload.json").write_text(
            json.dumps([{"rho": 0.5, "chi_i": 1.0}]),
            encoding="utf-8",
        )
        outputs = parse_tglf_output(tmp_path)
        assert outputs == []

    def test_parse_tglf_output_clamps_non_finite_numbers(self, tmp_path: Path) -> None:
        """NaN/inf inputs are coerced to finite defaults."""
        payload = (
            '{"rho_points":[NaN,0.7],'
            '"chi_i":[Infinity,2.4],'
            '"chi_e":[1.2,-Infinity],'
            '"gamma_max":[NaN,0.2]}'
        )
        (tmp_path / "non_finite.json").write_text(payload, encoding="utf-8")
        outputs = parse_tglf_output(tmp_path)
        assert len(outputs) == 2
        assert outputs[0].rho == pytest.approx(0.5)
        assert outputs[0].chi_i == pytest.approx(0.0)
        assert outputs[0].chi_e == pytest.approx(1.2)
        assert outputs[0].gamma_max == pytest.approx(0.0)
        assert outputs[1].rho == pytest.approx(0.7)
        assert outputs[1].chi_i == pytest.approx(2.4)
        assert outputs[1].chi_e == pytest.approx(0.0)
        assert outputs[1].gamma_max == pytest.approx(0.2)

    def test_parse_tglf_output_caps_oversized_vectors(self, tmp_path: Path) -> None:
        """Oversized vector payloads should be capped to bounded output length."""
        cap = tglf_mod._TGLF_MAX_PARSED_VECTOR_LENGTH
        n_points = cap + 64
        payload = {
            "rho_points": [float(i) for i in range(n_points)],
            "chi_i": [float(i) for i in range(n_points)],
            "chi_e": [float(i) * 0.5 for i in range(n_points)],
            "gamma_max": [0.1 for _ in range(n_points)],
            "q_i": [1.0 for _ in range(n_points)],
            "q_e": [0.5 for _ in range(n_points)],
        }
        (tmp_path / "oversized.json").write_text(json.dumps(payload), encoding="utf-8")
        outputs = parse_tglf_output(tmp_path)
        assert len(outputs) == cap
        assert outputs[0].rho == pytest.approx(0.0)
        assert outputs[-1].rho == pytest.approx(float(cap - 1))
        assert outputs[-1].chi_i == pytest.approx(float(cap - 1))


# ── 6. Write Input File ──────────────────────────────────────────────

class TestWriteInputFile:

    def test_write_tglf_input_file(self, tmp_path: Path) -> None:
        """write_tglf_input_file creates a readable input.tglf file."""
        deck = TGLFInputDeck(rho=0.5, q=1.5, kappa=1.7, R_LTi=6.0)
        path = write_tglf_input_file(deck, tmp_path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "Q_LOC" in text
        assert "KAPPA_LOC" in text
        assert "RLTS_1" in text  # R_LTi is written as RLTS_1

    def test_write_tglf_input_file_values_present(self, tmp_path: Path) -> None:
        """Written file should contain the numerical values from the deck."""
        deck = TGLFInputDeck(rho=0.4, q=2.0, kappa=1.85, R_LTi=8.5)
        path = write_tglf_input_file(deck, tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "2.000000" in text  # q value
        assert "1.850000" in text  # kappa value


# ── 7. TGLFBenchmark Comparison ──────────────────────────────────────

class TestTGLFBenchmark:

    def test_benchmark_compare(self) -> None:
        """TGLFBenchmark.compare produces a TGLFComparisonResult."""
        benchmark = TGLFBenchmark()
        rho_grid = np.linspace(0, 1, 50)
        our_chi_i = 3.0 * rho_grid ** 2
        our_chi_e = 2.0 * rho_grid ** 2

        tglf_outputs = [
            TGLFOutput(rho=0.3, chi_i=0.27, chi_e=0.18),
            TGLFOutput(rho=0.5, chi_i=0.75, chi_e=0.50),
            TGLFOutput(rho=0.7, chi_i=1.47, chi_e=0.98),
        ]
        result = benchmark.compare(our_chi_i, our_chi_e, rho_grid, tglf_outputs)
        assert isinstance(result, TGLFComparisonResult)
        assert len(result.rho_points) == 3
        assert np.isfinite(result.rms_error_chi_i)
        assert np.isfinite(result.rms_error_chi_e)

    def test_benchmark_compare_empty_reference_outputs(self) -> None:
        """Empty TGLF outputs should return default zeroed comparison metrics."""
        benchmark = TGLFBenchmark()
        rho_grid = np.linspace(0, 1, 10)
        our_chi_i = np.linspace(0.1, 1.0, 10)
        our_chi_e = np.linspace(0.2, 0.8, 10)
        result = benchmark.compare(our_chi_i, our_chi_e, rho_grid, [])
        assert isinstance(result, TGLFComparisonResult)
        assert result.rho_points == []
        assert result.rms_error_chi_i == 0.0
        assert result.rms_error_chi_e == 0.0

    def test_benchmark_perfect_match(self) -> None:
        """When our values match TGLF exactly, RMS error should be very small.

        Note: np.interp on a 50-point grid introduces small interpolation
        error (~1e-4) for quadratic functions, so we allow tolerance < 1e-3.
        """
        benchmark = TGLFBenchmark()
        rho_grid = np.linspace(0, 1, 50)
        chi = 2.0 * rho_grid ** 2

        tglf_outputs = [
            TGLFOutput(rho=r, chi_i=2.0 * r ** 2, chi_e=2.0 * r ** 2)
            for r in [0.2, 0.4, 0.6, 0.8]
        ]
        result = benchmark.compare(chi, chi, rho_grid, tglf_outputs)
        # Linear interpolation of a quadratic on a 50-point grid gives ~1e-4 error
        assert result.rms_error_chi_i < 1e-3
        assert result.rms_error_chi_e < 1e-3
        # Correlation should be nearly perfect
        assert result.correlation_chi_i > 0.999
        assert result.correlation_chi_e > 0.999

    def test_benchmark_generate_markdown_table(self) -> None:
        """generate_comparison_table returns a markdown string."""
        benchmark = TGLFBenchmark()
        result = TGLFComparisonResult(
            case_name="Test",
            rms_error_chi_i=0.5,
            rms_error_chi_e=0.3,
            correlation_chi_i=0.95,
            correlation_chi_e=0.92,
            max_rel_error_chi_i=0.2,
            max_rel_error_chi_e=0.15,
        )
        table = benchmark.generate_comparison_table([result])
        assert "Test" in table
        assert "0.500" in table  # RMS chi_i
        assert "|" in table

    def test_benchmark_generate_latex_table(self) -> None:
        """generate_latex_table returns a LaTeX string."""
        benchmark = TGLFBenchmark()
        result = TGLFComparisonResult(case_name="ITG")
        latex = benchmark.generate_latex_table([result])
        assert "\\begin{table}" in latex
        assert "ITG" in latex
        assert "\\end{table}" in latex


# ── 8. Reference Data ────────────────────────────────────────────────

class TestReferenceData:

    def test_reference_cases_exist(self) -> None:
        """REFERENCE_CASES dict has the expected entries."""
        assert "ITG-dominated" in REFERENCE_CASES
        assert "TEM-dominated" in REFERENCE_CASES
        assert "ETG-dominated" in REFERENCE_CASES

    def test_reference_cases_structure(self) -> None:
        """Each reference case has rho_points, chi_i, chi_e, gamma_max."""
        for name, data in REFERENCE_CASES.items():
            assert "rho_points" in data, f"Missing rho_points in {name}"
            assert "chi_i" in data, f"Missing chi_i in {name}"
            assert "chi_e" in data, f"Missing chi_e in {name}"
            assert "gamma_max" in data, f"Missing gamma_max in {name}"
            assert len(data["rho_points"]) == len(data["chi_i"])

    def test_write_reference_data(self, tmp_path: Path) -> None:
        """write_reference_data writes JSON files for each reference case."""
        write_reference_data(tmp_path)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == len(REFERENCE_CASES)
        # Check that files are valid JSON
        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            assert "case_name" in data
            assert "rho_points" in data
