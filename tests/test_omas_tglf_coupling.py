# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for P2.3: OMAS/TGLF Live Coupling."""

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.io.imas_connector import (
    HAS_OMAS,
    geqdsk_to_imas_equilibrium,
    ids_to_omas_core_profiles,
    ids_to_omas_equilibrium,
    omas_core_profiles_to_ids,
    omas_equilibrium_to_ids,
)
from scpn_fusion.io import imas_connector_omas
from scpn_fusion.core.eqdsk import GEqdsk
from scpn_fusion.core.tglf_interface import (
    TGLFInputDeck,
    TGLFOutput,
    TGLFBenchmark,
    run_tglf_binary,
    write_tglf_input_file,
    _parse_tglf_run_output,
)


# ── OMAS connector tests ──────────────────────────────────────────────


class TestOMASConnector:
    """Test OMAS conversion functions (graceful skip if omas not installed)."""

    def test_has_omas_is_boolean(self) -> None:
        assert isinstance(HAS_OMAS, bool)

    @pytest.mark.skipif(not HAS_OMAS, reason="omas not installed")
    def test_ids_to_omas_roundtrip(self) -> None:
        """If omas is installed, test full roundtrip."""
        eq = _make_test_geqdsk()
        ids_dict = geqdsk_to_imas_equilibrium(eq, time_s=0.0, shot=1, run=0)
        try:
            ods = ids_to_omas_equilibrium(ids_dict)
        except LookupError as exc:
            pytest.skip(f"IMAS schema incompatibility: {exc}")
        ids_back = omas_equilibrium_to_ids(ods)
        assert len(ids_back["time_slice"]) == 1
        gq = ids_back["time_slice"][0]["global_quantities"]
        assert abs(gq["ip"] - eq.current) < 1e-6

    def test_ids_to_omas_raises_without_omas(self) -> None:
        """When omas is not installed, should raise ImportError."""
        if HAS_OMAS:
            pytest.skip("omas is installed, can't test missing import")
        eq = _make_test_geqdsk()
        ids_dict = geqdsk_to_imas_equilibrium(eq, time_s=0.0, shot=1, run=0)
        with pytest.raises(ImportError, match="omas"):
            ids_to_omas_equilibrium(ids_dict)

    def test_omas_to_ids_raises_without_omas(self) -> None:
        """When omas is not installed, should raise ImportError."""
        if HAS_OMAS:
            pytest.skip("omas is installed, can't test missing import")
        with pytest.raises(ImportError, match="omas"):
            omas_equilibrium_to_ids({})

    def test_core_profiles_to_omas_raises_without_omas(self) -> None:
        """The core_profiles bridge preserves the optional OMAS dependency boundary."""
        if HAS_OMAS:
            pytest.skip("omas is installed, can't test missing import")
        with pytest.raises(ImportError, match="omas"):
            ids_to_omas_core_profiles(_make_core_profiles_ids())

    def test_omas_to_core_profiles_raises_without_omas(self) -> None:
        """The reverse core_profiles bridge also fails closed without OMAS."""
        if HAS_OMAS:
            pytest.skip("omas is installed, can't test missing import")
        with pytest.raises(ImportError, match="omas"):
            omas_core_profiles_to_ids({})

    def test_core_profiles_ods_mapping_roundtrip(self) -> None:
        """The shared core_profiles bridge maps IMAS arrays through ODS paths."""
        ids = _make_core_profiles_ids()
        ods = imas_connector_omas._populate_omas_core_profiles({}, ids)
        roundtrip = imas_connector_omas._omas_core_profiles_to_ids_unchecked(ods)

        assert roundtrip["ids_properties"]["homogeneous_time"] == 1
        assert roundtrip["time"] == [5.0]
        profile = roundtrip["profiles_1d"][0]
        assert profile["grid"]["rho_tor_norm"] == [0.0, 0.5, 1.0]
        assert profile["electrons"]["temperature"] == [1000.0, 700.0, 100.0]
        assert profile["electrons"]["density"] == [1.0e20, 8.0e19, 2.0e19]


# ── TGLF input deck writer ────────────────────────────────────────────


class TestTGLFInputWriter:
    def test_writes_input_file(self) -> None:
        deck = TGLFInputDeck()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tglf_input_file(deck, tmpdir)
            assert path.exists()
            assert path.name == "input.tglf"
            content = path.read_text()
            assert "SIGN_BT" in content
            assert "Q_LOC" in content

    def test_input_contains_deck_values(self) -> None:
        deck = TGLFInputDeck(rho=0.5, q=2.5, R_LTi=8.0, kappa=1.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tglf_input_file(deck, tmpdir)
            content = path.read_text()
            assert "Q_LOC = 2.500000" in content
            assert "RLTS_1 = 8.000000" in content
            assert "KAPPA_LOC = 1.800000" in content

    def test_creates_output_dir(self) -> None:
        deck = TGLFInputDeck()
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "nested" / "dir"
            path = write_tglf_input_file(deck, subdir)
            assert path.exists()
            assert subdir.exists()


# ── TGLF output parser ────────────────────────────────────────────────


class TestTGLFOutputParser:
    def test_parse_key_value_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.tglf.run"
            out_path.write_text("CHI_I = 1.5\nCHI_E = 0.8\nGAMMA_MAX = 0.12\nOTHER = ignored\n")
            result = _parse_tglf_run_output(out_path, rho=0.5)
            assert result.rho == pytest.approx(0.5)
            assert result.chi_i == pytest.approx(1.5)
            assert result.chi_e == pytest.approx(0.8)
            assert result.gamma_max == pytest.approx(0.12)

    def test_parse_chieff_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.tglf.run"
            out_path.write_text("CHIEFF_I = 2.0\nCHIEFF_E = 1.0\n")
            result = _parse_tglf_run_output(out_path, rho=0.3)
            assert result.chi_i == pytest.approx(2.0)
            assert result.chi_e == pytest.approx(1.0)

    def test_parse_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.tglf.run"
            out_path.write_text("")
            result = _parse_tglf_run_output(out_path, rho=0.5)
            assert result.chi_i == 0.0
            assert result.chi_e == 0.0

    def test_parse_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.tglf.run"
            out_path.write_text("# comment\nno_equals_here\nCHI_I = notanumber\nCHI_E = 1.5\n")
            result = _parse_tglf_run_output(out_path, rho=0.5)
            assert result.chi_i == 0.0  # couldn't parse
            assert result.chi_e == pytest.approx(1.5)

    def test_parse_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.tglf.run"
            out_path.write_text("CHI_I = NaN\nCHI_E = Infinity\nGAMMA_MAX = -Infinity\n")
            result = _parse_tglf_run_output(out_path, rho=0.5)
            assert result.chi_i == 0.0
            assert result.chi_e == 0.0
            assert result.gamma_max == 0.0


# ── TGLF binary execution (mocked) ────────────────────────────────────


class TestTGLFBinaryExecution:
    def test_run_tglf_binary_not_found(self) -> None:
        deck = TGLFInputDeck()
        with pytest.raises(FileNotFoundError, match="TGLF binary not found"):
            run_tglf_binary(deck, "/nonexistent/path/to/tglf")

    @pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
    def test_run_tglf_binary_rejects_invalid_timeout(self, timeout_s: float) -> None:
        deck = TGLFInputDeck()
        with pytest.raises(ValueError, match="timeout_s must be finite and > 0."):
            run_tglf_binary(
                deck,
                "/nonexistent/path/to/tglf",
                timeout_s=timeout_s,
            )

    @pytest.mark.parametrize("max_retries", [-1, 11, 1.5, True])
    def test_run_tglf_binary_rejects_invalid_max_retries(self, max_retries: Any) -> None:
        deck = TGLFInputDeck()
        with pytest.raises(ValueError, match="max_retries must be an integer in \\[0, 10\\]."):
            run_tglf_binary(
                deck,
                "/nonexistent/path/to/tglf",
                max_retries=max_retries,
            )


# ── TGLFBenchmark comparison ──────────────────────────────────────────


class TestTGLFBenchmark:
    def test_compare_identical(self) -> None:
        benchmark = TGLFBenchmark()
        rho = np.linspace(0.1, 0.9, 7)
        chi_i = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        chi_e = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        tglf_outputs = [
            TGLFOutput(rho=r, chi_i=ci, chi_e=ce) for r, ci, ce in zip(rho, chi_i, chi_e)
        ]
        result = benchmark.compare(chi_i, chi_e, rho, tglf_outputs)
        assert result.rms_error_chi_i < 1e-10
        assert result.rms_error_chi_e < 1e-10

    def test_compare_different(self) -> None:
        benchmark = TGLFBenchmark()
        rho = np.linspace(0.1, 0.9, 7)
        our_chi_i = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        our_chi_e = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        tglf_outputs = [
            TGLFOutput(rho=r, chi_i=ci * 1.1, chi_e=ce * 1.2)
            for r, ci, ce in zip(rho, our_chi_i, our_chi_e)
        ]
        result = benchmark.compare(our_chi_i, our_chi_e, rho, tglf_outputs)
        assert result.rms_error_chi_i > 0.0
        assert result.rms_error_chi_e > 0.0


# ── pyproject.toml optional dependency ─────────────────────────────────


class TestPyprojectConfig:
    def test_full_physics_in_pyproject(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        content = pyproject_path.read_text()
        assert "full-physics" in content
        assert "omas" in content


# ── Helper ─────────────────────────────────────────────────────────────


def _make_core_profiles_ids() -> dict[str, object]:
    """Create a minimal IMAS core_profiles payload for OMAS bridge tests."""
    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": "unit-test core_profiles",
        },
        "time": [5.0],
        "profiles_1d": [
            {
                "time": 5.0,
                "grid": {
                    "rho_tor_norm": [0.0, 0.5, 1.0],
                },
                "electrons": {
                    "temperature": [1000.0, 700.0, 100.0],
                    "density": [1.0e20, 8.0e19, 2.0e19],
                },
            }
        ],
    }


def _make_test_geqdsk() -> GEqdsk:
    """Create a minimal GEqdsk for testing."""
    nw, nh = 5, 5
    return GEqdsk(
        description="test",
        nw=nw,
        nh=nh,
        rdim=4.0,
        zdim=4.0,
        rcentr=6.2,
        rleft=4.2,
        zmid=0.0,
        rmaxis=6.2,
        zmaxis=0.0,
        simag=0.1,
        sibry=0.0,
        bcentr=5.3,
        current=15e6,
        fpol=np.ones(nw),
        pres=np.linspace(1e4, 0, nw),
        ffprime=np.zeros(nw),
        pprime=np.zeros(nw),
        qpsi=np.linspace(1.0, 3.0, nw),
        psirz=np.random.default_rng(42).random((nh, nw)),
        rbdry=np.array([6.2, 7.0, 6.2, 5.4]),
        zbdry=np.array([1.0, 0.0, -1.0, 0.0]),
        rlim=np.array([]),
        zlim=np.array([]),
    )
