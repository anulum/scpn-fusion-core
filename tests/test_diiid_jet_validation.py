# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — DIII-D/JET Validation Runner Tests
# ──────────────────────────────────────────────────────────────────────
"""
Regression tests for validation/run_diiid_jet_validation.py.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import validation.run_diiid_jet_validation as diiid_jet_mod
from scpn_fusion.core.eqdsk import read_geqdsk

ROOT = Path(__file__).resolve().parents[1]
DIIID_DIR = ROOT / "validation" / "reference_data" / "diiid"
SAMPLE_EQ = DIIID_DIR / "diiid_lmode_1MA.geqdsk"

pytestmark = pytest.mark.skipif(
    not SAMPLE_EQ.exists(),
    reason="DIII-D reference data not available",
)


class TestGSOperator:
    def test_rejects_axis_length_mismatch(self) -> None:
        psi = np.zeros((5, 5), dtype=np.float64)
        R = np.linspace(1.0, 2.0, 4)
        Z = np.linspace(-1.0, 1.0, 5)
        with pytest.raises(ValueError, match="axis lengths"):
            diiid_jet_mod.gs_operator(psi, R, Z)

    def test_rejects_non_monotonic_axis(self) -> None:
        psi = np.zeros((5, 5), dtype=np.float64)
        R = np.array([1.0, 1.2, 1.2, 1.6, 1.9], dtype=np.float64)
        Z = np.linspace(-1.0, 1.0, 5)
        with pytest.raises(ValueError, match="strictly increasing"):
            diiid_jet_mod.gs_operator(psi, R, Z)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_values(self, bad_value: float) -> None:
        psi = np.zeros((5, 5), dtype=np.float64)
        R = np.linspace(1.0, 2.0, 5)
        Z = np.linspace(-1.0, 1.0, 5)
        psi[2, 2] = bad_value
        with pytest.raises(ValueError, match="finite"):
            diiid_jet_mod.gs_operator(psi, R, Z)


class TestValidateFile:
    def test_validate_file_smoke(self) -> None:
        out = diiid_jet_mod.validate_file(SAMPLE_EQ, "DIII-D")
        assert out.file == SAMPLE_EQ.name
        assert out.device == "DIII-D"
        assert np.isfinite(out.psi_rmse_norm)
        assert np.isfinite(out.psi_relative_l2)
        assert np.isfinite(out.gs_residual_l2)

    def test_rejects_degenerate_psi_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eq = copy.deepcopy(read_geqdsk(SAMPLE_EQ))
        eq.sibry = eq.simag
        monkeypatch.setattr(diiid_jet_mod, "read_geqdsk", lambda _: eq)
        with pytest.raises(ValueError, match="degenerate psi range"):
            diiid_jet_mod.validate_file(SAMPLE_EQ, "DIII-D")

    def test_rejects_profile_length_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eq = copy.deepcopy(read_geqdsk(SAMPLE_EQ))
        eq.pprime = eq.pprime[:-1]
        monkeypatch.setattr(diiid_jet_mod, "read_geqdsk", lambda _: eq)
        with pytest.raises(ValueError, match="pprime"):
            diiid_jet_mod.validate_file(SAMPLE_EQ, "DIII-D")

    def test_rejects_psirz_shape_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eq = copy.deepcopy(read_geqdsk(SAMPLE_EQ))
        eq.psirz = eq.psirz[:, :-1]
        monkeypatch.setattr(diiid_jet_mod, "read_geqdsk", lambda _: eq)
        with pytest.raises(ValueError, match="psirz"):
            diiid_jet_mod.validate_file(SAMPLE_EQ, "DIII-D")
