# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — QuaLiKiz Solver Tests
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from scpn_fusion.core.gk_interface import GKLocalParams
from scpn_fusion.core.gk_qualikiz import (
    QuaLiKizSolver,
    _try_qualikiz_python,
)


@pytest.fixture()
def default_params():
    return GKLocalParams(R_L_Ti=6.0, R_L_Te=6.0, R_L_ne=2.0, q=1.4, s_hat=0.8)


# ── _try_qualikiz_python ────────────────────────────────────────────


class TestTryQualikizPython:
    def test_returns_none_when_import_fails(self, default_params):
        result = _try_qualikiz_python(default_params)
        assert result is None

    def test_returns_output_with_mock_module(self, default_params, monkeypatch):
        mock_mod = types.ModuleType("qualikiz_tools")
        mock_mod.run = MagicMock(  # type: ignore[attr-defined]
            return_value={"chi_i": 2.5, "chi_e": 1.8, "D_e": 0.3}
        )
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            result = _try_qualikiz_python(default_params)
            assert result is not None
            assert result.converged
            assert result.chi_i == pytest.approx(2.5)
            assert result.chi_e == pytest.approx(1.8)
            assert result.D_e == pytest.approx(0.3)
            assert result.dominant_mode == "ITG"
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)

    def test_handles_attribute_error(self, default_params, monkeypatch):
        mock_mod = types.ModuleType("qualikiz_tools")
        # No 'run' attribute
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            result = _try_qualikiz_python(default_params)
            assert result is None
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)

    def test_handles_key_error(self, default_params, monkeypatch):
        mock_mod = types.ModuleType("qualikiz_tools")
        mock_mod.run = MagicMock(return_value={})  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            result = _try_qualikiz_python(default_params)
            # Should succeed but with 0.0 values via .get() defaults
            assert result is not None
            assert result.chi_i == 0.0
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)

    def test_small_a_clamped(self, monkeypatch):
        params = GKLocalParams(R_L_Ti=5.0, R_L_Te=5.0, R_L_ne=2.0, q=1.4, s_hat=0.8, a=0.0)
        mock_mod = types.ModuleType("qualikiz_tools")
        mock_mod.run = MagicMock(return_value={"chi_i": 1.0, "chi_e": 0.8, "D_e": 0.1})  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            result = _try_qualikiz_python(params)
            assert result is not None
            # Verify Rmaj was computed without ZeroDivisionError
            call_kwargs = mock_mod.run.call_args  # type: ignore[attr-defined]
            assert call_kwargs is not None
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)


# ── QuaLiKizSolver ──────────────────────────────────────────────────


class TestQuaLiKizSolver:
    def test_is_available_false_without_module(self):
        solver = QuaLiKizSolver()
        # qualikiz_tools not installed in CI
        assert not solver.is_available()

    def test_is_available_true_with_mock(self, monkeypatch):
        mock_mod = types.ModuleType("qualikiz_tools")
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            solver = QuaLiKizSolver()
            assert solver.is_available()
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)

    def test_prepare_input_creates_dir(self, tmp_path, default_params):
        solver = QuaLiKizSolver(work_dir=tmp_path)
        result_path = solver.prepare_input(default_params)
        assert result_path.exists()
        assert solver._last_params is default_params

    def test_prepare_input_auto_tmpdir(self, default_params):
        solver = QuaLiKizSolver()
        result_path = solver.prepare_input(default_params)
        assert result_path.exists()

    def test_run_without_prepare_returns_unconverged(self, tmp_path):
        solver = QuaLiKizSolver()
        out = solver.run(tmp_path)
        assert not out.converged

    def test_run_with_python_api(self, tmp_path, default_params, monkeypatch):
        mock_mod = types.ModuleType("qualikiz_tools")
        mock_mod.run = MagicMock(  # type: ignore[attr-defined]
            return_value={"chi_i": 3.0, "chi_e": 2.0, "D_e": 0.5}
        )
        monkeypatch.setitem(sys.modules, "qualikiz_tools", mock_mod)
        try:
            solver = QuaLiKizSolver(work_dir=tmp_path)
            solver.prepare_input(default_params)
            out = solver.run(tmp_path)
            assert out.converged
            assert out.chi_i == pytest.approx(3.0)
        finally:
            monkeypatch.delitem(sys.modules, "qualikiz_tools", raising=False)

    def test_run_fallback_unconverged(self, tmp_path, default_params):
        solver = QuaLiKizSolver(work_dir=tmp_path)
        solver.prepare_input(default_params)
        out = solver.run(tmp_path)
        assert not out.converged

    def test_run_from_params(self, default_params):
        solver = QuaLiKizSolver()
        out = solver.run_from_params(default_params)
        assert not out.converged
