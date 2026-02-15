# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Compat Wrapper Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core import _rust_compat


class _DummyRustKernel:
    def __init__(self, _config_path: str) -> None:
        self._method = "sor"
        self._psi = np.zeros((5, 5), dtype=float)
        self._j_phi = np.zeros((5, 5), dtype=float)
        self._r = np.linspace(1.0, 3.0, 5, dtype=float)
        self._z = np.linspace(-1.0, 1.0, 5, dtype=float)

    def grid_shape(self) -> tuple[int, int]:
        return (5, 5)

    def get_r(self) -> list[float]:
        return self._r.tolist()

    def get_z(self) -> list[float]:
        return self._z.tolist()

    def get_psi(self) -> np.ndarray:
        return self._psi.copy()

    def get_j_phi(self) -> np.ndarray:
        return self._j_phi.copy()

    def solve_equilibrium(self) -> object:
        return object()

    def calculate_thermodynamics(self, _p_aux_mw: float) -> object:
        return object()

    def set_solver_method(self, method: str) -> None:
        aliases = {
            "sor": "sor",
            "picard_sor": "sor",
            "multigrid": "multigrid",
            "picard_multigrid": "multigrid",
            "mg": "multigrid",
        }
        if method not in aliases:
            raise ValueError("Unknown solver method")
        self._method = aliases[method]

    def solver_method(self) -> str:
        return self._method


def _write_minimal_config(path: Path) -> None:
    path.write_text("{}", encoding="utf-8")


def test_rust_wrapper_solver_method_forwarding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    assert wrapper.solver_method() == "sor"

    wrapper.set_solver_method("mg")
    assert wrapper.solver_method() == "multigrid"

    wrapper.set_solver_method("picard_sor")
    assert wrapper.solver_method() == "sor"


def test_rust_wrapper_solver_method_propagates_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    with pytest.raises(ValueError, match="Unknown solver method"):
        wrapper.set_solver_method("invalid")
