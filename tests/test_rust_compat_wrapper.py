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


class _DummyNonMonotonicRKernel(_DummyRustKernel):
    def get_r(self) -> list[float]:
        out = self._r.copy()
        out[2] = out[1]
        return out.tolist()


class _DummySolveBadStateKernel(_DummyRustKernel):
    def solve_equilibrium(self) -> object:
        self._psi = np.full((4, 5), 0.0, dtype=float)
        return object()


class _DummySolveNanStateKernel(_DummyRustKernel):
    def solve_equilibrium(self) -> object:
        self._psi = self._psi.copy()
        self._psi[0, 0] = np.nan
        return object()


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


def test_rust_wrapper_rejects_non_monotonic_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyNonMonotonicRKernel, raising=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        _rust_compat.RustAcceleratedKernel(str(cfg))


def test_rust_wrapper_tracks_state_sync_failure_on_bad_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummySolveBadStateKernel, raising=False)
    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    with pytest.raises(RuntimeError, match="state sync failed"):
        wrapper.solve_equilibrium()
    assert wrapper.state_sync_failures == 1
    assert wrapper.last_state_sync_error is not None
    assert "shape" in wrapper.last_state_sync_error


def test_rust_wrapper_tracks_state_sync_failure_on_nonfinite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummySolveNanStateKernel, raising=False)
    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    with pytest.raises(RuntimeError, match="state sync failed"):
        wrapper.solve_equilibrium()
    assert wrapper.state_sync_failures == 1
    assert wrapper.last_state_sync_error is not None
    assert "finite" in wrapper.last_state_sync_error


def test_rust_snn_pool_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    pool = _rust_compat.RustSnnPool(n_neurons=20, gain=5.0, window_size=10, seed=17)
    assert pool.backend == "numpy_fallback"
    out = 0.0
    for _ in range(60):
        out = pool.step(2.0)
    assert np.isfinite(out)
    assert out > 0.0


def test_rust_snn_controller_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    ctrl = _rust_compat.RustSnnController(target_r=6.2, target_z=0.0, seed=17)
    assert ctrl.backend == "numpy_fallback"

    out_r = 0.0
    out_z = 0.0
    for _ in range(60):
        out_r, out_z = ctrl.step(5.8, 0.2)  # positive R error, negative Z error

    assert np.isfinite(out_r)
    assert np.isfinite(out_z)
    assert out_r > 0.0
    assert out_z < 0.0


def test_rust_snn_pool_strict_mode_raises_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ImportError, match="allow_numpy_fallback=False"):
        _rust_compat.RustSnnPool(allow_numpy_fallback=False)


def test_rust_snn_controller_strict_mode_raises_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ImportError, match="allow_numpy_fallback=False"):
        _rust_compat.RustSnnController(allow_numpy_fallback=False)
