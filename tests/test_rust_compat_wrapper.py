# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Rust Compat Wrapper Tests
"""Tests for the optional Rust compatibility wrapper facade."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core import _rust_compat

FloatArray = NDArray[np.float64]


class _DummyRustKernel:
    """Minimal PyFusionKernel stand-in for wrapper contract tests."""

    def __init__(self, _config_path: str) -> None:
        """Create deterministic Rust-like grid arrays and solver state."""
        self._method = "sor"
        self._psi = np.zeros((5, 5), dtype=np.float64)
        self._j_phi = np.zeros((5, 5), dtype=np.float64)
        self._r = np.linspace(1.0, 3.0, 5, dtype=np.float64)
        self._z = np.linspace(-1.0, 1.0, 5, dtype=np.float64)

    def grid_shape(self) -> tuple[int, int]:
        """Return the Rust-style `(nr, nz)` grid shape."""
        return (5, 5)

    def get_r(self) -> list[float]:
        """Return the R coordinates as the PyO3 binding would."""
        return [float(value) for value in self._r]

    def get_z(self) -> list[float]:
        """Return the Z coordinates as the PyO3 binding would."""
        return [float(value) for value in self._z]

    def get_psi(self) -> FloatArray:
        """Return a copy of the poloidal-flux state."""
        return self._psi.copy()

    def get_j_phi(self) -> FloatArray:
        """Return a copy of the toroidal-current state."""
        return self._j_phi.copy()

    def solve_equilibrium(self) -> object:
        """Return a Rust-like opaque equilibrium result."""
        self._psi = np.arange(25, dtype=np.float64).reshape(5, 5)
        return object()

    def calculate_thermodynamics(self, _p_aux_mw: float) -> object:
        """Return a Rust-like opaque thermodynamics result."""
        return object()

    def set_solver_method(self, method: str) -> None:
        """Accept the supported solver aliases."""
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
        """Return the normalized solver method."""
        return self._method


class _DummyNonMonotonicRKernel(_DummyRustKernel):
    """Rust-kernel stand-in with a non-monotonic R axis."""

    def get_r(self) -> list[float]:
        """Return an invalid R axis with a repeated coordinate."""
        out = self._r.copy()
        out[2] = out[1]
        return [float(value) for value in out]


class _DummyShortRKernel(_DummyRustKernel):
    """Rust-kernel stand-in with an axis length mismatch."""

    def get_r(self) -> list[float]:
        """Return too few R coordinates for the declared grid."""
        return [float(value) for value in self._r[:-1]]


class _DummyNonFiniteRKernel(_DummyRustKernel):
    """Rust-kernel stand-in with a non-finite R axis."""

    def get_r(self) -> list[float]:
        """Return an R axis containing NaN."""
        out = self._r.copy()
        out[2] = np.nan
        return [float(value) for value in out]


class _DummySmallGridKernel(_DummyRustKernel):
    """Rust-kernel stand-in with an unusably small grid."""

    def grid_shape(self) -> tuple[int, int]:
        """Return an invalid one-cell grid."""
        return (1, 1)


class _DummySolveBadStateKernel(_DummyRustKernel):
    """Rust-kernel stand-in that returns a bad state shape after solve."""

    def solve_equilibrium(self) -> object:
        """Return a wrong-shaped Psi array after the solve."""
        self._psi = np.full((4, 5), 0.0, dtype=np.float64)
        return object()


class _DummySolveNanStateKernel(_DummyRustKernel):
    """Rust-kernel stand-in that returns non-finite state after solve."""

    def solve_equilibrium(self) -> object:
        """Return a NaN-contaminated Psi array after the solve."""
        self._psi = self._psi.copy()
        self._psi[0, 0] = np.nan
        return object()


class _DummyVacuumKernel:
    """FusionKernel stand-in for legacy vacuum-field delegation."""

    def __init__(self, config_path: str) -> None:
        """Remember the delegated config path."""
        self.config_path = config_path

    def calculate_vacuum_field(self) -> FloatArray:
        """Return a deterministic vacuum-field array."""
        return np.eye(3, dtype=np.float64)


class _DummyRustSnnPool:
    """Rust SNN pool stand-in exposed through a fake scpn_fusion_rs module."""

    def __init__(self, n_neurons: int, gain: float, _window_size: int) -> None:
        """Store constructor values surfaced by wrapper properties."""
        self.n_neurons = int(n_neurons)
        self.gain = float(gain)

    def step(self, error: float) -> float:
        """Return a scaled Rust-like control output."""
        return float(error) * self.gain


class _DummyRustSnnController:
    """Rust SNN controller stand-in exposed through a fake scpn_fusion_rs module."""

    def __init__(self, target_r: float, target_z: float) -> None:
        """Store target positions surfaced by wrapper properties."""
        self.target_r = float(target_r)
        self.target_z = float(target_z)

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        """Return target-minus-measured control signals."""
        return self.target_r - float(measured_r), self.target_z - float(measured_z)


def _write_minimal_config(path: Path) -> None:
    """Write the minimal config consumed by checked_json_load."""
    path.write_text("{}", encoding="utf-8")


def _load_no_rust_compat(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Execute the compatibility module with the Rust extension blocked."""
    source_path = Path(_rust_compat.__file__)
    spec = importlib.util.spec_from_file_location("_rust_compat_no_rs_test", source_path)
    assert spec is not None
    assert isinstance(spec.loader, importlib.machinery.SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", None)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rust_wrapper_solver_method_forwarding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wrapper forwards solver-method aliases to the Rust kernel."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    assert wrapper.solver_method() == "sor"

    wrapper.set_solver_method("mg")
    assert wrapper.solver_method() == "multigrid"

    wrapper.set_solver_method("picard_sor")
    assert wrapper.solver_method() == "sor"


def test_rust_wrapper_solver_method_propagates_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wrapper preserves invalid solver-method errors from Rust."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    with pytest.raises(ValueError, match="Unknown solver method"):
        wrapper.set_solver_method("invalid")


def test_rust_wrapper_rejects_non_monotonic_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wrapper rejects Rust grids whose axes are not strictly increasing."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyNonMonotonicRKernel, raising=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        _rust_compat.RustAcceleratedKernel(str(cfg))


@pytest.mark.parametrize(
    ("kernel_cls", "match"),
    [
        (_DummyShortRKernel, "1-D with length"),
        (_DummyNonFiniteRKernel, "finite values"),
        (_DummySmallGridKernel, "grid shape"),
    ],
)
def test_rust_wrapper_rejects_invalid_grid_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kernel_cls: type[_DummyRustKernel],
    match: str,
) -> None:
    """Wrapper rejects Rust grids with invalid shape, length, or values."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", kernel_cls, raising=False)
    with pytest.raises(ValueError, match=match):
        _rust_compat.RustAcceleratedKernel(str(cfg))


def test_rust_wrapper_success_paths_and_facade_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wrapper exposes solve, field, X-point, thermodynamics, and save facades."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)
    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))

    result = wrapper.solve_equilibrium()
    assert result is not None
    assert wrapper.B_R.shape == wrapper.Psi.shape
    assert wrapper.B_Z.shape == wrapper.Psi.shape

    wrapper.cfg = {"dimensions": {"Z_min": 0.0}}
    point, psi_value = wrapper.find_x_point(wrapper.Psi)
    assert len(point) == 2
    assert np.isfinite(psi_value)

    wrapper.cfg = {"dimensions": {"Z_min": -10.0}}
    no_mask_point, no_mask_psi = wrapper.find_x_point(wrapper.Psi)
    assert no_mask_point == (0.0, 0.0)
    assert no_mask_psi == pytest.approx(float(np.min(wrapper.Psi)))

    assert wrapper.calculate_thermodynamics(2.5) is not None

    output_path = tmp_path / "state.npz"
    wrapper.save_results(str(output_path))
    assert output_path.is_file()


def test_rust_wrapper_rejects_invalid_b_field_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """B-field computation rejects malformed and non-finite Psi arrays."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)
    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))

    wrapper.Psi = np.zeros((4, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="Psi shape mismatch"):
        wrapper.compute_b_field()

    wrapper.Psi = np.zeros((5, 5), dtype=np.float64)
    wrapper.Psi[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        wrapper.compute_b_field()


def test_rust_wrapper_delegates_vacuum_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy vacuum-field facade delegates to the Python FusionKernel."""
    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)
    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))

    import scpn_fusion.core.fusion_kernel as fusion_kernel_mod

    monkeypatch.setattr(fusion_kernel_mod, "FusionKernel", _DummyVacuumKernel)
    np.testing.assert_allclose(wrapper.calculate_vacuum_field(), np.eye(3, dtype=np.float64))


def test_rust_wrapper_tracks_state_sync_failure_on_bad_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wrapper records state-sync failures for bad Rust state shapes."""
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
    """Wrapper records state-sync failures for non-finite Rust state."""
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
    """SNN pool uses the deterministic NumPy backend when Rust is absent."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    pool = _rust_compat.RustSnnPool(n_neurons=20, gain=5.0, window_size=10, seed=17)
    assert pool.backend == "numpy_fallback"
    out = 0.0
    for _ in range(60):
        out = pool.step(2.0)
    assert np.isfinite(out)
    assert out > 0.0


def test_rust_snn_pool_rust_backend_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SNN pool exposes Rust backend properties and representation."""
    fake_rs = types.ModuleType("scpn_fusion_rs")
    fake_rs.__dict__["PySnnPool"] = _DummyRustSnnPool
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_rs)
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", True)

    pool = _rust_compat.RustSnnPool(n_neurons=7, gain=3.0, window_size=5)
    assert pool.backend == "rust"
    assert pool.n_neurons == 7
    assert pool.gain == pytest.approx(3.0)
    assert pool.step(2.0) == pytest.approx(6.0)
    assert repr(pool) == "RustSnnPool(n_neurons=7, gain=3.0, backend='rust')"


@pytest.mark.parametrize(
    ("n_neurons", "gain", "window_size", "match"),
    [
        (0, 1.0, 3, "n_neurons"),
        (5, float("nan"), 3, "gain"),
        (5, 1.0, 0, "window_size"),
    ],
)
def test_rust_snn_pool_numpy_fallback_rejects_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    n_neurons: int,
    gain: float,
    window_size: int,
    match: str,
) -> None:
    """NumPy SNN pool fallback validates construction inputs."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ValueError, match=match):
        _rust_compat.RustSnnPool(
            n_neurons=n_neurons,
            gain=gain,
            window_size=window_size,
        )


def test_rust_snn_pool_numpy_fallback_rejects_nonfinite_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy SNN pool fallback rejects non-finite error signals."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    pool = _rust_compat.RustSnnPool(n_neurons=5, gain=1.0, window_size=3)
    with pytest.raises(ValueError, match="error_signal"):
        pool.step(float("nan"))


def test_rust_snn_controller_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """SNN controller uses paired NumPy pools when Rust is absent."""
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


def test_rust_snn_controller_rust_backend_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SNN controller exposes Rust backend targets and representation."""
    fake_rs = types.ModuleType("scpn_fusion_rs")
    fake_rs.__dict__["PySnnController"] = _DummyRustSnnController
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_rs)
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", True)

    ctrl = _rust_compat.RustSnnController(target_r=6.1, target_z=-0.2)
    assert ctrl.backend == "rust"
    assert ctrl.target_r == pytest.approx(6.1)
    assert ctrl.target_z == pytest.approx(-0.2)
    assert ctrl.step(5.9, 0.1) == pytest.approx((0.2, -0.3))
    assert repr(ctrl) == "RustSnnController(target_r=6.1, target_z=-0.2, backend='rust')"


def test_rust_snn_controller_numpy_fallback_rejects_nonfinite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy SNN controller fallback rejects non-finite targets and measurements."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ValueError, match="target_r and target_z"):
        _rust_compat.RustSnnController(target_r=float("nan"), target_z=0.0)

    ctrl = _rust_compat.RustSnnController(target_r=6.2, target_z=0.0)
    with pytest.raises(ValueError, match="measured_r and measured_z"):
        ctrl.step(float("nan"), 0.0)


def test_rust_snn_pool_strict_mode_raises_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict SNN pool mode rejects absent Rust bindings."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ImportError, match="allow_numpy_fallback=False"):
        _rust_compat.RustSnnPool(allow_numpy_fallback=False)


def test_rust_snn_controller_strict_mode_raises_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict SNN controller mode rejects absent Rust bindings."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    with pytest.raises(ImportError, match="allow_numpy_fallback=False"):
        _rust_compat.RustSnnController(allow_numpy_fallback=False)


def test_fusion_kernel_alias_resolves_in_both_backends() -> None:
    """``from _rust_compat import FusionKernel`` resolves with or without Rust."""
    from scpn_fusion.core._rust_compat import FusionKernel

    assert isinstance(FusionKernel, type)
    if not _rust_compat.RUST_BACKEND:
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        assert FusionKernel is PyFusionKernel


def test_fusion_kernel_alias_is_bound_module_attribute() -> None:
    """Consumers can import ``FusionKernel`` without relying on module getattr."""
    assert "FusionKernel" in vars(_rust_compat)
    assert isinstance(_rust_compat.FusionKernel, type)


def test_module_getattr_returns_fusion_kernel_alias() -> None:
    """The module getattr fallback returns the bound FusionKernel alias."""
    module_getattr = cast(Any, _rust_compat).__getattr__
    assert module_getattr("FusionKernel") is _rust_compat.FusionKernel


def test_module_getattr_rejects_unknown_attribute_without_rust() -> None:
    """The lazy module ``__getattr__`` fallback serves only ``FusionKernel``."""
    if _rust_compat.RUST_BACKEND:
        pytest.skip("module __getattr__ fallback is only defined when Rust is absent")
    with pytest.raises(AttributeError, match="nonexistent_symbol"):
        cast(Any, _rust_compat).nonexistent_symbol


def test_rust_available_reports_boolean() -> None:
    """Rust availability helper exposes the import-time backend flag."""
    assert _rust_compat._rust_available() is _rust_compat.RUST_BACKEND


def test_legacy_rust_shafranov_config_path_delegates_to_python_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy config-path Shafranov helper delegates to Python vacuum fields."""
    if not _rust_compat.RUST_BACKEND:
        pytest.skip("legacy config-path wrapper exists only when Rust helpers are importable")

    cfg = tmp_path / "cfg.json"
    _write_minimal_config(cfg)
    import scpn_fusion.core.fusion_kernel as fusion_kernel_mod

    monkeypatch.setattr(fusion_kernel_mod, "FusionKernel", _DummyVacuumKernel)
    np.testing.assert_allclose(
        _rust_compat.rust_shafranov_bv(str(cfg)),
        np.eye(3, dtype=np.float64),
    )


def test_rust_helper_numeric_delegates_to_imported_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric Shafranov and tearing helpers delegate to imported Rust symbols."""
    if not _rust_compat.RUST_BACKEND:
        pytest.skip("Rust helper delegation exists only when Rust helpers are importable")

    def fake_shafranov(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        assert args == (6.2, 2.0, 15.0)
        assert kwargs == {}
        return (1.0, 2.0, 3.0)

    def fake_tearing(
        steps: int, seed: int | None, beta_p: float, w_crit: float
    ) -> tuple[int, int | None, float, float]:
        return steps, seed, beta_p, w_crit

    monkeypatch.setattr(_rust_compat, "shafranov_bv", fake_shafranov, raising=False)
    monkeypatch.setattr(_rust_compat, "simulate_tearing_mode", fake_tearing, raising=False)

    assert _rust_compat.rust_shafranov_bv(6.2, 2.0, 15.0) == (1.0, 2.0, 3.0)
    assert _rust_compat.rust_simulate_tearing_mode(4, seed=9, beta_p=0.7, w_crit=0.04) == (
        4,
        9,
        0.7,
        0.04,
    )


def test_no_rust_helper_shims_raise_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-Rust compatibility helper bodies fail closed with ImportError."""
    module = _load_no_rust_compat(monkeypatch)
    module_any = cast(Any, module)

    with pytest.raises(ImportError, match="maturin develop"):
        module_any.rust_shafranov_bv(6.2, 2.0, 15.0)
    with pytest.raises(ImportError, match="maturin develop"):
        module_any.rust_solve_coil_currents()
    with pytest.raises(ImportError, match="maturin develop"):
        module_any.rust_measure_magnetics()
    with pytest.raises(ImportError, match="maturin develop"):
        module_any.rust_simulate_tearing_mode(4)


def test_rust_multigrid_vcycle_returns_none_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust multigrid shim returns None when Rust is unavailable."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    source = np.zeros((2, 2), dtype=np.float64)
    assert _rust_compat.rust_multigrid_vcycle(source, source, 1.0, 2.0, -1.0, 1.0, 2, 2) is None


def test_rust_multigrid_vcycle_delegates_to_pyo3_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust multigrid shim delegates to the exposed PyO3 function."""
    source = np.ones((2, 2), dtype=np.float64)
    psi_bc = np.zeros((2, 2), dtype=np.float64)
    expected = (psi_bc.copy(), 0.0, 3, True)

    def fake_multigrid_vcycle(
        source_arg: FloatArray,
        psi_bc_arg: FloatArray,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        nr: int,
        nz: int,
        tol: float,
        max_cycles: int,
    ) -> tuple[FloatArray, float, int, bool]:
        np.testing.assert_allclose(source_arg, source)
        np.testing.assert_allclose(psi_bc_arg, psi_bc)
        assert (r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles) == (
            1.0,
            2.0,
            -1.0,
            1.0,
            2,
            2,
            1.0e-6,
            500,
        )
        return expected

    fake_rs = types.ModuleType("scpn_fusion_rs")
    fake_rs.__dict__["multigrid_vcycle"] = fake_multigrid_vcycle
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_rs)
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", True)

    result = _rust_compat.rust_multigrid_vcycle(source, psi_bc, 1.0, 2.0, -1.0, 1.0, 2, 2)
    assert result is not None
    np.testing.assert_allclose(result[0], expected[0])
    assert result[1:] == expected[1:]
