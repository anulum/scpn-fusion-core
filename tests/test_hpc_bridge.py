# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HPC Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for HPC bridge safety and low-copy behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.hpc.hpc_bridge import HPCBridge, _as_contiguous_f64
from scpn_fusion.hpc import hpc_bridge as hpc_mod


class _DummyLib:
    def __init__(self) -> None:
        self.called = False
        self.called_converged = False
        self.last_size = None
        self.last_iterations = None
        self.last_max_iterations = None
        self.last_omega = None
        self.last_tolerance = None
        self.destroyed = None
        self.boundary_value = None
        self.last_psi_ref = None

    def run_step(self, solver_ptr, j_array, psi_array, size, iterations) -> None:
        self.called = True
        self.last_size = int(size)
        self.last_iterations = int(iterations)
        self.last_psi_ref = psi_array
        psi_array[:] = 0.5 * j_array

    def run_step_converged(
        self,
        solver_ptr,
        j_array,
        psi_array,
        size,
        max_iterations,
        omega,
        tolerance,
        final_delta_ptr,
    ) -> int:
        self.called_converged = True
        self.last_size = int(size)
        self.last_max_iterations = int(max_iterations)
        self.last_omega = float(omega)
        self.last_tolerance = float(tolerance)
        self.last_psi_ref = psi_array
        psi_array[:] = 0.25 * j_array
        final_delta_ptr._obj.value = 2.5e-4
        return 7

    def destroy_solver(self, solver_ptr) -> None:
        self.destroyed = solver_ptr

    def set_boundary_dirichlet(self, solver_ptr, value) -> None:
        self.boundary_value = float(value)


class _DummyDeleteLib:
    def __init__(self) -> None:
        self.deleted = None

    def delete_solver(self, solver_ptr) -> None:
        self.deleted = solver_ptr


def _make_bridge(nr: int = 2, nz: int = 3) -> HPCBridge:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = _DummyLib()
    bridge.solver_ptr = 12345
    bridge.loaded = True
    bridge._destroy_symbol = "destroy_solver"
    bridge._has_converged_api = True
    bridge._has_boundary_api = True
    bridge.nr = nr
    bridge.nz = nz
    return bridge


def test_as_contiguous_f64_reuses_float64_c_contiguous() -> None:
    arr = np.zeros((3, 2), dtype=np.float64)
    out = _as_contiguous_f64(arr)
    assert out is arr


def test_as_contiguous_f64_converts_dtype_and_layout() -> None:
    arr = np.asfortranarray(np.zeros((3, 2), dtype=np.float32))
    out = _as_contiguous_f64(arr)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous


def test_solve_returns_none_when_solver_not_initialized() -> None:
    bridge = _make_bridge()
    bridge.solver_ptr = None
    result = bridge.solve(np.zeros((3, 2), dtype=np.float64))
    assert result is None


def test_solve_raises_on_shape_mismatch() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.solve(np.zeros((2, 2), dtype=np.float64))


def test_solve_rejects_non_2d_input() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    with pytest.raises(ValueError, match="must be a 2D array"):
        bridge.solve(np.zeros((6,), dtype=np.float64))


def test_solve_rejects_nonfinite_input() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bad = np.arange(6, dtype=np.float64).reshape(3, 2)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        bridge.solve(bad)


def test_solve_runs_and_returns_expected_shape() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve(j_phi, iterations=17)

    assert out is not None
    assert out.shape == (3, 2)
    assert np.allclose(out, 0.5 * j_phi)
    assert bridge.lib.called
    assert bridge.lib.last_size == 6
    assert bridge.lib.last_iterations == 17


def test_solve_into_reuses_output_buffer() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_into(j_phi, out_buf, iterations=5)

    assert out is out_buf
    assert bridge.lib.last_psi_ref is out_buf
    assert np.allclose(out_buf, 0.5 * j_phi)
    assert bridge.lib.last_iterations == 5


def test_solve_into_rejects_noncontiguous_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((2, 3), dtype=np.float64).T
    with pytest.raises(ValueError, match="C-contiguous"):
        bridge.solve_into(j_phi, out_buf, iterations=3)


def test_solve_into_rejects_wrong_shape_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.solve_into(j_phi, out_buf, iterations=3)


def test_solve_into_rejects_non_ndarray_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    with pytest.raises(ValueError, match="numpy.ndarray"):
        bridge.solve_into(j_phi, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], iterations=3)  # type: ignore[arg-type]


def test_solve_until_converged_uses_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve_until_converged(j_phi, max_iterations=200, tolerance=1e-5, omega=1.7)
    assert out is not None
    psi, iters, delta = out
    assert psi.shape == (3, 2)
    assert np.allclose(psi, 0.25 * j_phi)
    assert iters == 7
    assert abs(delta - 2.5e-4) < 1e-12
    assert bridge.lib.called_converged
    assert bridge.lib.last_max_iterations == 200
    assert abs(bridge.lib.last_omega - 1.7) < 1e-12
    assert abs(bridge.lib.last_tolerance - 1e-5) < 1e-12


def test_solve_until_converged_into_reuses_output_buffer() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_until_converged_into(
        j_phi,
        out_buf,
        max_iterations=33,
        tolerance=1e-7,
        omega=1.5,
    )
    assert out is not None
    iters, delta = out
    assert iters == 7
    assert abs(delta - 2.5e-4) < 1e-12
    assert bridge.lib.last_psi_ref is out_buf
    assert np.allclose(out_buf, 0.25 * j_phi)
    assert bridge.lib.last_max_iterations == 33


def test_solve_until_converged_into_sanitizes_nonfinite_params() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_until_converged_into(
        j_phi,
        out_buf,
        max_iterations=0,
        tolerance=float("nan"),
        omega=float("inf"),
    )
    assert out is not None
    iters, delta = out
    assert iters == 7
    assert abs(delta - 2.5e-4) < 1e-12
    assert bridge.lib.last_max_iterations == 1
    assert abs(bridge.lib.last_omega - 1.8) < 1e-12
    assert abs(bridge.lib.last_tolerance - 0.0) < 1e-12


def test_solve_until_converged_falls_back_without_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_converged_api = False
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve_until_converged(j_phi, max_iterations=12)
    assert out is not None
    psi, iters, delta = out
    assert psi.shape == (3, 2)
    assert np.allclose(psi, 0.5 * j_phi)
    assert iters == 12
    assert np.isnan(delta)


def test_solve_until_converged_into_fallback_without_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_converged_api = False
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_until_converged_into(j_phi, out_buf, max_iterations=9)
    assert out is not None
    iters, delta = out
    assert iters == 9
    assert np.isnan(delta)
    assert np.allclose(out_buf, 0.5 * j_phi)
    assert bridge.lib.last_iterations == 9


def test_set_boundary_dirichlet_calls_native_symbol() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge.set_boundary_dirichlet(1.25)
    assert bridge.lib.boundary_value == 1.25


def test_set_boundary_dirichlet_noop_without_support() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_boundary_api = False
    bridge.set_boundary_dirichlet(0.5)
    assert bridge.lib.boundary_value is None


def test_close_releases_solver_pointer() -> None:
    bridge = _make_bridge()
    bridge.close()
    assert bridge.solver_ptr is None
    assert bridge.lib.destroyed == 12345


def test_close_supports_delete_solver_alias() -> None:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = _DummyDeleteLib()
    bridge.solver_ptr = 999
    bridge.loaded = True
    bridge._destroy_symbol = "delete_solver"
    bridge.close()
    assert bridge.solver_ptr is None
    assert bridge.lib.deleted == 999


def test_init_prefers_env_override_path(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = "/tmp/scpn_solver_override.so"

    def _raise_cdll(_path: str):
        raise OSError("no library")

    monkeypatch.setenv("SCPN_SOLVER_LIB", expected)
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", _raise_cdll)
    bridge = HPCBridge()
    assert bridge.lib_path == expected
    assert not bridge.loaded


def test_init_uses_package_local_default_without_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_cdll(_path: str):
        raise OSError("no library")

    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)
    monkeypatch.setattr(hpc_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", _raise_cdll)

    bridge = HPCBridge()
    expected = str(Path(hpc_mod.__file__).resolve().parent / "libscpn_solver.so")
    assert bridge.lib_path == expected
    assert not bridge.loaded


def test_compile_cpp_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCPN_ALLOW_NATIVE_BUILD", raising=False)
    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_builds_in_package_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_run(cmd, check):  # type: ignore[no-untyped-def]
        calls["cmd"] = list(cmd)
        calls["check"] = check

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    monkeypatch.setattr(hpc_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hpc_mod.subprocess, "run", _fake_run)

    out = hpc_mod.compile_cpp()
    assert out is not None
    assert Path(out).name == "libscpn_solver.so"
    assert Path(out).parent.name == "bin"
    assert calls["check"] is True
    assert isinstance(calls["cmd"], list)
    assert calls["cmd"][0] == "g++"
