# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HPC Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for HPC bridge safety and low-copy behavior."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.hpc.hpc_bridge import HPCBridge, _as_contiguous_f64


class _DummyLib:
    def __init__(self) -> None:
        self.called = False
        self.last_size = None
        self.last_iterations = None
        self.destroyed = None

    def run_step(self, solver_ptr, j_array, psi_array, size, iterations) -> None:
        self.called = True
        self.last_size = int(size)
        self.last_iterations = int(iterations)
        psi_array[:] = 0.5 * j_array

    def destroy_solver(self, solver_ptr) -> None:
        self.destroyed = solver_ptr


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
