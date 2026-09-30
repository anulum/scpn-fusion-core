# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Native Solver Transaction Tests
"""Exercise solver transactions through a real compiled native library."""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scpn_fusion.hpc.hpc_bridge import HPCBridge
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel


@pytest.fixture
def native_library(tmp_path: Path) -> Path:
    """Build and attest the actual bundled solver.

    Parameters
    ----------
    tmp_path : Path
        Isolated native build directory.

    Returns
    -------
    Path
        Trusted compiled shared library.
    """
    source = Path(__file__).resolve().parents[1] / "src/scpn_fusion/hpc/solver.cpp"
    library = tmp_path / "libscpn_solver.so"
    coverage_flags = ["--coverage"] if os.environ.get("SCPN_NATIVE_COVERAGE") == "1" else []
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *coverage_flags,
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        timeout=60,
    )
    library.with_suffix(".so.sha256").write_text(
        hashlib.sha256(library.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    return library


def test_restore_replays_native_history(native_library: Path) -> None:
    """Restore all native history before replaying a nontrivial source.

    Parameters
    ----------
    native_library : Path
        Real trusted solver library.
    """
    with HPCBridge(str(native_library)) as bridge:
        assert bridge.is_available()
        bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0), boundary_value=0.2)
        initial_source = np.arange(25, dtype=np.float64).reshape(5, 5) / 100
        assert bridge.solve(initial_source, iterations=2) is not None
        checkpoint = bridge.snapshot_state()
        next_source = np.flip(initial_source).copy()
        expected = bridge.solve(next_source, iterations=3)
        assert expected is not None
        bridge.set_boundary_dirichlet(7.0)
        assert bridge.solve(np.ones((5, 5)), iterations=9) is not None
        bridge.restore_state(checkpoint)
        restored = bridge.snapshot_state()
        np.testing.assert_array_equal(restored.psi, checkpoint.psi)
        np.testing.assert_array_equal(restored.j_phi, checkpoint.j_phi)
        assert restored.boundary_value == checkpoint.boundary_value
        actual = bridge.solve(next_source, iterations=3)
        assert actual is not None
        np.testing.assert_array_equal(actual, expected)


def test_snapshot_is_owned(native_library: Path) -> None:
    """Keep exported history immutable and independent of later native work.

    Parameters
    ----------
    native_library : Path
        Real trusted solver library.
    """
    with HPCBridge(str(native_library)) as bridge:
        bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0))
        checkpoint = bridge.snapshot_state()
        with pytest.raises(ValueError):
            checkpoint.psi.setflags(write=True)
        assert bridge.solve(np.ones((5, 5)), iterations=3) is not None
        assert not checkpoint.psi.any()
        assert not checkpoint.j_phi.any()


def test_foreign_checkpoint_preserves_native_state(native_library: Path) -> None:
    """Refuse a checkpoint from another solver before touching native history.

    Parameters
    ----------
    native_library : Path
        Real trusted solver library.
    """
    with HPCBridge(str(native_library)) as first, HPCBridge(str(native_library)) as second:
        first.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0))
        second.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0), boundary_value=0.3)
        before = second.snapshot_state()
        with pytest.raises(ValueError, match="checkpoint belongs to a different solver"):
            second.restore_state(first.snapshot_state())
        after = second.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value


def test_invalid_checkpoint_preserves_native_state(native_library: Path) -> None:
    """Reject a nonfinite checkpoint before altering any native field.

    Parameters
    ----------
    native_library : Path
        Real trusted solver library.
    """
    with HPCBridge(str(native_library)) as bridge:
        bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0), boundary_value=0.2)
        before = bridge.snapshot_state()
        invalid = replace(before, boundary_value=float("nan"))
        with pytest.raises(ValueError, match="only finite values"):
            bridge.restore_state(invalid)
        after = bridge.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value


def test_native_abi_refuses_invalid_state_without_writes(native_library: Path) -> None:
    """Exercise null, size and finite-value guards through the real C ABI.

    Parameters
    ----------
    native_library : Path
        Real trusted solver library.
    """
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library = ctypes.CDLL(str(native_library))
    library.export_solver_state.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        double_pointer,
        ctypes.c_int,
        double_pointer,
    ]
    library.export_solver_state.restype = ctypes.c_int
    library.import_solver_state.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        double_pointer,
        ctypes.c_int,
        ctypes.c_double,
    ]
    library.import_solver_state.restype = ctypes.c_int
    with HPCBridge(str(native_library)) as bridge:
        bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0), boundary_value=0.2)
        before = bridge.snapshot_state()
        psi = np.full((5, 5), -123.0)
        source = np.full((5, 5), -456.0)
        p = psi.ctypes.data_as(double_pointer)
        j = source.ctypes.data_as(double_pointer)
        boundary = ctypes.c_double(-789.0)
        b = ctypes.pointer(boundary)
        for args in (
            (None, p, j, 25, b),
            (bridge.solver_ptr, p, j, 0, b),
            (bridge.solver_ptr, None, j, 25, b),
            (bridge.solver_ptr, p, None, 25, b),
            (bridge.solver_ptr, p, j, 25, None),
            (bridge.solver_ptr, p, j, 24, b),
        ):
            assert library.export_solver_state(*args) == 0
            np.testing.assert_array_equal(psi, np.full((5, 5), -123.0))
            np.testing.assert_array_equal(source, np.full((5, 5), -456.0))
            assert boundary.value == -789.0
        for import_args in (
            (None, p, j, 25, 0.0),
            (bridge.solver_ptr, p, j, 0, 0.0),
            (bridge.solver_ptr, None, j, 25, 0.0),
            (bridge.solver_ptr, p, None, 25, 0.0),
            (bridge.solver_ptr, p, j, 24, 0.0),
            (bridge.solver_ptr, p, j, 25, float("nan")),
        ):
            assert library.import_solver_state(*import_args) == 0
        for array in (psi, source):
            saved = array[2, 2]
            array[2, 2] = np.nan
            assert library.import_solver_state(bridge.solver_ptr, p, j, 25, 0.0) == 0
            array[2, 2] = saved
        after = bridge.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value


def native_outer_kernel(tmp_path: Path) -> FusionKernel:
    """Construct a real kernel using the explicitly configured native library.

    Parameters
    ----------
    tmp_path : Path
        Isolated configuration directory.

    Returns
    -------
    FusionKernel
        Five by five zero-source kernel with strict inner residual checks.
    """
    configuration = {
        "reactor_name": "Native outer transaction negative control",
        "grid_resolution": [5, 5],
        "dimensions": {"R_min": 4, "R_max": 8, "Z_min": -4, "Z_max": 4},
        "physics": {"plasma_current_target": 0.0, "vacuum_permeability": 4e-7 * np.pi},
        "coils": [{"name": "CS", "r": 3, "z": 0, "current": 0}],
        "solver": {
            "max_iterations": 2,
            "convergence_threshold": 1e-12,
            "relaxation_factor": 1.0,
            "solver_method": "sor",
            "require_gs_residual": True,
            "gs_residual_threshold": 1e-12,
            "fail_on_diverge": True,
        },
    }
    path = tmp_path / "native_outer.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    return FusionKernel(path)


def test_native_outer_rejected_trials_restore_complete_state(
    native_library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore real native history when nonuniform-wall trials fail inner checks.

    Parameters
    ----------
    native_library : Path
        Actual compiled and attested library.
    tmp_path : Path
        Isolated configuration directory.
    monkeypatch : pytest.MonkeyPatch
        Set the supported native-library environment override only.
    """
    monkeypatch.setenv("SCPN_SOLVER_LIB", str(native_library))
    kernel = native_outer_kernel(tmp_path)
    try:
        assert kernel.hpc.is_available()
        before = kernel.hpc.snapshot_state()
        coils = CoilSet(
            positions=[(3.0, 0.0), (9.0, 0.0)],
            currents=np.zeros(2),
            turns=[1, 1],
            current_limits=np.full(2, 2e6),
            target_flux_points=np.array([[5.0, -4.0], [7.0, -4.0]]),
            target_flux_values=np.array([0.1, 0.2]),
        )
        result = kernel.solve_free_boundary(
            coils, max_outer_iter=4, tol=0.0, optimize_shape=True, tikhonov_alpha=0.0
        )
        assert result["outer_iterations"] == 4
        assert result["outer_outcome"] == "no_acceptable_step"
        assert result["accepted_steps"] == 0
        assert all(row["reason"] == "inner_not_converged" for row in result["trial_log"])
        assert result["canonical_admission"] == "not_evaluated"
        np.testing.assert_array_equal(coils.currents, np.zeros(2))
        np.testing.assert_array_equal(kernel.Psi, np.zeros((5, 5)))
        after = kernel.hpc.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value
    finally:
        kernel.hpc.close()


def test_closed_library_refuses_snapshot_and_restore(native_library: Path) -> None:
    """Refuse both checkpoint operations after the native context is closed.

    Parameters
    ----------
    native_library : Path
        Actual compiled and attested library.
    """
    bridge = HPCBridge(str(native_library))
    bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0))
    checkpoint = bridge.snapshot_state()
    bridge.close()
    assert not bridge.supports_state_snapshot()
    with pytest.raises(RuntimeError, match="does not support state checkpoints"):
        bridge.snapshot_state()
    with pytest.raises(RuntimeError, match="does not support state checkpoints"):
        bridge.restore_state(checkpoint)


def test_native_initial_failure_restores_preexisting_history(
    native_library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback nonfinite Python fields and nontrivial native source history.

    Parameters
    ----------
    native_library : Path
        Real compiled native solver.
    tmp_path : Path
        Isolated configuration directory.
    monkeypatch : pytest.MonkeyPatch
        Set the supported native-library override only.
    """
    monkeypatch.setenv("SCPN_SOLVER_LIB", str(native_library))
    kernel = native_outer_kernel(tmp_path)
    try:
        assert kernel.hpc.solve(np.ones((5, 5)), iterations=2) is not None
        before = kernel.hpc.snapshot_state()
        psi_before = kernel.Psi.copy()
        source_before = kernel.J_phi.copy()
        kernel.cfg["solver"]["relaxation_factor"] = float("nan")
        coils = CoilSet(positions=[(3.0, 0.0)], currents=np.array([1e5]), turns=[1])
        result = kernel.solve_free_boundary(coils, max_outer_iter=2)
        assert result["state_origin"] == "pre_call_unsolved"
        assert result["inner_status"] == "failed"
        np.testing.assert_array_equal(kernel.Psi, psi_before)
        np.testing.assert_array_equal(kernel.J_phi, source_before)
        assert not hasattr(kernel, "B_R")
        after = kernel.hpc.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value
    finally:
        kernel.hpc.close()


def test_legacy_library_refuses_snapshot(
    native_library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise ordinary solves and pre-mutation refusal with checkpoint exports hidden.

    Parameters
    ----------
    native_library : Path
        Build the current real implementation before making an ABI variant.
    tmp_path : Path
        Isolated actual library and configuration directory.
    monkeypatch : pytest.MonkeyPatch
        Set the supported native-library override only.
    """
    assert native_library.is_file()
    source = Path(__file__).resolve().parents[1] / "src/scpn_fusion/hpc/solver.cpp"
    exports = tmp_path / "ordinary_solver.exports"
    exports.write_text(
        "{ global: create_solver; run_step; run_step_converged; "
        "set_boundary_dirichlet; destroy_solver; delete_solver; local: *; };\n",
        encoding="utf-8",
    )
    library = tmp_path / "ordinary_solver.so"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            f"-Wl,--version-script={exports}",
            "-o",
            str(library),
        ],
        check=True,
        timeout=60,
    )
    library.with_suffix(".so.sha256").write_text(
        hashlib.sha256(library.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("SCPN_SOLVER_LIB", str(library))
    kernel = native_outer_kernel(tmp_path)
    try:
        assert kernel.hpc.is_available()
        assert not kernel.hpc.supports_state_snapshot()
        solved = kernel.hpc.solve(np.ones((5, 5)), iterations=2)
        assert solved is not None and np.isfinite(solved).all()
        with pytest.raises(RuntimeError, match="does not support state checkpoints"):
            kernel.hpc.snapshot_state()
        before = kernel.Psi.copy()
        source_before = kernel.J_phi.copy()
        coils = CoilSet(positions=[(3.0, 0.0)], currents=np.array([1e5]), turns=[1])
        with pytest.raises(RuntimeError, match="requires state checkpoint support"):
            kernel.solve_free_boundary(coils)
        np.testing.assert_array_equal(kernel.Psi, before)
        np.testing.assert_array_equal(kernel.J_phi, source_before)
        np.testing.assert_array_equal(coils.currents, [1e5])
        assert not hasattr(kernel, "B_R")
    finally:
        kernel.hpc.close()


def test_native_refuses_grid_metadata_drift_without_state_writes(native_library: Path) -> None:
    """Reject public grid metadata inconsistent with the actual native context.

    Parameters
    ----------
    native_library : Path
        Real compiled and attested solver.
    """
    with HPCBridge(str(native_library)) as bridge:
        bridge.initialize(5, 5, (4.0, 8.0), (-4.0, 4.0), boundary_value=0.2)
        assert bridge.solve(np.ones((5, 5)), iterations=2) is not None
        before = bridge.snapshot_state()
        # Deliberate caller metadata corruption; the C ABI still owns 25 cells.
        bridge.nr = 6
        with pytest.raises(RuntimeError, match="checkpoint export failed"):
            bridge.snapshot_state()
        mismatched = replace(before, psi=np.zeros((5, 6)), j_phi=np.zeros((5, 6)))
        with pytest.raises(RuntimeError, match="checkpoint restore failed"):
            bridge.restore_state(mismatched)
        bridge.nr = 5
        after = bridge.snapshot_state()
        np.testing.assert_array_equal(after.psi, before.psi)
        np.testing.assert_array_equal(after.j_phi, before.j_phi)
        assert after.boundary_value == before.boundary_value
