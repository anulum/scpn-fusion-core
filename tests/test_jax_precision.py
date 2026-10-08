# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Precision Boundary Tests
"""Exercise declared precision through fresh real model processes."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import coverage
import pytest


_PROBE = """
import json, os, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import numpy as np
import jax
requested = os.environ["JAX_ENABLE_X64"] == "1"
assert jax.config.x64_enabled == requested
from scpn_fusion.core import gpu_runtime, jax_solvers, jax_gk_nonlinear
from scpn_fusion.control import jax_traceable_runtime
from scpn_fusion.core import frc_rigid_rotor_solver
from scpn_fusion.core.jax_precision import JaxPrecisionRefusal
assert jax.config.x64_enabled == requested
model = sys.argv[1]
observed = []

def profile(frame, event, argument):
    if event != "return":
        return
    module = frame.f_globals.get("__name__")
    name = frame.f_code.co_name
    if module == gpu_runtime.__name__ and name == "_jax_multigrid":
        if isinstance(argument, np.ndarray):
            observed.append(str(argument.dtype))
    elif module == jax_traceable_runtime.__name__ and name in (
        "_simulate_jax", "_simulate_jax_batch"
    ):
        hist = frame.f_locals.get("hist")
        if hist is not None:
            observed.append(str(hist.dtype))

def execute():
    if model == "gpu":
        result = gpu_runtime.GPURuntimeBridge(seed=42).benchmark_equilibrium_latency(
            backend="jax", trials=16, grid_size=32
        )
        assert result.backend == "jax" and result.trials == 16
    elif model == "transport":
        n = 10
        result = jax_solvers.thomas_solve(
            np.zeros(n-1), np.ones(n), np.zeros(n-1), np.arange(n, dtype=np.float64)
        )
        np.testing.assert_array_equal(result, np.arange(n, dtype=np.float64))
        observed.append(str(result.dtype))
        rho = np.linspace(0.01, 1.0, 32)
        values = np.full(32, 5.0)
        chi = np.full(32, 0.5)
        source = np.zeros(32)
        dr = float(rho[1]-rho[0])
        for result in (
            jax_solvers.diffusion_rhs(values, chi, rho, dr),
            jax_solvers.crank_nicolson_step(values, chi, source, rho, dr, 0.01),
            jax_solvers.crank_nicolson_step_jax(values, chi, source, rho, dr, 0.01),
            jax_solvers.batched_crank_nicolson(
                np.stack([values, values]), chi, source, rho, dr, 0.01
            ),
        ):
            observed.append(str(result.dtype))
            assert np.all(np.isfinite(np.asarray(result)))
    elif model == "control":
        commands = np.array([0.5, 1.2, -0.7, 0.1, 0.0])
        spec = jax_traceable_runtime.TraceableRuntimeSpec(
            dt_s=0.002, tau_s=0.010, gain=2.0, command_limit=1.0
        )
        single = jax_traceable_runtime.run_traceable_control_loop(
            commands, initial_state=0.25, spec=spec, backend="jax"
        )
        batch = jax_traceable_runtime.run_traceable_control_batch(
            np.stack([commands, commands]), initial_state=np.array([0.25, 0.25]),
            spec=spec, backend="jax"
        )
        assert single.backend_used == batch.backend_used == "jax"
        np.testing.assert_array_equal(batch.state_history[0], single.state_history)
    elif model == "frc":
        result = frc_rigid_rotor_solver.frc_no_rotation_jax_observables(
            np.linspace(0.0, 2.0, 401), n0=1.0e19, T_i_eV=200.0,
            T_e_eV=180.0, R_s=0.3, B_ext=0.5, delta=0.02
        )
        for name in ("B_z", "psi", "energy_J"):
            observed.append(str(result[name].dtype))
            assert np.all(np.isfinite(np.asarray(result[name])))
    elif model == "gk":
        from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig
        cfg = NonlinearGKConfig(
            n_kx=4, n_ky=4, n_theta=8, n_vpar=4, n_mu=3, n_species=2,
            kinetic_electrons=True, electromagnetic=True, nonlinear=True,
            collisions=False, hyper_coeff=0.0, dt=0.005, n_steps=3,
            save_interval=1, cfl_adapt=False
        )
        result = jax_gk_nonlinear.JaxNonlinearGKSolver(cfg).run()
        assert result.final_state is not None and result.time.size > 0
        assert np.all(np.isfinite(result.final_state.f))
        observed.append(str(result.final_state.f.dtype))
    else:
        raise AssertionError("Unknown public model")

previous = sys.getprofile()
try:
    if model in ("gpu", "control"):
        sys.setprofile(profile)
    try:
        execute()
    except JaxPrecisionRefusal as refusal:
        assert not requested
        assert refusal.reason_code == "unsupported_precision"
        assert refusal.required_dtype == "float64" and refusal.x64_enabled is False
        status = "refused"
    else:
        assert requested and observed
        assert set(observed) <= {"float64", "complex128"}
        status = "completed"
finally:
    sys.setprofile(previous)
assert jax.config.x64_enabled == requested
print(json.dumps({"model":model, "requested_x64":requested,
    "effective_x64_after":jax.config.x64_enabled, "actual_dtypes":observed,
    "status":status, "jax_version":jax.__version__, "worker_pid":os.getpid(),
    "actual_backend":jax.default_backend() if requested else None}), flush=True)
"""


@pytest.mark.parametrize("enabled", [False, True], ids=["x64_disabled", "x64_enabled"])
@pytest.mark.parametrize("model", ["gpu", "transport", "control", "frc", "gk"])
def test_public_model_preserves_requested_precision(
    tmp_path: Path, model: str, enabled: bool
) -> None:
    """Keep import/call configuration and enforce actual declared model precision.

    Parameters
    ----------
    tmp_path : Path
        Explicit finite-job output namespace for original child evidence.
    model : str
        Real public fixed-precision model surface exercised in the fresh child.
    enabled : bool
        Application-selected X64 mode before any model import or execution.
    """
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1" if enabled else "0"
    original = subprocess.run(
        [sys.executable, "-B", "-c", _PROBE, model],
        env=environment,
        capture_output=True,
        timeout=35,
        check=False,
    )
    (tmp_path / "precision.stdout").write_bytes(original.stdout)
    (tmp_path / "precision.stderr").write_bytes(original.stderr)
    assert original.returncode == 0, original.stderr.decode(errors="replace")
    record = json.loads(original.stdout)
    assert record["requested_x64"] is enabled and record["effective_x64_after"] is enabled
    assert record["status"] == ("completed" if enabled else "refused")
    assert type(record["worker_pid"]) is int and record["worker_pid"] != os.getpid()
    assert record["actual_backend"] == ("cpu" if enabled else None)


def test_missing_jax_preserves_original_numpy_model(tmp_path: Path) -> None:
    """Run the original gyrokinetic fallback in an actual image without JAX.

    Parameters
    ----------
    tmp_path : Path
        Own directory containing installed dependencies with JAX absent.
    """
    dependencies = tmp_path / "dependencies_without_jax"
    dependencies.mkdir()
    specification = importlib.util.find_spec("numpy")
    assert specification is not None and specification.origin is not None
    installed_dependencies = Path(specification.origin).resolve().parent.parent
    for entry in installed_dependencies.iterdir():
        if entry.name.startswith("jax") or entry.suffix == ".pth":
            continue
        (dependencies / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(dependencies)))
    active_measurement = coverage.Coverage.current()
    if active_measurement is not None and active_measurement.config.config_file is not None:
        environment["COVERAGE_PROCESS_START"] = active_measurement.config.config_file
    probe = """
import importlib.util, json, os, resource, runpy
resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import coverage
coverage.process_startup()
assert importlib.util.find_spec("jax") is None
from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available
from scpn_fusion.core.gk_nonlinear import NonlinearGKSolver
assert not jax_available()
original = runpy.run_path("tests/test_gk_nonlinear.py")
solver = JaxNonlinearGKSolver(original["_FAST_CFG"])
state = NonlinearGKSolver(original["_FAST_CFG"]).init_state()
diagnostics = solver.nonlinear_invariant_diagnostics(state)
assert diagnostics.finite
original["TestJaxFallback"]().test_jax_solver_runs()
print(json.dumps({"jax_available": jax_available(), "original_numpy_case": True}))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-S", "-c", probe],
        cwd=root,
        env=environment,
        capture_output=True,
        timeout=35,
        check=False,
    )
    (tmp_path / "missing_jax.stdout").write_bytes(completed.stdout)
    (tmp_path / "missing_jax.stderr").write_bytes(completed.stderr)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert json.loads(completed.stdout) == {
        "jax_available": False,
        "original_numpy_case": True,
    }
