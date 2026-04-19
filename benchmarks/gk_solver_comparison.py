# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Benchmark Script
"""
End-to-End Benchmark for the Nonlinear GK Solver.
Compares the pure Python (NumPy) implementation with the Rust extension.
"""

import time
import numpy as np

from scpn_fusion.core.gk_nonlinear import (
    NonlinearGKSolver,
    NonlinearGKConfig,
    gk_nonlinear_step_numpy,
    gk_nonlinear_step_rust,
)

def run_benchmark():
    # Small grid for rapid test
    cfg = NonlinearGKConfig()
    cfg.n_steps = 20  # Fast benchmark

    print("Initializing solver state...")
    solver = NonlinearGKSolver(cfg)
    state = solver.init_state(amplitude=1e-5)
    
    # Separate states to ensure no sharing
    state_numpy = state
    state_rust = solver.init_state(amplitude=1e-5)
    state_rust.f = state.f.copy()
    state_rust.phi = state.phi.copy()
    
    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    print("Warming up backends...")
    dt = 0.01
    gk_nonlinear_step_numpy(solver, state_numpy.f, state_numpy.phi, 0.0, dt)
    try:
        gk_nonlinear_step_rust(solver, state_rust.f, state_rust.phi, 0.0, dt)
        rust_available = True
    except ImportError:
        print("Rust backend not available. Only running NumPy.")
        rust_available = False

    # ---------------------------------------------------------
    # NumPy Benchmark
    # ---------------------------------------------------------
    print(f"\nRunning NumPy backend for {cfg.n_steps} steps...")
    start_time = time.perf_counter()
    for _ in range(cfg.n_steps):
        dt = solver._cfl_dt(state_numpy)
        f_new, phi_new = gk_nonlinear_step_numpy(solver, state_numpy.f, state_numpy.phi, state_numpy.time, dt)
        state_numpy.f = f_new
        state_numpy.phi = phi_new
        state_numpy.time += dt
    numpy_time = time.perf_counter() - start_time
    print(f"NumPy time: {numpy_time:.4f} seconds")

    # ---------------------------------------------------------
    # Rust Benchmark
    # ---------------------------------------------------------
    if rust_available:
        print(f"\nRunning Rust backend for {cfg.n_steps} steps...")
        start_time = time.perf_counter()
        for _ in range(cfg.n_steps):
            dt = solver._cfl_dt(state_rust)
            f_new, phi_new = gk_nonlinear_step_rust(solver, state_rust.f, state_rust.phi, state_rust.time, dt)
            state_rust.f = f_new
            state_rust.phi = phi_new
            state_rust.time += dt
        rust_time = time.perf_counter() - start_time
        print(f"Rust time: {rust_time:.4f} seconds")
        
        speedup = numpy_time / rust_time
        print(f"\n>>> Rust backend speedup: {speedup:.2f}x")
        
        # Parity check
        f_diff = np.max(np.abs(state_numpy.f - state_rust.f))
        phi_diff = np.max(np.abs(state_numpy.phi - state_rust.phi))
        print(f"\nMax difference in f: {f_diff:.4e}")
        print(f"Max difference in phi: {phi_diff:.4e}")
        
        if f_diff < 1e-10 and phi_diff < 1e-10:
            print("Parity Check: PASSED")
        else:
            print("Parity Check: FAILED (Numerical divergence detected)")
            
if __name__ == "__main__":
    run_benchmark()
