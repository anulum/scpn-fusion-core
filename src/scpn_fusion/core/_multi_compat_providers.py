# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Language Backend Providers and Bootstrap
"""Concrete Rust/NumPy/GPU tier providers and dispatcher bootstrap wiring.

This module holds the concrete backend adapters that bridge the existing
Rust/NumPy/GPU implementations into the uniform kernel signatures expected by
the dispatcher engine (:mod:`scpn_fusion.core._multi_compat`), the stateful
kernel-class loaders, and the two bootstrap routines that register them.

The engine imports this module at the bottom of its own module body and calls
:func:`_bootstrap_existing_backends` and :func:`_bootstrap_kernel_classes` so
registration still runs on ``import scpn_fusion.core._multi_compat``. Each tier
provider imports its backend lazily (inside the call), so a tier can be
registered even when its backend is absent — the availability probe in
:func:`scpn_fusion.core._multi_compat.dispatch` selects the fastest *available*
tier at call time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from scpn_fusion.core._multi_compat import BackendTier, register_kernel, register_kernel_class

# ---------------------------------------------------------------------------
# Kernel-class (factory) loaders and bootstrap
# ---------------------------------------------------------------------------


def _load_rust_equilibrium_kernel() -> type:
    """Load the Rust-accelerated equilibrium kernel class."""
    from scpn_fusion.core._rust_compat import RustAcceleratedKernel

    return RustAcceleratedKernel


def _load_numpy_equilibrium_kernel() -> type:
    """Load the pure-Python equilibrium kernel class."""
    from scpn_fusion.core.fusion_kernel import FusionKernel

    return FusionKernel


def _load_rust_hall_mhd() -> type:
    """Load the Rust reduced Hall-MHD discovery simulator class."""
    module = import_module("scpn_fusion_rs")
    simulator = module.PyHallMHD
    if not isinstance(simulator, type):
        raise TypeError("scpn_fusion_rs.PyHallMHD is not a class")
    return simulator


def _load_numpy_hall_mhd() -> type:
    """Load the pure-Python reduced Hall-MHD discovery simulator class."""
    from scpn_fusion.core.hall_mhd_discovery import HallMHD

    return HallMHD


def _load_rust_fokker_planck() -> type:
    """Load the Rust runaway-electron Fokker-Planck solver class."""
    module = import_module("scpn_fusion_rs")
    solver = module.PyFokkerPlanckSolver
    if not isinstance(solver, type):
        raise TypeError("scpn_fusion_rs.PyFokkerPlanckSolver is not a class")
    return solver


def _load_numpy_fokker_planck() -> type:
    """Load the NumPy-tier runaway-electron Fokker-Planck kernel class."""
    from scpn_fusion.control.fokker_planck_re import FokkerPlanckKernel

    return FokkerPlanckKernel


def _load_rust_mpc_controller() -> type:
    """Load the Rust canonical-configuration surrogate MPC controller class."""
    module = import_module("scpn_fusion_rs")
    controller = module.PyMpcController
    if not isinstance(controller, type):
        raise TypeError("scpn_fusion_rs.PyMpcController is not a class")
    return controller


def _load_numpy_mpc_controller() -> type:
    """Load the NumPy-tier canonical-configuration surrogate MPC kernel class."""
    from scpn_fusion.control.neural_surrogate_mpc import MpcKernel

    return MpcKernel


def _load_rust_fno() -> type:
    """Load the Rust FNO turbulence surrogate kernel class."""
    module = import_module("scpn_fusion_rs")
    controller = module.PyFnoController
    if not isinstance(controller, type):
        raise TypeError("scpn_fusion_rs.PyFnoController is not a class")
    from scpn_fusion.core.fno_training import _FnoRustKernel

    return _FnoRustKernel


def _load_numpy_fno() -> type:
    """Load the NumPy-tier FNO turbulence surrogate kernel class."""
    from scpn_fusion.core.fno_training import FnoKernel

    return FnoKernel


def _load_rust_design_evaluator() -> type:
    """Load the Rust reactor-design single-point evaluator kernel class."""
    module = import_module("scpn_fusion_rs")
    evaluate = module.py_evaluate_design
    if not callable(evaluate):
        raise TypeError("scpn_fusion_rs.py_evaluate_design is not callable")
    from scpn_fusion.core.global_design_scanner import _DesignEvaluatorRustKernel

    return _DesignEvaluatorRustKernel


def _load_numpy_design_evaluator() -> type:
    """Load the NumPy-tier reactor-design evaluator (full Python explorer)."""
    from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer

    return GlobalDesignExplorer


def _bootstrap_kernel_classes() -> None:
    """Register the stateful kernel classes with their backend tiers.

    The equilibrium kernel dispatches Rust → NumPy: ``RustAcceleratedKernel`` is
    a drop-in for ``FusionKernel`` (same attribute/method interface), so the
    fastest available class is interchangeable for callers.

    The Hall-MHD discovery simulator dispatches Rust → NumPy on the shared
    sim-loop protocol (``(n, eta, nu, *, seed, background_amplitude)``
    construction, ``step() -> (total_energy, zonal_energy)``, ``run``, and the
    ``energy_history`` sequence). Both tiers implement the same reconciled
    reduced Hall-MHD model (hyper-viscous ``-nu k^4 U``, resistive
    ``-eta k^2 psi``, optional static current-sheet drive); trajectories are
    statistically equivalent, not bit-exact, because the seeded RNG streams
    are language-native.

    The runaway-electron Fokker-Planck kernel dispatches Rust -> NumPy on the
    shared ``(np_grid, p_max)`` construction and ``step(dt, e_field, n_e,
    t_e_ev, z_eff) -> (n_re, current_re)`` contract (the NumPy tier is the
    :class:`~scpn_fusion.control.fokker_planck_re.FokkerPlanckKernel` adapter
    over ``FokkerPlanckSolver``; the Rust tier is ``PyFokkerPlanckSolver``).
    Both implement the identical MUSCL-Hancock advection / central-difference
    diffusion / operator-split source scheme, so the diagnostics are
    deterministic and agree to floating-point summation order in bounded
    regimes; in exponentially growing regimes those round-off differences
    amplify, as for any explicit scheme.

    The surrogate-MPC controller dispatches Rust -> NumPy on the shared
    ``(b_matrix, target)`` construction and ``plan(state) -> action`` contract
    (the NumPy tier is the
    :class:`~scpn_fusion.control.neural_surrogate_mpc.MpcKernel` adapter over
    ``ModelPredictiveController``; the Rust tier is ``PyMpcController``). Both
    run the identical gradient-descent planner over the linear surrogate
    ``x_{t+1} = x_t + B u_t`` at the canonical configuration, so ``plan``
    agrees to floating-point round-off (bit-exact in practice).

    The FNO turbulence surrogate dispatches Rust -> NumPy on the shared
    ``(weights_path)`` construction and ``predict`` / ``predict_and_suppress``
    contract (the NumPy tier is the
    :class:`~scpn_fusion.core.fno_training.FnoKernel` adapter over
    ``MultiLayerFNO``; the Rust tier wraps ``PyFnoController.from_npz``). Both
    run the identical spectral FNO forward over the same weight archive, so the
    prediction and suppression factor agree to floating-point round-off.

    The reactor-design evaluator dispatches Rust -> NumPy on the shared
    ``evaluate_design(R_maj, B_field, I_plasma) -> dict`` contract (the NumPy
    tier is the full :class:`~scpn_fusion.core.global_design_scanner.GlobalDesignExplorer`;
    the Rust tier is the
    :class:`~scpn_fusion.core.global_design_scanner._DesignEvaluatorRustKernel`
    over ``scpn_fusion_rs.py_evaluate_design``). Both run the identical
    physics-scaling surrogate — Troyon/H-mode ``beta_N`` shaping, Eich divertor
    scaling, and the HEAT-ML magnetic-shadow ridge attenuation with the same
    frozen weights and engineering-constraint caps — so ``evaluate_design``
    agrees to floating-point round-off (~1e-15 relative across the design
    envelope, ``Constraint_OK`` identical). The Monte Carlo ``run_scan`` driver
    stays NumPy-only (its rejection sampler uses a language-native RNG stream).
    """
    register_kernel_class("equilibrium_kernel", BackendTier.RUST, _load_rust_equilibrium_kernel)
    register_kernel_class("equilibrium_kernel", BackendTier.NUMPY, _load_numpy_equilibrium_kernel)
    register_kernel_class("hall_mhd_discovery", BackendTier.RUST, _load_rust_hall_mhd)
    register_kernel_class("hall_mhd_discovery", BackendTier.NUMPY, _load_numpy_hall_mhd)
    register_kernel_class("fokker_planck_re", BackendTier.RUST, _load_rust_fokker_planck)
    register_kernel_class("fokker_planck_re", BackendTier.NUMPY, _load_numpy_fokker_planck)
    register_kernel_class("neural_surrogate_mpc", BackendTier.RUST, _load_rust_mpc_controller)
    register_kernel_class("neural_surrogate_mpc", BackendTier.NUMPY, _load_numpy_mpc_controller)
    register_kernel_class("fno_turbulence", BackendTier.RUST, _load_rust_fno)
    register_kernel_class("fno_turbulence", BackendTier.NUMPY, _load_numpy_fno)
    register_kernel_class("global_design_scan", BackendTier.RUST, _load_rust_design_evaluator)
    register_kernel_class("global_design_scan", BackendTier.NUMPY, _load_numpy_design_evaluator)


# ---------------------------------------------------------------------------
# Function-kernel tier providers
# ---------------------------------------------------------------------------


def _numpy_shafranov_bv(
    r_geo: float,
    a_min: float,
    ip_ma: float,
    *,
    beta_p: float = 0.5,
    li: float = 0.8,
) -> float:
    """NumPy-tier provider for the ``shafranov_bv`` kernel.

    The import is deferred to call time so registration neither pulls in the
    control package nor forms an import cycle with this dispatcher.
    """
    from scpn_fusion.control.analytic_solver import shafranov_bv

    return shafranov_bv(r_geo, a_min, ip_ma, beta_p=beta_p, li=li)


def _rust_shafranov_bv(
    r_geo: float,
    a_min: float,
    ip_ma: float,
    *,
    beta_p: float = 0.5,
    li: float = 0.8,
) -> float:
    """Rust-tier provider for the ``shafranov_bv`` kernel.

    Returns the canonical vertical field :math:`B_v` [T] (the Rust pyfunction
    also returns the two diagnostic force-balance terms, which are dropped here
    so the tier is bit-exact interchangeable with :func:`_numpy_shafranov_bv`).
    """
    from scpn_fusion_rs import shafranov_bv as _rs_shafranov_bv

    bv, _term_log, _term_physics = _rs_shafranov_bv(r_geo, a_min, ip_ma, beta_p, li)
    return float(bv)


def _numpy_solve_coil_currents(
    green_func: Any,
    target_bv: float,
    *,
    ridge_lambda: float = 0.0,
) -> Any:
    """NumPy-tier provider for the ``solve_coil_currents`` kernel."""
    from scpn_fusion.control.analytic_solver import solve_coil_currents

    return solve_coil_currents(green_func, target_bv, ridge_lambda=ridge_lambda)


def _rust_solve_coil_currents(
    green_func: Any,
    target_bv: float,
    *,
    ridge_lambda: float = 0.0,
) -> Any:
    """Rust-tier provider for the ``solve_coil_currents`` kernel.

    Wraps the Rust list result back into a float64 array so the tier is
    type-compatible with :func:`_numpy_solve_coil_currents`.
    """
    import numpy as np

    from scpn_fusion_rs import solve_coil_currents as _rs_solve_coil_currents

    coils = _rs_solve_coil_currents(
        np.asarray(green_func, dtype=np.float64).tolist(),
        float(target_bv),
        float(ridge_lambda),
    )
    return np.asarray(coils, dtype=np.float64)


def _numpy_measure_magnetics(
    psi: Any,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> Any:
    """NumPy-tier provider for the ``measure_magnetics`` kernel."""
    from scpn_fusion.diagnostics.synthetic_sensors import measure_magnetics

    return measure_magnetics(psi, nr, nz, r_min, r_max, z_min, z_max)


def _rust_measure_magnetics(
    psi: Any,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> Any:
    """Rust-tier provider for the ``measure_magnetics`` kernel.

    Normalises the Rust result into a float64 array so the tier is
    type-compatible with :func:`_numpy_measure_magnetics`.
    """
    import numpy as np

    from scpn_fusion_rs import measure_magnetics as _rs_measure_magnetics

    measurements = _rs_measure_magnetics(
        np.asarray(psi, dtype=np.float64), nr, nz, r_min, r_max, z_min, z_max
    )
    return np.asarray(measurements, dtype=np.float64)


def _numpy_multigrid_solve(
    source: Any,
    psi_bc: Any,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    *,
    tol: float = 1e-6,
    max_cycles: int = 500,
) -> Any:
    """NumPy-tier provider for the ``multigrid_solve`` kernel.

    Returns ``(psi, residual, n_cycles, converged)`` from the free-function
    geometric multigrid solve.
    """
    from scpn_fusion.core.multigrid_solve import multigrid_solve

    return multigrid_solve(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=tol, max_cycles=max_cycles
    )


def _rust_multigrid_solve(
    source: Any,
    psi_bc: Any,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    *,
    tol: float = 1e-6,
    max_cycles: int = 500,
) -> Any:
    """Rust-tier provider for the ``multigrid_solve`` kernel.

    Normalises the Rust result into ``(psi: float64 array, residual, n_cycles,
    converged)`` so the tier is type-compatible with :func:`_numpy_multigrid_solve`.
    """
    import numpy as np

    from scpn_fusion.core._rust_compat import rust_multigrid_vcycle

    result = rust_multigrid_vcycle(
        source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol=tol, max_cycles=max_cycles
    )
    if result is None:
        raise RuntimeError("Rust multigrid backend is unavailable.")
    psi, residual, n_cycles, converged = result
    return np.asarray(psi, dtype=np.float64), float(residual), int(n_cycles), bool(converged)


def _numpy_simulate_tearing_mode(
    steps: int = 1000,
    *,
    seed: int | None = None,
    beta_p: float = 0.8,
    w_crit: float = 0.05,
) -> Any:
    """NumPy-tier provider for the ``simulate_tearing_mode`` kernel."""
    import numpy as np

    from scpn_fusion.control.disruption_risk_runtime import simulate_tearing_mode

    rng = np.random.default_rng(seed) if seed is not None else None
    return simulate_tearing_mode(steps, rng=rng, beta_p=beta_p, w_crit=w_crit)


def _rust_simulate_tearing_mode(
    steps: int = 1000,
    *,
    seed: int | None = None,
    beta_p: float = 0.8,
    w_crit: float = 0.05,
) -> Any:
    """Rust-tier provider for the ``simulate_tearing_mode`` kernel.

    Normalises the Rust ``(signal, label, ttd)`` tuple so the signal is a float64
    array, type-compatible with :func:`_numpy_simulate_tearing_mode`.
    """
    import numpy as np

    from scpn_fusion.core._rust_compat import rust_simulate_tearing_mode

    signal, label, ttd = rust_simulate_tearing_mode(steps, seed, beta_p, w_crit)
    return np.asarray(signal, dtype=np.float64), int(label), int(ttd)


def _numpy_kuramoto_step(
    theta: Any,
    omega: Any,
    *,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi: float = 0.0,
    wrap: bool = True,
) -> Any:
    """NumPy-tier provider for the ``kuramoto_step`` kernel."""
    from scpn_fusion.phase.kuramoto import _kuramoto_step_numpy

    return _kuramoto_step_numpy(
        theta, omega, dt=dt, K=K, alpha=alpha, zeta=zeta, psi=psi, wrap=wrap
    )


def _rust_kuramoto_step(
    theta: Any,
    omega: Any,
    *,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi: float = 0.0,
    wrap: bool = True,
) -> Any:
    """Rust-tier provider for the ``kuramoto_step`` kernel.

    Normalises the PyO3 dict payload so the tier is type-compatible with
    :func:`_numpy_kuramoto_step` (float64 arrays, same keys).
    """
    import numpy as np

    from scpn_fusion_rs import py_kuramoto_step

    result = py_kuramoto_step(
        np.asarray(theta, dtype=np.float64).ravel(),
        np.asarray(omega, dtype=np.float64).ravel(),
        float(dt),
        float(K),
        alpha=float(alpha),
        zeta=float(zeta),
        psi=float(psi),
        wrap=bool(wrap),
    )
    result["theta1"] = np.asarray(result["theta1"], dtype=np.float64)
    result["dtheta"] = np.asarray(result["dtheta"], dtype=np.float64)
    return result


def _numpy_kuramoto_run(
    theta: Any,
    omega: Any,
    *,
    n_steps: int,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi: float = 0.0,
    wrap: bool = True,
) -> Any:
    """NumPy-tier provider for the batched ``kuramoto_run`` kernel.

    Iterates the single-step NumPy kernel with a constant driver phase and
    records the per-step order parameter, matching the Rust batched tier's
    contract so the two are dispatch-interchangeable.
    """
    import numpy as np

    from scpn_fusion.phase.kuramoto import _kuramoto_step_numpy

    theta_state: Any = np.asarray(theta, dtype=np.float64).ravel().copy()
    omega_arr = np.asarray(omega, dtype=np.float64).ravel()
    r_hist = np.empty(int(n_steps), dtype=np.float64)
    psi_r_hist = np.empty(int(n_steps), dtype=np.float64)
    for step in range(int(n_steps)):
        out = _kuramoto_step_numpy(
            theta_state, omega_arr, dt=dt, K=K, alpha=alpha, zeta=zeta, psi=psi, wrap=wrap
        )
        r_hist[step] = out["R"]
        psi_r_hist[step] = out["Psi_r"]
        theta_state = np.asarray(out["theta1"], dtype=np.float64)
    return {
        "theta_final": theta_state,
        "R_hist": r_hist,
        "Psi_r_hist": psi_r_hist,
        "Psi": float(psi),
    }


def _rust_kuramoto_run(
    theta: Any,
    omega: Any,
    *,
    n_steps: int,
    dt: float,
    K: float,
    alpha: float = 0.0,
    zeta: float = 0.0,
    psi: float = 0.0,
    wrap: bool = True,
) -> Any:
    """Rust-tier provider for the batched ``kuramoto_run`` kernel.

    Normalises the PyO3 dict payload so the tier is type-compatible with
    :func:`_numpy_kuramoto_run` (float64 arrays, same keys).
    """
    import numpy as np

    from scpn_fusion_rs import py_kuramoto_run

    result = py_kuramoto_run(
        np.asarray(theta, dtype=np.float64).ravel(),
        np.asarray(omega, dtype=np.float64).ravel(),
        int(n_steps),
        float(dt),
        float(K),
        alpha=float(alpha),
        zeta=float(zeta),
        psi=float(psi),
        wrap=bool(wrap),
    )
    result["theta_final"] = np.asarray(result["theta_final"], dtype=np.float64)
    result["R_hist"] = np.asarray(result["R_hist"], dtype=np.float64)
    result["Psi_r_hist"] = np.asarray(result["Psi_r_hist"], dtype=np.float64)
    return result


def _numpy_upde_tick(
    theta_flat: Any,
    omega_flat: Any,
    offsets: Any,
    K: Any,
    alpha: Any,
    zeta: Any,
    *,
    dt: float,
    psi_global: float,
    actuation_gain: float = 1.0,
    pac_gamma: float = 0.0,
    wrap: bool = True,
) -> Any:
    """NumPy-tier provider for the ``upde_tick`` kernel."""
    from scpn_fusion.phase.upde import _upde_tick_numpy

    return _upde_tick_numpy(
        theta_flat,
        omega_flat,
        offsets,
        K,
        alpha,
        zeta,
        dt=dt,
        psi_global=psi_global,
        actuation_gain=actuation_gain,
        pac_gamma=pac_gamma,
        wrap=wrap,
    )


def _rust_upde_tick(
    theta_flat: Any,
    omega_flat: Any,
    offsets: Any,
    K: Any,
    alpha: Any,
    zeta: Any,
    *,
    dt: float,
    psi_global: float,
    actuation_gain: float = 1.0,
    pac_gamma: float = 0.0,
    wrap: bool = True,
) -> Any:
    """Rust-tier provider for the ``upde_tick`` kernel.

    Normalises the PyO3 dict payload so the tier is type-compatible with
    :func:`_numpy_upde_tick` (float64 arrays, same keys).
    """
    import numpy as np

    from scpn_fusion_rs import py_upde_tick

    result = py_upde_tick(
        np.asarray(theta_flat, dtype=np.float64).ravel(),
        np.asarray(omega_flat, dtype=np.float64).ravel(),
        [int(v) for v in np.asarray(offsets).ravel()],
        np.ascontiguousarray(K, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.asarray(zeta, dtype=np.float64).ravel(),
        float(dt),
        float(psi_global),
        actuation_gain=float(actuation_gain),
        pac_gamma=float(pac_gamma),
        wrap=bool(wrap),
    )
    for key in ("theta1", "dtheta", "R_layer", "Psi_layer", "V_layer"):
        result[key] = np.asarray(result[key], dtype=np.float64)
    return result


def _numpy_upde_run(
    theta_flat: Any,
    omega_flat: Any,
    offsets: Any,
    K: Any,
    alpha: Any,
    zeta: Any,
    *,
    n_steps: int,
    dt: float,
    psi_global: float,
    actuation_gain: float = 1.0,
    pac_gamma: float = 0.0,
    wrap: bool = True,
) -> Any:
    """NumPy-tier provider for the batched ``upde_run`` kernel."""
    import numpy as np

    from scpn_fusion.phase.upde import _upde_tick_numpy

    L = int(len(offsets)) - 1
    theta = np.array(theta_flat, dtype=np.float64, copy=True)
    r_layer_hist = np.empty((n_steps, L))
    r_global_hist = np.empty(n_steps)
    v_layer_hist = np.empty((n_steps, L))
    v_global_hist = np.empty(n_steps)
    for step in range(n_steps):
        out = _upde_tick_numpy(
            theta,
            omega_flat,
            offsets,
            K,
            alpha,
            zeta,
            dt=dt,
            psi_global=psi_global,
            actuation_gain=actuation_gain,
            pac_gamma=pac_gamma,
            wrap=wrap,
        )
        theta = out["theta1"]
        r_layer_hist[step] = out["R_layer"]
        r_global_hist[step] = out["R_global"]
        v_layer_hist[step] = out["V_layer"]
        v_global_hist[step] = out["V_global"]
    return {
        "theta_final": theta,
        "R_layer_hist": r_layer_hist,
        "R_global_hist": r_global_hist,
        "V_layer_hist": v_layer_hist,
        "V_global_hist": v_global_hist,
    }


def _rust_upde_run(
    theta_flat: Any,
    omega_flat: Any,
    offsets: Any,
    K: Any,
    alpha: Any,
    zeta: Any,
    *,
    n_steps: int,
    dt: float,
    psi_global: float,
    actuation_gain: float = 1.0,
    pac_gamma: float = 0.0,
    wrap: bool = True,
) -> Any:
    """Rust-tier provider for the batched ``upde_run`` kernel.

    The whole multi-tick loop executes in Rust (``fusion-phase``), which is
    where the batched tier earns its speedup over per-tick dispatch.
    """
    import numpy as np

    from scpn_fusion_rs import py_upde_run

    result = py_upde_run(
        np.asarray(theta_flat, dtype=np.float64).ravel(),
        np.asarray(omega_flat, dtype=np.float64).ravel(),
        [int(v) for v in np.asarray(offsets).ravel()],
        np.ascontiguousarray(K, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.asarray(zeta, dtype=np.float64).ravel(),
        int(n_steps),
        float(dt),
        float(psi_global),
        actuation_gain=float(actuation_gain),
        pac_gamma=float(pac_gamma),
        wrap=bool(wrap),
    )
    for key in ("theta_final", "R_layer_hist", "R_global_hist", "V_layer_hist", "V_global_hist"):
        result[key] = np.asarray(result[key], dtype=np.float64)
    return result


# Cache of GPU GS solver instances keyed by grid geometry. Device and
# pipeline construction costs ~10^2 ms, so re-solves on the same grid must
# not pay it again; the cache holds one wgpu device per distinct geometry.
_gpu_gs_solver_cache: dict[tuple[int, int, float, float, float, float], Any] = {}


def _numpy_gs_rb_sor_smooth(
    psi: Any,
    source: Any,
    r_left: float,
    r_right: float,
    z_bottom: float,
    z_top: float,
    *,
    omega: float = 1.3,
    n_sweeps: int = 50,
) -> Any:
    """NumPy-tier provider for the ``gs_rb_sor_smooth`` kernel.

    Runs ``n_sweeps`` Red-Black SOR sweeps of the toroidal GS* stencil on a
    copy of *psi* (Dirichlet boundary preserved) and returns the smoothed
    float64 array.
    """
    import numpy as np

    from scpn_fusion.core.multigrid_solve import mg_smooth

    psi_arr = np.array(psi, dtype=np.float64, copy=True)
    source_arr = np.asarray(source, dtype=np.float64)
    nz, nr = psi_arr.shape
    r_axis = np.linspace(r_left, r_right, nr)
    z_axis = np.linspace(z_bottom, z_top, nz)
    r_grid, _ = np.meshgrid(r_axis, z_axis)
    dr = (r_right - r_left) / (nr - 1)
    dz = (z_top - z_bottom) / (nz - 1)
    return mg_smooth(psi_arr, source_arr, r_grid, dr, dz, omega, n_sweeps)


def _gpu_gs_rb_sor_smooth(
    psi: Any,
    source: Any,
    r_left: float,
    r_right: float,
    z_bottom: float,
    z_top: float,
    *,
    omega: float = 1.3,
    n_sweeps: int = 50,
) -> Any:
    """GPU-tier provider for the ``gs_rb_sor_smooth`` kernel.

    Identical Red-Black SOR sweeps of the same toroidal GS* stencil executed
    as wgpu compute shaders in f32 (see ``fusion-gpu/src/gs_solver.wgsl``).
    The result is returned as float64 for tier interchangeability; agreement
    with the NumPy tier is bounded by f32 round-off, not by the algorithm.
    The wgpu device is cached per grid geometry.
    """
    import numpy as np

    from scpn_fusion_rs import PyGpuSolver

    psi_arr = np.asarray(psi, dtype=np.float64)
    source_arr = np.asarray(source, dtype=np.float64)
    nz, nr = psi_arr.shape
    key = (nr, nz, float(r_left), float(r_right), float(z_bottom), float(z_top))
    solver = _gpu_gs_solver_cache.get(key)
    if solver is None:
        solver = PyGpuSolver(nr, nz, r_left, r_right, z_bottom, z_top)
        _gpu_gs_solver_cache[key] = solver
    flat = solver.solve(
        psi_arr.astype(np.float32).ravel().tolist(),
        source_arr.astype(np.float32).ravel().tolist(),
        int(n_sweeps),
        float(omega),
    )
    return np.asarray(flat, dtype=np.float64).reshape(nz, nr)


def _bootstrap_existing_backends() -> None:
    """Register the function-kernels that have Rust and/or NumPy implementations.

    Tier providers import their backend lazily, so a tier can be registered even
    when its backend is absent — the availability probe in :func:`dispatch`
    selects the fastest *available* tier at call time. This bridges the existing
    implementations into the multi-tier dispatcher without an import cycle.
    """
    # shafranov_bv — canonical contract reconciled (A2 kernel #1). Both tiers are
    # bit-exact for the returned Bv, so registration is unconditional and the
    # NumPy tier guarantees `dispatch("shafranov_bv")` resolves without Rust.
    register_kernel("shafranov_bv", BackendTier.RUST, _rust_shafranov_bv)
    register_kernel("shafranov_bv", BackendTier.NUMPY, _numpy_shafranov_bv)

    # solve_coil_currents — canonical contract reconciled (A2 kernel #2). Both
    # tiers share the direct minimum-norm/ridge formula (tolerance-aware across
    # the Green's-norm reduction), and the NumPy tier resolves dispatch without
    # Rust.
    register_kernel("solve_coil_currents", BackendTier.RUST, _rust_solve_coil_currents)
    register_kernel("solve_coil_currents", BackendTier.NUMPY, _numpy_solve_coil_currents)

    # measure_magnetics — canonical contract reconciled (A2 kernel #4). Both tiers
    # evaluate the same noise-free bilinear stencil at the same probe positions
    # (tolerance-aware across the trig/position rounding), and the NumPy tier
    # resolves dispatch without Rust.
    register_kernel("measure_magnetics", BackendTier.RUST, _rust_measure_magnetics)
    register_kernel("measure_magnetics", BackendTier.NUMPY, _numpy_measure_magnetics)

    # multigrid_solve — canonical contract reconciled (A2 kernel #3). Both tiers
    # relax the identical toroidal GS* operator with the same Red-Black smoother
    # and grid transfers, converging to the same fixed point (agreement is
    # effectively bit-exact on the standard grids). The NumPy tier resolves
    # dispatch without Rust.
    register_kernel("multigrid_solve", BackendTier.RUST, _rust_multigrid_solve)
    register_kernel("multigrid_solve", BackendTier.NUMPY, _numpy_multigrid_solve)

    # simulate_tearing_mode — canonical contract reconciled (A2 kernel #5). Both
    # tiers run the full Modified Rutherford physics (bootstrap drive, Gaussian
    # process noise, island seeding); the deterministic per-step increment is
    # bit-exact and the stochastic trajectory is statistically equivalent across
    # independent RNG streams. The NumPy tier resolves dispatch without Rust.
    register_kernel("simulate_tearing_mode", BackendTier.RUST, _rust_simulate_tearing_mode)
    register_kernel("simulate_tearing_mode", BackendTier.NUMPY, _numpy_simulate_tearing_mode)

    # gs_rb_sor_smooth — fixed-sweep Red-Black SOR smoothing of the toroidal
    # GS* operator (W-2 kernel). The GPU tier runs the identical stencil as
    # wgpu compute shaders in f32; the NumPy tier is the float64 reference
    # (`multigrid_solve.mg_smooth`). Agreement is f32-round-off bounded, not
    # bit-exact. The GPU tier only exists when the extension is built with
    # `--features gpu` AND a physical adapter is present; the NumPy tier
    # guarantees dispatch resolves everywhere.
    register_kernel("gs_rb_sor_smooth", BackendTier.GPU, _gpu_gs_rb_sor_smooth)
    register_kernel("gs_rb_sor_smooth", BackendTier.NUMPY, _numpy_gs_rb_sor_smooth)

    # kuramoto_step / upde_tick — the SCPN phase-dynamics lane (M-3 kernels).
    # Both are deterministic (no RNG), so cross-tier agreement is bounded
    # only by floating-point summation order (~1e-14 relative). Driver
    # (Psi) resolution policy stays in the phase package; the kernels take
    # the resolved value. The NumPy tier guarantees dispatch resolves
    # everywhere.
    register_kernel("kuramoto_step", BackendTier.RUST, _rust_kuramoto_step)
    register_kernel("kuramoto_step", BackendTier.NUMPY, _numpy_kuramoto_step)
    register_kernel("kuramoto_run", BackendTier.RUST, _rust_kuramoto_run)
    register_kernel("kuramoto_run", BackendTier.NUMPY, _numpy_kuramoto_run)
    register_kernel("upde_tick", BackendTier.RUST, _rust_upde_tick)
    register_kernel("upde_tick", BackendTier.NUMPY, _numpy_upde_tick)
    register_kernel("upde_run", BackendTier.RUST, _rust_upde_run)
    register_kernel("upde_run", BackendTier.NUMPY, _numpy_upde_run)
