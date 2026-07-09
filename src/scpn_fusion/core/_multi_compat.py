# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Language Fallback Dispatcher
"""Multi-language backend dispatcher with fallback chain.

Probes available acceleration backends in priority order:

    Rust (PyO3) → GPU (wgpu) → Mojo → Julia (juliacall) → Go (cgo) → JAX → NumPy

Each backend tier is detected once at import time. Individual compute
kernels register themselves via :func:`register_kernel` and callers use
:func:`dispatch` to invoke the fastest available implementation.

The dispatcher integrates with :mod:`scpn_fusion.fallback_telemetry` to
record backend-selection events and enforce optional budget gates.

Usage::

    from scpn_fusion.core._multi_compat import dispatch, available_backends

    # Get the fastest available GK nonlinear solver
    solver = dispatch("gk_nonlinear_step")
    result = solver(state, dt)

    # Check what's available
    print(available_backends())
    # → {'rust': True, 'mojo': False, 'julia': True, 'go': False,
    #    'jax': True, 'numpy': True}
"""

from __future__ import annotations

import logging
import os
import threading
from importlib import import_module
from importlib.util import find_spec
from enum import IntEnum
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend tier enumeration (lower = faster)
# ---------------------------------------------------------------------------


class BackendTier(IntEnum):
    """Acceleration backend tiers, ordered fastest → slowest.

    Ordering is *relative within each kernel's registered tiers*: a kernel
    only dispatches across the tiers actually registered for it. GPU sits
    after RUST because wgpu dispatch overhead (one submit per Red-Black
    colour per sweep plus PCIe transfers) dominates at production
    equilibrium grid sizes; per-kernel benchmarks decide where GPU is
    registered at all (see ``validation/reports/gpu_gs_solver_benchmark.json``).
    """

    RUST = 0
    GPU = 1
    MOJO = 2
    JULIA = 3
    GO = 4
    JAX = 5
    NUMPY = 6


_TIER_NAMES: dict[BackendTier, str] = {
    BackendTier.RUST: "rust",
    BackendTier.GPU: "gpu",
    BackendTier.MOJO: "mojo",
    BackendTier.JULIA: "julia",
    BackendTier.GO: "go",
    BackendTier.JAX: "jax",
    BackendTier.NUMPY: "numpy",
}

# ---------------------------------------------------------------------------
# Backend availability probes (run once)
# ---------------------------------------------------------------------------

_probe_lock = threading.Lock()
_probed = False
_availability: dict[BackendTier, bool] = {}


def _probe_rust() -> bool:
    """Check if Rust PyO3 extension is importable."""
    try:
        import scpn_fusion_rs  # noqa: F401

        return True
    except ImportError:
        return False


def _probe_gpu() -> bool:
    """Check if the wgpu compute tier is available.

    True only when the Rust extension was built with ``--features gpu``
    (which exports ``py_gpu_available``) AND a physical GPU adapter is
    accepted by the runtime probe (CPU/software adapters are rejected by
    the Rust side unless ``SCPN_FUSION_GPU_ALLOW_CPU_ADAPTER=1``).
    """
    if os.environ.get("SCPN_DISABLE_GPU", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        import scpn_fusion_rs

        probe = getattr(scpn_fusion_rs, "py_gpu_available", None)
        if probe is None:
            return False
        return bool(probe())
    except Exception as exc:
        logger.debug("GPU backend probe failed; treating GPU as unavailable: %s", exc)
        return False


def _probe_mojo() -> bool:
    """Check if Mojo interop is available."""
    if os.environ.get("SCPN_DISABLE_MOJO", "").strip().lower() in ("1", "true", "yes"):
        return False
    return find_spec("scpn_fusion_mojo") is not None


def _probe_julia() -> bool:
    """Check if Julia is available via juliacall."""
    if os.environ.get("SCPN_DISABLE_JULIA", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        juliacall = cast(Any, import_module("juliacall"))
        return juliacall.Main is not None
    except (AttributeError, ImportError):
        return False


def _probe_go() -> bool:
    """Check if Go shared library is loadable."""
    if os.environ.get("SCPN_DISABLE_GO", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        import ctypes

        ctypes.CDLL("libscpn_fusion_go.so")
        return True
    except OSError:
        return False


def _probe_jax() -> bool:
    """Check if JAX is importable."""
    try:
        import jax  # noqa: F401

        return True
    except Exception as exc:
        logger.debug("JAX backend probe failed; treating JAX as unavailable: %s", exc)
        return False


def _ensure_probed() -> None:
    """Run backend probes once (thread-safe)."""
    global _probed
    if _probed:
        return
    with _probe_lock:
        if _probed:
            return
        _availability[BackendTier.RUST] = _probe_rust()
        _availability[BackendTier.GPU] = _probe_gpu()
        _availability[BackendTier.MOJO] = _probe_mojo()
        _availability[BackendTier.JULIA] = _probe_julia()
        _availability[BackendTier.GO] = _probe_go()
        _availability[BackendTier.JAX] = _probe_jax()
        _availability[BackendTier.NUMPY] = True  # always available

        available = [_TIER_NAMES[t] for t in sorted(_availability) if _availability[t]]
        logger.info("Multi-backend probe: %s", ", ".join(available))
        _probed = True


def available_backends() -> dict[str, bool]:
    """Return a mapping of backend name → availability."""
    _ensure_probed()
    return {_TIER_NAMES[t]: v for t, v in sorted(_availability.items())}


def is_available(tier: BackendTier) -> bool:
    """Check if a specific backend tier is available."""
    _ensure_probed()
    return _availability.get(tier, False)


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

# Maps kernel_name → sorted list of (tier, callable)
_registry_lock = threading.Lock()
_registry: dict[str, list[tuple[BackendTier, Callable[..., Any]]]] = {}

# Tracks which tier was selected for each kernel on first dispatch
_dispatch_cache: dict[str, tuple[BackendTier, Callable[..., Any]]] = {}
_rust_symbol_cache: dict[str, tuple[int, Callable[..., object]]] = {}


def _record_fallback_event_safe(
    *,
    kernel: str,
    selected_tier: str,
    fastest_tier: str,
) -> None:
    """Best-effort telemetry emission for backend fallback events."""
    try:
        from scpn_fusion.fallback_telemetry import record_fallback_event

        record_fallback_event(
            "multi_backend",
            f"{kernel}_fallback_to_{selected_tier}",
            context={
                "kernel": kernel,
                "selected_tier": selected_tier,
                "fastest_tier": fastest_tier,
            },
        )
    except Exception as exc:
        logger.debug(
            "Fallback telemetry skipped for kernel=%s selected=%s: %s",
            kernel,
            selected_tier,
            exc,
            exc_info=True,
        )


def register_kernel(
    name: str,
    tier: BackendTier,
    implementation: Callable[..., Any],
) -> None:
    """Register a kernel implementation for a given backend tier.

    Parameters
    ----------
    name : str
        Kernel identifier (e.g. ``"gk_nonlinear_step"``).
    tier : BackendTier
        Backend tier this implementation belongs to.
    implementation : callable
        The actual function to call.
    """
    with _registry_lock:
        if name not in _registry:
            _registry[name] = []
        _registry[name].append((tier, implementation))
        _registry[name].sort(key=lambda x: x[0])
        # Invalidate dispatch cache for this kernel
        _dispatch_cache.pop(name, None)


def dispatch(name: str) -> Callable[..., Any]:
    """Return the fastest available implementation for the named kernel.

    Raises
    ------
    KeyError
        If no implementation is registered for *name*.
    RuntimeError
        If all registered implementations belong to unavailable backends.
    """
    _ensure_probed()

    # Check cache first
    cached = _dispatch_cache.get(name)
    if cached is not None:
        return cached[1]

    with _registry_lock:
        # Re-check under lock
        cached = _dispatch_cache.get(name)
        if cached is not None:
            return cached[1]

        entries = _registry.get(name)
        if not entries:
            raise KeyError(f"No implementations registered for kernel {name!r}")

        for tier, impl in entries:
            if _availability.get(tier, False):
                _dispatch_cache[name] = (tier, impl)
                tier_name = _TIER_NAMES[tier]
                logger.debug("Dispatching %s → %s", name, tier_name)

                # Record the backend selection via fallback telemetry
                # (only if not the fastest tier, i.e. a fallback occurred)
                if tier != entries[0][0]:
                    _record_fallback_event_safe(
                        kernel=name,
                        selected_tier=tier_name,
                        fastest_tier=_TIER_NAMES[entries[0][0]],
                    )

                return impl

        available_tiers = [_TIER_NAMES[t] for t, _ in entries]
        raise RuntimeError(
            f"All registered backends for {name!r} are unavailable. "
            f"Registered tiers: {available_tiers}"
        )


def dispatch_rust_symbol(symbol_name: str) -> Callable[..., object]:
    """Resolve a callable symbol from the Rust PyO3 extension.

    This helper is the single import boundary for production code that needs a
    Rust-only symbol with no NumPy fallback. It keeps optional-extension loading
    behind the same backend-dispatch module used by function and class kernels,
    while preserving the extension's native ``ImportError`` and
    ``AttributeError`` failure contracts for CLI and probe callers.

    Parameters
    ----------
    symbol_name : str
        Name of the callable exported by ``scpn_fusion_rs``.

    Returns
    -------
    callable
        The requested Rust extension callable.

    Raises
    ------
    ImportError
        If the optional Rust extension is unavailable.
    AttributeError
        If the extension does not export ``symbol_name``.
    TypeError
        If the exported symbol is present but is not callable.
    """
    module = import_module("scpn_fusion_rs")
    module_id = id(module)
    cached = _rust_symbol_cache.get(symbol_name)
    if cached is not None and cached[0] == module_id:
        return cached[1]

    symbol = getattr(module, symbol_name)
    if not callable(symbol):
        raise TypeError(f"Rust symbol {symbol_name!r} is not callable.")
    callable_symbol = cast(Callable[..., object], symbol)
    _rust_symbol_cache[symbol_name] = (module_id, callable_symbol)
    return callable_symbol


def dispatch_tier(name: str) -> str:
    """Return the name of the backend tier selected for *name*."""
    _ensure_probed()
    cached = _dispatch_cache.get(name)
    if cached is not None:
        return _TIER_NAMES[cached[0]]

    # Force dispatch to populate cache
    dispatch(name)
    cached = _dispatch_cache.get(name)
    if cached is None:
        raise RuntimeError(f"dispatch_tier failed for {name!r}")
    return _TIER_NAMES[cached[0]]


def registered_kernels() -> dict[str, list[str]]:
    """Return all registered kernels and their available tiers."""
    _ensure_probed()
    with _registry_lock:
        result: dict[str, list[str]] = {}
        for name, entries in sorted(_registry.items()):
            result[name] = [
                f"{_TIER_NAMES[t]}{'*' if _availability.get(t, False) else ''}" for t, _ in entries
            ]
        return result


def registered_kernel_classes() -> dict[str, list[str]]:
    """Return all registered kernel classes and their available tiers.

    Mirrors :func:`registered_kernels` for the stateful class registry: each
    tier name carries a ``*`` suffix when its backend probe reports available.
    """
    _ensure_probed()
    with _class_registry_lock:
        result: dict[str, list[str]] = {}
        for name, entries in sorted(_class_registry.items()):
            result[name] = [
                f"{_TIER_NAMES[t]}{'*' if _availability.get(t, False) else ''}" for t, _ in entries
            ]
        return result


# ---------------------------------------------------------------------------
# Kernel-class (factory) registry — for stateful backends such as the
# equilibrium solver, where the unit of dispatch is a class, not a function.
# ---------------------------------------------------------------------------

_class_registry_lock = threading.Lock()
_class_registry: dict[str, list[tuple[BackendTier, Callable[[], type]]]] = {}
_class_dispatch_cache: dict[str, tuple[BackendTier, type]] = {}


def register_kernel_class(
    name: str,
    tier: BackendTier,
    loader: Callable[[], type],
) -> None:
    """Register a lazily-loaded kernel class for a backend tier.

    Parameters
    ----------
    name : str
        Kernel-class identifier (e.g. ``"equilibrium_kernel"``).
    tier : BackendTier
        Backend tier this class belongs to.
    loader : callable
        Zero-argument thunk returning the class. Deferring the import into the
        thunk keeps registration free of the heavy backend imports and avoids
        the import cycle between this module and the tier providers.
    """
    with _class_registry_lock:
        entries = _class_registry.setdefault(name, [])
        entries.append((tier, loader))
        entries.sort(key=lambda x: x[0])
        _class_dispatch_cache.pop(name, None)


def dispatch_kernel_class(name: str) -> type:
    """Return the fastest available registered class for *name*.

    Raises
    ------
    KeyError
        If no class is registered for *name*.
    RuntimeError
        If every registered tier is unavailable.
    """
    _ensure_probed()
    cached = _class_dispatch_cache.get(name)
    if cached is not None:
        return cached[1]
    with _class_registry_lock:
        cached = _class_dispatch_cache.get(name)
        if cached is not None:
            return cached[1]
        entries = _class_registry.get(name)
        if not entries:
            raise KeyError(f"No kernel class registered for {name!r}")
        for tier, loader in entries:
            if _availability.get(tier, False):
                cls = loader()
                _class_dispatch_cache[name] = (tier, cls)
                if tier != entries[0][0]:
                    _record_fallback_event_safe(
                        kernel=name,
                        selected_tier=_TIER_NAMES[tier],
                        fastest_tier=_TIER_NAMES[entries[0][0]],
                    )
                return cls
        available_tiers = [_TIER_NAMES[t] for t, _ in entries]
        raise RuntimeError(
            f"All registered backends for kernel class {name!r} are unavailable. "
            f"Registered tiers: {available_tiers}"
        )


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
    """
    register_kernel_class("equilibrium_kernel", BackendTier.RUST, _load_rust_equilibrium_kernel)
    register_kernel_class("equilibrium_kernel", BackendTier.NUMPY, _load_numpy_equilibrium_kernel)
    register_kernel_class("hall_mhd_discovery", BackendTier.RUST, _load_rust_hall_mhd)
    register_kernel_class("hall_mhd_discovery", BackendTier.NUMPY, _load_numpy_hall_mhd)


# ---------------------------------------------------------------------------
# Convenience: pre-register existing Rust and NumPy fallbacks
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
    register_kernel("upde_tick", BackendTier.RUST, _rust_upde_tick)
    register_kernel("upde_tick", BackendTier.NUMPY, _numpy_upde_tick)
    register_kernel("upde_run", BackendTier.RUST, _rust_upde_run)
    register_kernel("upde_run", BackendTier.NUMPY, _numpy_upde_run)


# Run bootstrap on import
_bootstrap_existing_backends()
_bootstrap_kernel_classes()
