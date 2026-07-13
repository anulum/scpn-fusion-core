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


# ---------------------------------------------------------------------------
# Bootstrap on import
# ---------------------------------------------------------------------------
# The concrete tier providers and their registrations live in the sibling
# module :mod:`scpn_fusion.core._multi_compat_providers`. It is imported and its
# bootstraps are run here, at the bottom of the engine module body, so every
# engine name it depends on (``BackendTier``, ``register_kernel``,
# ``register_kernel_class``) is already defined — the relationship is acyclic.
# Running the bootstraps on import preserves the historic behaviour that the
# kernels are registered as a side effect of ``import
# scpn_fusion.core._multi_compat``. The import is inside the thunk so it is a
# function-body import (not a top-level statement), keeping the engine's import
# block clean.


def _run_bootstrap() -> None:
    """Register the concrete providers from the sibling module on import."""
    from scpn_fusion.core import _multi_compat_providers

    _multi_compat_providers._bootstrap_existing_backends()
    _multi_compat_providers._bootstrap_kernel_classes()


_run_bootstrap()
