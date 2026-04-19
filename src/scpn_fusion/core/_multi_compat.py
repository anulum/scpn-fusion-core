# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Language Fallback Dispatcher
"""Multi-language backend dispatcher with fallback chain.

Probes available acceleration backends in priority order:

    Rust (PyO3) → Mojo → Julia (juliacall) → Go (cgo) → JAX → NumPy

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
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend tier enumeration (lower = faster)
# ---------------------------------------------------------------------------


class BackendTier(IntEnum):
    """Acceleration backend tiers, ordered fastest → slowest."""

    RUST = 0
    MOJO = 1
    JULIA = 2
    GO = 3
    JAX = 4
    NUMPY = 5


_TIER_NAMES: dict[BackendTier, str] = {
    BackendTier.RUST: "rust",
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


def _probe_mojo() -> bool:
    """Check if Mojo interop is available."""
    if os.environ.get("SCPN_DISABLE_MOJO", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        import scpn_fusion_mojo  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


def _probe_julia() -> bool:
    """Check if Julia is available via juliacall."""
    if os.environ.get("SCPN_DISABLE_JULIA", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        from juliacall import Main as jl  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
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
    except ImportError:
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
        _availability[BackendTier.MOJO] = _probe_mojo()
        _availability[BackendTier.JULIA] = _probe_julia()
        _availability[BackendTier.GO] = _probe_go()
        _availability[BackendTier.JAX] = _probe_jax()
        _availability[BackendTier.NUMPY] = True  # always available

        available = [
            _TIER_NAMES[t] for t in sorted(_availability) if _availability[t]
        ]
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
                    try:
                        from scpn_fusion.fallback_telemetry import (
                            record_fallback_event,
                        )

                        record_fallback_event(
                            "multi_backend",
                            f"{name}_fallback_to_{tier_name}",
                            context={
                                "kernel": name,
                                "selected_tier": tier_name,
                                "fastest_tier": _TIER_NAMES[entries[0][0]],
                            },
                        )
                    except Exception:
                        pass  # telemetry must never block dispatch

                return impl

        available_tiers = [_TIER_NAMES[t] for t, _ in entries]
        raise RuntimeError(
            f"All registered backends for {name!r} are unavailable. "
            f"Registered tiers: {available_tiers}"
        )


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
                f"{_TIER_NAMES[t]}{'*' if _availability.get(t, False) else ''}"
                for t, _ in entries
            ]
        return result


# ---------------------------------------------------------------------------
# Convenience: pre-register existing Rust and NumPy fallbacks
# ---------------------------------------------------------------------------

def _bootstrap_existing_backends() -> None:
    """Register kernels that already have Rust + Python implementations.

    Called lazily on first dispatch if the _rust_compat module is loadable.
    This bridges the existing two-tier system into the new multi-tier
    dispatcher without modifying _rust_compat.py.
    """
    try:
        from scpn_fusion.core._rust_compat import _RUST_AVAILABLE

        if _RUST_AVAILABLE:
            from scpn_fusion.core._rust_compat import (
                rust_shafranov_bv,
                rust_solve_coil_currents,
                rust_measure_magnetics,
                rust_simulate_tearing_mode,
                rust_multigrid_vcycle,
            )

            register_kernel("shafranov_bv", BackendTier.RUST, rust_shafranov_bv)
            register_kernel(
                "solve_coil_currents", BackendTier.RUST, rust_solve_coil_currents
            )
            register_kernel(
                "measure_magnetics", BackendTier.RUST, rust_measure_magnetics
            )
            register_kernel(
                "simulate_tearing_mode",
                BackendTier.RUST,
                rust_simulate_tearing_mode,
            )
            if rust_multigrid_vcycle is not None:
                register_kernel(
                    "multigrid_vcycle", BackendTier.RUST, rust_multigrid_vcycle
                )
    except ImportError:
        pass


# Run bootstrap on import
_bootstrap_existing_backends()
