# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gyrokinetic Solver Registry
"""String-keyed registry and factory for :class:`~scpn_fusion.core.gk_interface.GKSolverBase`.

Before this registry existed the concrete GK solvers (TGLF external, TGLF
native, CGYRO, GS2, GENE, QuaLiKiz) had no shared discovery surface —
callers imported classes directly or not at all. The registry gives
pipelines one canonical way to enumerate, probe, and construct solvers by
name, with lazy class loading so importing this module stays cheap.

Scope boundaries (deliberate, so the registry does not overstate the lane):

- ``core.tglf_interface`` is NOT a solver — it is the TransportSolver
  comparison framework (deck extraction from solver state, JSON reference
  parsing, benchmark tables) and keeps its own contract.
- ``core.gk_nonlinear.NonlinearGKSolver`` is NOT registered — it does not
  implement the :class:`GKSolverBase` deck-file contract (config-driven
  5D delta-f evolution, no ``prepare_input``/``run(path)``); it stays a
  direct-import research lane.

The canonical TGLF path is :func:`resolve_tglf_solver`: the GACODE ``tglf``
binary when it is on ``PATH`` (higher fidelity), otherwise the always
available native quasilinear model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from scpn_fusion.core.gk_interface import GKSolverBase

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()
_registry: dict[str, Callable[[], type[GKSolverBase]]] = {}
_class_cache: dict[str, type[GKSolverBase]] = {}


def register_gk_solver(name: str, loader: Callable[[], type[GKSolverBase]]) -> None:
    """Register a lazily-loaded GK solver class under a canonical name.

    Parameters
    ----------
    name : str
        Registry key (lower-case, e.g. ``"tglf-native"``). Re-registering
        an existing key replaces it (mirrors the dispatcher semantics used
        by ``core._multi_compat``) and invalidates the cached class.
    loader : callable
        Zero-argument thunk returning the solver class. Deferring the
        import keeps registration free of heavy module imports.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("GK solver registry key must be a non-empty string.")
    with _registry_lock:
        _registry[key] = loader
        _class_cache.pop(key, None)


def gk_solver_class(name: str) -> type[GKSolverBase]:
    """Return the solver class registered under *name* (loads lazily, caches).

    Raises
    ------
    KeyError
        If no solver is registered under *name*.
    TypeError
        If the loaded object is not a :class:`GKSolverBase` subclass.
    """
    key = name.strip().lower()
    with _registry_lock:
        cached = _class_cache.get(key)
        if cached is not None:
            return cached
        loader = _registry.get(key)
        if loader is None:
            raise KeyError(
                f"No GK solver registered under {name!r}. Registered: {sorted(_registry)}"
            )
    cls = loader()
    if not (isinstance(cls, type) and issubclass(cls, GKSolverBase)):
        raise TypeError(f"GK solver loader for {name!r} did not return a GKSolverBase subclass.")
    with _registry_lock:
        _class_cache[key] = cls
    return cls


def create_gk_solver(name: str, **kwargs: Any) -> GKSolverBase:
    """Instantiate the solver registered under *name* with constructor kwargs."""
    return gk_solver_class(name)(**kwargs)


def registered_gk_solvers() -> list[str]:
    """Return the sorted registry keys."""
    with _registry_lock:
        return sorted(_registry)


def available_gk_solvers() -> dict[str, bool]:
    """Probe every registered solver's availability with default construction.

    A solver whose class fails to load, fails to construct with defaults, or
    raises from ``is_available`` reports ``False`` — the probe is
    fail-closed and never raises.
    """
    result: dict[str, bool] = {}
    for key in registered_gk_solvers():
        try:
            result[key] = bool(gk_solver_class(key)().is_available())
        except Exception as exc:
            logger.debug("GK solver availability probe failed for %s: %s", key, exc)
            result[key] = False
    return result


def resolve_tglf_solver(**kwargs: Any) -> GKSolverBase:
    """Return the canonical TGLF solver: external GACODE binary, else native.

    The external ``tglf`` binary (registry key ``"tglf"``) is the
    higher-fidelity lane and wins when it is on ``PATH``; the native
    quasilinear model (``"tglf-native"``) is the always-available floor.
    Constructor *kwargs* are forwarded to whichever class is selected, so
    they must be valid for both (pass none for the default resolution).
    """
    external = create_gk_solver("tglf", **kwargs)
    if external.is_available():
        return external
    logger.info("TGLF binary not on PATH; resolving to the native quasilinear solver.")
    return create_gk_solver("tglf-native", **kwargs)


def _load_tglf() -> type[GKSolverBase]:
    """Load the external GACODE TGLF binary wrapper."""
    from scpn_fusion.core.gk_tglf import TGLFSolver

    return TGLFSolver


def _load_tglf_native() -> type[GKSolverBase]:
    """Load the native quasilinear TGLF-class model."""
    from scpn_fusion.core.gk_tglf_native import TGLFNativeSolver

    return TGLFNativeSolver


def _load_cgyro() -> type[GKSolverBase]:
    """Load the CGYRO external binary wrapper."""
    from scpn_fusion.core.gk_cgyro import CGYROSolver

    return CGYROSolver


def _load_gs2() -> type[GKSolverBase]:
    """Load the GS2 external binary wrapper."""
    from scpn_fusion.core.gk_gs2 import GS2Solver

    return GS2Solver


def _load_gene() -> type[GKSolverBase]:
    """Load the GENE external binary wrapper."""
    from scpn_fusion.core.gk_gene import GENESolver

    return GENESolver


def _load_qualikiz() -> type[GKSolverBase]:
    """Load the QuaLiKiz interface wrapper."""
    from scpn_fusion.core.gk_qualikiz import QuaLiKizSolver

    return QuaLiKizSolver


def _bootstrap_builtin_solvers() -> None:
    """Register the six built-in GKSolverBase implementations."""
    register_gk_solver("tglf", _load_tglf)
    register_gk_solver("tglf-native", _load_tglf_native)
    register_gk_solver("cgyro", _load_cgyro)
    register_gk_solver("gs2", _load_gs2)
    register_gk_solver("gene", _load_gene)
    register_gk_solver("qualikiz", _load_qualikiz)


_bootstrap_builtin_solvers()
