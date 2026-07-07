# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the string-keyed gyrokinetic solver registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from scpn_fusion.core import gk_registry
from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase


class _StubSolver(GKSolverBase):
    """Deck-contract stub used to exercise registry mechanics."""

    def __init__(self, available: bool = True) -> None:
        self.available = available

    def prepare_input(self, params: GKLocalParams) -> Path:
        return Path("/nonexistent/stub")

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        return GKOutput(chi_i=1.0, chi_e=0.5, D_e=0.1)

    def is_available(self) -> bool:
        return self.available


class _BrokenAvailabilitySolver(_StubSolver):
    """Stub whose availability probe raises."""

    def is_available(self) -> bool:
        raise RuntimeError("probe explosion")


@pytest.fixture
def scratch_key(request: pytest.FixtureRequest) -> Iterator[str]:
    """Provide an isolated registry key and clean it up afterwards."""
    key = f"test-{request.node.name}".lower()
    yield key
    with gk_registry._registry_lock:
        gk_registry._registry.pop(key, None)
        gk_registry._class_cache.pop(key, None)


def test_builtin_solvers_are_registered() -> None:
    """All six GKSolverBase implementations are discoverable by name."""
    names = gk_registry.registered_gk_solvers()
    assert names == ["cgyro", "gene", "gs2", "qualikiz", "tglf", "tglf-native"]


def test_gk_solver_class_loads_and_caches(scratch_key: str) -> None:
    """The loader thunk runs once; subsequent lookups hit the class cache."""
    calls: list[int] = []

    def loader() -> type[GKSolverBase]:
        calls.append(1)
        return _StubSolver

    gk_registry.register_gk_solver(scratch_key, loader)
    assert gk_registry.gk_solver_class(scratch_key) is _StubSolver
    assert gk_registry.gk_solver_class(scratch_key) is _StubSolver
    assert len(calls) == 1


def test_reregistration_replaces_and_invalidates_cache(scratch_key: str) -> None:
    """Re-registering a key replaces the loader and drops the cached class."""
    gk_registry.register_gk_solver(scratch_key, lambda: _StubSolver)
    assert gk_registry.gk_solver_class(scratch_key) is _StubSolver

    class _Second(_StubSolver):
        pass

    gk_registry.register_gk_solver(scratch_key, lambda: _Second)
    assert gk_registry.gk_solver_class(scratch_key) is _Second


def test_unknown_name_raises_key_error_listing_registered() -> None:
    """An unknown key raises KeyError naming the registered solvers."""
    with pytest.raises(KeyError, match="tglf-native"):
        gk_registry.gk_solver_class("no-such-solver")


def test_empty_key_is_rejected() -> None:
    """Registering an empty or whitespace key is a ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        gk_registry.register_gk_solver("   ", lambda: _StubSolver)


def test_loader_returning_non_solver_raises_type_error(scratch_key: str) -> None:
    """A loader that returns a non-GKSolverBase class fails loudly."""
    gk_registry.register_gk_solver(scratch_key, lambda: dict)  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="GKSolverBase"):
        gk_registry.gk_solver_class(scratch_key)


def test_create_gk_solver_forwards_constructor_kwargs(scratch_key: str) -> None:
    """Factory kwargs reach the solver constructor."""
    gk_registry.register_gk_solver(scratch_key, lambda: _StubSolver)
    solver = gk_registry.create_gk_solver(scratch_key, available=False)
    assert isinstance(solver, _StubSolver)
    assert solver.is_available() is False


def test_available_gk_solvers_reports_native_true_and_is_fail_closed(
    scratch_key: str,
) -> None:
    """The availability probe never raises and reports the native floor True."""
    gk_registry.register_gk_solver(scratch_key, lambda: _BrokenAvailabilitySolver)
    availability = gk_registry.available_gk_solvers()
    assert availability["tglf-native"] is True
    assert availability[scratch_key] is False  # raising probe degrades to False


def test_builtin_classes_all_satisfy_the_contract() -> None:
    """Every built-in registry entry loads a GKSolverBase subclass."""
    for name in gk_registry.registered_gk_solvers():
        cls = gk_registry.gk_solver_class(name)
        assert issubclass(cls, GKSolverBase), name


def test_resolve_tglf_solver_prefers_external_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the GACODE binary on PATH, the canonical TGLF path is external."""
    import scpn_fusion.core.gk_tglf as gk_tglf_mod

    monkeypatch.setattr(gk_tglf_mod.shutil, "which", lambda _name: "/usr/bin/tglf")
    solver = gk_registry.resolve_tglf_solver()
    assert type(solver).__name__ == "TGLFSolver"


def test_resolve_tglf_solver_falls_back_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the binary, the canonical TGLF path is the native model."""
    import scpn_fusion.core.gk_tglf as gk_tglf_mod

    monkeypatch.setattr(gk_tglf_mod.shutil, "which", lambda _name: None)
    solver = gk_registry.resolve_tglf_solver()
    assert type(solver).__name__ == "TGLFNativeSolver"
    assert solver.is_available() is True


def test_registry_surface_is_exported_from_core_package() -> None:
    """The package-level lazy exports resolve to the registry callables."""
    from scpn_fusion import core

    assert core.registered_gk_solvers is gk_registry.registered_gk_solvers
    assert core.create_gk_solver is gk_registry.create_gk_solver
    assert core.available_gk_solvers is gk_registry.available_gk_solvers
    assert core.resolve_tglf_solver is gk_registry.resolve_tglf_solver
