# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the Hall-MHD zonal-flow discovery solver and tearing search."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.core.hall_mhd_discovery as hall_mhd_discovery
from scpn_fusion.core.hall_mhd_discovery import HallMHD, spitzer_resistivity


def test_spitzer_resistivity_handles_nonpositive_temperature() -> None:
    """Non-positive electron temperature returns the resistivity floor."""
    assert spitzer_resistivity(0.0) == 1e-4
    assert spitzer_resistivity(-1.0) == 1e-4


def test_hall_mhd_step_returns_finite_energy() -> None:
    """One Hall-MHD step returns finite total and zonal energies."""
    sim = HallMHD(N=8, eta=1e-4, nu=1e-4)
    total, zonal = sim.step()
    assert np.isfinite(total)
    assert np.isfinite(zonal)
    assert total >= 0.0


def test_hall_mhd_parameter_sweep_small_grid() -> None:
    """A small eta/nu sweep returns aligned growth-rate columns."""
    sim = HallMHD(N=8)
    result = sim.parameter_sweep((1e-4, 2e-4), (1e-4, 2e-4), n_steps=2, sim_steps=2)
    assert set(result.keys()) == {"eta", "nu", "growth_rate"}
    assert len(result["eta"]) == 4
    assert len(result["nu"]) == 4
    assert len(result["growth_rate"]) == 4


def test_hall_mhd_tearing_threshold_bounds() -> None:
    """The tearing-threshold bisection brackets its estimate."""
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-5, 1e-4), n_bisect=2, sim_steps=2)
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_spitzer_resistivity_positive_temperature() -> None:
    """A positive electron temperature gives a finite positive resistivity."""
    eta = spitzer_resistivity(100.0, Z_eff=1.5)
    assert eta > 0.0
    assert np.isfinite(eta)


def test_parameter_sweep_uses_growth_history_window() -> None:
    """A long enough sweep populates the log-growth estimate from the history window."""
    sim = HallMHD(N=8)
    result = sim.parameter_sweep((1e-4, 2e-4), (1e-4, 2e-4), n_steps=2, sim_steps=15)
    assert len(result["growth_rate"]) == 4
    assert all(np.isfinite(g) for g in result["growth_rate"])


def test_find_tearing_threshold_brackets_with_history_window() -> None:
    """The bisection uses the long-history growth estimate and brackets the threshold."""
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-9, 1e-6), n_bisect=4, sim_steps=30)
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


class _GrowingSim:
    """Deterministic fake simulator whose energy grows every step."""

    def __init__(self) -> None:
        self.energy_history: list[float] = []
        self._energy = 1.0

    def step(self) -> tuple[float, float]:
        self._energy *= 1.5
        self.energy_history.append(self._energy)
        return self._energy, 0.0


class _DecayingSim:
    """Deterministic fake simulator whose energy decays every step."""

    def __init__(self) -> None:
        self.energy_history: list[float] = []
        self._energy = 1.0

    def step(self) -> tuple[float, float]:
        self._energy *= 0.5
        self.energy_history.append(self._energy)
        return self._energy, 0.0


def test_find_tearing_threshold_raises_floor_when_unstable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive growth estimate pushes the marginal-resistivity floor upward.

    With dissipative resistivity, sustained growth at the probed ``eta`` means
    the marginal value lies above it, so the bisection floor must rise.
    """
    monkeypatch.setattr(hall_mhd_discovery, "create_hall_mhd", lambda *a, **k: _GrowingSim())
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-6, 1e-2), n_bisect=3, sim_steps=25)
    # Every iteration sees growth > 0, so the floor climbs toward the ceiling.
    assert result["lo"] > 1e-6
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_find_tearing_threshold_lowers_ceiling_when_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative growth estimate pulls the marginal-resistivity ceiling down."""
    monkeypatch.setattr(hall_mhd_discovery, "create_hall_mhd", lambda *a, **k: _DecayingSim())
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-6, 1e-2), n_bisect=3, sim_steps=25)
    assert result["hi"] < 1e-2
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_run_discovery_sim_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zonal-flow discovery demo runs and renders both diagnostics."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(hall_mhd_discovery, "STEPS", 20)
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    hall_mhd_discovery.run_discovery_sim()

    assert "Hall_MHD_Discovery.png" in saved
    assert "Hall_MHD_Structure.png" in saved


def test_hall_mhd_dispatch_registers_both_tiers() -> None:
    """The class-kernel registry carries RUST and NUMPY Hall-MHD tiers."""
    from scpn_fusion.core import _multi_compat as multi

    kernels = multi.registered_kernel_classes()
    assert "hall_mhd_discovery" in kernels
    tiers = [tier.rstrip("*") for tier in kernels["hall_mhd_discovery"]]
    assert "rust" in tiers
    assert "numpy" in tiers


def test_create_hall_mhd_numpy_floor_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory resolves to the NumPy simulator when Rust is unavailable."""
    from scpn_fusion.core import _multi_compat as multi

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    monkeypatch.delitem(multi._class_dispatch_cache, "hall_mhd_discovery", raising=False)
    try:
        sim = hall_mhd_discovery.create_hall_mhd(8, seed=7)
        assert isinstance(sim, HallMHD)
        total, zonal = sim.step()
        assert np.isfinite(total) and np.isfinite(zonal)
    finally:
        multi._class_dispatch_cache.pop("hall_mhd_discovery", None)


def test_seeded_python_backend_is_deterministic() -> None:
    """Equal seeds reproduce the exact NumPy-backend trajectory."""
    first = HallMHD(N=16, seed=123)
    second = HallMHD(N=16, seed=123)
    for _ in range(5):
        assert first.step() == second.step()


def test_background_drive_injects_energy_relative_to_unforced() -> None:
    """The current-sheet drive sustains energy the unforced run dissipates."""
    driven = HallMHD(N=16, seed=5, background_amplitude=1.0)
    unforced = HallMHD(N=16, seed=5)
    for _ in range(60):
        driven.step()
        unforced.step()
    assert driven.energy_history[-1] > unforced.energy_history[-1]


def test_unforced_run_dissipates_energy() -> None:
    """With corrected resistive dissipation the unforced sandbox decays."""
    sim = HallMHD(N=16, seed=11)
    for _ in range(60):
        sim.step()
    history = np.asarray(sim.energy_history)
    assert history[-1] < history[0]
    assert bool(np.all(np.isfinite(history)))


def test_backend_physics_invariant_parity() -> None:
    """Both dispatched backends satisfy the shared physics invariants.

    Trajectories are statistically equivalent, not bit-exact (language-native
    RNG streams), so parity is asserted on the invariant contract: finite
    non-negative energies, a populated history, and unforced late-time decay.
    """
    from scpn_fusion.core import _multi_compat as multi

    backends = [multi._load_numpy_hall_mhd()]
    if multi.is_available(multi.BackendTier.RUST):
        backends.append(multi._load_rust_hall_mhd())
    for backend_cls in backends:
        sim = backend_cls(16, None, None, seed=3, background_amplitude=0.0)
        energies = [sim.step() for _ in range(40)]
        totals = np.asarray([e for e, _ in energies])
        zonals = np.asarray([z for _, z in energies])
        assert bool(np.all(np.isfinite(totals))), backend_cls.__name__
        assert bool(np.all(totals >= 0.0)), backend_cls.__name__
        assert bool(np.all(np.isfinite(zonals))), backend_cls.__name__
        assert len(sim.energy_history) == 40, backend_cls.__name__
        # Unforced runs must not gain energy over the window.
        assert totals[-1] <= totals[0], backend_cls.__name__
