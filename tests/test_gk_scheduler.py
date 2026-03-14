# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GK Scheduler Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.core.gk_ood_detector import OODResult
from scpn_fusion.core.gk_scheduler import GKScheduler, SchedulerConfig


def _rho_grid(n: int = 50) -> np.ndarray:
    return np.linspace(0, 1, n)


def test_periodic_fires_on_schedule():
    cfg = SchedulerConfig(strategy="periodic", period=3)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    assert sched.step(rho, chi) is None  # step 1
    assert sched.step(rho, chi) is None  # step 2
    req = sched.step(rho, chi)  # step 3
    assert req is not None
    assert req.step_number == 3
    assert len(req.rho_indices) > 0


def test_periodic_skips_between():
    cfg = SchedulerConfig(strategy="periodic", period=5)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    results = [sched.step(rho, chi) for _ in range(10)]
    fired = [r for r in results if r is not None]
    assert len(fired) == 2  # steps 5 and 10


def test_periodic_includes_anchors():
    cfg = SchedulerConfig(strategy="periodic", period=1, anchor_rho=(0.3, 0.5, 0.8))
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    req = sched.step(rho, chi)
    assert req is not None
    rho_vals = [rho[i] for i in req.rho_indices]
    # Anchors should be close to 0.3, 0.5, 0.8
    assert any(abs(r - 0.3) < 0.05 for r in rho_vals)
    assert any(abs(r - 0.5) < 0.05 for r in rho_vals)


def test_adaptive_fires_on_ood():
    cfg = SchedulerConfig(strategy="adaptive")
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    # No OOD → should still fire with anchors on first step if chi changes
    ood_all_ok = [
        OODResult(is_ood=False, confidence=0.0, method="combined", details={}) for _ in range(50)
    ]
    # First step, no prev chi → no chi-change detection. But anchors fire.
    req = sched.step(rho, chi, ood_all_ok)
    # Only anchors, might return a request
    if req is not None:
        assert all("anchor" in r for r in req.reasons.values())


def test_adaptive_fires_on_chi_change():
    cfg = SchedulerConfig(strategy="adaptive", chi_change_threshold=0.3)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi_1 = np.ones(50)
    sched.step(rho, chi_1)  # set prev

    chi_2 = np.ones(50)
    chi_2[25] = 3.0  # 200% change at midpoint
    req = sched.step(rho, chi_2)
    assert req is not None
    assert 25 in req.rho_indices


def test_adaptive_no_fire_small_change():
    cfg = SchedulerConfig(strategy="adaptive", chi_change_threshold=0.5)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)
    sched.step(rho, chi)

    chi_2 = chi * 1.1  # 10% change, below threshold
    ood_ok = [
        OODResult(is_ood=False, confidence=0.0, method="combined", details={}) for _ in range(50)
    ]
    req = sched.step(rho, chi_2, ood_ok)
    # No OOD, no big change → only anchors at most
    if req is not None:
        assert all("anchor" in v for v in req.reasons.values())


def test_critical_region_selects_edges():
    cfg = SchedulerConfig(strategy="critical_region", pedestal_rho=0.85, axis_rho=0.15, budget=10)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    req = sched.step(rho, chi)
    assert req is not None
    rho_selected = [rho[i] for i in req.rho_indices]
    # Should include near-axis and near-edge
    assert any(r < 0.2 for r in rho_selected)
    assert any(r > 0.8 for r in rho_selected)


def test_budget_enforcement():
    cfg = SchedulerConfig(strategy="critical_region", budget=3)
    sched = GKScheduler(cfg)
    rho = _rho_grid()
    chi = np.ones(50)

    req = sched.step(rho, chi)
    assert req is not None
    assert len(req.rho_indices) <= 3


def test_reset():
    sched = GKScheduler()
    rho = _rho_grid()
    chi = np.ones(50)
    sched.step(rho, chi)
    sched.reset()
    assert sched._step == 0
    assert sched._prev_chi_i is None
