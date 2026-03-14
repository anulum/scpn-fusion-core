# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Checkpoint Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the save/resume checkpoint API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_fusion.core.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    """Verify that state is preserved exactly across save/load."""
    path = tmp_path / "test_ckpt.json"

    solver_state = {
        "Psi": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nested": {"arr": np.array([0.1, 0.2])},
    }
    episode = 42
    metrics = {"accuracy": 0.99, "history": [0.1, 0.5, 0.8]}

    save_checkpoint(path, solver_state, episode, metrics)

    s_restored, e_restored, m_restored = load_checkpoint(path)

    assert e_restored == episode
    assert m_restored["accuracy"] == metrics["accuracy"]
    np.testing.assert_allclose(s_restored["Psi"], solver_state["Psi"])
    np.testing.assert_allclose(s_restored["nested"]["arr"], solver_state["nested"]["arr"])


def test_checkpoint_resume_simulation(tmp_path: Path):
    """Verify that a simulation can be resumed from a checkpoint."""
    path = tmp_path / "resume_ckpt.json"

    # 1. Run "Phase 1" of a campaign
    initial_metrics = {"total_reward": 100.0}
    state_at_ep_10 = {"last_val": 5.0}

    save_checkpoint(path, state_at_ep_10, 10, initial_metrics)

    # 2. Resume "Phase 2"
    state, ep, metrics = load_checkpoint(path)
    assert ep == 10

    # Simulate further work
    ep += 1
    metrics["total_reward"] += 50.0

    assert ep == 11
    assert metrics["total_reward"] == 150.0
