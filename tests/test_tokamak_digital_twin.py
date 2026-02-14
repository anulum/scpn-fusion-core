# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Digital Twin Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic smoke tests for tokamak_digital_twin runtime entry points."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.tokamak_digital_twin import run_digital_twin
from scpn_fusion.io.imas_connector import (
    digital_twin_summary_to_ids,
    ids_to_digital_twin_summary,
)


def test_run_digital_twin_returns_finite_summary_without_plot() -> None:
    summary = run_digital_twin(
        time_steps=24,
        seed=123,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "seed",
        "steps",
        "final_avg_temp",
        "final_reward",
        "final_action",
        "final_islands_px",
        "reward_mean_last_50",
        "plot_saved",
    ):
        assert key in summary
    assert summary["steps"] == 24
    assert summary["plot_saved"] is False
    assert np.isfinite(summary["final_avg_temp"])
    assert np.isfinite(summary["final_reward"])
    assert np.isfinite(summary["final_action"])
    assert np.isfinite(summary["reward_mean_last_50"])


def test_run_digital_twin_is_deterministic_for_fixed_seed() -> None:
    a = run_digital_twin(time_steps=20, seed=77, save_plot=False, verbose=False)
    b = run_digital_twin(time_steps=20, seed=77, save_plot=False, verbose=False)
    assert a["final_avg_temp"] == b["final_avg_temp"]
    assert a["final_reward"] == b["final_reward"]
    assert a["final_action"] == b["final_action"]
    assert a["final_islands_px"] == b["final_islands_px"]
    assert a["reward_mean_last_50"] == b["reward_mean_last_50"]


def test_ids_roundtrip_preserves_core_digital_twin_fields() -> None:
    summary = run_digital_twin(time_steps=18, seed=5, save_plot=False, verbose=False)
    ids_payload = digital_twin_summary_to_ids(summary, machine="ITER", shot=101, run=2)
    recovered = ids_to_digital_twin_summary(ids_payload)

    assert recovered["steps"] == summary["steps"]
    assert recovered["final_islands_px"] == summary["final_islands_px"]
    assert np.isfinite(recovered["final_avg_temp"])
    assert np.isfinite(recovered["final_reward"])
