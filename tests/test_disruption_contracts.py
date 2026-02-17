# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Contract Tests
# ----------------------------------------------------------------------
"""Tests for reusable disruption contracts."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_contracts import (
    run_disruption_episode,
    run_real_shot_replay,
)
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


def _build_replay_shot(n: int = 220) -> dict[str, NDArray[np.float64] | bool | int]:
    t = np.linspace(0.0, 0.22, n, dtype=np.float64)
    dBdt = 0.45 + 0.10 * np.sin(2.0 * np.pi * 5.0 * t)
    dBdt += 0.35 * np.exp(-(((t - 0.18) / 0.018) ** 2))
    n1 = 0.20 + 0.55 * np.exp(-(((t - 0.18) / 0.020) ** 2))
    n2 = 0.08 + 0.20 * np.exp(-(((t - 0.18) / 0.022) ** 2))
    return {
        "time_s": t,
        "Ip_MA": np.full(n, 12.0, dtype=np.float64),
        "beta_N": np.full(n, 2.2, dtype=np.float64),
        "n1_amp": n1,
        "n2_amp": n2,
        "dBdt_gauss_per_s": dBdt,
        "is_disruption": True,
        "disruption_time_idx": 200,
    }


def test_run_disruption_episode_contract_smoke() -> None:
    rng = np.random.default_rng(42)
    agent = FusionAIAgent(epsilon=0.05)
    explorer = GlobalDesignExplorer("dummy")
    base_tbr = float(
        BreedingBlanket(thickness_cm=260.0, li6_enrichment=1.0)
        .calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
        )
        .tbr
    )
    out = run_disruption_episode(
        rng=rng,
        rl_agent=agent,
        base_tbr=base_tbr,
        explorer=explorer,
    )
    assert 0.0 <= float(out["risk_before"]) <= 1.0
    assert 0.0 <= float(out["risk_after"]) <= 1.0
    assert float(out["q_proxy"]) > 0.0
    assert float(out["tbr_proxy"]) > 0.0
    assert float(out["halo_current_ma"]) >= 0.0
    assert float(out["runaway_beam_ma"]) >= 0.0
    assert float(out["argon_quantity_mol"]) >= 0.0
    assert float(out["xenon_quantity_mol"]) >= 0.0
    assert float(out["total_impurity_mol"]) >= float(out["neon_quantity_mol"])


def test_run_real_shot_replay_reports_impurity_cocktail_fields() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    shot_data = _build_replay_shot()
    out = run_real_shot_replay(
        shot_data=shot_data,
        rl_agent=agent,
        risk_threshold=0.55,
        spi_trigger_risk=0.72,
        window_size=96,
    )
    assert "argon_mol" in out
    assert "xenon_mol" in out
    assert "total_impurity_mol" in out
    assert float(out["total_impurity_mol"]) >= float(out["neon_mol"])


def test_run_real_shot_replay_rejects_trigger_below_threshold() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    with pytest.raises(ValueError, match="spi_trigger_risk must be >="):
        run_real_shot_replay(
            shot_data=_build_replay_shot(),
            rl_agent=agent,
            risk_threshold=0.75,
            spi_trigger_risk=0.70,
            window_size=96,
        )


def test_run_real_shot_replay_rejects_non_monotonic_time() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    shot_data = _build_replay_shot()
    bad_time = np.asarray(shot_data["time_s"], dtype=np.float64).copy()
    bad_time[120] = bad_time[119]
    shot_data["time_s"] = bad_time
    with pytest.raises(ValueError, match="strictly increasing"):
        run_real_shot_replay(
            shot_data=shot_data,
            rl_agent=agent,
            risk_threshold=0.55,
            spi_trigger_risk=0.72,
            window_size=96,
        )


def test_run_real_shot_replay_rejects_bad_vector_lengths_and_indices() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    shot_data = _build_replay_shot()
    shot_data["n1_amp"] = np.asarray(shot_data["n1_amp"], dtype=np.float64)[:-1]
    with pytest.raises(ValueError, match=r"shot_data\.n1_amp must have"):
        run_real_shot_replay(
            shot_data=shot_data,
            rl_agent=agent,
            risk_threshold=0.55,
            spi_trigger_risk=0.72,
            window_size=96,
        )


def test_run_disruption_episode_rejects_nonpositive_base_tbr() -> None:
    rng = np.random.default_rng(42)
    agent = FusionAIAgent(epsilon=0.05)
    explorer = GlobalDesignExplorer("dummy")
    with pytest.raises(ValueError, match="base_tbr must be > 0"):
        run_disruption_episode(
            rng=rng,
            rl_agent=agent,
            base_tbr=0.0,
            explorer=explorer,
        )


def test_run_real_shot_replay_rejects_nonpositive_base_tbr() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    with pytest.raises(ValueError, match="base_tbr must be > 0"):
        run_real_shot_replay(
            shot_data=_build_replay_shot(),
            rl_agent=agent,
            base_tbr=0.0,
            risk_threshold=0.55,
            spi_trigger_risk=0.72,
            window_size=96,
        )


def test_run_real_shot_replay_rejects_window_larger_than_shot() -> None:
    agent = FusionAIAgent(epsilon=0.05)
    shot_data = _build_replay_shot(n=64)
    shot_data["disruption_time_idx"] = 48
    with pytest.raises(ValueError, match="window_size must be <= number of samples"):
        run_real_shot_replay(
            shot_data=shot_data,
            rl_agent=agent,
            risk_threshold=0.55,
            spi_trigger_risk=0.72,
            window_size=96,
        )

    shot_data = _build_replay_shot()
    shot_data["disruption_time_idx"] = 1000
    with pytest.raises(ValueError, match="disruption_time_idx must be < number of samples"):
        run_real_shot_replay(
            shot_data=shot_data,
            rl_agent=agent,
            risk_threshold=0.55,
            spi_trigger_risk=0.72,
            window_size=96,
        )
