# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shielded Contract Comparison Tests
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from scpn_fusion.control.shielded_tokamak_env import ShieldedTokamakEnv  # noqa: E402
from validation.shielded_contract_comparison import (  # noqa: E402
    ContractMetrics,
    ShieldedEpisodeResult,
    aggregate_contract,
    evaluate_contract,
    generate_contract_table,
    pid_gym_policy,
    run_contract_comparison,
    save_contract_json,
)


class _Space:
    def __init__(self, shape):
        self.shape = shape


class ScriptedTokamak:
    def __init__(self, obs, terminate_at=None):
        self._obs = np.asarray(obs, dtype=np.float64)
        self._terminate_at = terminate_at
        self._idx = 0
        self.action_space = _Space((4,))
        self.observation_space = _Space((8,))

    def reset(self, **_):
        self._idx = 0
        return self._obs.copy(), {}

    def step(self, action):
        self._idx += 1
        terminated = self._terminate_at is not None and self._idx >= self._terminate_at
        info = {"disrupted": terminated}
        return self._obs.copy(), -1.0, terminated, False, info


def test_pid_gym_policy_shapes_and_signs():
    obs = np.array([6.0, 0.2, 5.0, 0.5, 0.4, -0.3, 0.0, -3.0])
    action = pid_gym_policy(obs, kp_r=0.5, kp_z=0.5, kp_beta=0.2, beta_target=1.0)
    assert action.shape == (4,)
    # radial coil tracks err_R (index 4) > 0 -> positive.
    assert action[1] > 0
    # top and bottom coils oppose on err_Z.
    assert action[0] == pytest.approx(-action[2])
    # beta below target -> positive heating.
    assert action[3] > 0


def test_pid_gym_policy_clips_to_unit_band():
    obs = np.array([6.0, 0.0, 5.0, -50.0, 100.0, -100.0, 0.0, -3.0])
    action = pid_gym_policy(obs)
    assert np.all(action <= 1.0) and np.all(action >= -1.0)


def test_evaluate_contract_counts_breaches_and_clamps():
    # Ip 20 MA breaches the current limit every step; non-zero position errors make
    # the PID demand non-zero coil deltas, so the freeze clamp actually intervenes.
    env = ShieldedTokamakEnv(ScriptedTokamak([6.0, 0.0, 20.0, 1.0, 0.5, 0.3, 0.0, -3.0]))
    result = evaluate_contract(pid_gym_policy, env, shot_duration=5)
    assert result.steps == 5
    assert result.limit_breach_steps == 5
    # Three coil components frozen each step.
    assert result.shield_clamp_events == 15
    assert not result.disrupted


def test_evaluate_contract_stops_on_termination():
    env = ShieldedTokamakEnv(
        ScriptedTokamak([6.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, -3.0], terminate_at=3)
    )
    result = evaluate_contract(pid_gym_policy, env, shot_duration=10)
    assert result.steps == 3
    assert result.disrupted


def test_aggregate_contract_empty_is_zeroed():
    metrics = aggregate_contract("PID", [])
    assert metrics.n_episodes == 0
    assert metrics.mean_reward == 0.0


def test_aggregate_contract_means():
    episodes = [
        ShieldedEpisodeResult(
            reward=-2.0, steps=5, disrupted=False, limit_breach_steps=1, shield_clamp_events=3
        ),
        ShieldedEpisodeResult(
            reward=-4.0, steps=5, disrupted=True, limit_breach_steps=3, shield_clamp_events=9
        ),
    ]
    metrics = aggregate_contract("RL-shielded", episodes)
    assert metrics.n_episodes == 2
    assert metrics.mean_reward == pytest.approx(-3.0)
    assert metrics.disruption_rate == pytest.approx(0.5)
    assert metrics.mean_limit_breach_steps == pytest.approx(2.0)
    assert metrics.mean_shield_clamp_events == pytest.approx(6.0)


def test_run_contract_comparison_requires_controllers():
    with pytest.raises(RuntimeError, match="No controllers"):
        run_contract_comparison({}, n_episodes=2, shot_duration=5)


def test_run_contract_comparison_aggregates_injected_runners():
    def stable(_duration: int) -> ShieldedEpisodeResult:
        return ShieldedEpisodeResult(-1.0, 5, False, 0, 0)

    def shielded(_duration: int) -> ShieldedEpisodeResult:
        return ShieldedEpisodeResult(-2.0, 5, False, 2, 6)

    results = run_contract_comparison(
        {"PID": stable, "RL-shielded": shielded}, n_episodes=3, shot_duration=5
    )
    assert results["PID"].n_episodes == 3
    assert results["PID"].mean_shield_clamp_events == 0.0
    assert results["RL-shielded"].mean_shield_clamp_events == pytest.approx(6.0)


def test_generate_contract_table_has_all_rows():
    results = {
        "PID": ContractMetrics("PID", n_episodes=3, mean_reward=-1.0),
        "RL-shielded": ContractMetrics("RL-shielded", n_episodes=3, mean_reward=-2.0),
    }
    table = generate_contract_table(results)
    assert "Controller" in table
    assert "PID" in table
    assert "RL-shielded" in table
    assert table.count("\n") == 3  # header + separator + 2 rows


def test_save_contract_json_roundtrip(tmp_path: Path):
    results = {"PID": ContractMetrics("PID", n_episodes=2, mean_reward=-1.5, disruption_rate=0.5)}
    out = tmp_path / "contract.json"
    save_contract_json(results, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["controllers"]["PID"]["n_episodes"] == 2
    assert payload["controllers"]["PID"]["disruption_rate"] == 0.5
