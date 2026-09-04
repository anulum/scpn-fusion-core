# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gymnasium Tokamak Env Tests

import json
import logging
import runpy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from scpn_fusion.control.gym_tokamak_env import TokamakEnv, register
from scpn_fusion._data_paths import default_iter_config_path
from scpn_fusion.core.fusion_kernel import FusionKernel


@pytest.fixture(scope="module", autouse=True)
def registered_environment() -> None:
    """Register once for the module's public Gymnasium workflow tests."""
    register()


class _AxisKernel:
    """Kernel stand-in that exposes ``find_magnetic_axis`` (the preferred path)."""

    Psi = np.zeros((4, 4), dtype=np.float64)
    R = np.linspace(4.0, 8.0, 4, dtype=np.float64)
    Z = np.linspace(-2.0, 2.0, 4, dtype=np.float64)
    cfg = {"physics": {"plasma_current_target": 15.0, "beta_scale": 2.0}}

    def find_magnetic_axis(self) -> tuple[float, float, float]:
        return 6.2, 0.1, 0.5

    def find_x_point(self, _psi: NDArray[np.float64]) -> tuple[tuple[float, float], None]:
        return (5.0, -3.0), None


def test_env_registration() -> None:
    """Verify that Tokamak-v0 is registered correctly."""
    env_ids = [spec.id for spec in gym.envs.registry.values()]
    assert "Tokamak-v0" in env_ids


def test_env_reset() -> None:
    """Verify that reset returns valid observation and info."""
    env = gym.make("Tokamak-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    assert obs.shape == (8,)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert np.all(np.isfinite(obs))
    assert env.unwrapped.render() is None


def test_env_step() -> None:
    """Verify that step returns valid values and terminates."""
    env = gym.make("Tokamak-v0", max_steps=5)
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "disrupted" in info


def test_env_compliance() -> None:
    """Verify that the environment complies with Gymnasium API standards."""
    env = gym.make("Tokamak-v0", max_steps=5)
    # This checks observation space, action space, and API behavior
    check_env(env.unwrapped)


def test_disruption_penalty(tmp_path: Path) -> None:
    """Verify that disruption (large error) results in a penalty."""
    cfg = json.loads(default_iter_config_path().read_text())
    cfg["target"] = {"R_axis": 12.0, "Z_axis": 0.0}
    path = tmp_path / "off_target.json"
    path.write_text(json.dumps(cfg))
    env = TokamakEnv(config_file=str(path))
    env.reset()

    action = np.zeros(4, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == pytest.approx(-float(np.hypot(obs[4], obs[5])) - 10.0)
    assert terminated and info["disrupted"]
    assert not truncated


def test_get_obs_prefers_find_magnetic_axis_when_kernel_exposes_it() -> None:
    """When the kernel exposes ``find_magnetic_axis`` the observation uses it directly.

    The default ITER kernel lacks the method, so the env normally falls back to the
    grid-argmax estimate; injecting a kernel that provides it exercises the preferred
    branch and the axis coordinates flow straight into the observation.
    """
    env = TokamakEnv(max_steps=5)
    env.controller.kernel = _AxisKernel()
    obs = env._get_obs()

    assert obs[0] == pytest.approx(6.2)  # curr_R from find_magnetic_axis
    assert obs[1] == pytest.approx(0.1)  # curr_Z from find_magnetic_axis
    assert obs[2] == pytest.approx(15.0)  # plasma_current_target from cfg
    assert obs[6] == pytest.approx(5.0)  # x-point R
    assert obs[7] == pytest.approx(-3.0)  # x-point Z


@pytest.fixture
def tracking_config(tmp_path: Path) -> Path:
    """Use the real ITER deck on a smaller grid with its initial axis as target."""
    cfg = json.loads(default_iter_config_path().read_text())
    cfg["grid_resolution"] = [32, 32]
    path = tmp_path / "gym_tracking.json"
    path.write_text(json.dumps(cfg))
    kernel = FusionKernel(str(path))
    kernel.solve_equilibrium()
    iz, ir = np.unravel_index(np.argmax(kernel.Psi), kernel.Psi.shape)
    cfg["target"] = {"R_axis": float(kernel.R[ir]), "Z_axis": float(kernel.Z[iz])}
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize("dt", [0.01, 0.05, 0.2])
def test_actions_follow_real_plant_offset_trajectory(tracking_config: Path, dt: float) -> None:
    """PF order, MA scaling, lag and saturation match an independent plant trace."""
    env = TokamakEnv(str(tracking_config), control_dt_s=dt, max_steps=8)
    env.reset(seed=27)
    plant = FusionKernel(str(tracking_config))
    plant.solve_equilibrium()
    initial = np.array([coil["current"] for coil in plant.cfg["coils"]])
    offsets = np.zeros(3)
    beta = 0.0
    actions = [
        [0.001, -0.002, 0.003, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    for step, values in enumerate(actions, 1):
        action = np.array(values, dtype=np.float32)
        command = np.clip(action[:3].astype(np.float64) * 0.1, -0.05, 0.05)
        offsets += np.clip((command - offsets) * dt / (0.06 + dt), -dt, dt)
        beta_command = max(1.0, min(5.0, 1.0 + float(action[3]) * 0.5))
        beta = max(
            1.0, min(5.0, beta + max(-dt, min(dt, (beta_command - beta) * dt / (0.06 + dt))))
        )
        expected_currents = initial.copy()
        expected_currents[[0, 2, 4]] += offsets
        for coil, current in zip(plant.cfg["coils"], expected_currents, strict=True):
            coil["current"] = float(current)
        plant.cfg["physics"]["beta_scale"] = beta
        plant.solve_equilibrium()

        obs, reward, terminated, truncated, info = env.step(action)
        actual_currents = [coil["current"] for coil in env.controller.kernel.cfg["coils"]]
        np.testing.assert_allclose(actual_currents, expected_currents, rtol=0, atol=1e-14)
        np.testing.assert_allclose(env.controller.kernel.Psi, plant.Psi, rtol=1e-12, atol=1e-12)
        iz, ir = np.unravel_index(np.argmax(plant.Psi), plant.Psi.shape)
        xp, _ = plant.find_x_point(plant.Psi)
        expected_obs = np.array(
            [
                plant.R[ir],
                plant.Z[iz],
                plant.cfg["physics"]["plasma_current_target"],
                beta,
                plant.cfg["target"]["R_axis"] - plant.R[ir],
                plant.cfg["target"]["Z_axis"] - plant.Z[iz],
                xp[0],
                xp[1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(obs, expected_obs)
        disrupted = bool(abs(obs[4]) > 0.5 or abs(obs[5]) > 0.5)
        assert reward == pytest.approx(-float(np.hypot(obs[4], obs[5])) - 10 * disrupted)
        assert terminated is disrupted
        assert info["disrupted"] is disrupted
        assert truncated is (step == 8)
    env.close()


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(3),
        np.zeros(5),
        np.zeros((1, 4)),
        np.array([np.nan, 0, 0, 0]),
        np.array([0, np.inf, 0, 0]),
        np.array([0, 0, -np.inf, 0]),
        np.array([0, 0, 0, 1.01]),
        np.array([-1.01, 0, 0, 0]),
    ],
)
def test_invalid_action_is_atomic_and_recovers(
    tracking_config: Path, invalid: NDArray[np.float64]
) -> None:
    """Rejected actions cannot advance either the plant or hidden actuator history."""
    env = TokamakEnv(str(tracking_config))
    reference = TokamakEnv(str(tracking_config))
    env.reset(seed=11)
    reference.reset(seed=11)
    before = env.controller.kernel.Psi.copy()
    with pytest.raises(ValueError, match="action"):
        env.step(invalid)
    assert env.current_step == 0
    np.testing.assert_array_equal(env.controller.kernel.Psi, before)
    for values in [[0.1, 0.2, -0.3, 1.0], [-0.2, 0.0, 0.1, -1.0]]:
        action = np.array(values, dtype=np.float32)
        actual = env.step(action)
        expected = reference.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert env.controller.kernel.cfg == reference.controller.kernel.cfg
    env.close()
    reference.close()


def test_seeded_reset_replays_actuator_state(tracking_config: Path) -> None:
    """Reset discards actuator lag and restores the configured coil baseline."""
    env = TokamakEnv(str(tracking_config))
    initial, _ = env.reset(seed=42)
    action = np.array([0.3, -0.2, 0.1, 1.0], dtype=np.float32)
    first = [env.step(action) for _ in range(3)]
    reset_obs, _ = env.reset(seed=42)
    np.testing.assert_array_equal(reset_obs, initial)
    assert env.current_step == 0
    for expected in first:
        actual = env.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    env.close()


@pytest.mark.parametrize("bbox", [None, "tight"])
def test_rgb_render_depicts_current_state_without_advancing(
    tracking_config: Path, bbox: str | None
) -> None:
    """RGB rendering is repeatable, reflects a new solve and leaves the plant alone."""
    env = TokamakEnv(str(tracking_config), render_mode="rgb_array")
    env.reset(seed=12)
    before = env.controller.kernel.Psi.copy()
    from matplotlib import rc_context

    with rc_context({"savefig.bbox": bbox}):
        image = env.render()
    assert image is not None
    assert image.shape == (400, 600, 3)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 10
    repeated = env.render()
    assert repeated is not None
    np.testing.assert_array_equal(repeated, image)
    np.testing.assert_array_equal(env.controller.kernel.Psi, before)
    assert env.current_step == 0
    env.step(np.array([0.2, 0.1, -0.2, 0.4], dtype=np.float32))
    updated = env.render()
    assert updated is not None
    assert not np.array_equal(updated, image)
    assert env.current_step == 1
    env.close()


def test_shield_wrapper_preserves_real_actuator_trajectory(tracking_config: Path) -> None:
    """The shield's float64 actions traverse the same real Gym actuator contract."""
    from scpn_fusion.control.shielded_tokamak_env import ShieldedTokamakEnv

    base = TokamakEnv(str(tracking_config))
    shield = ShieldedTokamakEnv(base)
    reference = TokamakEnv(str(tracking_config))
    shield.reset(seed=31)
    reference.reset(seed=31)
    for values in [[0.1, -0.2, 0.3, 1.0], [-0.1, 0.2, -0.3, -1.0]]:
        obs, reward, terminated, truncated, info = shield.step(np.array(values))
        expected = reference.step(info["shielded_action"])
        np.testing.assert_array_equal(obs, expected[0])
        assert base.controller.kernel.cfg == reference.controller.kernel.cfg
        assert reward == expected[1]
        assert truncated is expected[3]
        assert terminated is (expected[2] or info["shield_halt"])
    base.close()
    reference.close()


def test_module_example_runs_real_gym_episode(caplog: pytest.LogCaptureFixture) -> None:
    """The executable module completes real reset/action calls and logs finite rewards."""
    module = Path(__file__).parents[1] / "src/scpn_fusion/control/gym_tokamak_env.py"
    with caplog.at_level(logging.INFO), pytest.warns(UserWarning, match="Overriding environment"):
        runpy.run_path(str(module), run_name="__main__")
    assert any("Initial observation:" in record.message for record in caplog.records)
    rewards = [record for record in caplog.records if "Sampled action reward:" in record.message]
    assert rewards
    assert all(
        "nan" not in record.message.lower() and "inf" not in record.message.lower()
        for record in rewards
    )
