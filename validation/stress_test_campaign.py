# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""
Step 2.1: 1000-Shot Stress-Test Campaign.

Runs the complete PID, H-infinity, LQR, MPC, NMPC-JAX, LIF-NEF-SNN, and
Rust-PID registry across one declared tokamak plant contract. Unavailable
lanes remain visible and make the campaign incomplete.

Metrics per controller:
  - Radial and vertical tracking error (m)
  - P50, P95, P99 control-policy latency (µs)
  - Disruption rate
  - Disruption Extension Factor (DEF)
  - Magnetic-actuator current-offset effort (MA s)

Usage:
    python stress_test_campaign.py              # full 1000 episodes
    python stress_test_campaign.py --quick      # 10 episodes (CI)
    python stress_test_campaign.py --episodes 200
"""

from __future__ import annotations

import argparse
import os
import platform
import secrets
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, cast

ModelPredictiveController: Any | None
NeuralSurrogate: Any | None
NengoSNNController: Any | None
NengoSNNConfig: Any | None
PyRustIsoFluxController: Any | None

import numpy as np

# Setup paths
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion._data_paths import default_iter_config_path
from scpn_fusion.control.flight_sim_controllers import (
    get_flight_sim_offset_hinf_controller,
    get_flight_sim_offset_lqr_controller,
)
from scpn_fusion.control.tokamak_flight_sim import (
    CoilCurrentOffsetCommand,
    ControlObservation,
    IsoFluxController,
    map_axis_commands_to_coil_offsets,
)
from scpn_fusion.core.neural_equilibrium_kernel import NeuralEquilibriumKernel
from scpn_fusion.io.safe_loaders import checked_json_load
from validation.stress_campaign_contract import (
    RESULT_SCHEMA,
    CampaignResults,
    ControllerMetrics,
    EpisodeFailure,
    EpisodeResult,
    StressScenario,
    digest_json,
)
from validation.stress_campaign_recovery import (
    RecoveryStore,
    atomic_write_json,
    build_campaign_identity,
    sha256_file,
)

# Optional controller imports
_mpc_available = False
_snn_available = False

try:
    from scpn_fusion.control.neural_surrogate_mpc import (
        ModelPredictiveController as _ModelPredictiveController,
        NeuralSurrogate as _NeuralSurrogate,
    )

    _mpc_available = True
    ModelPredictiveController = cast(Any, _ModelPredictiveController)
    NeuralSurrogate = cast(Any, _NeuralSurrogate)
except ImportError:
    ModelPredictiveController = None
    NeuralSurrogate = None

try:
    from scpn_fusion.control.nengo_snn_wrapper import (
        NengoSNNController as _NengoSNNController,
        NengoSNNConfig as _NengoSNNConfig,
        nengo_available,
    )

    _snn_available = nengo_available()
    NengoSNNController = cast(Any, _NengoSNNController)
    NengoSNNConfig = cast(Any, _NengoSNNConfig)
except ImportError:
    NengoSNNController = None
    NengoSNNConfig = None

_rust_pid_available = False
try:
    from scpn_fusion_rs import PyRustIsoFluxController as _PyRustIsoFluxController

    _rust_pid_available = True
    PyRustIsoFluxController = cast(Any, _PyRustIsoFluxController)
except ImportError:
    PyRustIsoFluxController = None


class EpisodeRunner(Protocol):
    """Callable contract shared by controller episode runners."""

    def __call__(
        self,
        config_path: str | Path,
        shot_duration: int = 30,
        surrogate: bool = False,
        *,
        scenario: StressScenario,
        episode_index: int,
        episode_seed: int,
        evaluation_contract_digest: str,
    ) -> EpisodeResult:
        """Run one controller episode under the exact shared scenario."""


HINF_RESEARCH_ENV = "SCPN_ENABLE_HINF_RESEARCH"
EVALUATION_CONTRACT_SCHEMA = "scpn-fusion-core.stress-evaluation-contract.v2"


def build_evaluation_contract(
    config_path: Path,
    *,
    surrogate: bool,
    scenario: StressScenario,
) -> dict[str, Any]:
    """Define the common plant, observation, action, timing, and scoring contract."""
    config = checked_json_load(config_path)
    target = config.get("target", {})
    coils = config.get("coils", [])
    if not isinstance(coils, list) or len(coils) < 5:
        raise ValueError("stress evaluation requires at least five configured coils.")
    payload = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "evidence_scope": "wiring_only" if surrogate else "controller_comparison",
        "config_sha256": sha256_file(config_path),
        "plant_backend": (
            "scpn_fusion.core.neural_equilibrium_kernel.NeuralEquilibriumKernel"
            if surrogate
            else "scpn_fusion.core.fusion_kernel.FusionKernel"
        ),
        "control_dt_s": 0.01 if surrogate else 0.05,
        "axis_target_m": {
            "R": float(target.get("R_axis", 6.2)),
            "Z": float(target.get("Z_axis", 0.0)),
        },
        "ordered_coils": [
            str(coil.get("name", f"coil-{index}")) for index, coil in enumerate(coils)
        ],
        "observation": "noisy_RZ_axis_and_current_equilibrium_x_point",
        "action": "full_ordered_coil_current_offset_setpoint_ma",
        "axis_action_basis": {"radial": "PF3", "vertical": "PF5_minus_PF1"},
        "actuator": {
            "offset_limit_ma": 0.05,
            "slew_limit_ma_per_s": 1.0,
            "first_order_tau_s": 0.06,
            "pure_command_delay_s": scenario.actuator_delay_s,
        },
        "disturbance_trace": {
            "schema": "scpn-fusion-position-noise-v1",
            "channels": ["R_axis_m", "Z_axis_m"],
            "distribution": "independent_zero_mean_gaussian",
            "std_m": scenario.measurement_noise_std_m,
            "numpy_bit_generator": "PCG64",
        },
        "policy_call_count": "exactly_once_per_control_step",
        "policy_latency": "perf_counter_ns_around_policy_step_only",
        "tracking_error_quadrature": "trapezoidal_including_t0_and_final_state",
        "disruption": {
            "axis_absolute_error_threshold_m": 0.5,
            "sampling": "t0_then_each_post_actuation_state",
        },
        "actuator_effort": "integral_sum_absolute_applied_coil_offset_ma_s",
    }
    return {"digest": digest_json(payload), "payload": payload}


# Local flight-sim calibration for the H-infinity research lane.
#
# These values are measured from the same ITER flight-sim kernel and coil-command
# convention used by IsoFluxController. A positive radial PF3 command moves the
# axis outward. A positive vertical controller command is mapped to a negative
# top-coil command and a positive bottom-coil command; in the current flight-sim
# kernel that moves the axis downward, so the scalar H-infinity vertical channel
# must invert its command before handing it back to IsoFluxController.
HINF_RADIAL_POSITION_SENSITIVITY = 0.7559055118110241
HINF_VERTICAL_POSITION_SENSITIVITY = 0.0629921259842523


def _hinf_zero_delay_command(controller: Any, error: float, dt: float) -> float:
    """Return a same-sample H-infinity command after observer assimilation."""
    command = float(controller.step(error, dt))
    if abs(command) > 0.0 or abs(error) == 0.0:
        return command

    feedback = getattr(controller, "_Fd", None)
    state = getattr(controller, "state", None)
    if feedback is None or state is None:
        return command
    corrected = np.asarray(feedback) @ np.asarray(state)
    return float(corrected.reshape(-1)[0])


def _env_flag_enabled(name: str) -> bool:
    """Return True when an environment gate is explicitly enabled."""
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_isoflux_controller(
    config_path: str | Path,
    *,
    surrogate: bool,
    dt: float,
    scenario: StressScenario,
    episode_seed: int,
) -> IsoFluxController:
    kwargs: dict[str, Any] = {
        "verbose": False,
        "control_dt_s": dt,
        "measurement_noise_std_m": scenario.measurement_noise_std_m,
        "actuator_delay_s": scenario.actuator_delay_s,
        "rng_seed": episode_seed,
    }
    if surrogate:
        kwargs["kernel_factory"] = NeuralEquilibriumKernel
    return IsoFluxController(str(config_path), **kwargs)


class _TwoAxisPolicy:
    """Map one paired R/Z controller evaluation onto the common PF basis."""

    def __init__(
        self,
        n_coils: int,
        radial_step: Callable[[float], float],
        vertical_step: Callable[[float], float],
    ) -> None:
        self._n_coils = n_coils
        self._radial_step = radial_step
        self._vertical_step = vertical_step

    def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
        """Evaluate both axes exactly once for the common observation."""
        return map_axis_commands_to_coil_offsets(
            self._n_coils,
            self._radial_step(observation.radial_error_m),
            self._vertical_step(observation.vertical_error_m),
        )


def _python_episode_result(
    shot_result: dict[str, Any],
    shot_duration_s: int,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Convert a completed Python shot into time-resolved campaign metrics."""
    disrupted = bool(shot_result["disrupted"])
    t_disruption = min(float(shot_duration_s), float(shot_result["t_disruption_s"]))
    radial_error = float(shot_result["mean_abs_r_error_m"])
    vertical_error = float(shot_result["mean_abs_z_error_m"])
    return EpisodeResult(
        episode_index=episode_index,
        seed=episode_seed,
        scenario_digest=scenario.digest,
        evaluation_contract_digest=evaluation_contract_digest,
        disturbance_trace_digest=str(shot_result["disturbance_trace_digest"]),
        realized_measurement_noise_rms_m=float(shot_result["realized_measurement_noise_rms_m"]),
        mean_abs_r_error_m=radial_error,
        mean_abs_z_error_m=vertical_error,
        tracking_reward_m=-(radial_error + vertical_error),
        control_policy_latency_us=[
            float(value) for value in shot_result["control_policy_latency_us"]
        ],
        simulation_wall_time_us=float(shot_result["simulation_wall_time_us"]),
        disrupted=disrupted,
        t_disruption_s=t_disruption,
        magnetic_actuator_absolute_current_offset_integral_ma_s=float(
            shot_result["magnetic_actuator_absolute_current_offset_integral_ma_s"]
        ),
        mean_abs_coil_current_offset_tracking_error_ma=float(
            shot_result["mean_abs_coil_current_offset_tracking_error_ma"]
        ),
    )


def _run_pid_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run a single PID episode."""
    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


def _run_hinf_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run a single H-infinity episode.

    Uses two independent H-inf controllers (one per axis), each
    synthesized for the flight sim's quasi-static equilibrium dynamics
    (error + first-order actuator lag). The vertical command is sign-adapted
    to the IsoFluxController top/bottom coil-pair convention.
    """
    if not _env_flag_enabled(HINF_RESEARCH_ENV):
        raise RuntimeError(
            "H-infinity research lane is disabled by default; set "
            f"{HINF_RESEARCH_ENV}=1 or pass --enable-hinf-research."
        )

    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )

    hinf_R = get_flight_sim_offset_hinf_controller(
        position_sensitivity=HINF_RADIAL_POSITION_SENSITIVITY,
    )
    hinf_Z = get_flight_sim_offset_hinf_controller(
        position_sensitivity=-HINF_VERTICAL_POSITION_SENSITIVITY,
    )

    policy = _TwoAxisPolicy(
        len(ctrl.kernel.cfg["coils"]),
        lambda error: _hinf_zero_delay_command(hinf_R, error, dt),
        lambda error: _hinf_zero_delay_command(hinf_Z, error, dt),
    )
    result = ctrl.run_shot(shot_duration=steps, save_plot=False, control_policy=policy)
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


def _run_lqr_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run one LQR episode matched to the MA offset-setpoint plant."""
    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )

    lqr_R = get_flight_sim_offset_lqr_controller(
        position_sensitivity=0.567,
    )
    lqr_Z = get_flight_sim_offset_lqr_controller(
        position_sensitivity=-0.05,
    )

    policy = _TwoAxisPolicy(
        len(ctrl.kernel.cfg["coils"]),
        lambda error: float(lqr_R.step(error, dt)),
        lambda error: float(lqr_Z.step(error, dt)),
    )
    result = ctrl.run_shot(shot_duration=steps, save_plot=False, control_policy=policy)
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


def _run_mpc_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run a single linear-surrogate MPC episode."""
    if not _mpc_available or ModelPredictiveController is None or NeuralSurrogate is None:
        raise RuntimeError("MPC controller is unavailable in this environment.")
    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )
    n_coils = len(ctrl.kernel.cfg.get("coils", []))
    if n_coils < 1:
        raise RuntimeError("ITER config is missing coil definitions for MPC.")

    calibration_kernel = type(ctrl.kernel)(str(config_path))
    calibrated_model = NeuralSurrogate(n_coils=n_coils, n_state=4, verbose=False)
    calibrated_model.train_on_kernel(calibration_kernel, perturbation=0.01)
    surrogate_model = NeuralSurrogate(n_coils=n_coils, n_state=2, verbose=False)
    surrogate_model.B = np.asarray(calibrated_model.B[:2, :], dtype=np.float64)
    mpc = ModelPredictiveController(
        surrogate_model,
        target_state=np.array([ctrl.target_R, ctrl.target_Z], dtype=np.float64),
        prediction_horizon=6,
        learning_rate=0.25,
        iterations=8,
        action_limit=2.0,
    )

    class MpcPolicy:
        """Apply the complete calibrated full-coil MPC action once per step."""

        def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
            state = np.array(
                [observation.measured_axis_r_m, observation.measured_axis_z_m],
                dtype=np.float64,
            )
            action = np.asarray(mpc.plan_trajectory(state), dtype=np.float64).reshape(-1)
            if action.shape != (n_coils,):
                raise ValueError("MPC returned an invalid full-coil action shape.")
            return CoilCurrentOffsetCommand(tuple(float(value) for value in action))

    result = ctrl.run_shot(shot_duration=steps, save_plot=False, control_policy=MpcPolicy())
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


def _run_nmpc_jax_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Reject the uncalibrated random-MLP NMPC implementation fail closed."""
    raise RuntimeError(
        "NMPC-JAX has no calibrated dynamics artifact or held-out validation gate; "
        "its random Xavier dynamics cannot produce scientific campaign evidence."
    )


def _run_snn_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run the pure-NumPy two-channel LIF/NEF SNN on the common plant."""
    if not _snn_available or NengoSNNController is None or NengoSNNConfig is None:
        raise RuntimeError("LIF-NEF-SNN controller is unavailable in this environment.")
    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )
    snn_config = NengoSNNConfig(n_neurons=200, n_channels=2, seed=episode_seed)
    snn = NengoSNNController(snn_config)
    substeps_float = dt / float(snn_config.dt)
    substeps = int(round(substeps_float))
    if not np.isclose(substeps_float, substeps, rtol=0.0, atol=1.0e-12):
        raise ValueError("plant control period must be an integer multiple of SNN dt.")

    class LifNefPolicy:
        """Advance both SNN channels through exactly one plant control interval."""

        def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
            error = np.array(
                [observation.radial_error_m, observation.vertical_error_m], dtype=np.float64
            )
            output = np.zeros(2, dtype=np.float64)
            for _ in range(substeps):
                output = np.asarray(snn.step(error), dtype=np.float64)
            return map_axis_commands_to_coil_offsets(
                len(observation.coil_currents_ma), float(output[0]), float(output[1])
            )

    result = ctrl.run_shot(shot_duration=steps, save_plot=False, control_policy=LifNefPolicy())
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


def _run_rust_pid_episode(
    config_path: str | Path,
    shot_duration: int = 30,
    surrogate: bool = False,
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str,
) -> EpisodeResult:
    """Run the Rust-native two-axis PID policy against the common Python plant."""
    if not _rust_pid_available or PyRustIsoFluxController is None:
        raise RuntimeError("Rust PID policy binding is unavailable in this environment.")
    dt = 0.01 if surrogate else 0.05
    steps = int(shot_duration / dt)
    ctrl = _build_isoflux_controller(
        config_path, surrogate=surrogate, dt=dt, scenario=scenario, episode_seed=episode_seed
    )
    rust_pid = PyRustIsoFluxController(ctrl.target_R, ctrl.target_Z)

    class RustPidPolicy:
        """Adapt the Rust PID pair to the common full-coil action basis."""

        def step(self, observation: ControlObservation) -> CoilCurrentOffsetCommand:
            radial, vertical = rust_pid.step(
                observation.measured_axis_r_m, observation.measured_axis_z_m
            )
            return map_axis_commands_to_coil_offsets(
                len(observation.coil_currents_ma), float(radial), float(vertical)
            )

    result = ctrl.run_shot(shot_duration=steps, save_plot=False, control_policy=RustPidPolicy())
    return _python_episode_result(
        result,
        shot_duration,
        scenario,
        episode_index,
        episode_seed,
        evaluation_contract_digest,
    )


CONTROLLERS: dict[str, EpisodeRunner] = {
    "PID": _run_pid_episode,
    "H-infinity": _run_hinf_episode,
    "LQR": _run_lqr_episode,
    "MPC": _run_mpc_episode,
    "NMPC-JAX": _run_nmpc_jax_episode,
    "LIF-NEF-SNN": _run_snn_episode,
    "Rust-PID": _run_rust_pid_episode,
}

POLICY_IMPLEMENTATIONS = {
    "PID": "python.IsoFluxController.PID.RZ",
    "H-infinity": "python.HInfinityController.offset_setpoint.RZ",
    "LQR": "python.LQRController.offset_setpoint.RZ",
    "MPC": "python.ModelPredictiveController.kernel_calibrated_linear.full_coil",
    "NMPC-JAX": "unavailable.random_untrained_DynamicsMLP",
    "LIF-NEF-SNN": "python.numpy_LIF_NEF.two_channel_1ms",
    "Rust-PID": "rust.fusion_control.pid.IsoFluxController.RZ_via_PyO3",
}


def _controller_unavailability_reason(controller_name: str) -> str | None:
    """Return an explicit reason when a registered lane cannot produce evidence."""
    if controller_name == "H-infinity" and not _env_flag_enabled(HINF_RESEARCH_ENV):
        return f"research_gate_disabled:{HINF_RESEARCH_ENV}"
    if controller_name == "MPC" and (
        not _mpc_available or ModelPredictiveController is None or NeuralSurrogate is None
    ):
        return "linear_mpc_dependency_unavailable"
    if controller_name == "NMPC-JAX":
        return "uncalibrated_dynamics_model:no_trained_artifact_or_held_out_gate"
    if controller_name == "LIF-NEF-SNN" and (
        not _snn_available or NengoSNNController is None or NengoSNNConfig is None
    ):
        return "lif_nef_snn_backend_unavailable"
    if controller_name == "Rust-PID" and (
        not _rust_pid_available or PyRustIsoFluxController is None
    ):
        return "rust_pid_policy_binding_unavailable"
    return None


def run_campaign(
    n_episodes: int = 1000,
    shot_duration: int = 30,
    config_path: str | Path | None = None,
    measurement_noise_std_m: float = 0.2,
    actuator_delay_ms: float = 50.0,
    surrogate: bool = False,
    controllers: list[str] | None = None,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> CampaignResults:
    """Run the full stress-test campaign across all available controllers.

    Parameters
    ----------
    n_episodes : int
        Number of episodes per controller (default: 1000).
    shot_duration : int
        Simulated shot duration in seconds.
    config_path : Path or str or None
        Path to ITER config JSON; defaults to the bundled validation config.
    measurement_noise_std_m : float
        Standard deviation of independent radial and vertical position
        measurement noise, in metres.
    actuator_delay_ms : float
        Pure actuator-command transport delay in milliseconds.
    surrogate : bool
        Whether to use the neural equilibrium surrogate (fast-path).
    controllers : list[str] | None
        Optional ordered subset of controller names to run. By default all
        registered controllers run.
    seed : int or None
        Unsigned 64-bit master seed. A cryptographically generated seed is
        recorded when omitted, so every run remains replayable.
    checkpoint_dir : Path or str or None
        Recovery directory. Defaults to an identity-scoped path below
        ``.cache/stress_campaign``.
    resume : bool
        Resume only checkpoints whose exact campaign identity matches.
    """
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes < 1:
        raise ValueError("n_episodes must be >= 1.")
    if isinstance(shot_duration, bool) or not isinstance(shot_duration, int) or shot_duration < 1:
        raise ValueError("shot_duration must be >= 1 second.")
    if resume and seed is None:
        raise ValueError("--resume requires the exact recorded --seed value.")
    if seed is None:
        seed = secrets.randbits(64)
    scenario = StressScenario(
        measurement_noise_std_m=float(measurement_noise_std_m),
        actuator_delay_s=float(actuator_delay_ms) / 1000.0,
        master_seed=seed,
    )

    if config_path is None:
        config_path = default_iter_config_path()
    resolved_config_path = Path(config_path).resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"campaign config does not exist: {resolved_config_path}")
    evaluation_contract = build_evaluation_contract(
        resolved_config_path,
        surrogate=surrogate,
        scenario=scenario,
    )

    print("=== Controller Stress-Test Campaign ===")
    print(f"Episodes: {n_episodes} | Shot duration: {shot_duration}s")
    print(
        f"Measurement noise sigma: {scenario.measurement_noise_std_m:.6g}m | "
        f"Actuator command delay: {scenario.actuator_delay_s * 1000.0:.6g}ms"
    )
    print(f"Master seed: {scenario.master_seed} | Scenario: {scenario.digest}")
    print(f"Surrogate: {'Enabled' if surrogate else 'Disabled'}")
    print(
        "H-infinity research lane: "
        + ("Enabled" if _env_flag_enabled(HINF_RESEARCH_ENV) else "Disabled")
    )
    controller_registry = CONTROLLERS
    if controllers is not None:
        if not controllers:
            raise ValueError("controllers must contain at least one controller name.")
        if len(controllers) != len(set(controllers)):
            raise ValueError("controllers must not contain duplicate names.")
        unknown = [name for name in controllers if name not in CONTROLLERS]
        if unknown:
            available = ", ".join(CONTROLLERS.keys())
            raise ValueError(f"Unknown controller(s): {', '.join(unknown)}. Available: {available}")
        controller_registry = {name: CONTROLLERS[name] for name in controllers}

    print(f"Controllers: {', '.join(controller_registry.keys())}")

    software_versions = {
        "numpy": _lib_version("numpy"),
        "jax": _lib_version("jax"),
        "nengo": _lib_version("nengo"),
        "scpn_fusion_rs": "present" if _rust_pid_available else "absent",
        "scpn_fusion_rs_binary_sha256": _module_binary_sha256("scpn_fusion_rs"),
    }
    campaign_identity = build_campaign_identity(
        repo_root=repo_root,
        config_path=resolved_config_path,
        scenario=scenario,
        controllers=list(controller_registry),
        requested_episodes=n_episodes,
        shot_duration_s=shot_duration,
        surrogate=surrogate,
        software_versions=software_versions,
        execution_options={
            "hinf_research_enabled": _env_flag_enabled(HINF_RESEARCH_ENV),
        },
        evaluation_contract=evaluation_contract,
        controller_implementations={
            name: POLICY_IMPLEMENTATIONS.get(
                name,
                f"injected.{getattr(runner, '__module__', 'unknown')}.{getattr(runner, '__name__', type(runner).__name__)}",
            )
            for name, runner in controller_registry.items()
        },
    )
    if checkpoint_dir is None:
        checkpoint_dir = repo_root / ".cache" / "stress_campaign" / campaign_identity["digest"]
    recovery = RecoveryStore(Path(checkpoint_dir), campaign_identity, require_existing=resume)
    results = CampaignResults(
        scenario=scenario,
        campaign_identity=campaign_identity,
    )
    for controller_name in controller_registry:
        restored = recovery.load_lane(controller_name) if resume else None
        results[controller_name] = restored or ControllerMetrics(
            name=controller_name,
            requested_episodes=n_episodes,
            scenario_digest=scenario.digest,
            evaluation_contract_digest=evaluation_contract["digest"],
            policy_implementation=campaign_identity["payload"]["controller_implementations"][
                controller_name
            ],
        )
        if not resume:
            recovery.write_lane(results[controller_name])
    attempted_at_start = sum(lane.n_episodes + lane.failed_episodes for lane in results.values())
    started_monotonic = time.monotonic()
    recovery.write_progress(
        lanes=results,
        active_controller=None,
        active_episode_index=None,
        active_episode_seed=None,
        started_monotonic=started_monotonic,
        attempted_at_start=attempted_at_start,
    )

    for ctrl_name, run_fn in controller_registry.items():
        print(f"\n--- Running {ctrl_name} ({n_episodes} episodes) ---")
        metrics = results[ctrl_name]
        unavailable_reason = _controller_unavailability_reason(ctrl_name)
        if unavailable_reason is not None:
            metrics.status = "unavailable"
            metrics.reason = unavailable_reason
            recovery.write_lane(metrics)
            recovery.write_progress(
                lanes=results,
                active_controller=None,
                active_episode_index=None,
                active_episode_seed=None,
                started_monotonic=started_monotonic,
                attempted_at_start=attempted_at_start,
            )
            continue

        attempted_indices = {item.episode_index for item in metrics.episodes} | {
            item.episode_index for item in metrics.failures
        }
        if len(attempted_indices) == n_episodes:
            metrics.finalize(float(shot_duration))
            recovery.write_lane(metrics)
            continue
        metrics.status = "running"

        for ep in range(n_episodes):
            if ep in attempted_indices:
                continue
            episode_seed = scenario.episode_seed(ep)
            recovery.write_lane(metrics)
            recovery.write_progress(
                lanes=results,
                active_controller=ctrl_name,
                active_episode_index=ep,
                active_episode_seed=episode_seed,
                started_monotonic=started_monotonic,
                attempted_at_start=attempted_at_start,
            )
            try:
                with recovery.progress_heartbeat(
                    lanes=results,
                    active_controller=ctrl_name,
                    active_episode_index=ep,
                    active_episode_seed=episode_seed,
                    started_monotonic=started_monotonic,
                    attempted_at_start=attempted_at_start,
                ):
                    episode = run_fn(
                        resolved_config_path,
                        shot_duration,
                        surrogate=surrogate,
                        scenario=scenario,
                        episode_index=ep,
                        episode_seed=episode_seed,
                        evaluation_contract_digest=evaluation_contract["digest"],
                    )
                if episode.episode_index != ep:
                    raise ValueError("runner returned a mismatched episode_index.")
                if episode.seed != episode_seed:
                    raise ValueError("runner returned a mismatched episode seed.")
                if episode.scenario_digest != scenario.digest:
                    raise ValueError("runner returned a mismatched scenario digest.")
                if episode.evaluation_contract_digest != evaluation_contract["digest"]:
                    raise ValueError("runner returned a mismatched evaluation contract digest.")
                expected_trace = next(
                    (
                        prior.disturbance_trace_digest
                        for name, lane in results.items()
                        if name != ctrl_name
                        for prior in lane.episodes
                        if prior.episode_index == ep
                    ),
                    None,
                )
                if (
                    expected_trace is not None
                    and episode.disturbance_trace_digest != expected_trace
                ):
                    raise ValueError("runner returned a mismatched common disturbance trace.")
                metrics.episodes.append(episode)
            except Exception as e:
                print(f"  Episode {ep} failed: {e}")
                traceback.print_exc()
                metrics.failures.append(
                    EpisodeFailure(
                        episode_index=ep,
                        seed=episode_seed,
                        exception_type=type(e).__name__,
                        message=str(e),
                        backend=ctrl_name,
                        stage="episode_runner",
                        traceback_text=traceback.format_exc(limit=20)[-16_000:],
                    )
                )

            metrics.n_episodes = len(metrics.episodes)
            metrics.failed_episodes = len(metrics.failures)
            recovery.write_lane(metrics)
            recovery.write_progress(
                lanes=results,
                active_controller=ctrl_name,
                active_episode_index=None,
                active_episode_seed=None,
                started_monotonic=started_monotonic,
                attempted_at_start=attempted_at_start,
            )

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"  Episode {ep + 1}/{n_episodes}")

        metrics.finalize(float(shot_duration))
        recovery.write_lane(metrics)
        recovery.write_progress(
            lanes=results,
            active_controller=None,
            active_episode_index=None,
            active_episode_seed=None,
            started_monotonic=started_monotonic,
            attempted_at_start=attempted_at_start,
        )

    return results


def generate_summary_table(results: dict[str, ControllerMetrics]) -> str:
    """Generate a markdown summary table."""

    def render(value: float | None, spec: str) -> str:
        """Render absent metrics honestly instead of substituting zero."""
        return "N/A" if value is None else format(value, spec)

    lines = [
        "| Controller | Status | Episodes | Failures | Mean reward (m) | Std reward (m) "
        "| Mean R err (m) | Mean Z err (m) | P50 policy (us) | P95 policy (us) "
        "| P99 policy (us) | Disrupt rate | DEF | Mean actuator effort (MA s) |",
        "|------------|--------|----------|----------|-----------------|----------------"
        "|----------------|----------------|-----------------|-----------------"
        "|-----------------|--------------|-----|-----------------------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.status} | {m.n_episodes}/{m.requested_episodes} "
            f"| {m.failed_episodes} | {render(m.mean_tracking_reward_m, '.4f')} "
            f"| {render(m.std_tracking_reward_m, '.4f')} "
            f"| {render(m.mean_abs_r_error_m, '.4f')} "
            f"| {render(m.mean_abs_z_error_m, '.4f')} "
            f"| {render(m.p50_control_policy_latency_us, '.0f')} "
            f"| {render(m.p95_control_policy_latency_us, '.0f')} "
            f"| {render(m.p99_control_policy_latency_us, '.0f')} "
            f"| {render(m.disruption_rate, '.2%')} | {render(m.mean_def, '.2f')} "
            f"| {render(m.mean_magnetic_actuator_absolute_current_offset_integral_ma_s, '.4f')} |"
        )
    return "\n".join(lines)


def derive_hinf_graduation_status(results: dict[str, ControllerMetrics]) -> dict[str, Any]:
    """Compute explicit graduation criteria for H-infinity default-lane promotion."""
    pid = results.get("PID")
    hinf = results.get("H-infinity")
    if pid is None or hinf is None:
        return {
            "available": False,
            "eligible_for_default_lane": False,
            "reason": "missing_pid_or_hinf_metrics",
        }
    if not pid.comparable or not hinf.comparable:
        return {
            "available": True,
            "eligible_for_default_lane": False,
            "reason": "pid_or_hinf_lane_incomplete_or_incomparable",
            "pid_status": pid.status,
            "hinf_status": hinf.status,
        }
    if pid.scenario_digest != hinf.scenario_digest:
        return {
            "available": True,
            "eligible_for_default_lane": False,
            "reason": "scenario_identity_mismatch",
        }
    if pid.evaluation_contract_digest != hinf.evaluation_contract_digest:
        return {
            "available": True,
            "eligible_for_default_lane": False,
            "reason": "evaluation_contract_identity_mismatch",
        }
    if isinstance(results, CampaignResults):
        scope = (
            results.campaign_identity.get("payload", {})
            .get("evaluation_contract", {})
            .get("payload", {})
            .get("evidence_scope")
        )
        if scope != "controller_comparison":
            return {
                "available": True,
                "eligible_for_default_lane": False,
                "reason": "evaluation_contract_not_promotion_eligible",
                "evidence_scope": scope or "unknown",
            }
    pid_traces = {item.episode_index: item.disturbance_trace_digest for item in pid.episodes}
    hinf_traces = {item.episode_index: item.disturbance_trace_digest for item in hinf.episodes}
    if pid_traces != hinf_traces:
        return {
            "available": True,
            "eligible_for_default_lane": False,
            "reason": "disturbance_trace_identity_mismatch",
        }

    pid_p95 = pid.p95_control_policy_latency_us
    pid_disruption_rate = pid.disruption_rate
    pid_reward = pid.mean_tracking_reward_m
    hinf_p95 = hinf.p95_control_policy_latency_us
    hinf_disruption_rate = hinf.disruption_rate
    hinf_reward = hinf.mean_tracking_reward_m
    if (
        pid_p95 is None
        or pid_disruption_rate is None
        or pid_reward is None
        or hinf_p95 is None
        or hinf_disruption_rate is None
        or hinf_reward is None
    ):
        return {
            "available": True,
            "eligible_for_default_lane": False,
            "reason": "complete_lane_missing_aggregate_metrics",
        }

    latency_ratio = hinf_p95 / max(pid_p95, 1.0e-12)
    disruption_delta = hinf_disruption_rate - pid_disruption_rate
    reward_delta = hinf_reward - pid_reward

    checks: dict[str, dict[str, Any]] = {
        "research_gate_enabled": {
            "value": bool(_env_flag_enabled(HINF_RESEARCH_ENV)),
            "required": True,
            "passes": bool(_env_flag_enabled(HINF_RESEARCH_ENV)),
        },
        "episodes": {
            "value": int(hinf.n_episodes),
            "required_min": 100,
            "passes": bool(hinf.n_episodes >= 100),
        },
        "disruption_delta": {
            "value": round(disruption_delta, 6),
            "required_max": 0.01,
            "passes": bool(disruption_delta <= 0.01),
        },
        "latency_p95_ratio": {
            "value": round(latency_ratio, 6),
            "required_max": 3.0,
            "passes": bool(latency_ratio <= 3.0),
        },
        "reward_delta": {
            "value": round(reward_delta, 6),
            "required_min": -0.05,
            "passes": bool(reward_delta >= -0.05),
        },
    }

    return {
        "available": True,
        "eligible_for_default_lane": bool(all(c["passes"] for c in checks.values())),
        "checks": checks,
        "baseline_pid": {
            "n_episodes": int(pid.n_episodes),
            "mean_tracking_reward_m": pid_reward,
            "p95_control_policy_latency_us": pid_p95,
            "disruption_rate": pid_disruption_rate,
        },
        "candidate_hinf": {
            "n_episodes": int(hinf.n_episodes),
            "mean_tracking_reward_m": hinf_reward,
            "p95_control_policy_latency_us": hinf_p95,
            "disruption_rate": hinf_disruption_rate,
        },
    }


def _cpu_model() -> str:
    """Return the real CPU model string for host provenance (never fabricated)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _lib_version(module_name: str) -> str:
    """Return an installed module's version, or 'absent' if it is not importable."""
    try:
        module = __import__(module_name)
    except Exception:
        return "absent"
    return str(getattr(module, "__version__", "present"))


def _module_binary_sha256(module_name: str) -> str:
    """Fingerprint an imported extension/module file, or report its absence."""
    try:
        module = __import__(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            return "absent"
        module_path = Path(module_file).resolve()
        return sha256_file(module_path)
    except (AttributeError, ImportError, OSError, TypeError):
        return "absent"


def _git_sha() -> str:
    """Return the current Git commit from repository metadata, or ``unknown``."""
    try:
        git_path = repo_root / ".git"
        if git_path.is_file():
            marker, raw_path = git_path.read_text(encoding="utf-8").strip().split(":", 1)
            if marker != "gitdir":
                return "unknown"
            git_path = (repo_root / raw_path.strip()).resolve()
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if len(head) == 40 else "unknown"
        reference = head.removeprefix("ref: ")
        loose_reference = git_path / reference
        if loose_reference.is_file():
            return loose_reference.read_text(encoding="utf-8").strip()
        packed_references = git_path / "packed-refs"
        if packed_references.is_file():
            for line in packed_references.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, packed_reference = line.split(" ", 1)
                    if packed_reference == reference:
                        return sha
    except (OSError, UnicodeError, ValueError):
        return "unknown"
    return "unknown"


def collect_provenance(
    *,
    n_episodes: int,
    shot_duration: int,
    seed: int | None,
    controllers: list[str],
    timestamp_utc: str,
    scenario: StressScenario | None = None,
    campaign_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-documenting provenance block for a measurement run.

    The host CPU model is read from the machine actually running the campaign
    (``/proc/cpuinfo`` / ``platform``); it is NEVER hard-coded, so latency numbers
    are always attributable to the box that produced them. Run the same command on
    a second host (e.g. a cloud instance) to obtain an independent measurement.
    """
    return {
        "schema": "scpn-fusion-core.stress-test-campaign-provenance.v3",
        "timestamp_utc": timestamp_utc,
        "git_sha": _git_sha(),
        "host": {
            "cpu_model": _cpu_model(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": _lib_version("numpy"),
            "jax": _lib_version("jax"),
            "nengo": _lib_version("nengo"),
            "scpn_fusion_rs": "present" if _rust_pid_available else "absent",
            "scpn_fusion_rs_binary_sha256": _module_binary_sha256("scpn_fusion_rs"),
        },
        "methodology": {
            "n_episodes": int(n_episodes),
            "shot_duration_s": int(shot_duration),
            "seed": seed,
            "scenario": scenario.to_dict() if scenario is not None else None,
            "scenario_digest": scenario.digest if scenario is not None else None,
            "campaign_identity": campaign_identity,
            "latency_metric": (
                "perf_counter_ns around exactly one control-policy step; "
                "p50/p95/p99 over every recorded control-step sample"
            ),
            "controllers": controllers,
            "note": (
                "Latency is host-dependent; reproduce on any host with "
                "`python validation/stress_test_campaign.py --output <path> [--seed N]` "
                "and compare the provenance.host block."
            ),
        },
    }


def campaign_is_complete_and_comparable(results: CampaignResults) -> bool:
    """Require every lane, evaluation contract, and per-episode trace to match."""
    if not results or any(not lane.comparable for lane in results.values()):
        return False
    if {lane.scenario_digest for lane in results.values()} != {results.scenario.digest}:
        return False
    if len({lane.evaluation_contract_digest for lane in results.values()}) != 1:
        return False
    requested_counts = {lane.requested_episodes for lane in results.values()}
    if len(requested_counts) != 1:
        return False
    requested = requested_counts.pop()
    for episode_index in range(requested):
        traces = {
            episode.disturbance_trace_digest
            for lane in results.values()
            for episode in lane.episodes
            if episode.episode_index == episode_index
        }
        if len(traces) != 1:
            return False
    return True


def campaign_promotion_status(results: CampaignResults) -> tuple[bool, str | None]:
    """Return promotion eligibility independently of computational completion."""
    if not campaign_is_complete_and_comparable(results):
        return False, "campaign_incomplete_or_incomparable"
    scope = (
        results.campaign_identity.get("payload", {})
        .get("evaluation_contract", {})
        .get("payload", {})
        .get("evidence_scope")
    )
    if scope != "controller_comparison":
        return False, f"evaluation_scope:{scope or 'unknown'}"
    return True, None


def save_results_json(
    results: CampaignResults,
    path: Path,
    provenance: dict[str, Any],
) -> None:
    """Persist only scenario-bound campaign results with host provenance."""
    if not isinstance(results, CampaignResults):
        raise TypeError("results must be a scenario-bound CampaignResults instance.")
    if not provenance:
        raise ValueError("provenance is required for a recovery-grade result.")
    campaign_complete = campaign_is_complete_and_comparable(results)
    promotion_eligible, promotion_reason = campaign_promotion_status(results)
    if not campaign_complete:
        campaign_status = "incomplete"
    elif promotion_eligible:
        campaign_status = "complete"
    else:
        campaign_status = "complete_wiring_only"
    data: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "provenance": provenance,
        "campaign_status": campaign_status,
        "campaign_complete": campaign_complete,
        "promotion_eligible": promotion_eligible,
        "promotion_ineligibility_reason": promotion_reason,
        "scenario": results.scenario.to_dict(),
        "scenario_digest": results.scenario.digest,
        "campaign_identity": results.campaign_identity,
        "controllers": {},
        "hinf_graduation": derive_hinf_graduation_status(results),
    }
    for name, m in results.items():
        data["controllers"][name] = m.to_dict(include_episodes=True)
    atomic_write_json(path, data)
    print(f"Results saved to {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the stress-campaign command-line parser."""
    parser = argparse.ArgumentParser(description="Controller Stress-Test Campaign")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes per controller (default: 1000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 episodes for CI validation",
    )
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Use neural equilibrium surrogate for ~1000x faster loop",
    )
    parser.add_argument(
        "--shot-duration",
        type=int,
        default=30,
        help="Simulated shot duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Path to the FusionKernel / IsoFlux config JSON consumed by each "
            "controller episode. Defaults to the bundled validation config when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--enable-hinf-research",
        action="store_true",
        help=(
            "Enable H-infinity research lane. Disabled by default because "
            "this controller remains an experimental policy path."
        ),
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default=None,
        help=(
            "Comma-separated ordered subset of controllers to run, for example "
            "PID,H-infinity,LQR,Rust-PID."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Unsigned 64-bit master seed. If omitted, a generated seed is recorded "
            "for exact scientific replay. Latency remains host-dependent."
        ),
    )
    parser.add_argument(
        "--measurement-noise-std-m",
        type=float,
        default=0.2,
        help="Gaussian R/Z measurement-noise standard deviation in metres (default: 0.2)",
    )
    parser.add_argument(
        "--actuator-delay-ms",
        type=float,
        default=50.0,
        help="Pure actuator command delay in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Recovery directory (default: identity-scoped directory below .cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exact-identity lane checkpoints; reject any input/source/environment drift",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, ControllerMetrics]:
    """Parse arguments and run the stress-test campaign entrypoint."""
    args = build_arg_parser().parse_args(argv)

    if args.quick:
        args.episodes = 10
    if args.enable_hinf_research:
        os.environ[HINF_RESEARCH_ENV] = "1"
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    selected_controllers = (
        [name.strip() for name in args.controllers.split(",") if name.strip()]
        if args.controllers
        else None
    )
    results = run_campaign(
        n_episodes=args.episodes,
        shot_duration=args.shot_duration,
        config_path=args.config_path,
        surrogate=args.surrogate,
        controllers=selected_controllers,
        measurement_noise_std_m=args.measurement_noise_std_m,
        actuator_delay_ms=args.actuator_delay_ms,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    print("\n" + generate_summary_table(results))

    if args.output:
        provenance = collect_provenance(
            n_episodes=args.episodes,
            shot_duration=args.shot_duration,
            seed=results.scenario.master_seed,
            controllers=list(results.keys()),
            timestamp_utc=timestamp_utc,
            scenario=results.scenario,
            campaign_identity=results.campaign_identity,
        )
        save_results_json(results, Path(args.output), provenance=provenance)

    if not campaign_is_complete_and_comparable(results):
        raise SystemExit(2)
    return results


if __name__ == "__main__":
    main()
