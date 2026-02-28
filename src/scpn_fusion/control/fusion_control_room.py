# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Control Room
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

logger = logging.getLogger(__name__)

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    try:
        from scpn_fusion.core.fusion_kernel import FusionKernel
    except (ImportError, OSError):  # pragma: no cover - optional kernel path
        FusionKernel = None

RESOLUTION = 60
SIM_DURATION = 200
FPS = 10
_RENDER_OUTPUT_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
_KERNEL_INIT_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError, AttributeError)
_KERNEL_COIL_UPDATE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
    IndexError,
)
_KERNEL_SOLVE_EXCEPTIONS = (RuntimeError, ValueError, TypeError, FloatingPointError)


class TokamakPhysicsEngine:
    """
    Reduced Grad-Shafranov geometry model with optional kernel-Psi ingestion.
    """

    def __init__(
        self,
        size: int = RESOLUTION,
        *,
        seed: int = 42,
        kernel: Optional[Any] = None,
    ) -> None:
        size = int(size)
        if size < 16:
            raise ValueError("size must be >= 16.")
        self.size = size
        self.rng = np.random.default_rng(int(seed))
        self.kernel = kernel

        self.R = np.linspace(1.0, 5.0, self.size)
        self.Z = np.linspace(-3.0, 3.0, self.size)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.density = np.zeros((self.size, self.size), dtype=np.float64)

        # Plasma Parameters (ITER-like)
        self.R0 = 3.0
        self.a = 1.0
        self.kappa = 1.7
        self.delta = 0.33

        # State
        self.z_pos = 0.0
        self.v_drift = 0.0

    def _kernel_psi(self) -> Optional[np.ndarray]:
        if self.kernel is None or not hasattr(self.kernel, "Psi"):
            return None
        psi = np.asarray(getattr(self.kernel, "Psi"), dtype=np.float64)
        if psi.ndim != 2 or psi.shape[0] < 8 or psi.shape[1] < 8:
            return None
        return psi

    def solve_flux_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return `(density, psi)` from kernel state when available, otherwise
        from analytic Miller-parameterized geometry.
        """
        psi_kernel = self._kernel_psi()
        if psi_kernel is not None:
            psi_min = float(np.min(psi_kernel))
            psi_max = float(np.max(psi_kernel))
            psi_span = max(psi_max - psi_min, 1.0e-9)
            psi = (psi_kernel - psi_min) / psi_span
        else:
            rho_sq = (
                (self.RR - self.R0) ** 2
                + ((self.ZZ - self.z_pos) / self.kappa) ** 2
                - 2.0
                * self.delta
                * (self.RR - self.R0)
                * ((self.RR - self.R0) ** 2)
            )
            psi = rho_sq / (self.a**2)
            psi = np.clip(psi, 0.0, None)

        core_term = np.maximum(1.0 - psi, 0.0)
        self.density = np.where(psi < 1.0, core_term**1.5, 0.0)
        noise = self.rng.normal(0.0, 0.05, size=self.density.shape) * self.density
        self.density = np.clip(self.density + noise, 0.0, None)
        return self.density, psi

    def step_dynamics(self, coil_action_top: float, coil_action_bottom: float) -> float:
        """
        Reduced vertical-displacement dynamics.
        """
        instability_growth = 0.1
        control_force = (float(coil_action_bottom) - float(coil_action_top)) * 0.2
        disturbance = float(self.rng.normal(0.0, 0.01))
        if float(self.rng.random()) < 0.05:
            disturbance += 0.2

        accel = (instability_growth * self.z_pos) + control_force + disturbance
        self.v_drift += accel
        self.z_pos += self.v_drift
        self.v_drift *= 0.9
        return float(self.z_pos)


class DiagnosticSystem:
    """Noisy vertical-position probe."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def measure_position(self, true_z: float) -> float:
        return float(true_z + self._rng.normal(0.0, 0.05))


class KalmanObserver:
    """
    Linear Kalman Filter for robust plasma position state estimation.
    Harden state estimation against sensor noise and dropout.
    """
    def __init__(self, dt: float = 0.1):
        # State: [z, v_z]
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 0.1
        
        # Transition Matrix (Linear drift model)
        self.A = np.array([[1.0, dt],
                           [0.0, 0.9]]) # 0.9 matches drift damping in engine
        
        # Measurement Matrix (We only measure position z)
        self.H = np.array([[1.0, 0.0]])
        
        # Covariance Matrices
        self.Q = np.eye(2) * 0.01 # Process Noise
        self.R = np.array([[0.05]]) # Measurement Noise
        
    def update(self, measured_z: float, dropout: bool = False) -> float:
        """Predict-correct cycle. Returns filtered Z-position."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        if not dropout and np.isfinite(measured_z):
            y = measured_z - (self.H @ self.x)  # innovation
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            self.x = self.x + K @ y
            self.P = (np.eye(2) - K @ self.H) @ self.P
        else:
            self.P *= 1.2  # covariance inflation on dropout
            
        return float(self.x[0])


class NeuralController:
    """Hardened PID control policy for vertical stabilization."""

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        
        # PID Gains (Hardened defaults)
        self.kp = 5.0
        self.ki = 0.5
        self.kd = 2.0
        
        # Filter constant for derivative (tau_d)
        # Prevents noise spikes from being amplified
        self.tau_d = 0.05 
        
        # Anti-windup limit
        self.integral_limit = 2.0

    def compute_action(self, measured_z: float) -> tuple[float, float]:
        error = -float(measured_z)

        p_term = self.kp * error

        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral_error

        # d/dt low-pass: y(t) = α·x(t) + (1-α)·y(t-dt)
        raw_derivative = (error - self.prev_error) / self.dt
        alpha = self.dt / (self.tau_d + self.dt)
        self.filtered_derivative = alpha * raw_derivative + (1.0 - alpha) * self.filtered_derivative
        d_term = self.kd * self.filtered_derivative
        
        self.prev_error = error

        pid_out = p_term + i_term + d_term
        
        # Actuator Saturation (tanh)
        force = float(np.tanh(pid_out))
        if force > 0.0:
            return 0.0, abs(force)
        return abs(force), 0.0


def _render_outputs(
    frames: list[dict[str, Any]],
    history_z: list[float],
    history_top: list[float],
    history_bot: list[float],
    *,
    save_animation: bool,
    save_report: bool,
    output_gif: str,
    output_report: str,
) -> tuple[bool, Optional[str], bool, Optional[str]]:
    fig = plt.figure(figsize=(12, 8), facecolor="#1e1e1e")
    gs = fig.add_gridspec(2, 2)

    ax_plasma = fig.add_subplot(gs[:, 0])
    ax_plasma.set_facecolor("black")
    ax_plasma.set_title("Tokamak Cross-Section (Live)", color="white")
    extent = [
        float(frames[0]["r_min"]),
        float(frames[0]["r_max"]),
        float(frames[0]["z_min"]),
        float(frames[0]["z_max"]),
    ]
    im = ax_plasma.imshow(
        np.asarray(frames[0]["density"]),
        extent=extent,
        origin="lower",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        animated=True,
    )

    ax_trace = fig.add_subplot(gs[0, 1])
    ax_trace.set_facecolor("#2e2e2e")
    ax_trace.set_title("Vertical Displacement (Z-Pos)", color="white")
    ax_trace.set_ylim(-1.5, 1.5)
    ax_trace.grid(True, color="#444")
    line_z, = ax_trace.plot([], [], "cyan", lw=2, animated=True)
    line_setpoint, = ax_trace.plot([], [], "r--", alpha=0.5, animated=True)
    ax_trace.set_xlim(0, max(50, len(frames)))

    ax_coils = fig.add_subplot(gs[1, 1])
    ax_coils.set_facecolor("#2e2e2e")
    ax_coils.set_title("Poloidal Field Coil Currents", color="white")
    ax_coils.set_ylim(0, 1.1)
    bar_top = ax_coils.bar([0], [history_top[0]], color="red", label="Top Coil")
    bar_bot = ax_coils.bar([1], [history_bot[0]], color="blue", label="Bottom Coil")
    ax_coils.set_xticks([0, 1])
    ax_coils.set_xticklabels(["Top", "Bottom"], color="white")
    ax_coils.legend()

    top_marker, = ax_plasma.plot(3.0, 2.9, "s", color="red", markersize=20, alpha=0.3)
    bot_marker, = ax_plasma.plot(3.0, -2.9, "s", color="blue", markersize=20, alpha=0.3)
    wall = plt.Rectangle(
        (extent[0], extent[2] + 0.2),
        extent[1] - extent[0],
        (extent[3] - extent[2]) - 0.4,
        linewidth=2,
        edgecolor="gray",
        facecolor="none",
    )
    ax_plasma.add_patch(wall)

    def update(frame_idx: int):
        rec = frames[frame_idx]
        im.set_data(np.asarray(rec["density"]))
        line_z.set_data(range(frame_idx + 1), history_z[: frame_idx + 1])
        line_setpoint.set_data(range(frame_idx + 1), [0.0] * (frame_idx + 1))

        top_val = float(history_top[frame_idx])
        bot_val = float(history_bot[frame_idx])
        bar_top[0].set_height(top_val)
        bar_bot[0].set_height(bot_val)
        top_marker.set_alpha(0.3 + (0.7 * top_val))
        bot_marker.set_alpha(0.3 + (0.7 * bot_val))
        ax_plasma.set_title(f"Plasma Shape (t={frame_idx})", color="white")

        return (
            im,
            line_z,
            line_setpoint,
            bar_top[0],
            bar_bot[0],
            top_marker,
            bot_marker,
        )

    animation_saved = False
    animation_error: Optional[str] = None
    if save_animation:
        try:
            ani = FuncAnimation(
                fig,
                update,
                frames=len(frames),
                interval=100,
                blit=True,
            )
            ani.save(output_gif, writer=PillowWriter(fps=FPS))
            animation_saved = True
        except _RENDER_OUTPUT_EXCEPTIONS as exc:
            animation_error = str(exc)

    report_saved = False
    report_error: Optional[str] = None
    if save_report:
        try:
            update(len(frames) - 1)
            plt.tight_layout()
            fig.savefig(output_report)
            report_saved = True
        except _RENDER_OUTPUT_EXCEPTIONS as exc:
            report_error = str(exc)

    plt.close(fig)
    return animation_saved, animation_error, report_saved, report_error


def run_control_room(
    sim_duration: int = SIM_DURATION,
    *,
    seed: int = 42,
    save_animation: bool = True,
    save_report: bool = True,
    output_gif: str = "SCPN_Fusion_Control_Room.gif",
    output_report: str = "SCPN_Fusion_Status_Report.png",
    verbose: bool = True,
    kernel_factory: Optional[Callable[[str], Any]] = None,
    config_file: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the control-room loop and return deterministic summary metrics.
    """
    steps = int(sim_duration)
    if steps < 1:
        raise ValueError("sim_duration must be >= 1.")
    rng = np.random.default_rng(int(seed))

    kernel = None
    kernel_error = None
    psi_source = "analytic"
    if kernel_factory is not None or (config_file is not None and FusionKernel is not None):
        if kernel_factory is None and FusionKernel is not None:
            kernel_factory = FusionKernel
        cfg = (
            str(config_file)
            if config_file is not None
            else str(Path(__file__).resolve().parents[3] / "iter_config.json")
        )
        try:
            kernel = kernel_factory(cfg) if kernel_factory is not None else None
            if kernel is not None:
                psi_source = "kernel"
        except _KERNEL_INIT_EXCEPTIONS as exc:
            kernel = None
            kernel_error = str(exc)
            psi_source = "analytic"

    if verbose:
        logger.info("--- SCPN FUSION CONTROL ROOM: Grad-Shafranov VDE Simulation ---")
        if psi_source == "kernel":
            logger.info("Using kernel-backed Psi source.")
        elif kernel_error:
            logger.warning("Kernel init failed, fallback to analytic Psi: %s", kernel_error)

    reactor = TokamakPhysicsEngine(seed=seed, kernel=kernel)
    sensors = DiagnosticSystem(rng=rng)
    observer = KalmanObserver()
    ai = NeuralController()

    history_z: list[float] = []
    history_z_measured: list[float] = []
    history_top: list[float] = []
    history_bot: list[float] = []
    frames: list[dict[str, Any]] = []

    top_action = 0.0
    bot_action = 0.0

    for frame in range(steps):
        if kernel is not None and hasattr(kernel, "cfg"):
            try:
                coils = kernel.cfg.get("coils", [])
                if len(coils) >= 5:
                    coils[0]["current"] = float(coils[0].get("current", 0.0)) + top_action
                    coils[4]["current"] = float(coils[4].get("current", 0.0)) + bot_action
            except _KERNEL_COIL_UPDATE_EXCEPTIONS:
                pass

        if kernel is not None and hasattr(kernel, "solve_equilibrium"):
            try:
                kernel.solve_equilibrium()
            except _KERNEL_SOLVE_EXCEPTIONS:
                pass

        true_z = reactor.step_dynamics(top_action, bot_action)
        density, psi = reactor.solve_flux_surfaces()
        
        # Sensor measurement with potential dropout
        raw_measured_z = sensors.measure_position(true_z)
        dropout = (frame % 20 == 0) # Simulate periodic glitch
        
        # State estimation
        filtered_z = observer.update(raw_measured_z, dropout=dropout)
        
        # AI uses filtered state
        top_action, bot_action = ai.compute_action(filtered_z)

        history_z.append(float(true_z))
        history_z_measured.append(float(raw_measured_z))
        history_top.append(float(top_action))
        history_bot.append(float(bot_action))
        frames.append(
            {
                "density": np.asarray(density, dtype=np.float64).copy(),
                "psi": np.asarray(psi, dtype=np.float64).copy(),
                "r_min": float(np.min(reactor.R)),
                "r_max": float(np.max(reactor.R)),
                "z_min": float(np.min(reactor.Z)),
                "z_max": float(np.max(reactor.Z)),
            }
        )

    animation_saved = False
    animation_error: Optional[str] = None
    report_saved = False
    report_error: Optional[str] = None
    if save_animation or save_report:
        (
            animation_saved,
            animation_error,
            report_saved,
            report_error,
        ) = _render_outputs(
            frames,
            history_z,
            history_top,
            history_bot,
            save_animation=save_animation,
            save_report=save_report,
            output_gif=output_gif,
            output_report=output_report,
        )

    z_arr = np.asarray(history_z, dtype=np.float64)
    top_arr = np.asarray(history_top, dtype=np.float64)
    bot_arr = np.asarray(history_bot, dtype=np.float64)
    return {
        "seed": int(seed),
        "steps": int(steps),
        "psi_source": psi_source,
        "kernel_error": kernel_error,
        "final_z": float(z_arr[-1]),
        "mean_abs_z": float(np.mean(np.abs(z_arr))),
        "max_abs_z": float(np.max(np.abs(z_arr))),
        "mean_top_action": float(np.mean(top_arr)),
        "mean_bottom_action": float(np.mean(bot_arr)),
        "animation_saved": bool(animation_saved),
        "animation_error": animation_error,
        "report_saved": bool(report_saved),
        "report_error": report_error,
    }


if __name__ == "__main__":
    summary = run_control_room()
    print(
        "Control-room run complete "
        f"(psi_source={summary['psi_source']}, steps={summary['steps']})."
    )
