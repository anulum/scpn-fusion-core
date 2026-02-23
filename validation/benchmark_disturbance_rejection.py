# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disturbance Rejection Benchmark
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Benchmark SNN vs MPC vs H-infinity vs PID on three tokamak
disturbance-rejection scenarios.

Scenarios
---------
1. **VDE** -- Vertical Displacement Event with exponential growth
   rate gamma=100/s.  Duration: 2 s, dt=1e-4 s.
2. **Density ramp** -- Linear density ramp from 0.5 to 1.2 Greenwald
   fraction over 2 s.  Duration: 4 s, dt=1e-4 s.
3. **ELM pacing** -- 10 Hz periodic bursts, each causing a 5 % drop
   in beta_N.  Duration: 3 s, dt=1e-4 s.

Controllers
-----------
- **PID**: Proportional-Integral-Derivative with tuned gains and
  anti-windup.
- **H-infinity**: Riccati-synthesised robust controller from
  ``get_radial_robust_controller()`` with LQR fallback.
- **MPC**: Model Predictive Control with 10-step horizon and
  quadratic cost.
- **SNN**: Spiking neural network via ``SpikingControllerPool``
  from neuro_cybernetic_controller module.

Metrics (per controller per scenario)
--------------------------------------
- ISE: Integral of Squared Error
- Settling time: time to reach and stay within 5 % of setpoint
- Peak overshoot: maximum deviation from setpoint
- Control effort: integral of |u|
- Wall-clock time

Outputs
-------
- Comparison table to stdout (markdown format)
- ``<output-dir>/benchmark_disturbance_rejection.json``
- ``<output-dir>/benchmark_disturbance_rejection.md``
- Time-series overlay plots (if matplotlib available)

Usage::

    python validation/benchmark_disturbance_rejection.py
    python validation/benchmark_disturbance_rejection.py --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- make ``src/`` importable when running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Defensive imports for controller modules
# ---------------------------------------------------------------------------
_hinf_available = False
_snn_available = False

try:
    from scpn_fusion.control.h_infinity_controller import (
        HInfinityController,
        get_radial_robust_controller,
    )
    _hinf_available = True
except ImportError:
    warnings.warn(
        "H-infinity controller not available (scpn_fusion.control."
        "h_infinity_controller could not be imported). "
        "H-infinity will be skipped.",
        stacklevel=1,
    )

try:
    from scipy.linalg import solve_continuous_are
    _scipy_are_available = True
except ImportError:
    _scipy_are_available = False

try:
    from scpn_fusion.control.neuro_cybernetic_controller import (
        SpikingControllerPool,
    )
    _snn_available = True
except Exception:
    warnings.warn(
        "SNN controller not available (scpn_fusion.control."
        "neuro_cybernetic_controller could not be imported). "
        "SNN will be skipped.",
        stacklevel=1,
    )

# Optional matplotlib for plotting
_mpl_available = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_available = True
except ImportError:
    pass


# ===================================================================
# Simulation timestep and durations (shared across all controllers)
# ===================================================================
DT = 1.0e-4  # 100 us timestep for all scenarios

SCENARIO_DURATIONS = {
    "VDE": 2.0,           # 2 s
    "Density ramp": 4.0,  # 4 s
    "ELM pacing": 3.0,    # 3 s
}


# ===================================================================
# Controller protocol and implementations
# ===================================================================

class ControllerProtocol(Protocol):
    """Minimal controller interface."""

    def step(self, error: float, dt: float) -> float: ...
    def reset(self) -> None: ...


# -------------------------------------------------------------------
# PID Controller
# -------------------------------------------------------------------

class PIDController:
    """Textbook PID controller with anti-windup clamp.

    Tuned gains are chosen for the linearised vertical stability
    plant with growth rate gamma=100/s.
    """

    def __init__(
        self,
        kp: float = 1.5e4,
        ki: float = 3.0e3,
        kd: float = 1.5e2,
        u_max: float = 1.0e6,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.u_max = float(u_max)
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_step = True

    def step(self, error: float, dt: float) -> float:
        e = float(error)
        dt = float(dt)

        p_term = self.kp * e

        self._integral += e * dt
        i_term = self.ki * self._integral

        if self._first_step:
            d_term = 0.0
            self._first_step = False
        else:
            d_term = self.kd * (e - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = e

        u = p_term + i_term + d_term

        u_sat = float(np.clip(u, -self.u_max, self.u_max))
        if abs(u) > self.u_max:
            self._integral -= e * dt
        return u_sat

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_step = True


# -------------------------------------------------------------------
# LQR Robust Controller (fallback for H-infinity)
# -------------------------------------------------------------------

class LQRRobustController:
    """LQR-based robust controller with observer for vertical stability.

    Uses the same 2-state plant as ``get_radial_robust_controller()``.
    Falls back to this when the H-infinity ARE solver encounters
    singular matrices.
    """

    def __init__(self, gamma_growth: float = 100.0) -> None:
        self.gamma_growth = float(gamma_growth)

        self.A = np.array([
            [0.0, 1.0],
            [gamma_growth ** 2, -10.0],
        ])
        self.B = np.array([[0.0], [1.0]])
        self.C = np.array([[1.0, 0.0]])

        if not _scipy_are_available:
            raise RuntimeError("scipy.linalg.solve_continuous_are unavailable")

        Q = np.diag([10000.0, 10.0])
        R = np.array([[0.01]])
        X = solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R) @ self.B.T @ X

        A_cl = self.A - self.B @ self.K
        cl_eigs = np.linalg.eigvals(A_cl)

        fastest = float(np.min(np.real(cl_eigs)))
        p1 = 5.0 * fastest
        p2 = 5.0 * fastest - 50.0

        alpha1 = -(p1 + p2)
        alpha0 = p1 * p2
        tr_a = self.A[0, 0] + self.A[1, 1]
        det_a = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
        a01 = self.A[0, 1]
        self.L_obs = np.array([
            [alpha1 + tr_a],
            [(alpha0 - det_a) / a01 + alpha1 + tr_a],
        ])

        self.n = 2
        self.x_hat = np.zeros(2)
        self._is_stable = bool(np.all(np.real(cl_eigs) < 0))

    def step(self, error: float, dt: float) -> float:
        z_meas = -float(error)
        y = np.array([z_meas])
        y_hat = self.C @ self.x_hat
        innovation = y - y_hat

        u = float((-self.K @ self.x_hat).flat[0])

        dx_hat = (
            self.A @ self.x_hat
            + self.B.flatten() * u
            + self.L_obs.flatten() * float(innovation.flat[0])
        )
        self.x_hat = self.x_hat + dx_hat * dt
        return u

    def reset(self) -> None:
        self.x_hat = np.zeros(self.n)

    @property
    def is_stable(self) -> bool:
        return self._is_stable


# -------------------------------------------------------------------
# MPC Controller (quadratic cost, 10-step horizon)
# -------------------------------------------------------------------

class MPCController:
    """Model Predictive Controller with quadratic cost.

    Uses a simple linearised plant model internally and optimises a
    10-step horizon via iterative gradient descent on the quadratic
    cost J = sum_{k=0}^{H-1} [ Q * e(k)^2 + R * u(k)^2 ].

    The plant model matches the linearised vertical stability system:
        x = [z, dz/dt]
        A = [[0, 1], [gamma^2, -10]]
        B = [[0], [1]]
    """

    def __init__(
        self,
        gamma_growth: float = 100.0,
        horizon: int = 10,
        q_weight: float = 1.0e4,
        r_weight: float = 1.0e-2,
        iterations: int = 15,
        learning_rate: float = 0.1,
        u_max: float = 1.0e6,
    ) -> None:
        self.A = np.array([
            [0.0, 1.0],
            [gamma_growth ** 2, -10.0],
        ])
        self.B = np.array([0.0, 1.0])
        self.C = np.array([1.0, 0.0])

        self.horizon = int(horizon)
        self.q_weight = float(q_weight)
        self.r_weight = float(r_weight)
        self.iterations = int(iterations)
        self.lr = float(learning_rate)
        self.u_max = float(u_max)

        self._prev_error = 0.0
        self._first_step = True
        self._x_hat = np.zeros(2)

    def step(self, error: float, dt: float) -> float:
        e = float(error)
        dt_f = float(dt)

        # Simple observer: reconstruct velocity from error derivative
        if self._first_step:
            de_dt = 0.0
            self._first_step = False
        else:
            de_dt = (e - self._prev_error) / dt_f if dt_f > 0 else 0.0
        self._prev_error = e

        # Map error to plant state: z = -error (target=0), dz/dt ~ -de/dt
        self._x_hat[0] = -e
        self._x_hat[1] = -de_dt

        # Optimise control sequence over horizon
        u_seq = np.zeros(self.horizon)
        for _ in range(self.iterations):
            # Forward simulate and compute gradients
            x = self._x_hat.copy()
            grad = np.zeros(self.horizon)
            # Forward pass: store states
            states = np.zeros((self.horizon + 1, 2))
            states[0] = x.copy()
            for k in range(self.horizon):
                dx = self.A @ states[k] + self.B * u_seq[k]
                states[k + 1] = states[k] + dx * dt_f

            # Backward pass: compute gradients
            # Cost: J = sum Q * z(k)^2 + R * u(k)^2
            # dJ/du(k) = R * 2 * u(k) + dz(k+1)/du(k) * Q * 2 * z(k+1)
            # For simplicity, use single-step sensitivity:
            #   dz(k+1)/du(k) ~ B[0]*dt (direct) but second-order also matters
            # We use accumulated sensitivity via chain rule
            for k in range(self.horizon):
                z_k1 = states[k + 1, 0]
                # Sensitivity of z at step k+1 to u at step k
                # dz_{k+1}/du_k = B_z * dt = 0 * dt = 0 (B[0]=0)
                # but dz_dot_{k+1}/du_k = B[1]*dt = dt
                # and dz_{k+2}/du_k includes gamma^2 * dz_dot contribution
                # For simplicity, use finite-difference-like sensitivity
                # through the chain: dJ/du_k ~ 2*R*u_k + 2*Q*z_{k+1}*dz/du
                sens_z = self.B[1] * dt_f * dt_f  # dz ~ B[1]*dt*dt through A
                grad[k] = (
                    2.0 * self.r_weight * u_seq[k]
                    + 2.0 * self.q_weight * z_k1 * sens_z
                )

            u_seq -= self.lr * grad
            u_seq = np.clip(u_seq, -self.u_max, self.u_max)

        # Apply first action (receding horizon)
        return float(u_seq[0])

    def reset(self) -> None:
        self._prev_error = 0.0
        self._first_step = True
        self._x_hat = np.zeros(2)


# -------------------------------------------------------------------
# SNN Controller Wrapper
# -------------------------------------------------------------------

class SNNControllerWrapper:
    """Wraps SpikingControllerPool with the step(error, dt) / reset() API.

    The SNN pool operates in discrete spike-rate mode. We scale the
    error signal and multiply the output by a gain factor tuned for
    the vertical-stability plant.
    """

    def __init__(
        self,
        n_neurons: int = 50,
        gain: float = 5.0e3,
        tau_window: int = 20,
        seed: int = 42,
    ) -> None:
        if not _snn_available:
            raise RuntimeError("SpikingControllerPool is not available")
        self._pool = SpikingControllerPool(
            n_neurons=n_neurons,
            gain=1.0,
            tau_window=tau_window,
            seed=seed,
            allow_numpy_fallback=True,
        )
        self._gain = float(gain)

    def step(self, error: float, dt: float) -> float:
        # The pool expects a raw error signal; we scale output for plant
        raw = self._pool.step(float(error))
        return raw * self._gain

    def reset(self) -> None:
        # Re-initialise spike history
        for _ in range(self._pool.window_size):
            self._pool.history_pos.append(0)
            self._pool.history_neg.append(0)
        self._pool.last_rate_pos = 0.0
        self._pool.last_rate_neg = 0.0
        if self._pool._v_pos is not None:
            self._pool._v_pos[:] = 0.0
        if self._pool._v_neg is not None:
            self._pool._v_neg[:] = 0.0


# ===================================================================
# Linearised plant model
# ===================================================================

class LinearPlant:
    """Linearised vertical-stability plant for disturbance-rejection tests.

    State: x = [z, dz/dt] (vertical position and velocity).

    dx/dt = A x + B u + B_d d(t)

    Plant model matches ``get_radial_robust_controller()``:
        A = [[0, 1], [gamma^2, -10]]
        B = [[0], [1]]
        B_d = [[0], [0.5]]
    """

    def __init__(self, gamma_growth: float = 100.0) -> None:
        self.gamma_growth = float(gamma_growth)
        self.A = np.array([
            [0.0, 1.0],
            [gamma_growth ** 2, -10.0],
        ])
        self.B = np.array([0.0, 1.0])
        self.B_d = np.array([0.0, 0.5])
        self.x = np.zeros(2)

    def step(self, u: float, d: float, dt: float) -> float:
        """Advance one timestep. Returns measured position z."""
        dx = self.A @ self.x + self.B * u + self.B_d * d
        self.x = self.x + dx * dt
        return float(self.x[0])

    @property
    def z(self) -> float:
        return float(self.x[0])

    @property
    def dz(self) -> float:
        return float(self.x[1])

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        if x0 is not None:
            self.x = np.array(x0, dtype=float)
        else:
            self.x = np.zeros(2)


# ===================================================================
# Disturbance scenario definitions
# ===================================================================

def _disturbance_vde(t: float) -> float:
    """VDE: exponentially growing vertical displacement.

    At t=0 a 1 cm displacement already exists (via x0). The
    disturbance models an external vertical kick that excites the
    unstable mode: a strong impulse in the first 1 ms, then zero
    (the instability itself provides the exponential growth via the
    plant dynamics with gamma=100/s).
    """
    if t < 1.0e-3:
        return 5000.0  # impulse kick
    return 0.0


def _disturbance_density_ramp(t: float) -> float:
    """Density ramp: linear ramp from 0.5 to 1.2 Greenwald fraction over 2 s.

    The density change creates a position-disturbing force proportional
    to the departure from nominal density (1.0).
    """
    t_ramp = 2.0
    n_start = 0.5
    n_end = 1.2
    n_frac = n_end if t >= t_ramp else n_start + (n_end - n_start) * (t / t_ramp)
    return 200.0 * (n_frac - 1.0)


def _disturbance_elm_pacing(t: float) -> float:
    """ELM pacing: 10 Hz periodic energy bursts.

    Each ELM causes a 5 % beta_N drop, modelled as a short (0.5 ms)
    disturbance pulse every 100 ms.
    """
    period = 0.1       # 10 Hz
    pulse_width = 0.5e-3
    phase = t % period
    if phase < pulse_width:
        return 1000.0  # impulse during ELM burst
    return 0.0


# Scenario registry
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "VDE": {
        "disturbance": _disturbance_vde,
        "duration_s": SCENARIO_DURATIONS["VDE"],
        "dt_s": DT,
        "target_z": 0.0,
        "x0": np.array([0.01, 0.0]),  # 1 cm initial displacement
        "settling_threshold": 0.05,
        "description": (
            "Vertical Displacement Event: gamma=100/s exponential "
            "instability, impulsive kick. Duration: 2 s."
        ),
    },
    "Density ramp": {
        "disturbance": _disturbance_density_ramp,
        "duration_s": SCENARIO_DURATIONS["Density ramp"],
        "dt_s": DT,
        "target_z": 0.0,
        "x0": np.array([0.0, 0.0]),
        "settling_threshold": 0.05,
        "description": (
            "Linear density ramp 0.5 to 1.2 Greenwald fraction "
            "over 2 s, then 2 s settling. Duration: 4 s."
        ),
    },
    "ELM pacing": {
        "disturbance": _disturbance_elm_pacing,
        "duration_s": SCENARIO_DURATIONS["ELM pacing"],
        "dt_s": DT,
        "target_z": 0.0,
        "x0": np.array([0.0, 0.0]),
        "settling_threshold": 0.05,
        "description": (
            "ELM pacing at 10 Hz, 5 % beta_N drop per burst, "
            "recovery tracking. Duration: 3 s."
        ),
    },
}


# ===================================================================
# Metrics dataclass
# ===================================================================

@dataclass
class ScenarioMetrics:
    """Aggregated metrics from one (controller, scenario) pair."""
    controller: str
    scenario: str
    ise: float                 # Integral of Squared Error
    settling_time_s: float     # Time to within 5 % of setpoint permanently
    peak_overshoot: float      # Max |deviation| from setpoint
    control_effort: float      # Integral of |u|
    wall_clock_s: float        # Wall-clock time for this run
    stable: bool               # True if plant stayed bounded

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# Simulation engine
# ===================================================================

def _compute_settling_time(
    times: np.ndarray,
    errors: np.ndarray,
    threshold_frac: float,
    reference_amplitude: float,
) -> float:
    """Compute time at which |error| stays below threshold permanently.

    Returns total duration if never settles.
    """
    if reference_amplitude == 0.0:
        reference_amplitude = 1.0
    band = threshold_frac * abs(reference_amplitude)
    if band == 0.0:
        band = 1.0e-6

    abs_errors = np.abs(errors)
    n = len(abs_errors)
    last_exceed = -1
    for i in range(n - 1, -1, -1):
        if abs_errors[i] > band:
            last_exceed = i
            break
    if last_exceed < 0:
        return 0.0
    if last_exceed >= n - 1:
        return float(times[-1])
    return float(times[last_exceed + 1])


@dataclass
class TraceData:
    """Time-series data from a simulation run, for plotting."""
    times: np.ndarray
    positions: np.ndarray
    errors: np.ndarray
    controls: np.ndarray


def run_scenario(
    controller_name: str,
    controller: Any,
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
) -> Tuple[ScenarioMetrics, TraceData]:
    """Run a single (controller, scenario) pair and return metrics + traces."""
    dist_fn = scenario_cfg["disturbance"]
    duration = float(scenario_cfg["duration_s"])
    dt = float(scenario_cfg["dt_s"])
    target = float(scenario_cfg["target_z"])
    x0 = np.array(scenario_cfg["x0"], dtype=float)
    settle_thresh = float(scenario_cfg["settling_threshold"])

    n_steps = int(round(duration / dt))

    plant = LinearPlant(gamma_growth=100.0)
    plant.reset(x0.copy())
    controller.reset()

    times = np.zeros(n_steps)
    errors = np.zeros(n_steps)
    controls = np.zeros(n_steps)
    positions = np.zeros(n_steps)

    stable = True

    t_wall_start = time.perf_counter()

    for k in range(n_steps):
        t = k * dt
        times[k] = t

        z_meas = plant.z
        error = target - z_meas
        errors[k] = error
        positions[k] = z_meas

        u = controller.step(error, dt)
        controls[k] = u

        d = dist_fn(t)
        plant.step(u, d, dt)

        if abs(plant.z) > 10.0:
            stable = False
            errors[k + 1:] = error
            controls[k + 1:] = u
            positions[k + 1:] = plant.z
            times[k + 1:] = np.arange(k + 1, n_steps) * dt
            break

    wall_clock = time.perf_counter() - t_wall_start

    ise = float(np.trapz(errors ** 2, times))

    ref_amp = max(
        abs(x0[0]),
        float(np.max(np.abs(positions[:min(100, n_steps)]))),
        1.0e-3,
    )
    settling_time = _compute_settling_time(times, errors, settle_thresh, ref_amp)

    peak_overshoot = float(np.max(np.abs(positions - target)))

    control_effort = float(np.trapz(np.abs(controls), times))

    metrics = ScenarioMetrics(
        controller=controller_name,
        scenario=scenario_name,
        ise=ise,
        settling_time_s=settling_time,
        peak_overshoot=peak_overshoot,
        control_effort=control_effort,
        wall_clock_s=wall_clock,
        stable=stable,
    )
    traces = TraceData(
        times=times,
        positions=positions,
        errors=errors,
        controls=controls,
    )
    return metrics, traces


# ===================================================================
# Controller factory
# ===================================================================

def _build_hinf_controller(gamma_growth: float = 100.0) -> Any:
    """Build the H-infinity controller with LQR fallback."""
    if not _hinf_available:
        raise RuntimeError("H-infinity module not importable")
    try:
        ctrl = get_radial_robust_controller(gamma_growth=gamma_growth)
        print("    [H-infinity] Riccati synthesis: OK")
        return ctrl
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(
            f"    [H-infinity] ARE failed ({exc}); using LQR fallback"
        )
        ctrl = LQRRobustController(gamma_growth=gamma_growth)
        assert ctrl.is_stable, "LQR closed-loop must be stable"
        return ctrl


def build_controllers() -> Dict[str, Any]:
    """Instantiate all available benchmark controllers.

    Controllers that cannot be imported are skipped with a warning.
    """
    controllers: Dict[str, Any] = {}

    # PID -- always available (pure Python/NumPy)
    controllers["PID"] = PIDController(
        kp=1.5e4, ki=3.0e3, kd=1.5e2,
    )
    print("    [PID] Initialised: Kp=1.5e4, Ki=3.0e3, Kd=1.5e2")

    # H-infinity
    if _hinf_available:
        try:
            controllers["H-infinity"] = _build_hinf_controller(
                gamma_growth=100.0,
            )
        except Exception as exc:
            warnings.warn(f"H-infinity controller skipped: {exc}")
    else:
        print("    [H-infinity] SKIPPED (import failed)")

    # MPC -- always available (pure Python/NumPy)
    controllers["MPC"] = MPCController(
        gamma_growth=100.0,
        horizon=10,
        q_weight=1.0e4,
        r_weight=1.0e-2,
        iterations=15,
        learning_rate=0.1,
    )
    print("    [MPC] Initialised: horizon=10, Q=1e4, R=1e-2")

    # SNN
    if _snn_available:
        try:
            controllers["SNN"] = SNNControllerWrapper(
                n_neurons=50,
                gain=5.0e3,
                tau_window=20,
                seed=42,
            )
            print(
                f"    [SNN] Initialised: 50 neurons, "
                f"backend={controllers['SNN']._pool.backend}"
            )
        except Exception as exc:
            warnings.warn(f"SNN controller skipped: {exc}")
    else:
        print("    [SNN] SKIPPED (import failed)")

    return controllers


# ===================================================================
# Output formatting
# ===================================================================

def _fmt_sci(v: float, width: int = 12) -> str:
    """Format a number in scientific notation for table display."""
    if abs(v) < 1.0e-3 or abs(v) >= 1.0e4:
        return f"{v:>{width}.3e}"
    return f"{v:>{width}.5f}"


def generate_stdout_table(all_metrics: List[ScenarioMetrics]) -> str:
    """Generate markdown comparison table for stdout."""
    lines: List[str] = []
    lines.append("")
    lines.append(
        "| Controller   | Scenario      | ISE          "
        "| Settle (s)  | Peak Ovshoot | Ctrl Effort  "
        "| Wall (s)    | Stable |"
    )
    lines.append(
        "|--------------|---------------|-------------:"
        "|------------:|-------------:|-------------:"
        "|------------:|--------|"
    )

    for m in all_metrics:
        lines.append(
            f"| {m.controller:<12} "
            f"| {m.scenario:<13} "
            f"| {m.ise:>12.3e} "
            f"| {m.settling_time_s:>11.4f} "
            f"| {m.peak_overshoot:>12.3e} "
            f"| {m.control_effort:>12.3e} "
            f"| {m.wall_clock_s:>11.4f} "
            f"| {'Yes' if m.stable else 'No':>6} |"
        )

    return "\n".join(lines)


def generate_markdown_report(all_metrics: List[ScenarioMetrics]) -> str:
    """Generate full markdown report for file output."""
    lines: List[str] = [
        "# Disturbance Rejection Benchmark",
        "",
        "ITER-like parameters: Ip=15 MA, BT=5.3 T, R=6.2 m, a=2.0 m, "
        "gamma_growth=100/s",
        "",
        "## Results",
        "",
    ]

    # Per-scenario tables
    scenarios_seen: List[str] = []
    for m in all_metrics:
        if m.scenario not in scenarios_seen:
            scenarios_seen.append(m.scenario)

    for scenario in scenarios_seen:
        sc_metrics = [m for m in all_metrics if m.scenario == scenario]
        cfg = SCENARIOS[scenario]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append(f"_{cfg['description']}_")
        lines.append(
            f"Duration: {cfg['duration_s']:.1f} s | "
            f"dt: {cfg['dt_s']:.0e} s | "
            f"Steps: {int(cfg['duration_s'] / cfg['dt_s']):,}"
        )
        lines.append("")
        lines.append(
            "| Controller | ISE | Settling Time (s) | Peak Overshoot "
            "| Control Effort | Wall-Clock (s) | Stable |"
        )
        lines.append(
            "|------------|----:|-------------------:|---------------:"
            "|---------------:|---------------:|--------|"
        )
        for m in sc_metrics:
            lines.append(
                f"| {m.controller} "
                f"| {m.ise:.3e} "
                f"| {m.settling_time_s:.4f} "
                f"| {m.peak_overshoot:.3e} "
                f"| {m.control_effort:.3e} "
                f"| {m.wall_clock_s:.4f} "
                f"| {'Yes' if m.stable else 'No'} |"
            )
        lines.append("")

    # Metrics definitions
    lines.append("## Metrics Definitions")
    lines.append("")
    lines.append("- **ISE**: Integral of Squared Error (lower is better)")
    lines.append(
        "- **Settling Time**: Time to reach and stay within 5 % band "
        "(lower is better)"
    )
    lines.append(
        "- **Peak Overshoot**: Maximum absolute deviation from setpoint "
        "(lower is better)"
    )
    lines.append(
        "- **Control Effort**: Integral of |u| over time "
        "(lower = more efficient)"
    )
    lines.append(
        "- **Wall-Clock**: Real execution time in seconds "
        "(lower = faster)"
    )
    lines.append(
        "- **Stable**: Whether the plant state remained bounded "
        "during the scenario"
    )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    for scenario in scenarios_seen:
        sc_metrics = [m for m in all_metrics if m.scenario == scenario]
        stable_m = [m for m in sc_metrics if m.stable]
        if stable_m:
            best = min(stable_m, key=lambda m: m.ise)
            lines.append(
                f"- **{scenario}**: Best ISE = {best.controller} "
                f"({best.ise:.3e})"
            )
        else:
            lines.append(f"- **{scenario}**: ALL CONTROLLERS UNSTABLE")
    lines.append("")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"*Generated: {ts}*")
    return "\n".join(lines)


def generate_json_results(all_metrics: List[ScenarioMetrics]) -> Dict[str, Any]:
    """Build a JSON-serialisable results dictionary."""
    return {
        "benchmark": "disturbance_rejection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation_dt_s": DT,
        "scenarios": {
            name: {
                "duration_s": cfg["duration_s"],
                "dt_s": cfg["dt_s"],
                "steps": int(cfg["duration_s"] / cfg["dt_s"]),
                "description": cfg["description"],
            }
            for name, cfg in SCENARIOS.items()
        },
        "results": [m.to_dict() for m in all_metrics],
    }


# ===================================================================
# Plotting
# ===================================================================

def save_overlay_plots(
    all_traces: Dict[Tuple[str, str], TraceData],
    all_metrics: List[ScenarioMetrics],
    output_dir: Path,
) -> List[str]:
    """Save per-scenario time-series overlay plots.

    Returns list of saved file paths.
    """
    if not _mpl_available:
        print("  matplotlib not available -- skipping plots")
        return []

    saved: List[str] = []

    scenarios_seen: List[str] = []
    controllers_seen: List[str] = []
    for m in all_metrics:
        if m.scenario not in scenarios_seen:
            scenarios_seen.append(m.scenario)
        if m.controller not in controllers_seen:
            controllers_seen.append(m.controller)

    colours = {
        "PID": "#2196F3",
        "H-infinity": "#FF5722",
        "MPC": "#4CAF50",
        "SNN": "#9C27B0",
    }

    for scenario in scenarios_seen:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"Disturbance Rejection Benchmark: {scenario}",
            fontsize=14,
            fontweight="bold",
        )

        ax_pos = axes[0]
        ax_err = axes[1]
        ax_ctrl = axes[2]

        ax_pos.set_ylabel("Position z (m)")
        ax_pos.set_title("Plasma Vertical Position")
        ax_pos.axhline(0.0, color="k", linestyle="--", alpha=0.3, label="Setpoint")
        ax_pos.grid(True, alpha=0.3)

        ax_err.set_ylabel("Error (m)")
        ax_err.set_title("Tracking Error")
        ax_err.axhline(0.0, color="k", linestyle="--", alpha=0.3)
        ax_err.grid(True, alpha=0.3)

        ax_ctrl.set_ylabel("Control u")
        ax_ctrl.set_title("Control Effort")
        ax_ctrl.set_xlabel("Time (s)")
        ax_ctrl.axhline(0.0, color="k", linestyle="--", alpha=0.3)
        ax_ctrl.grid(True, alpha=0.3)

        for ctrl_name in controllers_seen:
            key = (ctrl_name, scenario)
            if key not in all_traces:
                continue
            trace = all_traces[key]
            colour = colours.get(ctrl_name, "#607D8B")

            # Downsample for plotting if too many points
            n_pts = len(trace.times)
            stride = max(1, n_pts // 5000)

            t_plot = trace.times[::stride]
            ax_pos.plot(
                t_plot,
                trace.positions[::stride],
                color=colour,
                label=ctrl_name,
                linewidth=0.8,
                alpha=0.9,
            )
            ax_err.plot(
                t_plot,
                trace.errors[::stride],
                color=colour,
                label=ctrl_name,
                linewidth=0.8,
                alpha=0.9,
            )
            ax_ctrl.plot(
                t_plot,
                trace.controls[::stride],
                color=colour,
                label=ctrl_name,
                linewidth=0.8,
                alpha=0.9,
            )

        ax_pos.legend(loc="upper right", fontsize=8)
        ax_err.legend(loc="upper right", fontsize=8)
        ax_ctrl.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        safe_name = scenario.lower().replace(" ", "_")
        plot_path = output_dir / f"benchmark_{safe_name}.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        saved.append(str(plot_path))
        print(f"  Plot saved: {plot_path}")

    return saved


# ===================================================================
# Main
# ===================================================================

def main(output_dir: Optional[str] = None) -> None:
    """Run all (controller x scenario) pairs and output results."""
    if output_dir is None:
        out_path = REPO_ROOT / "artifacts"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  DISTURBANCE REJECTION BENCHMARK")
    print("  SNN vs MPC vs H-infinity vs PID")
    print(f"  Simulation dt = {DT:.0e} s")
    print("=" * 72)
    print()

    # Build controllers
    print("Initialising controllers...")
    controllers = build_controllers()
    print(f"  Active controllers: {', '.join(controllers.keys())}")
    print()

    all_metrics: List[ScenarioMetrics] = []
    all_traces: Dict[Tuple[str, str], TraceData] = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        n_steps = int(scenario_cfg["duration_s"] / scenario_cfg["dt_s"])
        print(f"--- Scenario: {scenario_name} ---")
        print(f"    {scenario_cfg['description']}")
        print(
            f"    Duration: {scenario_cfg['duration_s']:.1f} s, "
            f"dt: {scenario_cfg['dt_s']:.0e} s, "
            f"steps: {n_steps:,}"
        )
        print()

        for ctrl_name, ctrl in controllers.items():
            metrics, traces = run_scenario(
                ctrl_name, ctrl, scenario_name, scenario_cfg,
            )
            all_metrics.append(metrics)
            all_traces[(ctrl_name, scenario_name)] = traces

            status = "STABLE" if metrics.stable else "UNSTABLE"
            print(
                f"    {ctrl_name:<12}: ISE={metrics.ise:.3e}  "
                f"settle={metrics.settling_time_s:.4f}s  "
                f"peak={metrics.peak_overshoot:.3e}  "
                f"effort={metrics.control_effort:.3e}  "
                f"wall={metrics.wall_clock_s:.3f}s  [{status}]"
            )
        print()

    # Print comparison table to stdout
    table_str = generate_stdout_table(all_metrics)
    print(table_str)
    print()

    # JSON
    json_path = out_path / "benchmark_disturbance_rejection.json"
    json_data = generate_json_results(all_metrics)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON results written to: {json_path}")

    # Markdown
    md_path = out_path / "benchmark_disturbance_rejection.md"
    md_content = generate_markdown_report(all_metrics)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown table written to: {md_path}")

    # Plots
    print()
    print("Generating overlay plots...")
    saved_plots = save_overlay_plots(all_traces, all_metrics, out_path)
    if saved_plots:
        print(f"  {len(saved_plots)} plot(s) saved to {out_path}")
    print()

    print("=" * 72)
    print("  VERDICT SUMMARY")
    print("=" * 72)

    for scenario_name in SCENARIOS:
        scenario_results = [
            m for m in all_metrics if m.scenario == scenario_name
        ]
        stable_results = [m for m in scenario_results if m.stable]
        if stable_results:
            best = min(stable_results, key=lambda m: m.ise)
            print(
                f"  {scenario_name:<16}: Best ISE = {best.controller} "
                f"({best.ise:.3e})"
            )
        else:
            print(f"  {scenario_name:<16}: ALL CONTROLLERS UNSTABLE")

    # Overall winner by ISE wins
    wins: Dict[str, int] = {}
    for scenario_name in SCENARIOS:
        stable_results = [
            m
            for m in all_metrics
            if m.scenario == scenario_name and m.stable
        ]
        if stable_results:
            best = min(stable_results, key=lambda m: m.ise)
            wins[best.controller] = wins.get(best.controller, 0) + 1

    if wins:
        overall_winner = max(wins, key=lambda k: wins[k])
        print(
            f"\n  Overall winner (by ISE wins): {overall_winner} "
            f"({wins[overall_winner]}/{len(SCENARIOS)} scenarios)"
        )
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SNN vs MPC vs H-infinity vs PID on three "
            "tokamak disturbance-rejection scenarios."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for output artifacts (JSON, markdown, plots). "
            "Defaults to <repo>/artifacts/"
        ),
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
