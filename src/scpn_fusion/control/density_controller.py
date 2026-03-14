# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Real-time Density Profile Control
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ParticleTransportModel:
    def __init__(self, n_rho: int = 50, R0: float = 6.2, a: float = 2.0):
        self.n_rho = n_rho
        self.R0 = R0
        self.a = a
        self.rho = np.linspace(0.0, 1.0, n_rho)
        self.drho = self.rho[1] - self.rho[0]

        self.D = np.ones(n_rho) * 1.0  # m^2/s
        self.V_pinch = -np.ones(n_rho) * 0.1  # m/s (inward)

        # Geometry factors
        self.V = 2.0 * np.pi**2 * self.R0 * (self.a * self.rho) ** 2
        self.V_prime = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho

    def set_transport(self, D: np.ndarray, V_pinch: np.ndarray) -> None:
        self.D = D
        self.V_pinch = V_pinch

    def gas_puff_source(self, rate: float, penetration_depth: float = 0.03) -> np.ndarray:
        """rate in particles/s. Source localized at edge."""
        # Simple exponential decay from edge
        decay = np.exp(-(1.0 - self.rho) / penetration_depth)
        decay /= np.sum(decay * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * decay)

    def pellet_source(
        self, speed_ms: float, radius_mm: float, launch_angle_deg: float = 0.0
    ) -> np.ndarray:
        """Simple NGS mock. Penetrates deeper with higher speed and radius."""
        if radius_mm <= 0.0:
            return np.zeros(self.n_rho)

        N_pellet = (4.0 / 3.0 * np.pi * (radius_mm * 1e-3) ** 3) * 6e28  # solid D2 density
        pen_rho = max(0.2, 1.0 - 0.1 * (speed_ms / 1000.0) * (radius_mm / 2.0))

        # Gaussian deposition
        dep = np.exp(-((self.rho - pen_rho) ** 2) / (0.1**2))
        dep /= np.sum(dep * self.V_prime * self.drho) + 1e-10
        return np.asarray(N_pellet * dep)

    def nbi_source(self, beam_energy_keV: float, power_MW: float) -> np.ndarray:
        """Broad core source."""
        if power_MW <= 0.0:
            return np.zeros(self.n_rho)

        I_beam = power_MW * 1e6 / (beam_energy_keV * 1e3)
        rate = I_beam / 1.6e-19

        dep = np.exp(-((self.rho - 0.3) ** 2) / (0.3**2))
        dep /= np.sum(dep * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * dep)

    def cryopump_sink(self, pump_speed: float, ne_edge: float) -> np.ndarray:
        """Extracts particles from the edge."""
        # This is a rate, not a profile shape itself, but we'll return the sink profile
        sink = np.zeros(self.n_rho)
        sink[-1] = pump_speed * ne_edge / (self.V_prime[-1] * self.drho + 1e-10)
        return sink

    def recycling_source(self, outflux: float, recycling_coeff: float = 0.97) -> np.ndarray:
        return self.gas_puff_source(outflux * recycling_coeff, penetration_depth=0.02)

    def step(self, ne: np.ndarray, sources: np.ndarray, dt: float) -> np.ndarray:
        # Explicit forward-Euler diffusion: CFL requires dt < drho^2 / (2 * D_max)
        D_max = np.max(self.D)
        if D_max > 0.0:
            dt_cfl = (self.drho * self.a) ** 2 / (2.0 * D_max)
            if dt > dt_cfl:
                dt = dt_cfl

        flux = np.zeros(self.n_rho + 1)
        for i in range(1, self.n_rho):
            grad_n = (ne[i] - ne[i - 1]) / self.drho
            n_face = 0.5 * (ne[i] + ne[i - 1])
            D_face = 0.5 * (self.D[i] + self.D[i - 1])
            V_face = 0.5 * (self.V_pinch[i] + self.V_pinch[i - 1])

            # 1/a to convert d/drho to d/dr
            flux[i] = -D_face * grad_n / self.a + V_face * n_face

        flux[0] = 0.0

        # Outflux at edge
        flux[-1] = -self.D[-1] * (0.0 - ne[-1]) / self.drho / self.a + self.V_pinch[-1] * ne[-1]

        dne_dt = np.zeros(self.n_rho)
        for i in range(self.n_rho):
            Vp = max(self.V_prime[i], 1e-6)
            Vp_plus = (
                self.V_prime[i]
                if i == self.n_rho - 1
                else 0.5 * (self.V_prime[i] + self.V_prime[i + 1])
            )
            Vp_minus = 0.0 if i == 0 else 0.5 * (self.V_prime[i] + self.V_prime[i - 1])

            div_flux = (Vp_plus * flux[i + 1] - Vp_minus * flux[i]) / (Vp * self.drho * self.a)
            dne_dt[i] = -div_flux + sources[i]

        ne_new = ne + dne_dt * dt
        return np.asarray(np.maximum(ne_new, 1e16))


@dataclass
class ActuatorCommand:
    gas_puff_rate: float
    pellet_freq: float
    pellet_speed: float
    cryo_pump_speed: float


class DensityController:
    def __init__(self, model: ParticleTransportModel, dt_control: float = 0.001):
        self.model = model
        self.dt = dt_control
        self.ne_target = np.zeros(model.n_rho)

        self.n_GW = 1.0e20
        self.gas_max = 1e22
        self.pellet_freq_max = 10.0
        self.pump_max = 10.0

        self.integral_error = 0.0

    def set_target(self, ne_target: np.ndarray) -> None:
        self.ne_target = ne_target

    def set_constraints(
        self, n_GW: float, gas_max: float, pellet_freq_max: float, pump_max: float
    ) -> None:
        self.n_GW = n_GW
        self.gas_max = gas_max
        self.pellet_freq_max = pellet_freq_max
        self.pump_max = pump_max

    def greenwald_fraction(self, ne: np.ndarray, I_p_MA: float, a: float) -> float:
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_tot = np.sum(ne * self.model.V_prime * self.model.drho)
        n_avg = N_tot / vol

        n_GW = I_p_MA / (math.pi * a**2) * 1e20
        return float(n_avg / n_GW)

    def step(self, ne_measured: np.ndarray) -> ActuatorCommand:
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_meas = np.sum(ne_measured * self.model.V_prime * self.model.drho)
        N_targ = np.sum(self.ne_target * self.model.V_prime * self.model.drho)

        error = N_targ - N_meas
        self.integral_error += error * self.dt

        # simple PI
        Kp = 10.0
        Ki = 1.0
        cmd = Kp * error + Ki * self.integral_error

        f_gw = (N_meas / vol) / self.n_GW

        gas = 0.0
        pellet = 0.0
        pump = 0.0

        if f_gw > 0.95:
            # Over Greenwald -> max pump, no source
            pump = self.pump_max
        elif cmd > 0:
            # Need fueling
            gas = min(cmd, self.gas_max)
            # If error is very large, use pellets
            if cmd > self.gas_max * 0.5:
                pellet = min(self.pellet_freq_max, (cmd - self.gas_max * 0.5) / 1e21)
        else:
            # Need pumping
            pump = min(self.pump_max, -cmd / 1e20)

        return ActuatorCommand(gas, pellet, 500.0, pump)


class KalmanDensityEstimator:
    def __init__(self, n_rho: int, n_chords: int = 8):
        self.n_rho = n_rho
        self.n_chords = n_chords
        self.x = np.zeros(n_rho)
        self.P = np.eye(n_rho) * 1e38

        # Q and R
        self.Q = np.eye(n_rho) * 1e36  # Process noise
        self.R = np.eye(n_chords) * 1e34  # Meas noise

    def measurement_matrix(self, chord_angles: np.ndarray) -> np.ndarray:
        # Mock Abel transform matrix
        C = np.zeros((self.n_chords, self.n_rho))
        for i in range(self.n_chords):
            impact = i / self.n_chords
            for j in range(self.n_rho):
                rho = j / self.n_rho
                if rho > impact:
                    C[i, j] = 2.0 * rho / math.sqrt(rho**2 - impact**2 + 1e-6)
        return C

    def predict(self, ne: np.ndarray, dt: float) -> np.ndarray:
        # Simplified prediction: static + noise
        self.x = ne
        self.P = self.P + self.Q * dt
        return self.x

    def update(
        self, ne_pred: np.ndarray, measurements: np.ndarray, chord_angles: np.ndarray
    ) -> np.ndarray:
        C = self.measurement_matrix(chord_angles)

        # K = P C^T (C P C^T + R)^-1
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)

        y_pred = C @ ne_pred
        inn = measurements - y_pred

        self.x = ne_pred + K @ inn
        self.P = (np.eye(self.n_rho) - K @ C) @ self.P

        return self.x


@dataclass
class PelletSchedule:
    times: list[float]
    speeds: list[float]
    sizes: list[float]


class FuelingOptimizer:
    def optimize_pellet_sequence(
        self, ne_current: np.ndarray, ne_target: np.ndarray, n_pellets: int, time_horizon: float
    ) -> PelletSchedule:
        # Simplistic shooting: spread pellets evenly
        if n_pellets <= 0:
            return PelletSchedule([], [], [])

        dt = time_horizon / (n_pellets + 1)
        times = [dt * (i + 1) for i in range(n_pellets)]
        speeds = [500.0] * n_pellets
        sizes = [2.0] * n_pellets
        return PelletSchedule(times, speeds, sizes)
