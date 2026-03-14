# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Alpha Particle Guiding-Center Orbit Following
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EnsembleResult:
    loss_fraction: float
    heating_profile: np.ndarray
    current_drive: float
    n_passing: int
    n_trapped: int
    n_lost: int


class GuidingCenterOrbit:
    """
    (R, Z, phi, v_par, mu) integration using RK4.
    """

    def __init__(
        self, m_amu: float, Z: int, E_keV: float, pitch_angle: float, R0_init: float, Z0_init: float
    ):
        self.m = m_amu * 1.67e-27
        self.Z_charge = Z * 1.602e-19
        self.E_J = E_keV * 1e3 * 1.602e-19

        self.v_tot = math.sqrt(2.0 * self.E_J / self.m)
        self.v_par = self.v_tot * math.cos(pitch_angle)
        v_perp = self.v_tot * math.sin(pitch_angle)

        self.R = R0_init
        self.Z = Z0_init
        self.phi = 0.0

        # We need initial B to compute mu
        # Since B is passed in step, we will defer mu calculation
        self.mu = -1.0
        self.v_perp_0 = v_perp

    def _eom(self, state: np.ndarray, B_field: Callable) -> np.ndarray:
        R, Z, phi, v_par = state

        # Evaluate B field
        B_R, B_Z, B_phi = B_field(R, Z)
        B_mag = math.sqrt(B_R**2 + B_Z**2 + B_phi**2)

        if self.mu < 0:
            self.mu = self.m * self.v_perp_0**2 / (2.0 * B_mag)

        omega_c = self.Z_charge * B_mag / self.m

        # Need grad B for drift. Simple finite difference
        eps = 1e-4
        B_R_plus, B_Z_plus, B_phi_plus = B_field(R + eps, Z)
        B_mag_R_plus = math.sqrt(B_R_plus**2 + B_Z_plus**2 + B_phi_plus**2)
        dB_dR = (B_mag_R_plus - B_mag) / eps

        B_R_zplus, B_Z_zplus, B_phi_zplus = B_field(R, Z + eps)
        B_mag_Z_plus = math.sqrt(B_R_zplus**2 + B_Z_zplus**2 + B_phi_zplus**2)
        dB_dZ = (B_mag_Z_plus - B_mag) / eps

        # B cross grad B drift components
        # B x grad B = (B_y dB_z - B_z dB_y, B_z dB_x - B_x dB_z, B_x dB_y - B_y dB_x)
        # B = (B_R, B_phi, B_Z). grad B = (dB_dR, 0, dB_dZ)
        # B x grad B = (B_phi * dB_dZ - B_Z * 0, B_Z * dB_dR - B_R * dB_dZ, B_R * 0 - B_phi * dB_dR)
        bxg_R = B_phi * dB_dZ
        bxg_phi = B_Z * dB_dR - B_R * dB_dZ
        bxg_Z = -B_phi * dB_dR

        drift_coeff = (v_par**2 + self.mu * B_mag) / (omega_c * B_mag**2)

        dR_dt = v_par * B_R / B_mag + drift_coeff * bxg_R
        dZ_dt = v_par * B_Z / B_mag + drift_coeff * bxg_Z
        dphi_dt = v_par * B_phi / (R * B_mag) + drift_coeff * bxg_phi / R

        # Mirror force: dv_par / dt = -mu/m * (B . grad B) / B
        # B . grad B = B_R * dB_dR + B_Z * dB_dZ
        b_dot_grad_b = B_R * dB_dR + B_Z * dB_dZ
        dv_par_dt = -(self.mu / self.m) * b_dot_grad_b / B_mag

        return np.array([dR_dt, dZ_dt, dphi_dt, dv_par_dt])

    def step(self, B_field: Callable, dt: float) -> tuple[float, float, float, float]:
        state = np.array([self.R, self.Z, self.phi, self.v_par])

        # RK4
        k1 = self._eom(state, B_field)
        k2 = self._eom(state + 0.5 * dt * k1, B_field)
        k3 = self._eom(state + 0.5 * dt * k2, B_field)
        k4 = self._eom(state + dt * k3, B_field)

        state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.R, self.Z, self.phi, self.v_par = state_new
        return float(self.R), float(self.Z), float(self.phi), float(self.v_par)


class OrbitClassifier:
    @staticmethod
    def classify(
        orbit_R: np.ndarray,
        orbit_Z: np.ndarray,
        v_par: np.ndarray,
        R_wall: float,
        Z_wall_upper: float,
    ) -> str:
        # Check lost
        if (
            np.any(orbit_R > R_wall)
            or np.any(np.abs(orbit_Z) > Z_wall_upper)
            or np.any(orbit_R < 0.1)
        ):
            return "lost"

        # Check trapped (v_par changes sign)
        v_par_signs = np.sign(v_par)
        if np.any(v_par_signs != v_par_signs[0]):
            return "trapped"

        return "passing"


class MonteCarloEnsemble:
    def __init__(self, n_particles: int, E_birth_keV: float, R0: float, a: float, B0: float):
        self.n_particles = n_particles
        self.E_birth_keV = E_birth_keV
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.particles: list[GuidingCenterOrbit] = []

    def initialize(self, ne_profile: np.ndarray, Te_profile: np.ndarray, rho: np.ndarray) -> None:
        self.particles = []
        for _ in range(self.n_particles):
            # Peaked centrally
            r = np.random.beta(2, 5) * self.a
            theta = np.random.uniform(0, 2 * np.pi)
            pitch = np.random.uniform(0, np.pi)

            R_init = self.R0 + r * math.cos(theta)
            Z_init = r * math.sin(theta)

            p = GuidingCenterOrbit(4.0, 2, self.E_birth_keV, pitch, R_init, Z_init)
            self.particles.append(p)

    def follow(self, B_field: Callable, n_bounces: int = 10, dt: float = 1e-7) -> EnsembleResult:
        n_pass = 0
        n_trap = 0
        n_lost = 0

        heating = np.zeros(50)

        for p in self.particles:
            R_trace = []
            Z_trace = []
            v_trace = []

            # Follow for fixed time
            for _ in range(100):
                p.step(B_field, dt)
                R_trace.append(p.R)
                Z_trace.append(p.Z)
                v_trace.append(p.v_par)

            c = OrbitClassifier.classify(
                np.array(R_trace),
                np.array(Z_trace),
                np.array(v_trace),
                self.R0 + self.a + 0.5,
                self.a + 0.5,
            )
            if c == "lost":
                n_lost += 1
            elif c == "trapped":
                n_trap += 1
            else:
                n_pass += 1

        frac = n_lost / max(self.n_particles, 1)

        return EnsembleResult(frac, heating, 0.0, n_pass, n_trap, n_lost)


def first_orbit_loss(
    R0: float, a: float, B0: float, Ip_MA: float, E_alpha_keV: float = 3520.0
) -> float:
    """
    Fraction of alphas lost on their first orbit.
    Scales with rho_alpha / a and 1 / Ip.
    """
    v_alpha = math.sqrt(2.0 * E_alpha_keV * 1e3 * 1.6e-19 / (4.0 * 1.67e-27))
    rho_alpha = 4.0 * 1.67e-27 * v_alpha / (2.0 * 1.6e-19 * B0)

    # Heuristic scaling
    loss = (rho_alpha / a) * (2.0 / max(Ip_MA, 0.1))
    return float(min(1.0, max(0.0, loss)))


class SlowingDown:
    @staticmethod
    def critical_velocity(Te_keV: float, ne_20: float) -> float:
        # v_c ~ Te^0.5
        Te_J = Te_keV * 1e3 * 1.602e-19
        m_e = 9.109e-31
        v_te = math.sqrt(2.0 * Te_J / m_e)
        return 0.1 * v_te  # Heuristic

    @staticmethod
    def tau_sd(Te_keV: float, ne_20: float, Z_eff: float) -> float:
        # tau_sd ~ Te^1.5 / ne
        tau = 0.1 * (Te_keV**1.5) / max(ne_20, 0.01)
        return float(tau)

    @staticmethod
    def heating_partition(v: float, v_c: float) -> tuple[float, float]:
        """(f_ion, f_electron)"""
        # High v -> heats electrons. Low v -> heats ions.
        f_ion = (v_c**3) / (v**3 + v_c**3)
        return float(f_ion), float(1.0 - f_ion)
