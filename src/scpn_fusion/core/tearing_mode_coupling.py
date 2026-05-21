# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled Tearing Mode Dynamics
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_fusion.core.sawtooth import SawtoothCycler


@dataclass
class CoupledResult:
    w1_trace: np.ndarray
    w2_trace: np.ndarray
    chirikov_trace: np.ndarray
    overlap_time: float
    disruption: bool


class ChirikovOverlap:
    @staticmethod
    def parameter(w1: float, w2: float, delta_r: float) -> float:
        if delta_r <= 0.0:
            return float("inf")
        return (w1 + w2) / (2.0 * delta_r)

    @staticmethod
    def is_stochastic(sigma: float) -> bool:
        return sigma > 1.0

    @staticmethod
    def stochastic_region_width(w1: float, w2: float, delta_r: float) -> float:
        sigma = ChirikovOverlap.parameter(w1, w2, delta_r)
        if sigma > 1.0:
            return delta_r + w1 / 2.0 + w2 / 2.0
        return 0.0


class CoupledTearingModes:
    def __init__(
        self,
        mode1: tuple[int, int],
        mode2: tuple[int, int],
        r_s1: float,
        r_s2: float,
        a: float,
        R0: float,
        B0: float,
    ):
        self.m1, self.n1 = mode1
        self.m2, self.n2 = mode2
        self.r_s1 = r_s1
        self.r_s2 = r_s2
        self.a = a
        self.R0 = R0
        self.B0 = B0

        self.delta_r = abs(r_s1 - r_s2)

        # MRE parameters
        self.a1 = 6.35
        self.mu_0 = 4.0 * np.pi * 1e-7

    def coupling_coefficient(self, m1: int, n1: int, m2: int, n2: int) -> float:
        """
        Spectral-coupling coefficient for mode interaction.

        Same-`n` modes couple strongly through resonant toroidal harmonics.
        Cross-`n` modes retain weaker nonlinear coupling via sidebands.
        """
        if self.R0 <= 0.0 or self.a <= 0.0:
            raise ValueError("R0 and a must be > 0 for mode-coupling evaluation.")
        if m1 <= 0 or m2 <= 0 or n1 <= 0 or n2 <= 0:
            raise ValueError("Mode numbers m,n must be strictly positive integers.")

        base = 0.5 * (self.a / self.R0)
        delta_m = abs(int(m1) - int(m2))
        delta_n = abs(int(n1) - int(n2))

        # Harmonic mismatch attenuation: nearest harmonics couple most strongly.
        spectral_penalty = np.exp(-0.4 * delta_m - 1.2 * delta_n)
        # Same toroidal family is more strongly phase-coupled.
        toroidal_factor = 1.0 if n1 == n2 else 0.25
        coeff = base * toroidal_factor * float(spectral_penalty)
        return float(max(coeff, 0.0))

    def delta_prime_1(self, w1: float, w2: float) -> float:
        # Base stability + nonlinear modification from w2
        # Typically linearly stable
        dp0 = -2.0 * self.m1 / max(self.r_s1, 1e-3)
        return dp0 + 0.5 * w2 / self.a

    def delta_prime_2(self, w1: float, w2: float) -> float:
        dp0 = -2.0 * self.m2 / max(self.r_s2, 1e-3)
        return dp0 + 0.5 * w1 / self.a

    def evolve(
        self,
        w1_0: float,
        w2_0: float,
        j_bs: float,
        j_phi: float,
        eta: float,
        dt: float,
        n_steps: int,
        seed_time: float = -1.0,
        seed_amplitude: float = 0.0,
    ) -> CoupledResult:
        if int(n_steps) < 1:
            raise ValueError("n_steps must be >= 1.")
        if not np.isfinite(dt) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        if not np.isfinite(eta) or float(eta) <= 0.0:
            raise ValueError("eta must be finite and > 0.")
        if not np.isfinite(j_phi) or float(j_phi) <= 0.0:
            raise ValueError("j_phi must be finite and > 0.")
        if not np.isfinite(j_bs) or float(j_bs) < 0.0:
            raise ValueError("j_bs must be finite and >= 0.")

        n_steps = int(n_steps)
        w1 = max(w1_0, 1e-6)
        w2 = max(w2_0, 1e-6)

        w1_trace = np.zeros(n_steps)
        w2_trace = np.zeros(n_steps)
        chir_trace = np.zeros(n_steps)

        tau_R1 = self.mu_0 * self.r_s1**2 / eta
        tau_R2 = self.mu_0 * self.r_s2**2 / eta

        C12 = self.coupling_coefficient(self.m1, self.n1, self.m2, self.n2)
        C21 = self.coupling_coefficient(self.m2, self.n2, self.m1, self.n1)

        j_ratio = j_bs / max(j_phi, 1e-6)

        overlap_time = -1.0
        disruption = False

        for i in range(n_steps):
            t = i * dt

            # Seeding
            if seed_time > 0 and abs(t - seed_time) <= dt:
                w1 = max(w1, seed_amplitude)

            # MRE for w1 (3/2)
            dp1 = self.delta_prime_1(w1, w2)
            # bs term with w_d=1e-3
            bs_term1 = self.a1 * j_ratio * (self.r_s1 / w1) * (w1**2 / (w1**2 + 1e-6))
            c_term1 = C12 * (w2**2) / (self.r_s1 * self.r_s2)
            dw1_dt = (self.r_s1 / tau_R1) * (self.r_s1 * dp1 + bs_term1 + c_term1)

            # MRE for w2 (2/1)
            dp2 = self.delta_prime_2(w1, w2)
            bs_term2 = self.a1 * j_ratio * (self.r_s2 / w2) * (w2**2 / (w2**2 + 1e-6))
            c_term2 = C21 * (w1**2) / (self.r_s1 * self.r_s2)
            dw2_dt = (self.r_s2 / tau_R2) * (self.r_s2 * dp2 + bs_term2 + c_term2)

            w1 += dw1_dt * dt
            w2 += dw2_dt * dt

            # Prevent shrinking to strictly zero to maintain ODE stability
            # Also prevent overflow
            w1 = min(max(w1, 1e-6), 2.0 * self.a)
            w2 = min(max(w2, 1e-6), 2.0 * self.a)

            w1_trace[i] = w1
            w2_trace[i] = w2

            sigma = ChirikovOverlap.parameter(w1, w2, self.delta_r)
            chir_trace[i] = sigma

            if sigma > 1.0 and not disruption:
                disruption = True
                overlap_time = t

        return CoupledResult(w1_trace, w2_trace, chir_trace, overlap_time, disruption)


class SawtoothNTMSeeding:
    def __init__(self, sawtooth_cycler: SawtoothCycler | None):
        self.st = sawtooth_cycler

    def seed_amplitude(self, crash_energy_MJ: float, r_s: float) -> float:
        """Seed island width scales roughly as sqrt(delta W)."""
        # w ~ sqrt(delta B_r) ~ sqrt(delta W)
        return 0.05 * math.sqrt(max(0.0, crash_energy_MJ))

    def seed_probability(self, crash_energy: float, threshold: float) -> float:
        if crash_energy < threshold:
            return 0.0
        prob = 1.0 - math.exp(-(crash_energy - threshold))
        return float(np.clip(prob, 0.0, 1.0))


@dataclass
class DisruptionPath:
    warning_time_ms: float
    avoidable: bool


class DisruptionTriggerAssessment:
    def __init__(self, coupled: CoupledTearingModes):
        self.coupled = coupled

    def run_scenario(
        self, j_bs: float, j_phi: float, omega_phi: float, seed_energy: float
    ) -> DisruptionPath:
        st_seed = SawtoothNTMSeeding(None)
        amp = st_seed.seed_amplitude(seed_energy, self.coupled.r_s1)

        eta = 1e-7
        dt = 0.01
        n_steps = 1000

        res = self.coupled.evolve(
            1e-6, 1e-6, j_bs, j_phi, eta, dt, n_steps, seed_time=0.1, seed_amplitude=amp
        )

        if not res.disruption:
            return DisruptionPath(-1.0, True)

        # If disruption occurs, could ECCD on 3/2 have stopped it?
        # A perfectly aligned ECCD drives w1 -> 0
        # If w1 -> 0, w2 doesn't get the cross-coupling drive
        res_controlled = self.coupled.evolve(
            1e-6, 1e-6, 0.0, j_phi, eta, dt, n_steps, seed_time=0.1, seed_amplitude=amp
        )
        avoidable = not res_controlled.disruption

        return DisruptionPath(res.overlap_time * 1000.0, avoidable)


class TearingModeStabilityMap:
    def __init__(
        self,
        *,
        mode1: tuple[int, int] = (3, 2),
        mode2: tuple[int, int] = (2, 1),
        r_s1: float = 0.5,
        r_s2: float = 0.8,
        a: float = 2.0,
        R0: float = 6.2,
        B0: float = 5.3,
    ):
        self.coupled = CoupledTearingModes(mode1, mode2, r_s1, r_s2, a, R0, B0)

    def scan_beta_li(self, beta_N_range: np.ndarray, li_range: np.ndarray) -> np.ndarray:
        beta_arr = np.asarray(beta_N_range, dtype=float)
        li_arr = np.asarray(li_range, dtype=float)
        if beta_arr.ndim != 1 or li_arr.ndim != 1:
            raise ValueError("beta_N_range and li_range must be one-dimensional arrays.")
        if beta_arr.size == 0 or li_arr.size == 0:
            raise ValueError("beta_N_range and li_range must not be empty.")
        if np.any(~np.isfinite(beta_arr)) or np.any(~np.isfinite(li_arr)):
            raise ValueError("beta_N_range and li_range must contain finite values.")
        if np.any(beta_arr < 0.0) or np.any(li_arr <= 0.0):
            raise ValueError("beta_N_range must be >= 0 and li_range must be > 0.")

        res = np.zeros((beta_arr.size, li_arr.size))

        for i, b in enumerate(beta_arr):
            for j, li in enumerate(li_arr):
                j_phi = 1.0e6 * float(np.clip(li, 0.25, 3.0))
                # Bootstrap drive increases with beta and current peaking.
                drive_ratio = float(np.clip(0.18 * b * li, 0.0, 2.5))
                j_bs = j_phi * drive_ratio
                seed_amp = 0.02 + 0.015 * float(np.clip(b, 0.0, 8.0))

                out = self.coupled.evolve(
                    w1_0=1e-6,
                    w2_0=1e-6,
                    j_bs=j_bs,
                    j_phi=j_phi,
                    eta=1e-7,
                    dt=0.01,
                    n_steps=400,
                    seed_time=0.1,
                    seed_amplitude=seed_amp,
                )
                res[i, j] = -1 if out.disruption else 1

        return res
