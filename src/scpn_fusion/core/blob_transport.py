# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SOL filament and blob transport utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class BlobDynamics:
    """Regime-dependent blob velocity model for scrape-off-layer filaments."""

    def __init__(self, R0: float, B0: float, Te_eV: float, Ti_eV: float, mi_amu: float = 2.0):
        self.R0 = R0
        self.B0 = B0
        self.Te = Te_eV
        self.Ti = Ti_eV
        self.mi = mi_amu * 1.67e-27
        self.e_charge = 1.602e-19

        T_tot_J = (self.Te + self.Ti) * self.e_charge
        self.c_s = math.sqrt(T_tot_J / self.mi)
        self.rho_s = self.c_s / (self.e_charge * self.B0 / self.mi)

    def critical_size(self, L_parallel: float) -> float:
        """Return the critical blob size delta_b* [m]."""
        if L_parallel <= 0.0:
            return float("inf")
        return float(2.0 * self.rho_s * (L_parallel / (self.R0 * self.rho_s)) ** 0.2)

    def max_velocity(self, L_parallel: float) -> float:
        """Return the maximum blob velocity v_b at delta_b* [m/s]."""
        delta_star = self.critical_size(L_parallel)
        return self.sheath_velocity(delta_star)

    def sheath_velocity(self, delta_b: float) -> float:
        """Return v_b in the sheath-connected regime (small blobs)."""
        if delta_b <= 0.0:
            return 0.0
        # Sheath-connected regime: v_b ~ delta_b^{-1/2} — Krasheninnikov (2001)
        return 2.0 * self.c_s * self.rho_s / (self.R0 * math.sqrt(delta_b / self.R0 + 1e-6))

    def inertial_velocity(self, delta_b: float) -> float:
        """Return v_b in the inertia-limited regime (large blobs)."""
        # v_b prop delta_b^1/2
        return self.c_s * math.sqrt(2.0 * delta_b / self.R0)

    def blob_velocity(self, delta_b: float, n_e: float, L_parallel: float) -> tuple[float, str]:
        """Return blob radial velocity and active sheath or inertial regime."""
        delta_star = self.critical_size(L_parallel)

        if delta_b < delta_star:
            v = self.sheath_velocity(delta_b)
            regime = "sheath"
        else:
            v = self.inertial_velocity(delta_b)
            regime = "inertial"

        return float(v), regime


@dataclass
class BlobPopulation:
    """Generated blob sizes, amplitudes, velocities, and event birth times."""

    sizes: FloatArray
    amplitudes: FloatArray
    velocities: FloatArray
    birth_times: FloatArray


class BlobEnsemble:
    """Generate stochastic blob populations and aggregate radial fluxes."""

    def __init__(self, dynamics: BlobDynamics, n_blobs: int = 1000):
        self.dynamics = dynamics
        self.n_blobs = n_blobs

    def generate(
        self,
        delta_b_mean: float,
        delta_b_sigma: float,
        amplitude_mean: float,
        waiting_time_mean: float,
        rng: np.random.Generator,
        *,
        amplitude_sigma_log: float = 0.5,
    ) -> BlobPopulation:
        """Generate a stochastic blob population from size and waiting-time laws."""
        # log-normal amplitudes
        sigma_amp = float(amplitude_sigma_log)
        if not np.isfinite(sigma_amp) or sigma_amp <= 0.0:
            raise ValueError("amplitude_sigma_log must be finite and positive")
        mu_amp = math.log(amplitude_mean) - 0.5 * sigma_amp**2
        amps = rng.lognormal(mu_amp, sigma_amp, self.n_blobs)

        # normal sizes
        sizes = rng.normal(delta_b_mean, delta_b_sigma, self.n_blobs)
        sizes = np.maximum(sizes, 1e-3)

        # exponential waiting times
        waits = rng.exponential(waiting_time_mean, self.n_blobs)
        births = np.cumsum(waits)

        # velocities
        vels = np.zeros(self.n_blobs)
        for i in range(self.n_blobs):
            v, _ = self.dynamics.blob_velocity(sizes[i], 1e19, L_parallel=10.0)
            vels[i] = v

        return BlobPopulation(sizes, amps, vels, births)

    def radial_flux(self, population: BlobPopulation) -> float:
        """Return time-averaged radial particle flux from the population."""
        # Gamma_blob = f_b * <delta n> * <v_b> * <delta_b> / normalization
        # We just compute the time-averaged flux over the generation window
        tot_time = population.birth_times[-1] if self.n_blobs > 0 else 1.0

        # approximate flux contribution
        flux_sum = np.sum(population.amplitudes * population.velocities * population.sizes)
        return float(flux_sum / tot_time)

    def heat_flux(self, population: BlobPopulation, Te_eV: float) -> float:
        """Return convective heat flux carried by the generated blob population."""
        gamma = self.radial_flux(population)
        return gamma * Te_eV * 1.602e-19 * 1.5  # ~ 3/2 k T


class SOLBlobProfile:
    """Analytic SOL profile estimates with blob-enhanced radial transport."""

    @staticmethod
    def radial_density(
        r: FloatArray,
        Gamma_blob: float,
        D_perp: float,
        lambda_n: float,
        *,
        n_sep: float = 1.0e19,
    ) -> FloatArray:
        """Return normalised radial density profile with blob transport broadening."""
        r_arr = np.asarray(r, dtype=float)
        gamma = float(Gamma_blob)
        d_perp = float(D_perp)
        lam = float(lambda_n)
        n_ref = float(n_sep)
        if r_arr.ndim != 1:
            raise ValueError("r must be a one-dimensional radial grid.")
        if not np.all(np.isfinite(r_arr)):
            raise ValueError("r must contain finite values.")
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("Gamma_blob must be finite and >= 0.")
        if not np.isfinite(d_perp) or d_perp <= 0.0:
            raise ValueError("D_perp must be finite and > 0.")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lambda_n must be finite and > 0.")
        if not np.isfinite(n_ref) or n_ref <= 0.0:
            raise ValueError("n_sep must be finite and > 0.")

        # Transport broadening from a dimensionless radial Péclet number:
        # Pe_r = (Gamma_blob / n_sep) * lambda_n / D_perp.
        # Effective decay length grows with convective blob transport.
        pe_r = max((gamma / n_ref) * lam / d_perp, 0.0)
        eff_lambda = lam * math.sqrt(1.0 + pe_r)
        return np.asarray(np.exp(-r_arr / eff_lambda))

    @staticmethod
    def wall_flux(
        r_wall: float,
        Gamma_blob: float,
        lambda_n: float,
        *,
        n_sep: float = 1.0e19,
        D_perp: float = 1.0,
    ) -> float:
        """Return attenuated blob flux reaching a wall radius."""
        r_w = float(r_wall)
        gamma = float(Gamma_blob)
        lam = float(lambda_n)
        n_ref = float(n_sep)
        d_perp = float(D_perp)
        if not np.isfinite(r_w) or r_w < 0.0:
            raise ValueError("r_wall must be finite and >= 0.")
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("Gamma_blob must be finite and >= 0.")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lambda_n must be finite and > 0.")
        if not np.isfinite(n_ref) or n_ref <= 0.0:
            raise ValueError("n_sep must be finite and > 0.")
        if not np.isfinite(d_perp) or d_perp <= 0.0:
            raise ValueError("D_perp must be finite and > 0.")

        pe_r = max((gamma / n_ref) * lam / d_perp, 0.0)
        eff_lambda = lam * math.sqrt(1.0 + pe_r)
        return float(gamma * math.exp(-r_w / eff_lambda))


@dataclass
class BlobEvent:
    """Detected blob event interval and derived amplitude, duration, and size."""

    start_idx: int
    end_idx: int
    peak_amplitude: float
    duration: float
    size_estimate: float


class BlobDetector:
    """Detect transient blob events and conditional-average their waveforms."""

    def detect_blobs(
        self, signal: FloatArray, dt: float = 1e-6, threshold: float = 2.5
    ) -> list[BlobEvent]:
        """Return threshold-crossing blob events from a one-dimensional signal."""
        mean_sig = np.mean(signal)
        std_sig = np.std(signal)

        if std_sig == 0:
            return []

        norm_sig = (signal - mean_sig) / std_sig

        events = []
        in_blob = False
        start = 0
        peak = 0.0

        for i, val in enumerate(norm_sig):
            if val > threshold and not in_blob:
                in_blob = True
                start = i
                peak = val
            elif val > threshold and in_blob:
                peak = max(peak, val)
            elif val <= 0.0 and in_blob:  # zero crossing ends it
                in_blob = False
                dur = (i - start) * dt
                # Size estimate based on typical velocity
                size = dur * 1000.0
                events.append(BlobEvent(start, i, float(peak * std_sig), dur, size))

        return events

    def conditional_average(
        self, signal: FloatArray, events: list[BlobEvent], window: int = 50
    ) -> FloatArray:
        """Return event-centred conditional average over a symmetric sample window."""
        if not events:
            return np.zeros(2 * window + 1)

        cond_avg = np.zeros(2 * window + 1)
        count = 0

        for ev in events:
            center = ev.start_idx + (ev.end_idx - ev.start_idx) // 2
            if center - window >= 0 and center + window < len(signal):
                cond_avg += signal[center - window : center + window + 1]
                count += 1

        if count > 0:
            cond_avg /= count

        return cond_avg
