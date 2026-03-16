# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Sawtooth Model (Porcelli trigger + Kadomtsev reconnection)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class SawtoothEvent:
    """Record of a single sawtooth crash: timing, radii, and energy release."""

    crash_time: float
    rho_1: float
    rho_mix: float
    T_drop: float
    seed_energy: float


class SawtoothMonitor:
    """Monitors the q-profile for Porcelli-like sawtooth trigger conditions."""

    def __init__(self, rho: np.ndarray, s_crit: float = 0.1):
        self.rho = rho
        self.s_crit = s_crit

    def find_q1_radius(self, q: np.ndarray) -> float | None:
        """Find the normalized radius where q=1."""
        if q[0] >= 1.0 or np.min(q) >= 1.0:
            return None

        q_diff = q - 1.0
        crossings = np.where(np.diff(np.sign(q_diff)))[0]
        if len(crossings) == 0:
            return None

        idx = crossings[0]
        r1, r2 = self.rho[idx], self.rho[idx + 1]
        q1, q2 = q[idx], q[idx + 1]

        if q1 == q2:
            return float(r1)

        frac = (1.0 - q1) / (q2 - q1)
        rho_1 = r1 + frac * (r2 - r1)
        return float(rho_1)

    def check_trigger(self, q: np.ndarray, shear: np.ndarray) -> bool:
        """Porcelli-like trigger based on shear at q=1."""
        rho_1 = self.find_q1_radius(q)
        if rho_1 is None:
            return False

        idx = np.searchsorted(self.rho, rho_1)
        if idx == 0:
            s1 = shear[0]
        elif idx >= len(self.rho):
            s1 = shear[-1]
        else:
            r1, r2 = self.rho[idx - 1], self.rho[idx]
            s_1, s_2 = shear[idx - 1], shear[idx]
            if r1 == r2:
                s1 = s_1
            else:
                frac = (rho_1 - r1) / (r2 - r1)
                s1 = s_1 + frac * (s_2 - s_1)

        return bool(s1 > self.s_crit)


def kadomtsev_crash(
    rho: np.ndarray, T: np.ndarray, n: np.ndarray, q: np.ndarray, R0: float, a: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Apply Kadomtsev reconnection. Returns (T_new, n_new, q_new, rho_1, rho_mix)."""
    monitor = SawtoothMonitor(rho)
    rho_1 = monitor.find_q1_radius(q)

    if rho_1 is None:
        return T.copy(), n.copy(), q.copy(), 0.0, 0.0

    # Helical flux proxy: dpsi*/drho = rho (1/q - 1)
    integrand = rho * (1.0 / np.maximum(q, 1e-6) - 1.0)

    psi_star = np.zeros_like(rho)
    for i in range(1, len(rho)):
        psi_star[i] = psi_star[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * (
            rho[i] - rho[i - 1]
        )

    idx_1 = np.searchsorted(rho, rho_1)
    rho_mix = rho[-1]

    for i in range(idx_1, len(rho)):
        if psi_star[i] <= 0.0:
            if i > 0 and psi_star[i - 1] > 0.0:
                frac = psi_star[i - 1] / (psi_star[i - 1] - psi_star[i])
                rho_mix = rho[i - 1] + frac * (rho[i] - rho[i - 1])
            else:
                rho_mix = rho[i]
            break

    idx_mix = np.searchsorted(rho, rho_mix)
    if idx_mix == 0:
        return T.copy(), n.copy(), q.copy(), rho_1, rho_mix

    rho_inner = rho[:idx_mix]

    def volume_average(prof: np.ndarray) -> float:
        prof_inner = prof[:idx_mix]
        vol_int = trapezoid(prof_inner * rho_inner, rho_inner)
        vol_tot = trapezoid(rho_inner, rho_inner)
        if vol_tot == 0:
            return float(prof_inner[0])
        return float(vol_int / vol_tot)

    T_mix = volume_average(T)
    n_mix = volume_average(n)

    T_new = T.copy()
    n_new = n.copy()
    q_new = q.copy()

    T_new[:idx_mix] = T_mix
    n_new[:idx_mix] = n_mix
    q_new[:idx_mix] = 1.01

    return T_new, n_new, q_new, rho_1, rho_mix


class SawtoothCycler:
    """Tracks and triggers sawtooth crashes."""

    def __init__(self, rho: np.ndarray, R0: float, a: float, s_crit: float = 0.1):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.monitor = SawtoothMonitor(rho, s_crit)
        self.time = 0.0

    def step(
        self, dt: float, q: np.ndarray, shear: np.ndarray, T: np.ndarray, n: np.ndarray
    ) -> SawtoothEvent | None:
        """Advance by dt; trigger Kadomtsev crash if shear at q=1 exceeds s_crit."""
        self.time += dt

        if self.monitor.check_trigger(q, shear):
            T_core_old = T[0]

            e_charge = 1.602e-19

            def plasma_energy(Te: np.ndarray, ne: np.ndarray) -> float:
                energy_dens = 1.5 * (ne * 1e19) * (Te * 1e3 * e_charge)
                vol_element = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho
                return float(trapezoid(energy_dens * vol_element, self.rho))

            W_before = plasma_energy(T, n)

            T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(
                self.rho, T, n, q, self.R0, self.a
            )

            W_after = plasma_energy(T_new, n_new)

            np.copyto(T, T_new)
            np.copyto(n, n_new)
            np.copyto(q, q_new)

            T_drop = T_core_old - T[0]
            seed_energy = max(W_before - W_after, 0.0)
            core_energy_drop = (
                1.5
                * (n[0] * 1e19)
                * (T_drop * 1e3 * e_charge)
                * (2 * np.pi**2 * self.R0 * (rho_1 * self.a) ** 2)
            )

            return SawtoothEvent(
                crash_time=self.time,
                rho_1=rho_1,
                rho_mix=rho_mix,
                T_drop=T_drop,
                seed_energy=float(core_energy_drop),
            )

        return None
