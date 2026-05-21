# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Error Field Amplification and Locked Mode Chain
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ErrorFieldSpectrum:
    def __init__(
        self,
        B0: float,
        n_corrections: int = 0,
        *,
        R0: float = 6.2,
        alignment_sensitivity_21: float = 0.01,
        alignment_sensitivity_32: float = 0.005,
    ):
        if not np.isfinite(B0) or B0 <= 0.0:
            raise ValueError("B0 must be finite and positive")
        if n_corrections < 0:
            raise ValueError("n_corrections must be non-negative")
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and positive")
        if not np.isfinite(alignment_sensitivity_21) or alignment_sensitivity_21 <= 0.0:
            raise ValueError("alignment_sensitivity_21 must be finite and positive")
        if not np.isfinite(alignment_sensitivity_32) or alignment_sensitivity_32 <= 0.0:
            raise ValueError("alignment_sensitivity_32 must be finite and positive")
        self.B0 = B0
        self.R0 = float(R0)
        self.alignment_sensitivity_21 = float(alignment_sensitivity_21)
        self.alignment_sensitivity_32 = float(alignment_sensitivity_32)
        self.n_corrections = n_corrections
        self.B_mn_components = {}
        correction_factor = self._intrinsic_correction_factor()
        self.B_mn_components[(2, 1)] = 1e-4 * B0 * correction_factor
        self.B_mn_components[(3, 2)] = 5e-5 * B0 * correction_factor

    def _intrinsic_correction_factor(self) -> float:
        return float(1.0 / (1.0 + 0.35 * self.n_corrections))

    def set_coil_misalignment(self, delta_R_mm: float, delta_Z_mm: float) -> None:
        if not np.isfinite(delta_R_mm) or not np.isfinite(delta_Z_mm):
            raise ValueError("coil misalignment values must be finite")
        shift_mag = math.sqrt(delta_R_mm**2 + delta_Z_mm**2) / 1000.0
        if shift_mag < 0.0:
            raise ValueError("coil misalignment magnitude must be non-negative")
        # Geometry-normalised shift coupling.
        rel_shift = shift_mag / self.R0
        correction_factor = self._intrinsic_correction_factor()
        self.B_mn_components[(2, 1)] = (
            self.alignment_sensitivity_21 * self.B0 * rel_shift * correction_factor
        )
        self.B_mn_components[(3, 2)] = (
            self.alignment_sensitivity_32 * self.B0 * rel_shift * correction_factor
        )

    def B_mn(self, m: int, n: int) -> float:
        return self.B_mn_components.get((m, n), 0.0)

    def corrected_B_mn(self, m: int, n: int, I_correction: float) -> float:
        if not np.isfinite(I_correction) or I_correction < 0.0:
            raise ValueError("I_correction must be finite and non-negative")
        if m < 1 or n < 1:
            raise ValueError("mode numbers m and n must be positive")
        B_raw = self.B_mn(m, n)
        if B_raw <= 0.0:
            return 0.0
        # Coil correction transfer: higher (m,n) has weaker coupling.
        i_scale = 50.0 * float(m * n)
        attenuation = 1.0 / (1.0 + I_correction / max(i_scale, 1e-9))
        return float(B_raw * attenuation)


class ResonantFieldAmplification:
    def __init__(self, beta_N: float, beta_N_nowall: float):
        if not np.isfinite(beta_N) or beta_N < 0.0:
            raise ValueError("beta_N must be finite and non-negative")
        if not np.isfinite(beta_N_nowall) or beta_N_nowall <= 0.0:
            raise ValueError("beta_N_nowall must be finite and positive")
        self.beta_N = beta_N
        self.beta_N_nowall = beta_N_nowall

    def amplification_factor(self) -> float:
        if self.beta_N >= self.beta_N_nowall:
            return float("inf")
        return 1.0 / (1.0 - self.beta_N / self.beta_N_nowall)

    def resonant_field(self, B_err: float) -> float:
        return B_err * self.amplification_factor()


@dataclass
class RotationEvolution:
    omega_trace: np.ndarray
    locked: bool
    lock_time: float


class ModeLocking:
    def __init__(
        self,
        R0: float,
        a: float,
        B0: float,
        Ip_MA: float,
        omega_phi_0: float,
        *,
        n_e19: float = 8.0,
        kappa: float = 1.7,
        inertia_factor: float = 1.0,
        I_eff_override: float | None = None,
    ):
        for name, value in {
            "R0": R0,
            "a": a,
            "B0": B0,
            "Ip_MA": Ip_MA,
            "omega_phi_0": omega_phi_0,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(n_e19) or n_e19 <= 0.0:
            raise ValueError("n_e19 must be finite and positive")
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")
        if not np.isfinite(inertia_factor) or inertia_factor <= 0.0:
            raise ValueError("inertia_factor must be finite and positive")
        if I_eff_override is not None and (
            not np.isfinite(I_eff_override) or I_eff_override <= 0.0
        ):
            raise ValueError("I_eff_override must be finite and positive when provided")
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.Ip = Ip_MA * 1e6
        self.omega_phi_0 = omega_phi_0
        self.n_e19 = float(n_e19)
        self.kappa = float(kappa)
        self.inertia_factor = float(inertia_factor)

        self.I_eff = (
            float(I_eff_override)
            if I_eff_override is not None
            else self._estimate_effective_inertia()
        )

    def _estimate_effective_inertia(self) -> float:
        """Estimate effective toroidal rotation inertia from plasma mass and geometry."""
        m_ion = 2.5 * 1.67e-27
        m_e = 9.11e-31
        n_m3 = self.n_e19 * 1.0e19
        volume = 2.0 * math.pi**2 * self.R0 * self.a**2 * self.kappa
        mass = n_m3 * (m_ion + m_e) * volume
        return float(max(self.inertia_factor * mass * self.R0**2, 1e-8))

    def em_torque(self, B_res: float, r_s: float, m: int, n: int) -> float:
        """Electromagnetic braking torque [N m]."""
        if not np.isfinite(B_res) or B_res < 0.0:
            raise ValueError("B_res must be finite and non-negative")
        if not np.isfinite(r_s) or r_s <= 0.0:
            raise ValueError("r_s must be finite and positive")
        if m < 1 or n < 1:
            raise ValueError("mode numbers m and n must be positive")
        mu_0 = 4.0 * math.pi * 1e-7
        resonant_area = 4.0 * math.pi**2 * self.R0 * r_s
        pitch_factor = (m / r_s) * n
        torque = resonant_area * pitch_factor * (B_res**2) / mu_0
        return torque

    def evolve_rotation(
        self, B_res: float, r_s: float, tau_visc: float, dt: float, n_steps: int
    ) -> RotationEvolution:
        if not np.isfinite(tau_visc) or tau_visc <= 0.0:
            raise ValueError("tau_visc must be finite and positive")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        omega = self.omega_phi_0
        omega_trace = np.zeros(n_steps)
        locked = False
        lock_time = -1.0

        T_em = self.em_torque(B_res, r_s, 2, 1)
        # Locking when equilibrium rotation is driven to zero
        omega_eq = self.omega_phi_0 - T_em * tau_visc / self.I_eff
        omega_crit = max(0.0, omega_eq) if omega_eq > 0 else 0.0

        for i in range(n_steps):
            if not locked:
                # d_omega/dt = -(omega - omega_0)/tau_visc - T_em / I_eff
                d_omega = -(omega - self.omega_phi_0) / tau_visc - T_em / self.I_eff
                omega += d_omega * dt

                if omega <= omega_crit:
                    locked = True
                    lock_time = i * dt
                    omega = 0.0
            else:
                omega = 0.0

            omega_trace[i] = omega

        return RotationEvolution(omega_trace, locked, lock_time)


@dataclass
class IslandGrowth:
    w_trace: np.ndarray
    overlap_time: float
    stochastic: bool


class LockedModeIsland:
    def __init__(self, r_s: float, m: int, n: int, a: float, R0: float, delta_prime: float):
        self.r_s = r_s
        self.m = m
        self.n = n
        self.a = a
        self.R0 = R0
        self.delta_prime = delta_prime

    def grow(
        self, w0: float, eta: float, dt: float, n_steps: int, delta_r_mn: float = 0.3
    ) -> IslandGrowth:
        """
        w grows on resistive timescale.
        dw/dt = (eta / mu0 r_s) * [r_s Delta' + C_lock r_s / w]
        """
        w = max(w0, 1e-4)
        w_trace = np.zeros(n_steps)
        mu_0 = 4.0 * math.pi * 1e-7

        C_lock = 1.0  # normalized locked-mode current drive; La Haye (2006), Eq. 8
        overlap_time = -1.0
        stochastic = False

        for i in range(n_steps):
            # MRE
            dw_dt = (eta / (mu_0 * self.r_s)) * (
                self.r_s * self.delta_prime + C_lock * self.r_s / w
            )
            w += dw_dt * dt
            w_trace[i] = w

            # Chirikov
            if not stochastic and w > delta_r_mn:
                stochastic = True
                overlap_time = i * dt

        return IslandGrowth(w_trace, overlap_time, stochastic)


@dataclass
class ChainResult:
    lock_time: float
    island_width_at_tq: float
    tq_trigger_time: float
    disruption: bool
    warning_time_ms: float


class ErrorFieldToDisruptionChain:
    def __init__(self, config: dict[str, float]):
        self.config = config

    def run(self, B_err_n1: float, omega_phi_0: float) -> ChainResult:
        R0 = self.config.get("R0", 6.2)
        a = self.config.get("a", 2.0)
        B0 = self.config.get("B0", 5.3)
        Ip = self.config.get("Ip_MA", 15.0)
        beta_N = self.config.get("beta_N", 2.0)
        beta_nowall = self.config.get("beta_N_nowall", 2.8)

        # 1. RFA
        rfa = ResonantFieldAmplification(beta_N, beta_nowall)
        B_res = rfa.resonant_field(B_err_n1)

        # 2. Mode Locking
        dt = 0.001
        n_steps = 2000  # 2 seconds
        tau_visc = 0.1
        r_s = a * 0.5

        locker = ModeLocking(R0, a, B0, Ip, omega_phi_0)
        rot_evol = locker.evolve_rotation(B_res, r_s, tau_visc, dt, n_steps)

        if not rot_evol.locked:
            return ChainResult(-1.0, 0.0, -1.0, False, -1.0)

        # 3. Island Growth
        eta = 1e-7  # Typical hot plasma resistivity
        lm = LockedModeIsland(r_s, 2, 1, a, R0, delta_prime=-1.0)  # Classically stable without lock

        # How many steps left after locking?
        steps_left = n_steps - int(rot_evol.lock_time / dt)
        if steps_left <= 0:
            return ChainResult(rot_evol.lock_time, 0.0, -1.0, False, -1.0)

        grow_res = lm.grow(w0=1e-3, eta=eta, dt=dt, n_steps=steps_left, delta_r_mn=0.2 * a)

        if not grow_res.stochastic:
            return ChainResult(rot_evol.lock_time, grow_res.w_trace[-1], -1.0, False, -1.0)

        tq_time = rot_evol.lock_time + grow_res.overlap_time
        warning_time = grow_res.overlap_time * 1000.0

        return ChainResult(
            lock_time=rot_evol.lock_time,
            island_width_at_tq=grow_res.w_trace[int(grow_res.overlap_time / dt)],
            tq_trigger_time=tq_time,
            disruption=True,
            warning_time_ms=warning_time,
        )
