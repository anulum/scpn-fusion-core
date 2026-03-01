# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — RF Heating
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel


def _require_finite_float(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if min_value is not None and out < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return out


def _require_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out

class RFHeatingSystem:
    """
    Simulates Ion Cyclotron Resonance Heating (ICRH).
    Uses Ray-Tracing to track EM waves launching from the antenna 
    and absorbing at the resonance layer.
    """
    def __init__(self, config_path):
        self.kernel = FusionKernel(config_path)
        self.kernel.solve_equilibrium() # Get B-field map
        
        # Physics Constants
        self.q_D = 1.602e-19 # Charge (Deuterium)
        self.m_D = 3.34e-27  # Mass (Deuterium)
        self.freq = 50e6     # 50 MHz (Standard ICRH freq)
        self.omega_wave = 2 * np.pi * self.freq
        
    def get_plasma_params(self, R, Z):
        """
        Returns B_mod, density, and derivatives at (R,Z).
        """
        # 1. Magnetic Field (Toroidal dominates)
        # B_tor ~ B0 * R0 / R
        B0 = 5.3 # Tesla at axis (ITER)
        R0 = 6.2
        B_tor = B0 * R0 / R
        
        # Poloidal field from kernel
        # We need grid lookup
        ir = int((R - self.kernel.R[0]) / self.kernel.dR)
        iz = int((Z - self.kernel.Z[0]) / self.kernel.dZ)
        
        if 0 <= ir < self.kernel.NR and 0 <= iz < self.kernel.NZ:
            B_R = self.kernel.B_R[iz, ir]
            B_Z = self.kernel.B_Z[iz, ir]
            psi_val = self.kernel.Psi[iz, ir]
        else:
            B_R, B_Z, psi_val = 0, 0, 0
            
        B_mod = np.sqrt(B_tor**2 + B_R**2 + B_Z**2)
        
        # 2. Density Profile (Parabolic)
        # n = n0 * (1 - psi_norm)
        # Reduced-order surrogate: Gaussian blob
        dist_sq = (R - R0)**2 + Z**2
        n_e = 1e20 * np.exp(-dist_sq / 2.0)
        
        # Derivatives of density (for refraction)
        dn_dR = -n_e * (R - R0) / 1.0
        dn_dZ = -n_e * Z / 1.0
        
        return B_mod, n_e, dn_dR, dn_dZ

    def dispersion_relation(self, R, Z, k_R, k_Z):
        """
        Calculates the local dispersion D(w, k) = 0.
        Harden with Warm Plasma thermal corrections for ICRH.
        """
        B_mod, n_e, _, _ = self.get_plasma_params(R, Z)
        
        if n_e < 1e18: return 1.0 # Vacuum
        
        # 1. Alfven speed (Cold Limit)
        mu0 = 4*np.pi*1e-7
        v_A = B_mod / np.sqrt(mu0 * n_e * self.m_D)
        
        # 2. Thermal correction (Warm Plasma)
        # For ICRH, thermal effects modify the k_perp dispersion
        T_ion_keV = 10.0 # Heuristic local Ti
        v_thi = np.sqrt(2.0 * T_ion_keV * 1.602e-16 / self.m_D)
        
        # Finite Larmor Radius (FLR) correction factor
        # omega^2 = k^2 * v_A^2 * (1 + 3/4 * (k_perp * rho_i)^2)
        rho_i = self.m_D * v_thi / (self.q_D * B_mod)
        k_sq = k_R**2 + k_Z**2
        
        flr_correction = 1.0 + 0.75 * (k_sq * rho_i**2)
        
        # Dispersion: D = k^2 * v_A^2 * flr_correction - omega^2
        D = k_sq * v_A**2 * flr_correction - self.omega_wave**2
        
        return D

    def ray_equations(self, state, t):
        """
        Hamiltonian Ray Tracing equations.
        dr/dt = dD/dk
        dk/dt = -dD/dr
        """
        R, Z, kR, kZ = state
        
        # Finite differences for derivatives of D
        # This implicitly handles refraction and reflection
        eps = 1e-3
        
        # dD/dk
        D_pkR = self.dispersion_relation(R, Z, kR + eps, kZ)
        D_mkR = self.dispersion_relation(R, Z, kR - eps, kZ)
        dD_dkR = (D_pkR - D_mkR) / (2*eps)
        
        D_pkZ = self.dispersion_relation(R, Z, kR, kZ + eps)
        D_mkZ = self.dispersion_relation(R, Z, kR, kZ - eps)
        dD_dkZ = (D_pkZ - D_mkZ) / (2*eps)
        
        # dD/dr
        D_pR = self.dispersion_relation(R + eps, Z, kR, kZ)
        D_mR = self.dispersion_relation(R - eps, Z, kR, kZ)
        dD_dR = (D_pR - D_mR) / (2*eps)
        
        D_pZ = self.dispersion_relation(R, Z + eps, kR, kZ)
        D_mZ = self.dispersion_relation(R, Z - eps, kR, kZ)
        dD_dZ = (D_pZ - D_mZ) / (2*eps)
        
        # Group Velocity (dr/dt)
        dR_dt = -dD_dkR
        dZ_dt = -dD_dkZ
        
        # Wavevector change (dk/dt)
        dkR_dt = dD_dR
        dkZ_dt = dD_dZ
        
        return [dR_dt, dZ_dt, dkR_dt, dkZ_dt]

    def trace_rays(self, n_rays=10):
        print("--- RF HEATING RAY TRACING ---")
        print(f"Frequency: {self.freq/1e6} MHz")
        
        # Antenna Position (Outboard midplane)
        R_ant = 9.0
        Z_ant_spread = np.linspace(-1.0, 1.0, n_rays)
        
        trajectories = []
        
        for i in range(n_rays):
            # Initial condition: Launch inward (kR < 0)
            k0 = 10.0 # Initial wavenumber
            init_state = [R_ant, Z_ant_spread[i], -k0, 0.0]
            
            t_span = np.linspace(0, 0.5, 100) # Short time (normalized)
            
            # Solve ODE
            sol = odeint(self.ray_equations, init_state, t_span)
            
            # Check Resonance
            # Resonance condition: omega = omega_ci = qB/m
            # B_res = omega * m / q
            B_res = self.omega_wave * self.m_D / self.q_D
            
            # Find where B matches B_res along path
            # (Post-processing check)
            
            trajectories.append(sol)
            
        print(f"Resonance Field B_res: {B_res:.2f} Tesla")
        return trajectories, B_res

    def plot_heating(self, trajectories, B_res):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Plasma Geometry
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, colors='gray', alpha=0.3)
        
        # 2. Resonance Layer (Vertical Line approx)
        # B_tor ~ B0*R0/R. B ~ B_res => R_res ~ B0*R0/B_res
        R_res = (5.3 * 6.2) / B_res
        ax.axvline(R_res, color='green', linestyle='--', linewidth=2, label='Cyclotron Resonance Layer')
        
        # 3. Rays
        for i, sol in enumerate(trajectories):
            R = sol[:, 0]
            Z = sol[:, 1]
            ax.plot(R, Z, 'r-', alpha=0.6)
            
            # Draw absorption point (intersection with Resonance)
            # Simple geometric check
            idx = np.abs(R - R_res).argmin()
            if np.abs(R[idx] - R_res) < 0.1:
                ax.plot(R[idx], Z[idx], 'y*', markersize=10, label='Energy Dump' if i==0 else "")
        
        # Antenna
        ax.plot([9.0]*len(trajectories), [t[0,1] for t in trajectories], 'ko', label='Antenna Array')
        
        ax.set_title(f"ICRH Wave Propagation ({self.freq/1e6} MHz)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim(2, 10)
        ax.set_ylim(-6, 6)
        ax.legend()
        
        plt.savefig("RF_Heating_Rays.png")
        print("Saved: RF_Heating_Rays.png")

    def compute_power_deposition(self, trajectories, P_rf_mw=20.0, n_radial_bins=50):
        """Compute radial power deposition profile from ray absorption.

        Uses cyclotron damping: the imaginary part of the wave vector causes
        exponential power decay along each ray path. The absorption coefficient
        is proportional to exp(-(omega - n*Omega_ci)^2 / (k_par * v_thi)^2).

        Parameters
        ----------
        trajectories : list of ndarray
            Ray paths from trace_rays().
        P_rf_mw : float
            Total injected RF power (MW).
        n_radial_bins : int
            Number of radial bins for the deposition profile.

        Returns
        -------
        rho_bins : ndarray
            Normalised radius bin centres (0–1).
        P_dep_mw_m3 : ndarray
            Power deposition density (MW/m^3) per radial bin.
        absorption_efficiency : float
            Fraction of injected power absorbed (0–1).
        """
        R0 = 6.2
        a = 2.0  # minor radius (m)
        B0 = 5.3
        T_ion_keV = 20.0

        # Ion thermal speed
        v_thi = np.sqrt(2.0 * T_ion_keV * 1e3 * 1.602e-19 / self.m_D)

        rho_bins = np.linspace(0.0, 1.0, n_radial_bins)
        P_dep = np.zeros(n_radial_bins)
        total_absorbed = 0.0
        P_per_ray = P_rf_mw / max(len(trajectories), 1)

        for sol in trajectories:
            R = sol[:, 0]
            Z = sol[:, 1]
            P_ray = P_per_ray  # power remaining in this ray (MW)

            for j in range(1, len(R)):
                if P_ray < 1e-6:
                    break

                R_mid = 0.5 * (R[j - 1] + R[j])
                Z_mid = 0.5 * (Z[j - 1] + Z[j])
                ds = np.sqrt((R[j] - R[j - 1]) ** 2 + (Z[j] - Z[j - 1]) ** 2)

                if ds < 1e-12:
                    continue

                # Local B and cyclotron frequency
                B_local = B0 * R0 / max(R_mid, 0.1)
                omega_ci = self.q_D * B_local / self.m_D

                # Cyclotron damping coefficient
                # k_imag ~ (omega_pi^2 / (c * omega)) * exp(-delta^2)
                # where delta = (omega - omega_ci) / (k_par * v_thi)
                delta = (self.omega_wave - omega_ci) / max(10.0 * v_thi, 1e6)
                damping = np.exp(-delta**2)

                # Absorption: dP = -alpha * P * ds
                # alpha ~ 0.5 * damping / a  (normalised so that strong resonance
                # absorbs over ~1 minor radius length scale)
                alpha = 0.5 * damping / max(a, 0.1)
                dP = P_ray * (1.0 - np.exp(-alpha * ds))
                P_ray -= dP
                total_absorbed += dP

                # Map to radial bin
                rho = np.sqrt((R_mid - R0) ** 2 + Z_mid**2) / a
                rho = min(rho, 1.0)
                bin_idx = min(int(rho * n_radial_bins), n_radial_bins - 1)

                # Volume of radial shell
                dr = 1.0 / n_radial_bins
                r_inner = rho_bins[bin_idx] * a
                dV = 2.0 * np.pi * R_mid * 2.0 * np.pi * r_inner * a * dr
                dV = max(dV, 1e-6)

                P_dep[bin_idx] += dP / dV  # MW / m^3

        efficiency = total_absorbed / max(P_rf_mw, 1e-12)
        return rho_bins, P_dep, float(np.clip(efficiency, 0.0, 1.0))


class ECRHHeatingSystem:
    """Electron Cyclotron Resonance Heating (ECRH) with ray-tracing absorption.

    Operates at 170 GHz (ITER ECRH frequency). Resonance occurs where
    omega = n * Omega_ce (electron cyclotron frequency), with n=1 (fundamental)
    or n=2 (second harmonic).

    Parameters
    ----------
    b0_tesla : float
        On-axis toroidal magnetic field (T).
    r0_major : float
        Major radius (m).
    freq_ghz : float
        ECRH frequency (GHz). Default: 170.
    harmonic : int
        Cyclotron harmonic number (1 or 2).
    """

    def __init__(
        self,
        b0_tesla: float = 5.3,
        r0_major: float = 6.2,
        freq_ghz: float = 170.0,
        harmonic: int = 1,
    ):
        self.B0 = _require_finite_float("b0_tesla", b0_tesla, min_value=0.1)
        self.R0 = _require_finite_float("r0_major", r0_major, min_value=0.1)
        self.freq = _require_finite_float("freq_ghz", freq_ghz, min_value=0.1) * 1e9
        self.omega = 2.0 * np.pi * self.freq
        self.harmonic = _require_int("harmonic", harmonic, 1)
        self.m_e = 9.109e-31
        self.q_e = 1.602e-19

    def resonance_radius(self) -> float:
        """Major radius where n*Omega_ce = omega."""
        B_res = self.omega * self.m_e / (self.harmonic * self.q_e)
        return self.B0 * self.R0 / B_res

    def compute_deposition(
        self,
        P_ecrh_mw: float = 20.0,
        n_radial_bins: int = 50,
        T_e_keV: float = 20.0,
        n_e: float = 1e20,
        launch_angle_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ECRH power deposition profile.

        Uses Gaussian deposition centred at the resonance layer with width
        determined by the Doppler broadening of the electron cyclotron resonance.

        Returns
        -------
        rho_bins : ndarray
            Normalised radius (0–1).
        P_dep_mw_m3 : ndarray
            Power deposition density.
        absorption_efficiency : float
            Fraction absorbed (0–1).
        """
        P_ecrh_mw = _require_finite_float("P_ecrh_mw", P_ecrh_mw, min_value=0.0)
        n_radial_bins = _require_int("n_radial_bins", n_radial_bins, 8)
        T_e_keV = _require_finite_float("T_e_keV", T_e_keV, min_value=0.01)
        n_e = _require_finite_float("n_e", n_e, min_value=1e16)
        launch_angle_deg = _require_finite_float(
            "launch_angle_deg",
            launch_angle_deg,
            min_value=-85.0,
            max_value=85.0,
        )

        a = 2.0  # minor radius
        R_res = self.resonance_radius()
        rho_res = abs(R_res - self.R0) / a

        # Thermal speed for Doppler width
        v_the = np.sqrt(2.0 * T_e_keV * 1e3 * self.q_e / self.m_e)
        theta = np.deg2rad(launch_angle_deg)
        obliquity = float(np.clip(np.cos(theta) ** 2, 0.05, 1.0))
        # Doppler width in rho: delta_rho ~ v_the / (omega * a)
        delta_rho = max(v_the / (self.omega * a) * 50.0 * (1.0 + 0.35 * abs(np.sin(theta))), 0.02)

        rho_bins = np.linspace(0.0, 1.0, n_radial_bins)
        P_dep = np.zeros(n_radial_bins)

        # Gaussian deposition profile centred at rho_res
        for i, rho in enumerate(rho_bins):
            r_shell = rho * a
            R_local = self.R0 + r_shell
            dV = 2.0 * np.pi * R_local * 2.0 * np.pi * r_shell * a / n_radial_bins
            dV = max(dV, 1e-6)
            dep = np.exp(-((rho - rho_res) ** 2) / (2.0 * delta_rho**2))
            P_dep[i] = dep / dV

        # Normalise so total = P_ecrh * absorption
        # Single-pass absorption: eta ~ 1 - exp(-tau_opt)
        # Optical depth for O-mode: tau ~ (omega_pe / omega)^2 * (n_e * L / ...)
        omega_pe = np.sqrt(n_e * self.q_e**2 / (self.m_e * 8.854e-12))
        resonance_overlap = 1.0 if rho_res <= 1.0 else float(np.exp(-((rho_res - 1.0) / 0.18) ** 2))
        tau_opt = (omega_pe / self.omega) ** 2 * 20.0 * self.harmonic * obliquity * resonance_overlap
        efficiency = float(np.clip(1.0 - np.exp(-tau_opt), 0.01, 0.99))

        total_dep = np.sum(P_dep)
        if total_dep > 1e-12:
            P_dep *= (P_ecrh_mw * efficiency) / total_dep

        return rho_bins, P_dep, efficiency


if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    rf = RFHeatingSystem(cfg)
    rays, B_res = rf.trace_rays()
    rf.plot_heating(rays, B_res)
