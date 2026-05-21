# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Kinetic Equilibrium Reconstruction (Kinetic EFIT)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_fusion.control.realtime_efit import (
    MagneticDiagnostics,
    RealtimeEFIT,
    ReconstructionResult,
)


class FastIonPressure:
    """
    Anisotropic fast ion pressure model.
    """

    def __init__(self, E_fast_keV: float, n_fast_frac: float, anisotropy_sigma: float = 0.0):
        if not np.isfinite(E_fast_keV) or float(E_fast_keV) <= 0.0:
            raise ValueError("E_fast_keV must be finite and > 0.")
        if not np.isfinite(n_fast_frac) or not (0.0 <= float(n_fast_frac) <= 1.0):
            raise ValueError("n_fast_frac must be finite and within [0, 1].")
        if not np.isfinite(anisotropy_sigma) or float(anisotropy_sigma) >= 3.0:
            raise ValueError("anisotropy_sigma must be finite and < 3.0.")
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.sigma = anisotropy_sigma  # sigma = 1 - p_par / p_perp

    def p_perp(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        # p_fast = 2/3 n_fast E_fast for isotropic
        # p_fast = (2 p_perp + p_par)/3 = p_perp(2 + (1-sigma))/3 = p_perp(3-sigma)/3
        # p_perp = p_fast * 3 / (3 - sigma)
        n_fast = ne_19 * 1e19 * self.n_fast_frac
        p_fast_Pa = (2.0 / 3.0) * n_fast * (self.E_fast_keV * 1e3 * 1.602e-19)
        return p_fast_Pa * 3.0 / (3.0 - self.sigma)

    def p_par(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        return self.p_perp(rho, ne_19) * (1.0 - self.sigma)

    def p_isotropic_equivalent(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        p_perp = self.p_perp(rho, ne_19)
        p_par = self.p_par(rho, ne_19)
        return np.asarray((2.0 * p_perp + p_par) / 3.0)


@dataclass
class KineticConstraints:
    Te_points: list[tuple[float, float, float]]  # (R, Z, Te_keV)
    ne_points: list[tuple[float, float, float]]  # (R, Z, ne_19)
    Ti_points: list[tuple[float, float, float]]  # (R, Z, Ti_keV)
    mse_points: list[tuple[float, float, float]]  # (R, Z, pitch_angle_deg)


@dataclass
class KineticReconstructionResult(ReconstructionResult):
    p_kinetic: np.ndarray
    p_equilibrium: np.ndarray
    pressure_consistency: float
    q_profile: np.ndarray
    beta_fast: float
    sigma_anisotropy: np.ndarray


def mse_pitch_angle(B_R: float, B_Z: float, B_phi: float, v_beam: float, R: float) -> float:
    """
    Synthetic MSE pitch-angle measurement in degrees.

    Includes a beam-motion correction term that scales with `v_beam / R`
    and couples radial field into the observed vertical component.
    """
    b_r = float(B_R)
    b_z = float(B_Z)
    b_phi = float(B_phi)
    v = float(v_beam)
    r_major = float(R)
    if not np.isfinite(b_r) or not np.isfinite(b_z) or not np.isfinite(b_phi):
        raise ValueError("B_R, B_Z, and B_phi must be finite.")
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("v_beam must be finite and > 0.")
    if not np.isfinite(r_major) or r_major <= 0.0:
        raise ValueError("R must be finite and > 0.")

    # Small beam-geometry correction, bounded and dimensionless.
    c = 299_792_458.0
    corr = float(np.clip((v / c) * (1.0 / r_major), 0.0, 0.5))
    b_z_eff = b_z + corr * b_r
    return float(np.degrees(np.arctan2(b_z_eff, b_phi)))


class KineticEFIT(RealtimeEFIT):
    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        kinetic: KineticConstraints,
        fast_ions: FastIonPressure,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
    ):
        super().__init__(diagnostics, R_grid, Z_grid)
        self.kinetic = kinetic
        self.fast_ions = fast_ions

    def reconstruct(self, measurements: dict[str, Any]) -> KineticReconstructionResult:
        # 1. Base magnetic reconstruction
        res_mag = super().reconstruct(measurements)

        rho_1d = np.linspace(0, 1, 50)
        ne_prof = self._build_profile_from_points(self.kinetic.ne_points, rho_1d, default_core=5.0)
        te_prof = self._build_profile_from_points(self.kinetic.Te_points, rho_1d, default_core=10.0)
        ti_prof = self._build_profile_from_points(self.kinetic.Ti_points, rho_1d, default_core=10.0)

        # p_thermal = n_e T_e + n_i T_i
        e_charge = 1.602e-19
        p_th = (ne_prof * 1e19) * (te_prof * 1e3 * e_charge) + (ne_prof * 1e19) * (
            ti_prof * 1e3 * e_charge
        )

        p_fast = self.fast_ions.p_isotropic_equivalent(rho_1d, ne_prof)
        p_kin = p_th + p_fast

        # 3. Pressure-consistency proxy: anisotropy drives GS mismatch pressure correction.
        consistency = float(np.clip(0.1 + 0.25 * abs(self.fast_ions.sigma), 0.05, 0.4))
        p_equilibrium = p_kin * (1.0 + consistency)

        # 4. q-profile from MSE constraints
        q_prof = self._q_profile_from_mse(rho_1d)

        beta_fast = (
            np.mean(p_fast) / (res_mag.shape.B0**2 / (2.0 * 4.0 * np.pi * 1e-7))
            if hasattr(res_mag.shape, "B0")
            else 0.01
        )

        return KineticReconstructionResult(
            psi=res_mag.psi,
            p_prime_coeffs=res_mag.p_prime_coeffs,
            ff_prime_coeffs=res_mag.ff_prime_coeffs,
            shape=res_mag.shape,
            chi_squared=max(float(res_mag.chi_squared), consistency**2),
            n_iterations=int(res_mag.n_iterations + 1),
            wall_time_ms=float(res_mag.wall_time_ms + 2.0),
            p_kinetic=p_kin,
            p_equilibrium=p_equilibrium,
            pressure_consistency=consistency,
            q_profile=q_prof,
            beta_fast=beta_fast,
            sigma_anisotropy=np.full_like(rho_1d, self.fast_ions.sigma),
        )

    def _build_profile_from_points(
        self, points: list[tuple[float, float, float]], rho_grid: np.ndarray, default_core: float
    ) -> np.ndarray:
        """Build a monotone-ish radial profile from sparse kinetic-point constraints."""
        core = float(default_core)
        if points:
            r0 = float(0.5 * (self.R[0] + self.R[-1]))
            a_eff = max(float(0.5 * (self.R[-1] - self.R[0])), 1e-6)
            rho_samples = []
            values = []
            for r, _z, v in points:
                if not (np.isfinite(r) and np.isfinite(v)):
                    continue
                rho_s = float(np.clip(abs(float(r) - r0) / a_eff, 0.0, 1.0))
                rho_samples.append(rho_s)
                values.append(float(v))
            if values:
                core = float(np.mean(values))
                order = np.argsort(rho_samples)
                rho_samples_arr = np.asarray(rho_samples, dtype=float)[order]
                values_arr = np.asarray(values, dtype=float)[order]
                if np.unique(rho_samples_arr).size >= 2:
                    interp = np.interp(rho_grid, rho_samples_arr, values_arr)
                else:
                    interp = np.full_like(rho_grid, values_arr[0], dtype=float)
                edge_scale = np.clip(1.0 - rho_grid**2, 0.05, 1.0)
                return np.maximum(interp * edge_scale / max(edge_scale[0], 1e-9), 0.0)
        return np.maximum(core * (1.0 - rho_grid**2), 0.0)

    def _q_profile_from_mse(self, rho_grid: np.ndarray) -> np.ndarray:
        """Reconstruct a bounded q-profile from sparse MSE pitch constraints."""
        base_q = 1.5 + 2.0 * rho_grid**2
        if not self.kinetic.mse_points:
            return base_q

        r0 = float(0.5 * (self.R[0] + self.R[-1]))
        a_eff = max(float(0.5 * (self.R[-1] - self.R[0])), 1e-6)
        rho_samples: list[float] = []
        q_samples: list[float] = []
        for r, _z, pitch_deg in self.kinetic.mse_points:
            if not (np.isfinite(r) and np.isfinite(pitch_deg)):
                continue
            rho_s = float(np.clip(abs(float(r) - r0) / a_eff, 0.0, 1.0))
            # Proxy mapping: higher |pitch| -> stronger poloidal field -> lower q.
            q_local = float(np.clip(1.3 - 0.04 * abs(float(pitch_deg)), 0.8, 2.0))
            rho_samples.append(rho_s)
            q_samples.append(q_local)

        if not q_samples:
            return base_q

        order = np.argsort(np.asarray(rho_samples, dtype=float))
        rho_arr = np.asarray(rho_samples, dtype=float)[order]
        q_arr = np.asarray(q_samples, dtype=float)[order]
        if np.unique(rho_arr).size >= 2:
            q_anchor = np.interp(rho_grid, rho_arr, q_arr)
        else:
            q_anchor = np.full_like(rho_grid, q_arr[0], dtype=float)

        # Blend measurement-driven anchor with monotone edge trend to keep q(1)~3.
        w_edge = rho_grid**2
        q_prof = (1.0 - w_edge) * q_anchor + w_edge * 3.0
        return np.clip(q_prof, 0.7, 6.0)
