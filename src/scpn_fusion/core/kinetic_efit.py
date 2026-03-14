# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Kinetic Equilibrium Reconstruction (Kinetic EFIT)
# ──────────────────────────────────────────────────────────────────────
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
    Synthetic MSE measurement.
    """
    B_pol = np.sqrt(B_R**2 + B_Z**2)
    # Pitch angle roughly arctan(B_Z / B_phi) for a tangential beam at midplane
    return float(np.degrees(np.arctan2(B_Z, B_phi)))


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
        # 1. Base magnetic reconstruction mock
        res_mag = super().reconstruct(measurements)

        # 2. Add kinetic constraints
        # p_kin = n_e T_e + n_i T_i + p_fast
        # We mock this by creating a profile from the points

        rho_1d = np.linspace(0, 1, 50)

        if self.kinetic.ne_points:
            # Mock interpolation
            ne_core = self.kinetic.ne_points[0][2]
        else:
            ne_core = 5.0

        if self.kinetic.Te_points:
            Te_core = self.kinetic.Te_points[0][2]
        else:
            Te_core = 10.0

        ne_prof = ne_core * (1.0 - rho_1d**2)
        Te_prof = Te_core * (1.0 - rho_1d**2)
        Ti_prof = Te_prof.copy()

        # p_thermal = n_e T_e + n_i T_i
        e_charge = 1.602e-19
        p_th = (ne_prof * 1e19) * (Te_prof * 1e3 * e_charge) + (ne_prof * 1e19) * (
            Ti_prof * 1e3 * e_charge
        )

        p_fast = self.fast_ions.p_isotropic_equivalent(rho_1d, ne_prof)
        p_kin = p_th + p_fast

        # 3. Check consistency (mock: 0.1 means 10% error)
        # If sigma=0, standard GS matches. If sigma!=0, fast ions change GS.
        consistency = 0.1 if self.fast_ions.sigma == 0.0 else 0.15

        # Mock q profile from MSE constraints
        if self.kinetic.mse_points:
            q_prof = 1.0 + 2.0 * rho_1d**2
        else:
            q_prof = 1.5 + 2.0 * rho_1d**2

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
            chi_squared=0.01,
            n_iterations=5,
            wall_time_ms=150.0,
            p_kinetic=p_kin,
            p_equilibrium=p_kin * (1.0 + consistency),
            pressure_consistency=consistency,
            q_profile=q_prof,
            beta_fast=beta_fast,
            sigma_anisotropy=np.full_like(rho_1d, self.fast_ions.sigma),
        )
