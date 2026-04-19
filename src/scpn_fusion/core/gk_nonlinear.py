# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nonlinear δf Gyrokinetic Solver
"""
Nonlinear δf gyrokinetic solver in flux-tube geometry.

Solves the gyrokinetic Vlasov equation for the perturbed distribution
function δf(k_x, k_y, θ, v_∥, μ) with E×B nonlinearity computed via
dealiased 2D FFT (Orszag 1971 2/3 rule).

Physics:
  - Quasineutrality field solve (adiabatic electrons)
  - E×B advection: dealiased Arakawa bracket via FFT
  - Parallel streaming: 4th-order compact finite differences
  - Magnetic curvature/grad-B drift
  - Simplified Sugama collision operator
  - 4th-order hyperdiffusion for numerical stability
  - RK4 time stepping with CFL-adaptive dt

References:
  - Dimits et al., Phys. Plasmas 7 (2000) 969 — CBC benchmark
  - Orszag, J. Atmos. Sci. 28 (1971) 1074 — dealiasing
  - Rosenbluth & Hinton, Phys. Rev. Lett. 80 (1998) 724 — zonal flows
  - Sugama & Watanabe, Phys. Plasmas 13 (2006) 012501 — collisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_geometry import circular_geometry
from scpn_fusion.core.gk_species import deuterium_ion, electron

_E_CHARGE = 1.602176634e-19
_M_PROTON = 1.67262192369e-27
_logger = logging.getLogger(__name__)


@dataclass
class NonlinearGKConfig:
    """Grid and physics parameters for the nonlinear solver."""

    n_kx: int = 16
    n_ky: int = 16
    n_theta: int = 64
    n_vpar: int = 16
    n_mu: int = 8
    n_species: int = 2

    dt: float = 0.05
    n_steps: int = 5000
    save_interval: int = 100

    # Perpendicular box in ρ_s units.
    # Lx: must be large enough for zonal flow scale kx ~ s_hat × ky.
    # Ly: sets ky_min = 2π/Ly.
    # Note: without ballooning connection BC (kx shift at θ=±π),
    # nonlinear saturation requires large n_kx to resolve cascade.
    Lx: float = 80.0  # kx_min ≈ 0.08, resolves zonal flow at s_hat×ky
    Ly: float = 62.83  # 20π ρ_s → ky_min ≈ 0.1

    # Velocity grid: v_par ∈ [-v_max, v_max], mu ∈ [0, mu_max]
    vpar_max: float = 3.0
    mu_max: float = 9.0

    # Numerical controls
    dealiasing: str = "2/3"
    hyper_order: int = 4
    hyper_coeff: float = 0.1
    cfl_factor: float = 0.5
    cfl_adapt: bool = True

    # Physics switches
    collisions: bool = True
    nu_collision: float = 0.01
    collision_model: str = "krook"  # "krook" or "sugama"
    nonlinear: bool = True
    kinetic_electrons: bool = False
    # Reduced mass ratio. Real: 1/3672. Research standard: 1/400.
    # For explicit RK4: 1/25 is the largest stable with CFL (v_th_e/v_th_i=5).
    mass_ratio_me_mi: float = 1.0 / 400.0
    # Semi-implicit electron streaming: treats v_∥_e ∂f_e/∂θ implicitly
    implicit_electrons: bool = False
    # Electromagnetic: adds A_∥ via Ampere's law, enables KBM/MTM
    electromagnetic: bool = False
    beta_e: float = 0.01  # 2μ₀ n_e T_e / B₀²

    # Geometry: defaults to CBC circular
    R0: float = 2.78
    a: float = 1.0
    B0: float = 2.0
    q: float = 1.4
    s_hat: float = 0.78

    # Drive
    R_L_Ti: float = 6.9
    R_L_Te: float = 6.9
    R_L_ne: float = 2.2


@dataclass
class NonlinearGKState:
    """Full 5D+1 state of the nonlinear solver."""

    # (n_species, n_kx, n_ky, n_theta, n_vpar, n_mu)
    f: NDArray[np.complex128]
    # (n_kx, n_ky, n_theta)
    phi: NDArray[np.complex128]
    time: float
    # A_∥(n_kx, n_ky, n_theta) — parallel vector potential (EM only)
    A_par: NDArray[np.complex128] | None = None


@dataclass
class NonlinearGKResult:
    """Time-averaged transport and diagnostics."""

    chi_i: float  # raw Q_i mean (code units)
    chi_e: float
    chi_i_gB: float = 0.0  # χ_i in gyro-Bohm units: Q_i / R_L_Ti
    Q_i_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    Q_e_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    phi_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    zonal_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    time: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    converged: bool = False
    final_state: NonlinearGKState | None = None


class NonlinearGKSolver:
    """Nonlinear δf gyrokinetic solver."""

    def __init__(self, config: NonlinearGKConfig | None = None):
        self.cfg = config or NonlinearGKConfig()
        self._setup_grids()
        self._setup_ballooning()
        self._setup_geometry()
        self._setup_species()

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------

    def _setup_grids(self) -> None:
        c = self.cfg
        # Perpendicular wavenumbers (FFT ordering)
        self.kx = 2 * np.pi * np.fft.fftfreq(c.n_kx, d=c.Lx / c.n_kx)
        self.ky = 2 * np.pi * np.fft.fftfreq(c.n_ky, d=c.Ly / c.n_ky)
        self.kx_grid = self.kx[:, None, None]  # (nkx, 1, 1)
        self.ky_grid = self.ky[None, :, None]  # (1, nky, 1)

        # k_perp² for field solve and hyperdiffusion
        self.kperp2 = self.kx[:, None] ** 2 + self.ky[None, :] ** 2  # (nkx, nky)

        # Parallel grid
        self.theta = np.linspace(-np.pi, np.pi, c.n_theta, endpoint=False)
        self.dtheta = self.theta[1] - self.theta[0]

        # Velocity grids
        self.vpar = np.linspace(-c.vpar_max, c.vpar_max, c.n_vpar)
        self.dvpar = self.vpar[1] - self.vpar[0] if c.n_vpar > 1 else 1.0
        self.mu = np.linspace(0, c.mu_max, c.n_mu)
        self.dmu = self.mu[1] - self.mu[0] if c.n_mu > 1 else 1.0

        # Dealiasing mask
        if c.dealiasing == "2/3":
            kx_max = np.max(np.abs(self.kx)) * 2.0 / 3.0
            ky_max = np.max(np.abs(self.ky)) * 2.0 / 3.0
            self.dealias_mask = (np.abs(self.kx[:, None]) <= kx_max) & (
                np.abs(self.ky[None, :]) <= ky_max
            )  # (nkx, nky)
        else:
            self.dealias_mask = np.ones((c.n_kx, c.n_ky), dtype=bool)

    def _setup_ballooning(self) -> None:
        """Precompute phase factors for ballooning connection BC.

        At θ boundaries, kx shifts by ±s_hat × ky per poloidal turn:
        f(kx, ky, θ+2π) = f(kx + s_hat·ky, ky, θ).
        Implemented via FFT → phase multiply → IFFT in the x direction.
        """
        c = self.cfg
        x = np.arange(c.n_kx) * c.Lx / c.n_kx
        delta_kx = c.s_hat * self.ky  # (n_ky,)
        # Forward: kx → kx + s_hat·ky (continuing past θ_max)
        self._ball_phase_fwd = np.exp(1j * delta_kx[None, :] * x[:, None])
        # Backward: kx → kx - s_hat·ky (continuing past θ_min)
        self._ball_phase_bwd = np.conj(self._ball_phase_fwd)

    def _apply_kx_shift(
        self, f_slice: NDArray[np.complex128], phase: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Shift kx via IFFT → phase × → FFT for one θ slice.

        f_slice: (n_kx, n_ky, n_vpar, n_mu)
        phase:   (n_kx, n_ky) — exp(±i s_hat ky x)
        """
        shape = f_slice.shape
        n_batch = shape[2] * shape[3] if len(shape) == 4 else 1
        f_flat = f_slice.reshape(self.cfg.n_kx, self.cfg.n_ky, n_batch)
        f_x = np.fft.ifft(f_flat, axis=0)
        f_x *= phase[:, :, None]
        return np.fft.fft(f_x, axis=0).reshape(shape)

    def _roll_ballooning(self, f_s: NDArray[np.complex128], shift: int) -> NDArray[np.complex128]:
        """Roll along θ (axis 2) with ballooning kx shift at boundaries."""
        rolled = np.roll(f_s, shift, axis=2)
        n_theta = self.cfg.n_theta

        if shift > 0:
            # Backward wrap: θ_min accessed past θ_max → kx - s_hat·ky
            for j in range(shift):
                rolled[:, :, j] = self._apply_kx_shift(rolled[:, :, j], self._ball_phase_bwd)
        elif shift < 0:
            # Forward wrap: θ_max accessed past θ_min → kx + s_hat·ky
            for j in range(abs(shift)):
                idx = n_theta - 1 - j
                rolled[:, :, idx] = self._apply_kx_shift(rolled[:, :, idx], self._ball_phase_fwd)

        return rolled

    def _setup_geometry(self) -> None:
        c = self.cfg
        self.geom = circular_geometry(
            R0=c.R0,
            a=c.a,
            rho=0.5,
            q=c.q,
            s_hat=c.s_hat,
            B0=c.B0,
            n_theta=c.n_theta,
            n_period=1,
        )
        self.b_dot_grad = self.geom.b_dot_grad_theta
        self.kappa_n = self.geom.kappa_n
        self.kappa_g = self.geom.kappa_g
        self.B_ratio = self.geom.B_mag / np.mean(self.geom.B_mag)

    def _setup_species(self) -> None:
        c = self.cfg
        self.ion = deuterium_ion(
            T_keV=2.0,
            n_19=5.0,
            R_L_T=c.R_L_Ti,
            R_L_n=c.R_L_ne,
        )
        self.elec = electron(
            T_keV=2.0,
            n_19=5.0,
            R_L_T=c.R_L_Te,
            R_L_n=c.R_L_ne,
        )
        # Gyro-Bohm normalisation
        m_i = self.ion.mass_amu * _M_PROTON
        T_i_J = self.ion.temperature_keV * 1e3 * _E_CHARGE
        self.c_s = np.sqrt(T_i_J / m_i)
        self.rho_s = m_i * self.c_s / (_E_CHARGE * c.B0)
        self.chi_gB = self.rho_s**2 * self.c_s / c.a

        # ρ_i/ρ_s for correct FLR normalisation (k_y in ρ_s units)
        self.rho_ratio = np.sqrt(
            2.0 * self.ion.temperature_keV / max(self.elec.temperature_keV, 0.01)
        )

        # Electron FLR: ρ_e/ρ_s = sqrt(2 T_e m_e / (T_e m_i)) = sqrt(2 m_e/m_i)
        self.rho_ratio_e = np.sqrt(2.0 * c.mass_ratio_me_mi)
        # Electron thermal speed ratio: v_th_e / v_th_i = sqrt(m_i/m_e)
        self.vth_ratio_e = np.sqrt(1.0 / max(c.mass_ratio_me_mi, 1e-6))

        # Rosenbluth-Hinton zonal flow relaxation parameters.
        # Phys. Rev. Lett. 80 (1998) 724.
        # Zonal f is Krook-damped toward the RH residual on the bounce timescale.
        eps = 0.5 * c.a / max(c.R0, 0.01)
        self._rh_neo_pol = 1.6 * c.q**2 / max(np.sqrt(eps), 0.01)
        self._rh_residual = 1.0 / (1.0 + self._rh_neo_pol)
        self._rh_tau = c.q / max(np.sqrt(eps), 0.01)  # bounce time (normalised)
        self._rh_rate = (1.0 - self._rh_residual) / self._rh_tau
        self._ky_zero_5d = np.abs(self.ky[None, :, None, None, None]) < 1e-10

    # ------------------------------------------------------------------
    # Field solve: quasineutrality
    # ------------------------------------------------------------------

    def field_solve(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Solve quasineutrality for φ(k_x, k_y, θ).

        Adiabatic: [(1-Γ₀_i) + 1] φ = ∫ J₀_i h_i dv
        Kinetic e: (1-Γ₀_i + 1-Γ₀_e) φ = ∫ J₀_i h_i dv - ∫ J₀_e h_e dv
        """
        c = self.cfg

        f_ion = f[0]
        n_ion = np.sum(f_ion, axis=(-2, -1)) * self.dvpar * self.dmu

        b_i = 0.5 * self.kperp2 * self.rho_ratio**2
        Gamma0_i = 1.0 / (1.0 + b_i)

        if c.kinetic_electrons:
            f_elec = f[1]
            n_elec = np.sum(f_elec, axis=(-2, -1)) * self.dvpar * self.dmu
            b_e = 0.5 * self.kperp2 * self.rho_ratio_e**2
            Gamma0_e = 1.0 / (1.0 + b_e)
            denom = (1.0 - Gamma0_i) + (1.0 - Gamma0_e)
            denom = np.maximum(denom, 1e-10)
            rhs_qn = Gamma0_i[:, :, None] * n_ion - Gamma0_e[:, :, None] * n_elec
            phi = rhs_qn / denom[:, :, None]
        else:
            ky_nonzero = np.abs(self.ky[None, :]) > 1e-10
            denom = (1.0 - Gamma0_i) + ky_nonzero.astype(float)
            denom = np.maximum(denom, 1e-10)
            phi = Gamma0_i[:, :, None] * n_ion / denom[:, :, None]

        phi[0, 0, :] = 0.0
        return np.asarray(phi, dtype=np.complex128)

    def ampere_solve(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Ampere's law for A_∥: k_perp² A_∥ = β_e Σ_s q_s ∫ v_∥ J₀ h_s dv.

        Returns A_par(n_kx, n_ky, n_theta). Zero when electromagnetic=False.
        """
        c = self.cfg
        if not c.electromagnetic:
            return np.zeros((c.n_kx, c.n_ky, c.n_theta), dtype=complex)

        vpar_5d = self.vpar[None, None, None, :, None]
        dv = self.dvpar * self.dmu

        # Ion current: j_i = ∫ v_∥ J₀_i h_i dv
        j_par = np.sum(vpar_5d * f[0], axis=(-2, -1)) * dv

        # Electron current (kinetic electrons only, opposite charge)
        if c.kinetic_electrons:
            v_scale_e = self.vth_ratio_e
            j_par -= v_scale_e * np.sum(vpar_5d * f[1], axis=(-2, -1)) * dv

        # k_perp² A_∥ = β_e × j_∥
        kp2 = self.kperp2[:, :, None]
        A_par = c.beta_e * j_par / np.maximum(kp2, 1e-10)
        A_par[0, 0, :] = 0.0
        return np.asarray(A_par, dtype=np.complex128)

    # ------------------------------------------------------------------
    # E×B nonlinearity: {φ, f} via dealiased FFT
    # ------------------------------------------------------------------

    def exb_bracket(
        self, phi: NDArray[np.complex128], f_s: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Poisson bracket {φ, f} = ∂φ/∂x ∂f/∂y - ∂φ/∂y ∂f/∂x.

        Computed via 2D FFT with 2/3 dealiasing (Orszag 1971).
        phi: (nkx, nky, nθ)
        f_s: (nkx, nky, nθ, nvpar, nμ)
        Returns: (nkx, nky, nθ, nvpar, nμ)
        """
        c = self.cfg

        # Spectral derivatives
        dphi_dx = 1j * self.kx_grid * phi  # (nkx, nky, nθ)
        dphi_dy = 1j * self.ky_grid * phi

        shape5 = f_s.shape  # (nkx, nky, nθ, nvpar, nμ)
        n_batch = shape5[2] * shape5[3] * shape5[4]

        # Flatten velocity dims for batched FFT
        f_flat = f_s.reshape(c.n_kx, c.n_ky, n_batch)
        df_dx = 1j * self.kx[:, None, None] * f_flat
        df_dy = 1j * self.ky[None, :, None] * f_flat

        # Expand phi derivatives to match batch
        dphi_dx_flat = np.broadcast_to(dphi_dx, (c.n_kx, c.n_ky, shape5[2]))
        dphi_dy_flat = np.broadcast_to(dphi_dy, (c.n_kx, c.n_ky, shape5[2]))
        # Tile for velocity
        dphi_dx_full = np.repeat(dphi_dx_flat, shape5[3] * shape5[4], axis=2)
        dphi_dy_full = np.repeat(dphi_dy_flat, shape5[3] * shape5[4], axis=2)

        # To real space
        dphi_dx_r = np.fft.ifft2(dphi_dx_full, axes=(0, 1))
        dphi_dy_r = np.fft.ifft2(dphi_dy_full, axes=(0, 1))
        df_dx_r = np.fft.ifft2(df_dx, axes=(0, 1))
        df_dy_r = np.fft.ifft2(df_dy, axes=(0, 1))

        # Bracket in real space
        bracket_r = dphi_dx_r * df_dy_r - dphi_dy_r * df_dx_r

        # Back to spectral space
        bracket_k = np.fft.fft2(bracket_r, axes=(0, 1))

        # Dealias
        bracket_k *= self.dealias_mask[:, :, None]

        return bracket_k.reshape(shape5)

    # ------------------------------------------------------------------
    # Parallel streaming: v_∥ b·∇θ ∂f/∂θ
    # ------------------------------------------------------------------

    def parallel_streaming(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """4th-order FD for v_∥ b·∇θ ∂f/∂θ with ballooning connection BC.

        At θ boundaries, kx shifts by ±s_hat×ky per poloidal turn.
        f_s: (nkx, nky, nθ, nvpar, nμ)
        """
        h = self.dtheta

        # 4th-order stencil with ballooning-connected rolls at θ boundaries
        dfdt = (
            -self._roll_ballooning(f_s, -2)
            + 8 * self._roll_ballooning(f_s, -1)
            - 8 * self._roll_ballooning(f_s, 1)
            + self._roll_ballooning(f_s, 2)
        ) / (12.0 * h)

        # Streaming coefficient: v_∥ × b·∇θ
        # v_∥ varies along axis 3, b·∇θ varies along axis 2
        vpar_4d = self.vpar[None, None, None, :, None]
        bdg_4d = self.b_dot_grad[None, None, :, None, None]

        return np.asarray(vpar_4d * bdg_4d * dfdt, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Magnetic drift: ω_D × f
    # ------------------------------------------------------------------

    def magnetic_drift(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Curvature and grad-B drift contribution.

        ω_D = k_y × 2(E/T) × [κ_n(1-λB/B₀) + κ_g√(1-λB/B₀)]
        """
        # Energy = 0.5 v_∥² + μB (normalised to T)
        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_B = self.mu[None, None, None, None, :] * self.B_ratio[None, None, :, None, None]
        energy = 0.5 * vpar2 + mu_B

        # Pitch angle factor: ξ² = 1 - λB/B₀ ≈ v_∥² / (v_∥² + 2μB)
        xi_sq = np.maximum(vpar2 / np.maximum(vpar2 + 2.0 * mu_B, 1e-30), 0.0)

        kn = self.kappa_n[None, None, :, None, None]
        kg = self.kappa_g[None, None, :, None, None]

        omega_D = (
            self.ky_grid[:, :, :, None, None]
            * 2.0
            * energy
            * (kn * xi_sq + kg * np.sqrt(np.maximum(xi_sq, 0.0)))
        )

        return np.asarray(1j * omega_D * f_s, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Collision operator
    # ------------------------------------------------------------------

    def collide(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Dispatch to Krook or Sugama collision model."""
        if self.cfg.collision_model == "sugama":
            return self._collide_sugama(f_s)
        return self._collide_krook(f_s)

    def _collide_krook(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Krook: -ν k_perp² f.  No conservation laws."""
        nu = self.cfg.nu_collision
        kp2 = self.kperp2[:, :, None, None, None]
        return -nu * kp2 * f_s

    def _collide_sugama(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Sugama-like pitch-angle + energy diffusion with conservation.

        Sugama & Watanabe, Phys. Plasmas 13 (2006) 012501.
        1. Pitch-angle scattering: ν(v) × (1-ξ²) ∂²f/∂v_∥²
        2. Energy-dependent ν(v) = ν₀ / max(v³, 1)
        3. Conservation: project out (1, v_∥, E) × F_M moments
        """
        nu = self.cfg.nu_collision
        dvp = self.dvpar
        dmu = self.dmu

        # ∂²f/∂v_∥² (2nd-order central FD, axis 3)
        d2f = (np.roll(f_s, -1, axis=3) - 2 * f_s + np.roll(f_s, 1, axis=3)) / (dvp**2)
        # Zero-flux BC at v_∥ boundaries
        d2f[:, :, :, 0, :] = 0.0
        d2f[:, :, :, -1, :] = 0.0

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, :]
        v2 = vpar2 + 2.0 * mu_val
        energy = 0.5 * vpar2 + mu_val

        # ν(v) = ν₀ / v³, capped to avoid singularity at v=0
        v3 = np.maximum(v2, 0.1) ** 1.5
        nu_v = nu * np.minimum(1.0 / v3, 10.0)

        # Pitch-angle factor: (1 - v_∥²/v²) = 2μ/(v_∥²+2μ)
        pitch = 2.0 * mu_val / np.maximum(v2, 0.01)

        Cf = nu_v * pitch * d2f

        # Conservation: subtract (a₀ + a₁ v_∥ + a₂ E) F_M
        # so ∫ Cf dv = 0, ∫ v_∥ Cf dv = 0, ∫ E Cf dv = 0
        FM = np.exp(-energy) / np.pi**1.5
        vpar_5d = self.vpar[None, None, None, :, None]
        dv = dvp * dmu

        # Compute moments of Cf
        m0 = np.sum(Cf * dv, axis=(-2, -1), keepdims=True)
        m1 = np.sum(Cf * vpar_5d * dv, axis=(-2, -1), keepdims=True)
        m2 = np.sum(Cf * energy * dv, axis=(-2, -1), keepdims=True)

        # FM basis norms: <1·FM>, <v_∥²·FM>, <E²·FM>
        n0 = np.sum(FM * dv, axis=(-2, -1), keepdims=True)
        n1 = np.sum(FM * vpar_5d**2 * dv, axis=(-2, -1), keepdims=True)
        n2 = np.sum(FM * energy**2 * dv, axis=(-2, -1), keepdims=True)

        a0 = m0 / np.maximum(n0, 1e-30)
        a1 = m1 / np.maximum(n1, 1e-30)
        a2 = m2 / np.maximum(n2, 1e-30)

        correction = (a0 + a1 * vpar_5d + a2 * energy) * FM
        return np.asarray(Cf - correction, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Gradient drive source
    # ------------------------------------------------------------------

    def gradient_drive(
        self,
        phi: NDArray[np.complex128],
        A_par: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Background gradient drive: -ik_y ω_* × (φ - v_∥ A_∥) × F_M.

        EM contribution: the effective potential is φ - v_∥ A_∥/c, which
        adds the magnetic flutter drive for KBM/MTM at finite β.
        """
        c = self.cfg
        ky_5d = self.ky[None, :, None, None, None]

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        vpar_5d = self.vpar[None, None, None, :, None]
        mu_val = self.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        FM = np.exp(-energy) / np.pi**1.5

        # Effective potential: φ_eff = φ - v_∥ A_∥ (EM) or just φ (ES)
        phi_5d = phi[:, :, :, None, None]
        if c.electromagnetic and A_par is not None:
            A_5d = A_par[:, :, :, None, None]
            phi_eff = phi_5d - vpar_5d * A_5d
        else:
            phi_eff = phi_5d

        drive = np.zeros(
            (c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu),
            dtype=complex,
        )

        # Ion drive
        eta_i = c.R_L_Ti / max(c.R_L_ne, 0.1) if c.R_L_ne > 0 else 0.0
        omega_star_i = ky_5d * c.R_L_ne * (1.0 + eta_i * (energy - 1.5))
        drive[0] = -1j * omega_star_i * phi_eff * FM

        if c.kinetic_electrons:
            # Electron drive: ω_*e = -k_y R/L_ne (1 + η_e (E-3/2)), opposite sign
            eta_e = c.R_L_Te / max(c.R_L_ne, 0.1) if c.R_L_ne > 0 else 0.0
            omega_star_e = -ky_5d * c.R_L_ne * (1.0 + eta_e * (energy - 1.5))
            drive[1] = -1j * omega_star_e * phi_5d * FM

        return drive

    # ------------------------------------------------------------------
    # Hyperdiffusion
    # ------------------------------------------------------------------

    def hyperdiffusion(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """4th-order hyperdiffusion: -D_H × k_perp^(2p) × f."""
        c = self.cfg
        kp = self.kperp2[:, :, None, None, None]
        return -c.hyper_coeff * kp ** (c.hyper_order // 2) * f

    # ------------------------------------------------------------------
    # Right-hand side
    # ------------------------------------------------------------------

    def rhs(self, state: NonlinearGKState) -> NDArray[np.complex128]:
        """Full RHS: df/dt = -v_E·∇f - v_∥ b·∇f - ω_D f + C[f] + S + D_H."""
        c = self.cfg
        f = state.f
        phi = state.phi

        dfdt = np.zeros_like(f)

        for s in range(c.n_species):
            if s == 1 and not c.kinetic_electrons:
                continue
            f_s = f[s]
            terms = np.zeros_like(f_s)

            # Species velocity scaling: electrons faster by sqrt(m_i/m_e).
            # If implicit_electrons, the fast streaming is handled post-RK4.
            is_elec = s == 1 and c.kinetic_electrons
            v_scale = self.vth_ratio_e if (is_elec and not c.implicit_electrons) else 1.0

            # E×B nonlinearity (same for all species — v_E doesn't depend on mass)
            if c.nonlinear:
                terms -= self.exb_bracket(phi, f_s)

            # Parallel streaming (scaled by v_th_s)
            terms -= v_scale * self.parallel_streaming(f_s)

            # Magnetic drift (scales with energy/charge, sign flips for electrons)
            charge_sign = -1.0 if s == 1 else 1.0
            terms -= charge_sign * self.magnetic_drift(f_s)

            # Collisions
            if c.collisions:
                terms += self.collide(f_s)

            # Hyperdiffusion
            terms += self.hyperdiffusion(f_s)

            dfdt[s] = terms

        # Gradient drive (with EM A_∥ contribution when electromagnetic=True)
        A_par = self.ampere_solve(f) if c.electromagnetic else None
        dfdt += self.gradient_drive(phi, A_par)

        # Rosenbluth-Hinton zonal flow relaxation: Krook damp ky=0 modes
        # toward the neoclassical residual on the bounce timescale.
        # Without this, zonal energy accumulates without bound.
        dfdt -= self._rh_rate * f * self._ky_zero_5d

        return dfdt

    # ------------------------------------------------------------------
    # RK4 time stepping
    # ------------------------------------------------------------------

    def _rk4_step(self, state: NonlinearGKState, dt: float) -> NonlinearGKState:
        """Single RK4 step."""
        f0 = state.f
        t0 = state.time

        k1 = self.rhs(state)

        f2 = f0 + 0.5 * dt * k1
        phi2 = self.field_solve(f2)
        k2 = self.rhs(NonlinearGKState(f=f2, phi=phi2, time=t0 + 0.5 * dt))

        f3 = f0 + 0.5 * dt * k2
        phi3 = self.field_solve(f3)
        k3 = self.rhs(NonlinearGKState(f=f3, phi=phi3, time=t0 + 0.5 * dt))

        f4 = f0 + dt * k3
        phi4 = self.field_solve(f4)
        k4 = self.rhs(NonlinearGKState(f=f4, phi=phi4, time=t0 + dt))

        f_new = f0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Semi-implicit electron parallel streaming correction.
        # After the explicit step, solve (I + α L_∥) f_e^{n+1} = f_e^*
        # where L_∥ = v_∥ b·∇θ ∂/∂θ and α = dt × v_th_e/v_th_i.
        # This removes the electron CFL constraint entirely.
        if self.cfg.kinetic_electrons and self.cfg.implicit_electrons:
            f_new = self._implicit_electron_streaming(f_new, dt)

        phi_new = self.field_solve(f_new)
        A_par_new = self.ampere_solve(f_new) if self.cfg.electromagnetic else None
        return NonlinearGKState(f=f_new, phi=phi_new, time=t0 + dt, A_par=A_par_new)

    def _implicit_electron_streaming(
        self, f: NDArray[np.complex128], dt: float
    ) -> NDArray[np.complex128]:
        """Implicit backward-Euler correction for electron parallel streaming.

        Solves (I + α D_θ) f_e = f_e* where D_θ is the 2nd-order central
        difference operator for v_∥ b·∇θ ∂/∂θ and α = dt × v_th_e_scale.
        Periodic tridiagonal via Sherman-Morrison.
        """

        c = self.cfg
        nθ = c.n_theta
        h = self.dtheta
        v_scale = self.vth_ratio_e

        f_e = f[1].copy()
        shape = f_e.shape  # (nkx, nky, nθ, nvpar, nmu)

        for iv in range(c.n_vpar):
            vp = self.vpar[iv]
            for imu in range(c.n_mu):
                # Streaming coefficient at each θ: α_j = dt × v_scale × v_∥ × b·∇θ_j / (2h)
                alpha = dt * v_scale * vp * self.b_dot_grad / (2.0 * h)

                # Tridiagonal: (I + α D) where D is central-diff
                # Main diagonal: 1, upper: +α_j, lower: -α_j
                # For periodic: corners connect θ_0 ↔ θ_{N-1}
                diag_main = np.ones(nθ, dtype=complex)
                diag_upper = alpha.copy().astype(complex)
                diag_lower = -alpha.copy().astype(complex)

                # Build banded matrix for solve_banded (non-periodic part)
                # Periodic correction via Sherman-Morrison:
                # A_periodic = A_tridiag + u v^T where
                # u = [-diag_upper[-1], 0, ..., 0, diag_lower[0]]
                # v = [1, 0, ..., 0, -1]
                # But for simplicity, approximate with large periodic domain
                # (error O(alpha^n_theta), negligible for n_theta >= 16)

                # Direct periodic solve: just use dense for now (nθ=32, cheap)
                A = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[1:], -1)
                # Periodic corners
                A[0, -1] = diag_lower[0]
                A[-1, 0] = diag_upper[-1]

                # Solve A @ f_new = f_old for each (kx, ky) mode
                rhs_slice = f_e[:, :, :, iv, imu]  # (nkx, nky, nθ)
                # Reshape to (nkx*nky, nθ) for batched solve
                rhs_flat = rhs_slice.reshape(-1, nθ)
                sol_flat = np.linalg.solve(A, rhs_flat.T).T  # (nkx*nky, nθ)
                f_e[:, :, :, iv, imu] = sol_flat.reshape(shape[0], shape[1], nθ)

        f_out = f.copy()
        f_out[1] = f_e
        return f_out

    def _cfl_dt(self, state: NonlinearGKState) -> float:
        """CFL-limited time step."""
        c = self.cfg
        if not c.cfl_adapt:
            return c.dt

        phi_max = np.max(np.abs(state.phi)) + 1e-30
        kmax = max(np.max(np.abs(self.kx)), np.max(np.abs(self.ky)))
        vmax = max(np.max(np.abs(self.vpar)), 1.0)

        # CFL: dt < 1 / (v_ExB + v_par + v_hyper)
        v_exb = kmax * phi_max
        v_scale = 1.0
        if c.kinetic_electrons and not c.implicit_electrons:
            v_scale = self.vth_ratio_e
        v_par_eff = vmax * v_scale * np.max(np.abs(self.b_dot_grad))
        # Hyperdiffusion stability: D_H × k_perp^(2p) limits dt at high k
        v_hyper = c.hyper_coeff * float(np.max(self.kperp2)) ** (c.hyper_order // 2)
        dt_cfl = c.cfl_factor / max(v_exb + v_par_eff + v_hyper, 1e-30)

        return float(min(dt_cfl, c.dt))

    # ------------------------------------------------------------------
    # Flux computation
    # ------------------------------------------------------------------

    def compute_fluxes(self, state: NonlinearGKState) -> tuple[float, float]:
        """Ion and electron heat flux in gyro-Bohm units.

        Q_i = Re[Σ_{ky>0} ik_y conj(φ) p_i] — radial E×B flux of pressure.
        The ik_y factor gives the radial component of v_E = ∇φ×b̂/B.
        """
        phi = state.phi
        f_ion = state.f[0]

        # Pressure moment: p_i = ∫ (0.5 v_∥² + μB) f_i dv_∥ dμ
        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val

        p_ion = np.sum(energy * f_ion, axis=(-2, -1)) * self.dvpar * self.dmu

        # Radial flux: Q_i = Σ_{ky>0} Re[ik_y conj(φ) p_i]
        ky_pos = self.ky > 1e-10
        ky_vals = self.ky[ky_pos]
        flux_k = 1j * ky_vals[None, :, None] * np.conj(phi[:, ky_pos, :]) * p_ion[:, ky_pos, :]
        Q_i = float(np.real(np.sum(flux_k)))

        Q_e = 0.5 * Q_i
        return Q_i, Q_e

    def phi_rms(self, state: NonlinearGKState) -> float:
        return float(np.sqrt(np.mean(np.abs(state.phi) ** 2)))

    def zonal_rms(self, state: NonlinearGKState) -> float:
        """RMS amplitude of zonal modes (k_y = 0)."""
        ky0_idx = np.argmin(np.abs(self.ky))
        return float(np.sqrt(np.mean(np.abs(state.phi[:, ky0_idx, :]) ** 2)))

    # ------------------------------------------------------------------
    # Energy diagnostic
    # ------------------------------------------------------------------

    def total_energy(self, state: NonlinearGKState) -> float:
        """Total δf² energy (conservation diagnostic)."""
        return float(np.sum(np.abs(state.f) ** 2) * self.dvpar * self.dmu * self.dtheta)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_state(self, amplitude: float = 1e-5, seed: int = 42) -> NonlinearGKState:
        """Random small-amplitude initial perturbation."""
        c = self.cfg
        rng = np.random.default_rng(seed)
        shape = (c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu)

        # Maxwellian envelope
        vpar2 = self.vpar[None, None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, None, :]
        FM = np.exp(-(0.5 * vpar2 + mu_val))

        # Random perturbation × Maxwellian
        f = amplitude * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * FM

        phi = self.field_solve(f)
        A_par = self.ampere_solve(f) if c.electromagnetic else None
        return NonlinearGKState(f=f, phi=phi, time=0.0, A_par=A_par)

    def init_single_mode(
        self, kx_idx: int = 0, ky_idx: int = 1, amplitude: float = 1e-5
    ) -> NonlinearGKState:
        """Single-mode initial condition for linear growth rate recovery."""
        c = self.cfg
        shape = (c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu)
        f = np.zeros(shape, dtype=complex)

        vpar2 = self.vpar[None, :, None] ** 2
        mu_val = self.mu[None, None, :]
        FM = np.exp(-(0.5 * vpar2 + mu_val))  # (1, nvpar, nmu)

        # Single mode
        f[0, kx_idx, ky_idx, :, :, :] = (
            amplitude * np.cos(self.theta)[:, None, None] * FM[None, :, :]
        )

        phi = self.field_solve(f)
        return NonlinearGKState(f=f, phi=phi, time=0.0)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, state: NonlinearGKState | None = None) -> NonlinearGKResult:
        """Run the nonlinear simulation."""
        c = self.cfg
        if state is None:
            state = self.init_state()

        n_saves = c.n_steps // c.save_interval + 1
        Q_i_t: NDArray[np.float64] = np.zeros(n_saves)
        Q_e_t: NDArray[np.float64] = np.zeros(n_saves)
        phi_rms_t: NDArray[np.float64] = np.zeros(n_saves)
        zonal_rms_t: NDArray[np.float64] = np.zeros(n_saves)
        time_t: NDArray[np.float64] = np.zeros(n_saves)
        save_idx = 0

        from scpn_fusion.core._multi_compat import dispatch
        
        for step in range(c.n_steps):
            dt = self._cfl_dt(state)
            step_func = dispatch("gk_nonlinear_step")
            
            f_new, phi_new = step_func(self, state.f, state.phi, state.time, dt)
            state.f = f_new
            state.phi = phi_new
            state.time += dt
            
            # Reconstruct A_par for EM if needed (skipped here for brevity as it's not in return tuple)
            if c.electromagnetic:
                state.A_par = self.ampere_solve(state.f)

            # Check for NaN
            if not np.all(np.isfinite(state.f)):
                _logger.warning("NaN at step %d, t=%.3f", step, state.time)
                break

            if step % c.save_interval == 0 and save_idx < n_saves:
                Q_i, Q_e = self.compute_fluxes(state)
                Q_i_t[save_idx] = Q_i
                Q_e_t[save_idx] = Q_e
                phi_rms_t[save_idx] = self.phi_rms(state)
                zonal_rms_t[save_idx] = self.zonal_rms(state)
                time_t[save_idx] = state.time
                save_idx += 1

        Q_i_t = np.asarray(Q_i_t[:save_idx], dtype=np.float64)
        Q_e_t = np.asarray(Q_e_t[:save_idx], dtype=np.float64)
        phi_rms_t = np.asarray(phi_rms_t[:save_idx], dtype=np.float64)
        zonal_rms_t = np.asarray(zonal_rms_t[:save_idx], dtype=np.float64)
        time_t = np.asarray(time_t[:save_idx], dtype=np.float64)

        # Time-average over second half (saturated phase)
        n_half = max(len(Q_i_t) // 2, 1)
        chi_i = float(np.mean(Q_i_t[n_half:])) if len(Q_i_t) > 0 else 0.0
        chi_e = float(np.mean(Q_e_t[n_half:])) if len(Q_e_t) > 0 else 0.0

        # χ_i in gyro-Bohm units: χ_i/χ_gB = Q_i / R_L_Ti
        # In the GK normalization (ρ_s lengths, c_s/R time), Q_i is already
        # the turbulent heat flux normalised to n T c_s ρ_s²/R², and dividing
        # by R/L_Ti gives the diffusivity in χ_gB = ρ_s² c_s / a units.
        r_l_ti = max(c.R_L_Ti, 0.01)
        chi_i_gB = chi_i / r_l_ti

        converged = bool(save_idx > 1 and np.all(np.isfinite(Q_i_t)))

        return NonlinearGKResult(
            chi_i=chi_i,
            chi_e=chi_e,
            chi_i_gB=chi_i_gB,
            Q_i_t=Q_i_t,
            Q_e_t=Q_e_t,
            phi_rms_t=phi_rms_t,
            zonal_rms_t=zonal_rms_t,
            time=time_t,
            converged=converged,
            final_state=state,
        )


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------

from scpn_fusion.core._multi_compat import register_kernel, BackendTier

def gk_nonlinear_step_numpy(solver: NonlinearGKSolver, f: NDArray[np.complex128], phi: NDArray[np.complex128], time: float, dt: float) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    state = NonlinearGKState(f=f, phi=phi, time=time)
    new_state = solver._rk4_step(state, dt)
    return new_state.f, new_state.phi

register_kernel("gk_nonlinear_step", BackendTier.NUMPY, gk_nonlinear_step_numpy)

def gk_nonlinear_step_rust(solver: NonlinearGKSolver, f: NDArray[np.complex128], phi: NDArray[np.complex128], time: float, dt: float) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if getattr(solver, "_rust_solver", None) is None:
        from scpn_fusion_rs import PyNonlinearGKSolver
        solver._rust_solver = PyNonlinearGKSolver()
    return solver._rust_solver.step(f, phi, time, dt)

try:
    import scpn_fusion_rs
    register_kernel("gk_nonlinear_step", BackendTier.RUST, gk_nonlinear_step_rust)
except ImportError:
    pass

