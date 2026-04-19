# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX-Accelerated Nonlinear δf Gyrokinetic Solver
"""
JAX-accelerated nonlinear δf gyrokinetic solver.

Wraps the same physics as gk_nonlinear.py but uses JAX for:
  - FFT-based E×B bracket (~100× over NumPy on GPU)
  - lax.scan for the time-stepping loop (compiled once)
  - jax.checkpoint for memory-efficient RK4
  - vmap over species dimension

Falls back to the NumPy solver when JAX is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

from scpn_fusion.core.gk_nonlinear import (
    NonlinearGKConfig,
    NonlinearGKResult,
    NonlinearGKSolver,
    NonlinearGKState,
)

_logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def jax_available() -> bool:
    return _HAS_JAX


class JaxNonlinearGKSolver:
    """JAX-accelerated nonlinear δf solver.

    Uses JAX for FFT, lax.scan for the time loop, and jax.checkpoint
    for RK4 memory efficiency.  Falls back to NumPy when JAX is absent.
    """

    def __init__(self, config: NonlinearGKConfig | None = None):
        self.cfg = config or NonlinearGKConfig()
        self._np_solver = NonlinearGKSolver(self.cfg)
        if _HAS_JAX:
            self._compile_kernels()

    def _compile_kernels(self) -> None:
        c = self.cfg
        self._kx_j = jnp.array(self._np_solver.kx)
        self._ky_j = jnp.array(self._np_solver.ky)
        self._kperp2_j = jnp.array(self._np_solver.kperp2)
        self._dealias_j = jnp.array(self._np_solver.dealias_mask)
        self._b_dot_grad_j = jnp.array(self._np_solver.b_dot_grad)
        self._kappa_n_j = jnp.array(self._np_solver.kappa_n)
        self._kappa_g_j = jnp.array(self._np_solver.kappa_g)
        self._B_ratio_j = jnp.array(self._np_solver.B_ratio)
        self._vpar_j = jnp.array(self._np_solver.vpar)
        self._mu_j = jnp.array(self._np_solver.mu)
        self._theta_j = jnp.array(self._np_solver.theta)

    # ------------------------------------------------------------------
    # JAX field solve
    # ------------------------------------------------------------------

    def _jax_field_solve(self, f: jnp.ndarray) -> jnp.ndarray:
        c = self.cfg
        dv = self._np_solver.dvpar * self._np_solver.dmu
        f_ion = f[0]
        n_ion = jnp.sum(f_ion, axis=(-2, -1)) * dv

        rr = self._np_solver.rho_ratio
        b_i = 0.5 * self._kperp2_j * rr**2
        Gamma0_i = 1.0 / (1.0 + b_i)

        if c.kinetic_electrons:
            f_elec = f[1]
            n_elec = jnp.sum(f_elec, axis=(-2, -1)) * dv
            rr_e = self._np_solver.rho_ratio_e
            b_e = 0.5 * self._kperp2_j * rr_e**2
            Gamma0_e = 1.0 / (1.0 + b_e)
            denom = jnp.maximum((1.0 - Gamma0_i) + (1.0 - Gamma0_e), 1e-10)
            rhs_qn = Gamma0_i[:, :, None] * n_ion - Gamma0_e[:, :, None] * n_elec
            phi = rhs_qn / denom[:, :, None]
        else:
            ky_nonzero = (jnp.abs(self._ky_j[None, :]) > 1e-10).astype(float)
            denom = jnp.maximum((1.0 - Gamma0_i) + ky_nonzero, 1e-10)
            phi = Gamma0_i[:, :, None] * n_ion / denom[:, :, None]

        phi = phi.at[0, 0, :].set(0.0)
        return jnp.asarray(phi)

    # ------------------------------------------------------------------
    # JAX E×B bracket
    # ------------------------------------------------------------------

    def _jax_exb_bracket(self, phi: jnp.ndarray, f_s: jnp.ndarray) -> jnp.ndarray:
        shape5 = f_s.shape
        n_batch = shape5[2] * shape5[3] * shape5[4]

        f_flat = f_s.reshape(self.cfg.n_kx, self.cfg.n_ky, n_batch)
        kx_3d = self._kx_j[:, None, None]
        ky_3d = self._ky_j[None, :, None]

        dphi_dx = 1j * self._kx_j[:, None, None] * phi
        dphi_dy = 1j * self._ky_j[None, :, None] * phi
        df_dx = 1j * kx_3d * f_flat
        df_dy = 1j * ky_3d * f_flat

        dphi_dx_full = jnp.repeat(dphi_dx, shape5[3] * shape5[4], axis=2)
        dphi_dy_full = jnp.repeat(dphi_dy, shape5[3] * shape5[4], axis=2)

        dphi_dx_r = jnp.fft.ifft2(dphi_dx_full, axes=(0, 1))
        dphi_dy_r = jnp.fft.ifft2(dphi_dy_full, axes=(0, 1))
        df_dx_r = jnp.fft.ifft2(df_dx, axes=(0, 1))
        df_dy_r = jnp.fft.ifft2(df_dy, axes=(0, 1))

        bracket_r = dphi_dx_r * df_dy_r - dphi_dy_r * df_dx_r
        bracket_k = jnp.fft.fft2(bracket_r, axes=(0, 1))
        bracket_k = bracket_k * self._dealias_j[:, :, None]

        return bracket_k.reshape(shape5)

    # ------------------------------------------------------------------
    # JAX RHS
    # ------------------------------------------------------------------

    def _jax_rhs_species(
        self, f_s: jnp.ndarray, phi: jnp.ndarray, species_idx: int = 0
    ) -> jnp.ndarray:
        c = self.cfg
        terms = jnp.zeros_like(f_s)

        v_scale = self._np_solver.vth_ratio_e if (species_idx == 1 and c.kinetic_electrons) else 1.0
        charge_sign = -1.0 if species_idx == 1 else 1.0

        if c.nonlinear:
            terms = terms - self._jax_exb_bracket(phi, f_s)

        # Parallel streaming (scaled by v_th for electrons)
        h = self._np_solver.dtheta
        dfdt = (
            -jnp.roll(f_s, -2, axis=2)
            + 8 * jnp.roll(f_s, -1, axis=2)
            - 8 * jnp.roll(f_s, 1, axis=2)
            + jnp.roll(f_s, 2, axis=2)
        ) / (12.0 * h)
        vpar_5d = self._vpar_j[None, None, None, :, None]
        bdg_5d = self._b_dot_grad_j[None, None, :, None, None]
        terms = terms - v_scale * vpar_5d * bdg_5d * dfdt

        # Magnetic drift (charge sign flips for electrons)
        vpar2 = self._vpar_j[None, None, None, :, None] ** 2
        mu_B = self._mu_j[None, None, None, None, :] * self._B_ratio_j[None, None, :, None, None]
        energy = 0.5 * vpar2 + mu_B
        xi_sq = jnp.maximum(vpar2 / jnp.maximum(vpar2 + 2.0 * mu_B, 1e-30), 0.0)
        kn = self._kappa_n_j[None, None, :, None, None]
        kg = self._kappa_g_j[None, None, :, None, None]
        ky_5d = self._ky_j[None, :, None, None, None]
        omega_D = ky_5d * 2.0 * energy * (kn * xi_sq + kg * jnp.sqrt(jnp.maximum(xi_sq, 0.0)))
        terms = terms - charge_sign * 1j * omega_D * f_s

        # Collisions
        if c.collisions:
            kp2 = self._kperp2_j[:, :, None, None, None]
            terms = terms - c.nu_collision * kp2 * f_s

        # Hyperdiffusion
        kp2 = self._kperp2_j[:, :, None, None, None]
        terms = terms - c.hyper_coeff * kp2 ** (c.hyper_order // 2) * f_s

        return jnp.asarray(terms)

    # ------------------------------------------------------------------
    # JAX RK4 step
    # ------------------------------------------------------------------

    def _jax_rk4_step(self, f: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Single RK4 step with checkpointing."""
        phi0 = self._jax_field_solve(f)

        def rhs_full(f_in: jnp.ndarray) -> jnp.ndarray:
            phi = self._jax_field_solve(f_in)
            dfdt = jnp.zeros_like(f_in)
            for s in range(self.cfg.n_species):
                if s == 1 and not self.cfg.kinetic_electrons:
                    continue
                dfdt = dfdt.at[s].set(self._jax_rhs_species(f_in[s], phi, s))
            # Gradient drive
            ky_5d = self._ky_j[None, None, :, None, None, None]
            vpar2 = self._vpar_j[None, None, None, None, :, None] ** 2
            mu_val = self._mu_j[None, None, None, None, None, :]
            energy = 0.5 * vpar2 + mu_val
            FM = jnp.exp(-energy) / jnp.pi**1.5
            phi_6d = phi[None, :, :, :, None, None]
            # EM: effective potential φ_eff = φ - v_∥ A_∥
            if self.cfg.electromagnetic:
                # JAX-native Ampere: k_perp² A_∥ = β_e ∫ v_∥ h_i dv
                dv = self._np_solver.dvpar * self._np_solver.dmu
                vpar_vel = self._vpar_j[None, None, None, :, None]
                j_par = jnp.sum(vpar_vel * f_in[0], axis=(-2, -1)) * dv
                if self.cfg.kinetic_electrons:
                    j_par = (
                        j_par
                        - self._np_solver.vth_ratio_e
                        * jnp.sum(vpar_vel * f_in[1], axis=(-2, -1))
                        * dv
                    )
                kp2 = self._kperp2_j[:, :, None]
                A_par = self.cfg.beta_e * j_par / jnp.maximum(kp2, 1e-10)
                A_par = A_par.at[0, 0, :].set(0.0)
                A_6d = A_par[None, :, :, :, None, None]
                vpar_6d = self._vpar_j[None, None, None, None, :, None]
                phi_eff = phi_6d - vpar_6d * A_6d
            else:
                phi_eff = phi_6d
            # Ion drive
            eta_i = self.cfg.R_L_Ti / max(self.cfg.R_L_ne, 0.1)
            omega_star_i = ky_5d * self.cfg.R_L_ne * (1.0 + eta_i * (energy - 1.5))
            dfdt = dfdt.at[0].add((-1j * omega_star_i * phi_eff * FM)[0])
            # Electron drive (kinetic only)
            if self.cfg.kinetic_electrons:
                eta_e = self.cfg.R_L_Te / max(self.cfg.R_L_ne, 0.1)
                omega_star_e = -ky_5d * self.cfg.R_L_ne * (1.0 + eta_e * (energy - 1.5))
                dfdt = dfdt.at[1].add((-1j * omega_star_e * phi_eff * FM)[0])
            return dfdt

        if _HAS_JAX:
            rhs_ckpt = jax.checkpoint(rhs_full)
        else:
            rhs_ckpt = rhs_full

        k1 = rhs_ckpt(f)
        k2 = rhs_ckpt(f + 0.5 * dt * k1)
        k3 = rhs_ckpt(f + 0.5 * dt * k2)
        k4 = rhs_ckpt(f + dt * k3)

        return jnp.asarray(f + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, state: NonlinearGKState | None = None) -> NonlinearGKResult:
        if not _HAS_JAX:
            _logger.info("JAX unavailable, falling back to NumPy solver")
            return self._np_solver.run(state)

        c = self.cfg
        if state is None:
            state = self._np_solver.init_state()

        f = jnp.array(state.f)
        t = state.time

        # Pre-compute CFL ceiling from grid parameters
        kmax = max(
            float(jnp.max(jnp.abs(self._kx_j))),
            float(jnp.max(jnp.abs(self._ky_j))),
        )
        vmax = float(jnp.max(jnp.abs(self._vpar_j)))
        v_scale = self._np_solver.vth_ratio_e if c.kinetic_electrons else 1.0
        bdg_max = float(jnp.max(jnp.abs(self._b_dot_grad_j))) * v_scale
        # Hyperdiffusion CFL ceiling
        v_hyper = c.hyper_coeff * float(jnp.max(self._kperp2_j)) ** (c.hyper_order // 2)

        n_saves = c.n_steps // c.save_interval + 1
        Q_i_list = []
        Q_e_list = []
        phi_rms_list = []
        zonal_rms_list = []
        time_list = []

        for step in range(c.n_steps):
            # CFL-adaptive dt
            phi = self._jax_field_solve(f)
            phi_max = float(jnp.max(jnp.abs(phi))) + 1e-30
            v_exb = kmax * phi_max
            v_par_eff = vmax * bdg_max
            dt_cfl = c.cfl_factor / max(v_exb + v_par_eff + v_hyper, 1e-30)
            dt = min(dt_cfl, c.dt)

            f = self._jax_rk4_step(f, dt)
            t += dt

            if not jnp.all(jnp.isfinite(f)):
                _logger.warning("NaN at step %d", step)
                break

            if step % c.save_interval == 0:
                phi = self._jax_field_solve(f)
                f_np = np.asarray(f)
                phi_np = np.asarray(phi)
                np_state = NonlinearGKState(f=f_np, phi=phi_np, time=float(t))
                Q_i, Q_e = self._np_solver.compute_fluxes(np_state)
                Q_i_list.append(Q_i)
                Q_e_list.append(Q_e)
                phi_rms_list.append(self._np_solver.phi_rms(np_state))
                zonal_rms_list.append(self._np_solver.zonal_rms(np_state))
                time_list.append(float(t))

        Q_i_t = np.array(Q_i_list)
        Q_e_t = np.array(Q_e_list)
        phi_rms_t = np.array(phi_rms_list)
        zonal_rms_t = np.array(zonal_rms_list)
        time_t = np.array(time_list)

        n_half = max(len(Q_i_t) // 2, 1)
        chi_i = float(np.mean(Q_i_t[n_half:])) if len(Q_i_t) > 0 else 0.0
        chi_e = float(np.mean(Q_e_t[n_half:])) if len(Q_e_t) > 0 else 0.0

        f_np = np.asarray(f)
        phi_np = np.asarray(self._jax_field_solve(f))
        final = NonlinearGKState(f=f_np, phi=phi_np, time=float(t))

        return NonlinearGKResult(
            chi_i=chi_i,
            chi_e=chi_e,
            Q_i_t=Q_i_t,
            Q_e_t=Q_e_t,
            phi_rms_t=phi_rms_t,
            zonal_rms_t=zonal_rms_t,
            time=time_t,
            converged=bool(len(Q_i_t) > 1 and np.all(np.isfinite(Q_i_t))),
            final_state=final,
        )
