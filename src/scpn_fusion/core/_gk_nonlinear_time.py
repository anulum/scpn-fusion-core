# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Time stepping, diagnostics, and initialization for nonlinear GK."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKResult, NonlinearGKState


_logger = logging.getLogger(__name__)


class NonlinearGKTimeMixin:
    """Time integration and diagnostics for :class:`NonlinearGKSolver`."""

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

        if self.cfg.kinetic_electrons and self.cfg.implicit_electrons:
            f_new = self._implicit_electron_streaming(f_new, dt)

        phi_new = self.field_solve(f_new)
        A_par_new = self.ampere_solve(f_new) if self.cfg.electromagnetic else None
        return NonlinearGKState(f=f_new, phi=phi_new, time=t0 + dt, A_par=A_par_new)

    def _implicit_electron_streaming(
        self, f: NDArray[np.complex128], dt: float
    ) -> NDArray[np.complex128]:
        """Implicit backward-Euler correction for electron parallel streaming."""
        c = self.cfg
        ntheta = c.n_theta
        h = self.dtheta
        v_scale = self.vth_ratio_e

        f_e = f[1].copy()
        shape = f_e.shape

        for iv in range(c.n_vpar):
            vp = self.vpar[iv]
            for imu in range(c.n_mu):
                alpha = dt * v_scale * vp * self.b_dot_grad / (2.0 * h)
                diag_main = np.ones(ntheta, dtype=complex)
                diag_upper = alpha.copy().astype(complex)
                diag_lower = -alpha.copy().astype(complex)

                A = np.diag(diag_main) + np.diag(diag_upper[:-1], 1) + np.diag(diag_lower[1:], -1)
                A[0, -1] = diag_lower[0]
                A[-1, 0] = diag_upper[-1]

                rhs_slice = f_e[:, :, :, iv, imu]
                rhs_flat = rhs_slice.reshape(-1, ntheta)
                sol_flat = np.linalg.solve(A, rhs_flat.T).T
                f_e[:, :, :, iv, imu] = sol_flat.reshape(shape[0], shape[1], ntheta)

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

        v_exb = kmax * phi_max
        v_scale = 1.0
        if c.kinetic_electrons and not c.implicit_electrons:
            v_scale = self.vth_ratio_e
        v_par_eff = vmax * v_scale * np.max(np.abs(self.b_dot_grad))
        v_hyper = c.hyper_coeff * float(np.max(self.kperp2)) ** (c.hyper_order // 2)
        dt_cfl = c.cfl_factor / max(v_exb + v_par_eff + v_hyper, 1e-30)
        return float(min(dt_cfl, c.dt))

    def compute_fluxes(self, state: NonlinearGKState) -> tuple[float, float]:
        """Ion and electron heat flux in gyro-Bohm units."""
        phi = state.phi
        f_ion = state.f[0]

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        p_ion = np.sum(energy * f_ion, axis=(-2, -1)) * self.dvpar * self.dmu

        ky_pos = self.ky > 1e-10
        ky_vals = self.ky[ky_pos]
        flux_k = 1j * ky_vals[None, :, None] * np.conj(phi[:, ky_pos, :]) * p_ion[:, ky_pos, :]
        Q_i = float(np.real(np.sum(flux_k)))
        Q_e = 0.5 * Q_i
        return Q_i, Q_e

    def phi_rms(self, state: NonlinearGKState) -> float:
        """RMS electrostatic potential amplitude across the spectral grid."""
        return float(np.sqrt(np.mean(np.abs(state.phi) ** 2)))

    def zonal_rms(self, state: NonlinearGKState) -> float:
        """RMS amplitude of zonal modes (k_y = 0)."""
        ky0_idx = np.argmin(np.abs(self.ky))
        return float(np.sqrt(np.mean(np.abs(state.phi[:, ky0_idx, :]) ** 2)))

    def total_energy(self, state: NonlinearGKState) -> float:
        """Total delta-f squared energy for conservation diagnostics."""
        return float(np.sum(np.abs(state.f) ** 2) * self.dvpar * self.dmu * self.dtheta)

    def init_state(self, amplitude: float = 1e-5, seed: int = 42) -> NonlinearGKState:
        """Random small-amplitude initial perturbation."""
        c = self.cfg
        rng = np.random.default_rng(seed)
        shape = (c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu)

        vpar2 = self.vpar[None, None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, None, :]
        FM = np.exp(-(0.5 * vpar2 + mu_val))
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
        FM = np.exp(-(0.5 * vpar2 + mu_val))

        f[0, kx_idx, ky_idx, :, :, :] = (
            amplitude * np.cos(self.theta)[:, None, None] * FM[None, :, :]
        )

        phi = self.field_solve(f)
        return NonlinearGKState(f=f, phi=phi, time=0.0)

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

        for step in range(c.n_steps):
            dt = self._cfl_dt(state)
            state = self._rk4_step(state, dt)

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

        n_half = max(len(Q_i_t) // 2, 1)
        chi_i = float(np.mean(Q_i_t[n_half:])) if len(Q_i_t) > 0 else 0.0
        chi_e = float(np.mean(Q_e_t[n_half:])) if len(Q_e_t) > 0 else 0.0
        chi_i_gB = chi_i / max(c.R_L_Ti, 0.01)
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


__all__ = ["NonlinearGKTimeMixin"]
