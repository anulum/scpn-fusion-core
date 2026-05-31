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

from scpn_fusion.core._gk_nonlinear_types import (
    NonlinearGKFieldEnergyDiagnostics,
    NonlinearGKInvariantDiagnostics,
    NonlinearGKResult,
    NonlinearGKState,
)


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
        B_par_new = self.magnetic_compression_solve(f_new) if self.cfg.electromagnetic else None
        return NonlinearGKState(
            f=f_new, phi=phi_new, time=t0 + dt, A_par=A_par_new, B_par=B_par_new
        )

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

    def heat_flux_spectra(
        self, state: NonlinearGKState
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ion and electron heat-flux spectra over the retained kx/ky grid."""
        phi = state.phi

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val

        ky_pos = self.ky > 1e-10
        ky_vals = self.ky[ky_pos]

        def heat_flux_spectrum(f_s: NDArray[np.complex128]) -> NDArray[np.float64]:
            pressure = np.sum(energy * f_s, axis=(-2, -1)) * self.dvpar * self.dmu
            spectrum = np.zeros((self.cfg.n_kx, self.cfg.n_ky), dtype=np.float64)
            if ky_vals.size:
                flux_k = (
                    1j
                    * ky_vals[None, :, None]
                    * np.conj(phi[:, ky_pos, :])
                    * pressure[:, ky_pos, :]
                )
                spectrum[:, ky_pos] = np.real(np.sum(flux_k, axis=-1))
            return spectrum

        Q_i_kxky = heat_flux_spectrum(state.f[0])
        Q_e_kxky = heat_flux_spectrum(state.f[1]) if self.cfg.kinetic_electrons else 0.5 * Q_i_kxky
        return Q_i_kxky, Q_e_kxky

    def compute_fluxes(self, state: NonlinearGKState) -> tuple[float, float]:
        """Ion and electron heat flux in gyro-Bohm units."""
        Q_i_kxky, Q_e_kxky = self.heat_flux_spectra(state)
        Q_i = float(np.sum(Q_i_kxky))
        Q_e = float(np.sum(Q_e_kxky))

        return Q_i, Q_e

    def phi_rms(self, state: NonlinearGKState) -> float:
        """RMS electrostatic potential amplitude across the spectral grid."""
        return float(np.sqrt(np.mean(np.abs(state.phi) ** 2)))

    def zonal_rms(self, state: NonlinearGKState) -> float:
        """RMS amplitude of zonal modes (k_y = 0)."""
        ky0_idx = np.argmin(np.abs(self.ky))
        return float(np.sqrt(np.mean(np.abs(state.phi[:, ky0_idx, :]) ** 2)))

    def zonal_flow_energy(self, state: NonlinearGKState) -> float:
        """Electrostatic zonal-flow energy retained in the k_y = 0 field slice."""
        ky0_idx = np.argmin(np.abs(self.ky))
        zonal_phi = state.phi[:, ky0_idx, :]
        kperp_weight = 1.0 + self.kperp2[:, ky0_idx, None]
        return 0.5 * float(np.sum(kperp_weight * np.abs(zonal_phi) ** 2) * self.dtheta)

    def particle_free_energy(self, state: NonlinearGKState) -> float:
        """Distribution-function free energy for conservation diagnostics."""
        self.validate_state(state)
        return float(np.sum(np.abs(state.f) ** 2) * self.dvpar * self.dmu * self.dtheta)

    def particle_free_energy_spectra(self, state: NonlinearGKState) -> NDArray[np.float64]:
        """Species-resolved particle free-energy spectra over retained kx/ky modes."""
        self.validate_state(state)
        spectra = np.sum(np.abs(state.f) ** 2, axis=(3, 4, 5)) * (
            self.dvpar * self.dmu * self.dtheta
        )
        return np.asarray(spectra, dtype=np.float64)

    def field_energy_spectra(
        self, state: NonlinearGKState
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Electromagnetic field-energy spectra over retained kx/ky modes."""
        self.validate_state(state)
        kperp_weight = 1.0 + self.kperp2
        beta = max(float(self.cfg.beta_e), 1e-30)
        phi_energy = 0.5 * np.asarray(
            np.sum(kperp_weight[:, :, None] * np.abs(state.phi) ** 2, axis=-1) * self.dtheta,
            dtype=np.float64,
        )
        a_parallel_energy = np.zeros_like(phi_energy)
        if state.A_par is not None:
            a_parallel_energy = 0.5 * np.asarray(
                np.sum(kperp_weight[:, :, None] * np.abs(state.A_par) ** 2, axis=-1)
                * self.dtheta
                / beta,
                dtype=np.float64,
            )
        b_parallel_energy = np.zeros_like(phi_energy)
        if state.B_par is not None:
            b_parallel_energy = 0.5 * np.asarray(
                np.sum(np.abs(state.B_par) ** 2, axis=-1) * self.dtheta / beta,
                dtype=np.float64,
            )
        return phi_energy, a_parallel_energy, b_parallel_energy

    def field_energy(self, state: NonlinearGKState) -> NonlinearGKFieldEnergyDiagnostics:
        """Electromagnetic field energy across phi, A_parallel, and B_parallel."""
        phi_energy_kxky, a_parallel_energy_kxky, b_parallel_energy_kxky = self.field_energy_spectra(
            state
        )
        phi_energy = float(np.sum(phi_energy_kxky))
        a_parallel_energy = float(np.sum(a_parallel_energy_kxky))
        b_parallel_energy = float(np.sum(b_parallel_energy_kxky))
        total = phi_energy + a_parallel_energy + b_parallel_energy
        finite = bool(
            np.isfinite(phi_energy)
            and np.isfinite(a_parallel_energy)
            and np.isfinite(b_parallel_energy)
            and np.isfinite(total)
        )
        return NonlinearGKFieldEnergyDiagnostics(
            phi=phi_energy,
            A_parallel=a_parallel_energy,
            B_parallel=b_parallel_energy,
            total=total,
            finite=finite,
        )

    def total_energy(self, state: NonlinearGKState) -> float:
        """Total particle plus electromagnetic field energy."""
        return self.particle_free_energy(state) + self.field_energy(state).total

    def nonlinear_invariant_diagnostics(
        self, state: NonlinearGKState
    ) -> NonlinearGKInvariantDiagnostics:
        """Check discrete nonlinear E x B free-energy and dealiasing invariants."""
        c = self.cfg
        active_species = c.n_species if c.kinetic_electrons else min(c.n_species, 1)
        production = 0.0
        norm_f_sq = 0.0
        norm_b_sq = 0.0
        high_k_max = 0.0

        for species_idx in range(active_species):
            f_s = state.f[species_idx]
            bracket = self.exb_bracket(state.phi, f_s)
            production += float(np.real(np.sum(np.conj(f_s) * bracket)))
            norm_f_sq += float(np.sum(np.abs(f_s) ** 2))
            norm_b_sq += float(np.sum(np.abs(bracket) ** 2))
            if np.any(~self.dealias_mask):
                high_k = np.abs(bracket[~self.dealias_mask, ...])
                if high_k.size:
                    high_k_max = max(high_k_max, float(np.max(high_k)))

        phase_volume = self.dvpar * self.dmu * self.dtheta
        production *= phase_volume
        norm_scale = max(np.sqrt(norm_f_sq * norm_b_sq) * phase_volume, 1e-30)
        relative = abs(production) / norm_scale
        finite = bool(np.isfinite(production) and np.isfinite(relative) and np.isfinite(high_k_max))
        passes = bool(finite and abs(production) <= 1e-8 and high_k_max <= 1e-12)
        return NonlinearGKInvariantDiagnostics(
            exb_free_energy_production=production,
            exb_relative_free_energy_production=float(relative),
            dealiased_high_k_max_abs=high_k_max,
            finite=finite,
            passes=passes,
        )

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
        B_par = self.magnetic_compression_solve(f) if c.electromagnetic else None
        return NonlinearGKState(f=f, phi=phi, time=0.0, A_par=A_par, B_par=B_par)

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
        A_par = self.ampere_solve(f) if c.electromagnetic else None
        B_par = self.magnetic_compression_solve(f) if c.electromagnetic else None
        return NonlinearGKState(f=f, phi=phi, time=0.0, A_par=A_par, B_par=B_par)

    def run(self, state: NonlinearGKState | None = None) -> NonlinearGKResult:
        """Run the nonlinear simulation."""
        c = self.cfg
        if state is None:
            state = self.init_state()

        n_saves = c.n_steps // c.save_interval + 1
        Q_i_t: NDArray[np.float64] = np.zeros(n_saves)
        Q_e_t: NDArray[np.float64] = np.zeros(n_saves)
        Q_i_kxky_t: NDArray[np.float64] = np.zeros((n_saves, c.n_kx, c.n_ky))
        Q_e_kxky_t: NDArray[np.float64] = np.zeros((n_saves, c.n_kx, c.n_ky))
        phi_rms_t: NDArray[np.float64] = np.zeros(n_saves)
        zonal_rms_t: NDArray[np.float64] = np.zeros(n_saves)
        zonal_flow_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        particle_free_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        particle_free_energy_species_kxky_t: NDArray[np.float64] = np.zeros(
            (n_saves, c.n_species, c.n_kx, c.n_ky)
        )
        phi_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        A_parallel_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        B_parallel_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        phi_energy_kxky_t: NDArray[np.float64] = np.zeros((n_saves, c.n_kx, c.n_ky))
        A_parallel_energy_kxky_t: NDArray[np.float64] = np.zeros((n_saves, c.n_kx, c.n_ky))
        B_parallel_energy_kxky_t: NDArray[np.float64] = np.zeros((n_saves, c.n_kx, c.n_ky))
        total_energy_t: NDArray[np.float64] = np.zeros(n_saves)
        exb_free_energy_production_t: NDArray[np.float64] = np.zeros(n_saves)
        exb_relative_free_energy_production_t: NDArray[np.float64] = np.zeros(n_saves)
        dealiased_high_k_max_abs_t: NDArray[np.float64] = np.zeros(n_saves)
        nonlinear_invariant_pass_t: NDArray[np.bool_] = np.zeros(n_saves, dtype=np.bool_)
        time_t: NDArray[np.float64] = np.zeros(n_saves)
        save_idx = 0

        for step in range(c.n_steps):
            dt = self._cfl_dt(state)
            state = self._rk4_step(state, dt)

            if not np.all(np.isfinite(state.f)):
                _logger.warning("NaN at step %d, t=%.3f", step, state.time)
                break

            if step % c.save_interval == 0 and save_idx < n_saves:
                Q_i_kxky, Q_e_kxky = self.heat_flux_spectra(state)
                Q_i = float(np.sum(Q_i_kxky))
                Q_e = float(np.sum(Q_e_kxky))
                Q_i_t[save_idx] = Q_i
                Q_e_t[save_idx] = Q_e
                Q_i_kxky_t[save_idx] = Q_i_kxky
                Q_e_kxky_t[save_idx] = Q_e_kxky
                phi_rms_t[save_idx] = self.phi_rms(state)
                zonal_rms_t[save_idx] = self.zonal_rms(state)
                zonal_flow_energy_t[save_idx] = self.zonal_flow_energy(state)
                particle_energy_species_kxky = self.particle_free_energy_spectra(state)
                particle_energy = float(np.sum(particle_energy_species_kxky))
                phi_energy_kxky, A_parallel_energy_kxky, B_parallel_energy_kxky = (
                    self.field_energy_spectra(state)
                )
                particle_free_energy_t[save_idx] = particle_energy
                particle_free_energy_species_kxky_t[save_idx] = particle_energy_species_kxky
                phi_energy_kxky_t[save_idx] = phi_energy_kxky
                A_parallel_energy_kxky_t[save_idx] = A_parallel_energy_kxky
                B_parallel_energy_kxky_t[save_idx] = B_parallel_energy_kxky
                phi_energy_t[save_idx] = float(np.sum(phi_energy_kxky))
                A_parallel_energy_t[save_idx] = float(np.sum(A_parallel_energy_kxky))
                B_parallel_energy_t[save_idx] = float(np.sum(B_parallel_energy_kxky))
                total_energy_t[save_idx] = (
                    particle_energy
                    + phi_energy_t[save_idx]
                    + A_parallel_energy_t[save_idx]
                    + B_parallel_energy_t[save_idx]
                )
                invariant = self.nonlinear_invariant_diagnostics(state)
                exb_free_energy_production_t[save_idx] = invariant.exb_free_energy_production
                exb_relative_free_energy_production_t[save_idx] = (
                    invariant.exb_relative_free_energy_production
                )
                dealiased_high_k_max_abs_t[save_idx] = invariant.dealiased_high_k_max_abs
                nonlinear_invariant_pass_t[save_idx] = invariant.passes
                time_t[save_idx] = state.time
                save_idx += 1

        Q_i_t = np.asarray(Q_i_t[:save_idx], dtype=np.float64)
        Q_e_t = np.asarray(Q_e_t[:save_idx], dtype=np.float64)
        Q_i_kxky_t = np.asarray(Q_i_kxky_t[:save_idx], dtype=np.float64)
        Q_e_kxky_t = np.asarray(Q_e_kxky_t[:save_idx], dtype=np.float64)
        phi_rms_t = np.asarray(phi_rms_t[:save_idx], dtype=np.float64)
        zonal_rms_t = np.asarray(zonal_rms_t[:save_idx], dtype=np.float64)
        zonal_flow_energy_t = np.asarray(zonal_flow_energy_t[:save_idx], dtype=np.float64)
        particle_free_energy_t = np.asarray(particle_free_energy_t[:save_idx], dtype=np.float64)
        particle_free_energy_species_kxky_t = np.asarray(
            particle_free_energy_species_kxky_t[:save_idx], dtype=np.float64
        )
        phi_energy_t = np.asarray(phi_energy_t[:save_idx], dtype=np.float64)
        A_parallel_energy_t = np.asarray(A_parallel_energy_t[:save_idx], dtype=np.float64)
        B_parallel_energy_t = np.asarray(B_parallel_energy_t[:save_idx], dtype=np.float64)
        phi_energy_kxky_t = np.asarray(phi_energy_kxky_t[:save_idx], dtype=np.float64)
        A_parallel_energy_kxky_t = np.asarray(A_parallel_energy_kxky_t[:save_idx], dtype=np.float64)
        B_parallel_energy_kxky_t = np.asarray(B_parallel_energy_kxky_t[:save_idx], dtype=np.float64)
        total_energy_t = np.asarray(total_energy_t[:save_idx], dtype=np.float64)
        exb_free_energy_production_t = np.asarray(
            exb_free_energy_production_t[:save_idx], dtype=np.float64
        )
        exb_relative_free_energy_production_t = np.asarray(
            exb_relative_free_energy_production_t[:save_idx], dtype=np.float64
        )
        dealiased_high_k_max_abs_t = np.asarray(
            dealiased_high_k_max_abs_t[:save_idx], dtype=np.float64
        )
        nonlinear_invariant_pass_t = np.asarray(
            nonlinear_invariant_pass_t[:save_idx], dtype=np.bool_
        )
        time_t = np.asarray(time_t[:save_idx], dtype=np.float64)

        n_half = max(len(Q_i_t) // 2, 1)
        chi_i = float(np.mean(Q_i_t[n_half:])) if len(Q_i_t) > 0 else 0.0
        chi_e = float(np.mean(Q_e_t[n_half:])) if len(Q_e_t) > 0 else 0.0
        late = slice(n_half, None)
        saturated_Q_i_kxky = (
            np.mean(Q_i_kxky_t[late], axis=0)
            if len(Q_i_kxky_t) > 0
            else np.zeros((c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_Q_e_kxky = (
            np.mean(Q_e_kxky_t[late], axis=0)
            if len(Q_e_kxky_t) > 0
            else np.zeros((c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_phi_rms = float(np.mean(phi_rms_t[late])) if len(phi_rms_t) > 0 else 0.0
        saturated_zonal_flow_energy = (
            float(np.mean(zonal_flow_energy_t[late])) if len(zonal_flow_energy_t) > 0 else 0.0
        )
        saturated_particle_free_energy_species_kxky = (
            np.mean(particle_free_energy_species_kxky_t[late], axis=0)
            if len(particle_free_energy_species_kxky_t) > 0
            else np.zeros((c.n_species, c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_phi_energy = float(np.mean(phi_energy_t[late])) if len(phi_energy_t) > 0 else 0.0
        saturated_A_parallel_energy = (
            float(np.mean(A_parallel_energy_t[late])) if len(A_parallel_energy_t) > 0 else 0.0
        )
        saturated_B_parallel_energy = (
            float(np.mean(B_parallel_energy_t[late])) if len(B_parallel_energy_t) > 0 else 0.0
        )
        saturated_phi_energy_kxky = (
            np.mean(phi_energy_kxky_t[late], axis=0)
            if len(phi_energy_kxky_t) > 0
            else np.zeros((c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_A_parallel_energy_kxky = (
            np.mean(A_parallel_energy_kxky_t[late], axis=0)
            if len(A_parallel_energy_kxky_t) > 0
            else np.zeros((c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_B_parallel_energy_kxky = (
            np.mean(B_parallel_energy_kxky_t[late], axis=0)
            if len(B_parallel_energy_kxky_t) > 0
            else np.zeros((c.n_kx, c.n_ky), dtype=np.float64)
        )
        saturated_total_energy = (
            float(np.mean(total_energy_t[late])) if len(total_energy_t) > 0 else 0.0
        )
        chi_i_gB = chi_i / max(c.R_L_Ti, 0.01)
        converged = bool(save_idx > 1 and np.all(np.isfinite(Q_i_t)))

        return NonlinearGKResult(
            chi_i=chi_i,
            chi_e=chi_e,
            chi_i_gB=chi_i_gB,
            Q_i_t=Q_i_t,
            Q_e_t=Q_e_t,
            Q_i_kxky_t=Q_i_kxky_t,
            Q_e_kxky_t=Q_e_kxky_t,
            saturated_Q_i_kxky=saturated_Q_i_kxky,
            saturated_Q_e_kxky=saturated_Q_e_kxky,
            saturated_phi_rms=saturated_phi_rms,
            saturated_zonal_flow_energy=saturated_zonal_flow_energy,
            saturated_particle_free_energy_species_kxky=saturated_particle_free_energy_species_kxky,
            saturated_phi_energy=saturated_phi_energy,
            saturated_A_parallel_energy=saturated_A_parallel_energy,
            saturated_B_parallel_energy=saturated_B_parallel_energy,
            saturated_phi_energy_kxky=saturated_phi_energy_kxky,
            saturated_A_parallel_energy_kxky=saturated_A_parallel_energy_kxky,
            saturated_B_parallel_energy_kxky=saturated_B_parallel_energy_kxky,
            saturated_total_energy=saturated_total_energy,
            phi_rms_t=phi_rms_t,
            zonal_rms_t=zonal_rms_t,
            zonal_flow_energy_t=zonal_flow_energy_t,
            particle_free_energy_t=particle_free_energy_t,
            particle_free_energy_species_kxky_t=particle_free_energy_species_kxky_t,
            phi_energy_t=phi_energy_t,
            A_parallel_energy_t=A_parallel_energy_t,
            B_parallel_energy_t=B_parallel_energy_t,
            phi_energy_kxky_t=phi_energy_kxky_t,
            A_parallel_energy_kxky_t=A_parallel_energy_kxky_t,
            B_parallel_energy_kxky_t=B_parallel_energy_kxky_t,
            total_energy_t=total_energy_t,
            exb_free_energy_production_t=exb_free_energy_production_t,
            exb_relative_free_energy_production_t=exb_relative_free_energy_production_t,
            dealiased_high_k_max_abs_t=dealiased_high_k_max_abs_t,
            nonlinear_invariant_pass_t=nonlinear_invariant_pass_t,
            kx_rhos=np.asarray(self.kx, dtype=np.float64),
            ky_rhos=np.asarray(self.ky, dtype=np.float64),
            theta_rad=np.asarray(self.theta, dtype=np.float64),
            vpar_vth=np.asarray(self.vpar, dtype=np.float64),
            mu_normalized=np.asarray(self.mu, dtype=np.float64),
            time=time_t,
            converged=converged,
            final_state=state,
        )


__all__ = ["NonlinearGKTimeMixin"]
