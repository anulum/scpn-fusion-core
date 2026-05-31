# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Field solves and differential operators for nonlinear GK."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKState


class NonlinearGKOperatorsMixin:
    """Physics operators used by the nonlinear GK time stepper."""

    def field_solve(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Solve quasineutrality for phi(k_x, k_y, theta)."""
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
        """Ampere's law for A_parallel; zero when electromagnetic mode is off."""
        c = self.cfg
        if not c.electromagnetic:
            return np.zeros((c.n_kx, c.n_ky, c.n_theta), dtype=complex)

        vpar_5d = self.vpar[None, None, None, :, None]
        dv = self.dvpar * self.dmu
        j_par = np.sum(vpar_5d * f[0], axis=(-2, -1)) * dv

        if c.kinetic_electrons:
            v_scale_e = self.vth_ratio_e
            j_par -= v_scale_e * np.sum(vpar_5d * f[1], axis=(-2, -1)) * dv

        kp2 = self.kperp2[:, :, None]
        A_par = c.beta_e * j_par / np.maximum(kp2, 1e-10)
        A_par[0, 0, :] = 0.0
        return np.asarray(A_par, dtype=np.complex128)

    def magnetic_compression_solve(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Perpendicular pressure-balance solve for B_parallel."""
        c = self.cfg
        if not c.electromagnetic:
            return np.zeros((c.n_kx, c.n_ky, c.n_theta), dtype=complex)

        mu_5d = self.mu[None, None, None, None, :]
        dv = self.dvpar * self.dmu
        p_perp = np.sum(mu_5d * f[0], axis=(-2, -1)) * dv
        if c.kinetic_electrons:
            p_perp += np.sum(mu_5d * f[1], axis=(-2, -1)) * dv

        denom = 1.0 + self.kperp2[:, :, None]
        B_par = -c.beta_e * p_perp / denom
        B_par[0, 0, :] = 0.0
        return np.asarray(B_par, dtype=np.complex128)

    def exb_bracket(
        self, phi: NDArray[np.complex128], f_s: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Poisson bracket {phi, f} using dealiased 2D FFTs."""
        c = self.cfg
        dphi_dx = 1j * self.kx_grid * phi
        dphi_dy = 1j * self.ky_grid * phi

        shape5 = f_s.shape
        n_batch = shape5[2] * shape5[3] * shape5[4]
        f_flat = f_s.reshape(c.n_kx, c.n_ky, n_batch)
        df_dx = 1j * self.kx[:, None, None] * f_flat
        df_dy = 1j * self.ky[None, :, None] * f_flat

        dphi_dx_flat = np.broadcast_to(dphi_dx, (c.n_kx, c.n_ky, shape5[2]))
        dphi_dy_flat = np.broadcast_to(dphi_dy, (c.n_kx, c.n_ky, shape5[2]))
        dphi_dx_full = np.repeat(dphi_dx_flat, shape5[3] * shape5[4], axis=2)
        dphi_dy_full = np.repeat(dphi_dy_flat, shape5[3] * shape5[4], axis=2)

        dphi_dx_r = np.fft.ifft2(dphi_dx_full, axes=(0, 1))
        dphi_dy_r = np.fft.ifft2(dphi_dy_full, axes=(0, 1))
        df_dx_r = np.fft.ifft2(df_dx, axes=(0, 1))
        df_dy_r = np.fft.ifft2(df_dy, axes=(0, 1))

        bracket_r = dphi_dx_r * df_dy_r - dphi_dy_r * df_dx_r
        bracket_k = np.fft.fft2(bracket_r, axes=(0, 1))
        bracket_k *= self.dealias_mask[:, :, None]
        return bracket_k.reshape(shape5)

    def nonlinear_exb_term(
        self, state: NonlinearGKState, *, return_diagnostics: bool = False
    ) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], object]:
        """Return the conservative nonlinear term ``-{phi, g}`` for active species."""
        self.validate_state(state)
        term = np.zeros_like(state.f)
        active_species = (
            self.cfg.n_species if self.cfg.kinetic_electrons else min(self.cfg.n_species, 1)
        )
        for species_idx in range(active_species):
            term[species_idx] = -self.exb_bracket(state.phi, state.f[species_idx])
        if return_diagnostics:
            return term, self.nonlinear_invariant_diagnostics(state)
        return term

    def parallel_streaming(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Fourth-order theta derivative with ballooning connection BC."""
        h = self.dtheta
        dfdt = (
            -self._roll_ballooning(f_s, -2)
            + 8 * self._roll_ballooning(f_s, -1)
            - 8 * self._roll_ballooning(f_s, 1)
            + self._roll_ballooning(f_s, 2)
        ) / (12.0 * h)

        vpar_4d = self.vpar[None, None, None, :, None]
        bdg_4d = self.b_dot_grad[None, None, :, None, None]
        return np.asarray(vpar_4d * bdg_4d * dfdt, dtype=np.complex128)

    def magnetic_drift(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Curvature and grad-B drift contribution."""
        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_B = self.mu[None, None, None, None, :] * self.B_ratio[None, None, :, None, None]
        energy = 0.5 * vpar2 + mu_B
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

    def collide(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Dispatch to Krook or Sugama-like collision model."""
        if self.cfg.collision_model == "sugama":
            return self._collide_sugama(f_s)
        return self._collide_krook(f_s)

    def _collide_krook(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Krook: -nu k_perp^2 f. No conservation laws."""
        nu = self.cfg.nu_collision
        kp2 = self.kperp2[:, :, None, None, None]
        return -nu * kp2 * f_s

    def _collide_sugama(self, f_s: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Sugama-like pitch-angle plus energy diffusion with conservation."""
        nu = self.cfg.nu_collision
        dvp = self.dvpar
        dmu = self.dmu

        d2f = (np.roll(f_s, -1, axis=3) - 2 * f_s + np.roll(f_s, 1, axis=3)) / (dvp**2)
        d2f[:, :, :, 0, :] = 0.0
        d2f[:, :, :, -1, :] = 0.0

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        mu_val = self.mu[None, None, None, None, :]
        v2 = vpar2 + 2.0 * mu_val
        energy = 0.5 * vpar2 + mu_val

        v3 = np.maximum(v2, 0.1) ** 1.5
        nu_v = nu * np.minimum(1.0 / v3, 10.0)
        pitch = 2.0 * mu_val / np.maximum(v2, 0.01)
        Cf = nu_v * pitch * d2f

        FM = np.exp(-energy) / np.pi**1.5
        vpar_5d = self.vpar[None, None, None, :, None]
        dv = dvp * dmu

        basis = (np.ones_like(energy), vpar_5d, energy)
        moments = np.stack(
            [np.sum(Cf * b * dv, axis=(-2, -1)) for b in basis],
            axis=0,
        )
        gram = np.array(
            [[np.sum(a * b * FM * dv) for b in basis] for a in basis],
            dtype=np.float64,
        )
        coeffs = np.tensordot(np.linalg.inv(gram), moments, axes=(1, 0))
        correction = (
            coeffs[0, ..., None, None]
            + coeffs[1, ..., None, None] * vpar_5d
            + coeffs[2, ..., None, None] * energy
        ) * FM
        return np.asarray(Cf - correction, dtype=np.complex128)

    def gradient_drive(
        self,
        phi: NDArray[np.complex128],
        A_par: NDArray[np.complex128] | None = None,
        B_par: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Background gradient drive with optional electromagnetic contribution."""
        c = self.cfg
        ky_5d = self.ky[None, :, None, None, None]

        vpar2 = self.vpar[None, None, None, :, None] ** 2
        vpar_5d = self.vpar[None, None, None, :, None]
        mu_val = self.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        FM = np.exp(-energy) / np.pi**1.5

        phi_5d = phi[:, :, :, None, None]
        if c.electromagnetic and A_par is not None:
            A_5d = A_par[:, :, :, None, None]
            phi_eff = phi_5d - vpar_5d * A_5d
        else:
            phi_eff = phi_5d
        if c.electromagnetic and B_par is not None:
            phi_eff = phi_eff + mu_val * B_par[:, :, :, None, None]

        drive = np.zeros(
            (c.n_species, c.n_kx, c.n_ky, c.n_theta, c.n_vpar, c.n_mu),
            dtype=complex,
        )

        eta_i = c.R_L_Ti / max(c.R_L_ne, 0.1) if c.R_L_ne > 0 else 0.0
        omega_star_i = ky_5d * c.R_L_ne * (1.0 + eta_i * (energy - 1.5))
        drive[0] = -1j * omega_star_i * phi_eff * FM

        if c.kinetic_electrons:
            eta_e = c.R_L_Te / max(c.R_L_ne, 0.1) if c.R_L_ne > 0 else 0.0
            omega_star_e = -ky_5d * c.R_L_ne * (1.0 + eta_e * (energy - 1.5))
            drive[1] = -1j * omega_star_e * phi_eff * FM

        return drive

    def hyperdiffusion(self, f: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Fourth-order hyperdiffusion: -D_H times k_perp^(2p) times f."""
        c = self.cfg
        kp = self.kperp2[:, :, None, None, None]
        return -c.hyper_coeff * kp ** (c.hyper_order // 2) * f

    def rhs(self, state: NonlinearGKState) -> NDArray[np.complex128]:
        """Full RHS for the nonlinear gyrokinetic Vlasov equation."""
        c = self.cfg
        f = state.f
        phi = state.phi
        dfdt = np.zeros_like(f)
        exb_terms = self.nonlinear_exb_term(state) if c.nonlinear else None

        for s in range(c.n_species):
            if s == 1 and not c.kinetic_electrons:
                continue
            f_s = f[s]
            terms = np.zeros_like(f_s)

            is_elec = s == 1 and c.kinetic_electrons
            v_scale = self.vth_ratio_e if (is_elec and not c.implicit_electrons) else 1.0

            if exb_terms is not None:
                terms += exb_terms[s]
            terms -= v_scale * self.parallel_streaming(f_s)

            charge_sign = -1.0 if s == 1 else 1.0
            terms -= charge_sign * self.magnetic_drift(f_s)

            if c.collisions:
                terms += self.collide(f_s)
            terms += self.hyperdiffusion(f_s)
            dfdt[s] = terms

        A_par = self.ampere_solve(f) if c.electromagnetic else None
        B_par = self.magnetic_compression_solve(f) if c.electromagnetic else None
        dfdt += self.gradient_drive(phi, A_par, B_par)
        dfdt -= self._rh_rate * f * self._ky_zero_5d
        return dfdt


__all__ = ["NonlinearGKOperatorsMixin"]
