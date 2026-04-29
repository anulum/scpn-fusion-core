# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Grid, geometry, and species setup for the nonlinear GK solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._gk_nonlinear_types import _E_CHARGE, _M_PROTON
from scpn_fusion.core.gk_geometry import circular_geometry
from scpn_fusion.core.gk_species import deuterium_ion, electron


class NonlinearGKSetupMixin:
    """Setup helpers used by :class:`NonlinearGKSolver`."""

    def _setup_grids(self) -> None:
        c = self.cfg
        self.kx = 2 * np.pi * np.fft.fftfreq(c.n_kx, d=c.Lx / c.n_kx)
        self.ky = 2 * np.pi * np.fft.fftfreq(c.n_ky, d=c.Ly / c.n_ky)
        self.kx_grid = self.kx[:, None, None]
        self.ky_grid = self.ky[None, :, None]
        self.kperp2 = self.kx[:, None] ** 2 + self.ky[None, :] ** 2

        self.theta = np.linspace(-np.pi, np.pi, c.n_theta, endpoint=False)
        self.dtheta = self.theta[1] - self.theta[0]

        self.vpar = np.linspace(-c.vpar_max, c.vpar_max, c.n_vpar)
        self.dvpar = self.vpar[1] - self.vpar[0] if c.n_vpar > 1 else 1.0
        self.mu = np.linspace(0, c.mu_max, c.n_mu)
        self.dmu = self.mu[1] - self.mu[0] if c.n_mu > 1 else 1.0

        if c.dealiasing == "2/3":
            kx_max = np.max(np.abs(self.kx)) * 2.0 / 3.0
            ky_max = np.max(np.abs(self.ky)) * 2.0 / 3.0
            self.dealias_mask = (np.abs(self.kx[:, None]) <= kx_max) & (
                np.abs(self.ky[None, :]) <= ky_max
            )
        else:
            self.dealias_mask = np.ones((c.n_kx, c.n_ky), dtype=bool)

    def _setup_ballooning(self) -> None:
        """Precompute phase factors for ballooning connection BC."""
        c = self.cfg
        x = np.arange(c.n_kx) * c.Lx / c.n_kx
        delta_kx = c.s_hat * self.ky
        self._ball_phase_fwd = np.exp(1j * delta_kx[None, :] * x[:, None])
        self._ball_phase_bwd = np.conj(self._ball_phase_fwd)

    def _apply_kx_shift(
        self, f_slice: NDArray[np.complex128], phase: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Shift kx via IFFT, phase multiply, and FFT for one theta slice."""
        shape = f_slice.shape
        n_batch = shape[2] * shape[3] if len(shape) == 4 else 1
        f_flat = f_slice.reshape(self.cfg.n_kx, self.cfg.n_ky, n_batch)
        f_x = np.fft.ifft(f_flat, axis=0)
        f_x *= phase[:, :, None]
        return np.fft.fft(f_x, axis=0).reshape(shape)

    def _roll_ballooning(self, f_s: NDArray[np.complex128], shift: int) -> NDArray[np.complex128]:
        """Roll along theta with ballooning kx shifts at periodic boundaries."""
        rolled = np.roll(f_s, shift, axis=2)
        n_theta = self.cfg.n_theta

        if shift > 0:
            for j in range(shift):
                rolled[:, :, j] = self._apply_kx_shift(rolled[:, :, j], self._ball_phase_bwd)
        elif shift < 0:
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
        m_i = self.ion.mass_amu * _M_PROTON
        T_i_J = self.ion.temperature_keV * 1e3 * _E_CHARGE
        self.c_s = np.sqrt(T_i_J / m_i)
        self.rho_s = m_i * self.c_s / (_E_CHARGE * c.B0)
        self.chi_gB = self.rho_s**2 * self.c_s / c.a

        self.rho_ratio = np.sqrt(
            2.0 * self.ion.temperature_keV / max(self.elec.temperature_keV, 0.01)
        )
        self.rho_ratio_e = np.sqrt(2.0 * c.mass_ratio_me_mi)
        self.vth_ratio_e = np.sqrt(1.0 / max(c.mass_ratio_me_mi, 1e-6))

        eps = 0.5 * c.a / max(c.R0, 0.01)
        self._rh_neo_pol = 1.6 * c.q**2 / max(np.sqrt(eps), 0.01)
        self._rh_residual = 1.0 / (1.0 + self._rh_neo_pol)
        self._rh_tau = c.q / max(np.sqrt(eps), 0.01)
        self._rh_rate = (1.0 - self._rh_residual) / self._rh_tau
        self._ky_zero_5d = np.abs(self.ky[None, :, None, None, None]) < 1e-10


__all__ = ["NonlinearGKSetupMixin"]
