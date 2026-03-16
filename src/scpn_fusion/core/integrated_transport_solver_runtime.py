# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Runtime Mixins
"""Runtime/time-advance mixins extracted from integrated transport solver monolith."""

from __future__ import annotations

import numpy as np
from scpn_fusion.core.integrated_transport_solver_adaptive import (
    AdaptiveTimeController,  # re-export for backward compatibility
)
from scpn_fusion.core.integrated_transport_solver_coupling import (
    TransportSolverCouplingMixin,
)
from scpn_fusion.core.integrated_transport_solver_runtime_physics import (
    TransportSolverRuntimePhysicsMixin,
    _physics_error_type,
)
from scpn_fusion.core.integrated_transport_solver_runtime_utils import (
    build_cn_tridiag as _build_cn_tridiag_impl,
    explicit_diffusion_rhs as _explicit_diffusion_rhs_impl,
    sanitize_with_fallback as _sanitize_with_fallback_impl,
    thomas_solve as _thomas_solve_impl,
)

__all__ = ["AdaptiveTimeController", "TransportSolverRuntimeMixin"]


class TransportSolverRuntimeMixin(
    TransportSolverRuntimePhysicsMixin,
    TransportSolverCouplingMixin,
):
    D_n: np.ndarray
    chi_e: np.ndarray
    chi_i: np.ndarray
    n_impurity: np.ndarray

    @staticmethod
    def _thomas_solve(a, b, c, d):
        """O(n) tridiagonal solver (Thomas algorithm).

        Solves  A x = d  where A is tridiagonal with sub-diagonal *a*,
        main diagonal *b*, and super-diagonal *c*.

        Parameters
        ----------
        a : array, length n-1  — sub-diagonal
        b : array, length n    — main diagonal
        c : array, length n-1  — super-diagonal
        d : array, length n    — right-hand side

        Returns
        -------
        x : array, length n
        """
        return _thomas_solve_impl(a, b, c, d)

    def _explicit_diffusion_rhs(self, T, chi):
        """Compute explicit diffusion operator L_h(T) = (1/r) d/dr(r chi dT/dr).

        Uses half-grid diffusivities and central differences on the
        interior, returning an array of the same length as *T*.
        """
        return _explicit_diffusion_rhs_impl(
            rho=self.rho,
            drho=float(self.drho),
            T=np.asarray(T, dtype=np.float64),
            chi=np.asarray(chi, dtype=np.float64),
        )

    def _build_cn_tridiag(self, chi, dt):
        """Build tridiagonal coefficients for the Crank-Nicolson LHS.

        The implicit system is:
            (I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*(S - Sink)

        Returns (a, b, c) sub/main/super diagonals for the interior points,
        padded to full grid size (BCs applied separately).
        """
        return _build_cn_tridiag_impl(
            rho=self.rho,
            drho=float(self.drho),
            chi=np.asarray(chi, dtype=np.float64),
            dt=float(dt),
        )

    @staticmethod
    def _sanitize_with_fallback(
        arr: np.ndarray,
        reference: np.ndarray,
        *,
        floor: float | None = None,
        ceil: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Replace non-finite entries and enforce optional lower/upper bounds."""
        return _sanitize_with_fallback_impl(
            np.asarray(arr, dtype=np.float64),
            np.asarray(reference, dtype=np.float64),
            floor=floor,
            ceil=ceil,
        )

    def _sanitize_runtime_state(self, *, label_prefix: str) -> int:
        """Keep runtime profiles and coefficients finite during transport stepping."""
        recovered_total = 0

        ti_fb = np.where(np.isfinite(self.Ti), self.Ti, 1.0)
        self.Ti, n_ti = self._sanitize_with_fallback(self.Ti, ti_fb, floor=0.01, ceil=1e3)
        recovered_total += n_ti
        self._record_recovery(f"{label_prefix}.Ti", n_ti)

        te_fb = np.where(np.isfinite(self.Te), self.Te, 1.0)
        self.Te, n_te = self._sanitize_with_fallback(self.Te, te_fb, floor=0.01, ceil=1e3)
        recovered_total += n_te
        self._record_recovery(f"{label_prefix}.Te", n_te)

        ne_fb = np.where(np.isfinite(self.ne), self.ne, 5.0)
        self.ne, n_ne = self._sanitize_with_fallback(self.ne, ne_fb, floor=0.1, ceil=1e3)
        recovered_total += n_ne
        self._record_recovery(f"{label_prefix}.ne", n_ne)

        chi_i_fb = np.where(np.isfinite(self.chi_i), self.chi_i, 0.5)
        self.chi_i, n_chi_i = self._sanitize_with_fallback(
            self.chi_i, chi_i_fb, floor=0.01, ceil=1e4
        )
        recovered_total += n_chi_i
        self._record_recovery(f"{label_prefix}.chi_i", n_chi_i)

        chi_e_fb = np.where(np.isfinite(self.chi_e), self.chi_e, 0.5)
        self.chi_e, n_chi_e = self._sanitize_with_fallback(
            self.chi_e, chi_e_fb, floor=0.01, ceil=1e4
        )
        recovered_total += n_chi_e
        self._record_recovery(f"{label_prefix}.chi_e", n_chi_e)

        dn_fb = np.where(np.isfinite(self.D_n), self.D_n, 0.1)
        self.D_n, n_dn = self._sanitize_with_fallback(self.D_n, dn_fb, floor=0.0, ceil=1e4)
        recovered_total += n_dn
        self._record_recovery(f"{label_prefix}.D_n", n_dn)

        imp_fb = np.where(np.isfinite(self.n_impurity), self.n_impurity, 0.0)
        self.n_impurity, n_imp = self._sanitize_with_fallback(
            self.n_impurity, imp_fb, floor=0.0, ceil=1e3
        )
        recovered_total += n_imp
        self._record_recovery(f"{label_prefix}.n_impurity", n_imp)

        if self.n_D is not None:
            n_d_fb = np.where(np.isfinite(self.n_D), self.n_D, 0.5)
            self.n_D, n_d = self._sanitize_with_fallback(self.n_D, n_d_fb, floor=0.001, ceil=1e3)
            recovered_total += n_d
            self._record_recovery(f"{label_prefix}.n_D", n_d)
        if self.n_T is not None:
            n_t_fb = np.where(np.isfinite(self.n_T), self.n_T, 0.5)
            self.n_T, n_t = self._sanitize_with_fallback(self.n_T, n_t_fb, floor=0.001, ceil=1e3)
            recovered_total += n_t
            self._record_recovery(f"{label_prefix}.n_T", n_t)
        if self.n_He is not None:
            n_he_fb = np.where(np.isfinite(self.n_He), self.n_He, 0.0)
            self.n_He, n_he = self._sanitize_with_fallback(self.n_He, n_he_fb, floor=0.0, ceil=1e3)
            recovered_total += n_he
            self._record_recovery(f"{label_prefix}.n_He", n_he)

        return recovered_total

    def evolve_profiles(
        self,
        dt: float,
        P_aux: float,
        enforce_conservation: bool = False,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> tuple[float, float]:
        """Advance Ti by one time step using Crank-Nicolson implicit diffusion.

        The scheme is unconditionally stable, allowing dt up to ~1.0 s
        without NaN.  The full equation solved is:

            (T^{n+1} - T^n)/dt = 0.5*[L_h(T^{n+1}) + L_h(T^n)] + S - Sink

        Parameters
        ----------
        dt : float
            Time step [s].
        P_aux : float
            Auxiliary heating power [MW].
        enforce_conservation : bool
            When True, raise :class:`PhysicsError` if the per-step energy
            conservation error exceeds 1%.
        enforce_numerical_recovery : bool
            When True, raise :class:`PhysicsError` if recoveries exceed the
            configured budget (`max_numerical_recoveries_per_step` or explicit
            `max_numerical_recoveries`).
        max_numerical_recoveries : int or None
            Optional per-call override for recovery budget.  ``None`` uses the
            solver-level default.
        """
        # Update EPED boundary condition first
        self.update_pedestal_bc()

        if (not np.isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")
        if not np.isfinite(P_aux):
            raise ValueError(f"P_aux must be finite, got {P_aux!r}")

        self._last_numerical_recovery_count = 0
        self._last_numerical_recovery_breakdown = {}
        self._last_numerical_recovery_count += self._sanitize_runtime_state(label_prefix="pre")
        Ti_old = self.Ti.copy()
        Te_old = self.Te.copy()

        if self.multi_ion:
            _S_He, P_rad_line_Wm3 = self._evolve_species(dt)
        else:
            self._evolve_impurity(dt)
            P_rad_line_Wm3 = np.zeros(self.nr)

        S_heat_i, S_heat_e_aux = self._compute_aux_heating_sources(P_aux)

        if self.multi_ion:  # radiation sinks
            # Use coronal tungsten radiation (already in W/m^3)
            # Convert to keV / s per 10^19:  P [W/m^3] / (n_e [10^19 m^-3] * 1e19 * e_keV_J)
            e_keV_J = 1.602176634e-16
            ne_safe = np.maximum(self.ne, 0.1) * 1e19
            S_rad_i = P_rad_line_Wm3 / (ne_safe * e_keV_J) * 0.5  # half to ions
        else:
            cooling_factor = 5.0
            S_rad_i = cooling_factor * self.ne * self.n_impurity * np.sqrt(self.Te + 0.1)

        net_source_i = S_heat_i - S_rad_i
        net_source_i, n_src_i = self._sanitize_with_fallback(
            net_source_i,
            np.zeros_like(net_source_i),
        )
        self._last_numerical_recovery_count += n_src_i
        self._record_recovery("cn.ion_net_source", n_src_i)

        Lh_explicit = self._explicit_diffusion_rhs(self.Ti, self.chi_i)
        Lh_explicit, n_lh_i = self._sanitize_with_fallback(
            Lh_explicit,
            np.zeros_like(Lh_explicit),
        )
        self._last_numerical_recovery_count += n_lh_i
        self._record_recovery("cn.ion_diffusion_rhs", n_lh_i)
        rhs = self.Ti + 0.5 * dt * Lh_explicit + dt * net_source_i
        rhs, n_rhs_i = self._sanitize_with_fallback(rhs, Ti_old, floor=0.01, ceil=1e3)
        self._last_numerical_recovery_count += n_rhs_i
        self._record_recovery("cn.ion_rhs", n_rhs_i)
        a, b, c = self._build_cn_tridiag(self.chi_i, dt)
        new_Ti = self._thomas_solve(a, b, c, rhs)

        new_Ti[0] = new_Ti[1]  # Neumann at core
        new_Ti[-1] = 0.1  # Dirichlet at edge
        self.Ti, n_ti_new = self._sanitize_with_fallback(new_Ti, Ti_old, floor=0.01, ceil=1e3)
        self._last_numerical_recovery_count += n_ti_new
        self._record_recovery("cn.ion_solution", n_ti_new)

        if self.multi_ion:  # electron temperature evolution
            # Independent electron temperature evolution
            # Electrons receive configured auxiliary-heating split.
            S_heat_e = S_heat_e_aux
            P_brem = self._bremsstrahlung_power_density(self.ne, Te_old, self._Z_eff)
            ne_safe_e = np.maximum(self.ne, 0.1) * 1e19
            S_brem_e = P_brem / (ne_safe_e * e_keV_J)
            # Tungsten radiation on electrons (other half)
            S_rad_e = P_rad_line_Wm3 / (ne_safe_e * e_keV_J) * 0.5

            # Electron-ion coupling (collisional equilibration)
            # tau_eq ~ 3e18 * Te^1.5 / (ne * ln_lambda)
            # Calibrated to approx 0.1s for ne=1e20, Te=10keV
            tau_eq = 0.01 * (Te_old**1.5) / np.maximum(self.ne / 10.0, 0.1)
            tau_eq = np.clip(tau_eq, 0.001, 1.0)
            S_equil = (self.Ti - Te_old) / tau_eq

            net_source_e = S_heat_e - S_rad_e - S_brem_e + S_equil
            net_source_e, n_src_e = self._sanitize_with_fallback(
                net_source_e,
                np.zeros_like(net_source_e),
            )
            self._last_numerical_recovery_count += n_src_e
            self._record_recovery("cn.electron_net_source", n_src_e)

            Lh_explicit_e = self._explicit_diffusion_rhs(Te_old, self.chi_e)
            Lh_explicit_e, n_lh_e = self._sanitize_with_fallback(
                Lh_explicit_e,
                np.zeros_like(Lh_explicit_e),
            )
            self._last_numerical_recovery_count += n_lh_e
            self._record_recovery("cn.electron_diffusion_rhs", n_lh_e)
            rhs_e = Te_old + 0.5 * dt * Lh_explicit_e + dt * net_source_e
            rhs_e, n_rhs_e = self._sanitize_with_fallback(rhs_e, Te_old, floor=0.01, ceil=1e3)
            self._last_numerical_recovery_count += n_rhs_e
            self._record_recovery("cn.electron_rhs", n_rhs_e)
            a_e, b_e, c_e = self._build_cn_tridiag(self.chi_e, dt)
            new_Te = self._thomas_solve(a_e, b_e, c_e, rhs_e)

            new_Te[0] = new_Te[1]
            new_Te[-1] = self.T_edge_keV  # EPED boundary condition
            self.Te, n_te_new = self._sanitize_with_fallback(new_Te, Te_old, floor=0.01, ceil=1e3)
            self._last_numerical_recovery_count += n_te_new
            self._record_recovery("cn.electron_solution", n_te_new)
        else:
            self.Te = self.Ti.copy()  # Assume equilibrated (legacy)

        # No auxiliary heating: forbid unphysical mean-temperature growth due to
        # numerical overshoot in the diffusion solve on coarse toy grids.
        if P_aux <= 0.0:
            mean_ti_old = float(np.mean(Ti_old))
            mean_ti_new = float(np.mean(self.Ti))
            if np.isfinite(mean_ti_old) and np.isfinite(mean_ti_new) and mean_ti_new > mean_ti_old:
                scale = mean_ti_old / max(mean_ti_new, 1e-12)
                self.Ti *= scale
                self.Ti[0] = self.Ti[1]
                self.Ti[-1] = max(0.1, self.T_edge_keV)
                self.Ti = np.maximum(0.01, self.Ti)
                if not self.multi_ion:
                    self.Te = self.Ti.copy()
                self._last_numerical_recovery_count += 1
                self._record_recovery("stability.zero_aux_overshoot_rescale", 1)

        # Energy conservation diagnostic
        if not self.multi_ion:
            e_keV_J = 1.602176634e-16
        dV = self._rho_volume_element()

        W_before = 1.5 * np.sum(self.ne * 1e19 * Ti_old * e_keV_J * dV)
        W_after = 1.5 * np.sum(self.ne * 1e19 * self.Ti * e_keV_J * dV)
        dW_source = dt * 1.5 * np.sum(self.ne * 1e19 * net_source_i * e_keV_J * dV)

        dW_actual = W_after - W_before
        self._last_conservation_error = abs(dW_actual - dW_source) / max(abs(W_before), 1e-10)
        if not np.isfinite(self._last_conservation_error):
            self._last_conservation_error = float("inf")

        if enforce_conservation and self._last_conservation_error > 0.01:
            raise _physics_error_type()(
                f"Energy conservation violated: relative error "
                f"{self._last_conservation_error:.4e} > 1% threshold. "
                f"W_before={W_before:.4e} J, W_after={W_after:.4e} J, "
                f"dW_source={dW_source:.4e} J."
            )

        self._last_numerical_recovery_count += self._sanitize_runtime_state(label_prefix="post")
        self._enforce_recovery_budget(
            enforce_numerical_recovery=enforce_numerical_recovery,
            max_numerical_recoveries=max_numerical_recoveries,
        )
        avg_ti: float = np.mean(self.Ti).item()
        core_ti: float = self.Ti[0].item()
        return avg_ti, core_ti
