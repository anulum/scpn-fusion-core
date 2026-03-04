# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Runtime Mixins
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runtime/time-advance mixins extracted from integrated transport solver monolith."""

from __future__ import annotations

import logging
import numpy as np
from typing import Any
from scpn_fusion.core.integrated_transport_solver_adaptive import (
    AdaptiveTimeController,  # re-export for backward compatibility
)
from scpn_fusion.core.integrated_transport_solver_coupling import (
    TransportSolverCouplingMixin,
)
from scpn_fusion.core.integrated_transport_solver_runtime_utils import (
    build_cn_tridiag as _build_cn_tridiag_impl,
    explicit_diffusion_rhs as _explicit_diffusion_rhs_impl,
    sanitize_with_fallback as _sanitize_with_fallback_impl,
    thomas_solve as _thomas_solve_impl,
)
_logger = logging.getLogger(__name__)
__all__ = ["AdaptiveTimeController", "TransportSolverRuntimeMixin"]


def _physics_error_type() -> type[RuntimeError]:
    """Lazily resolve PhysicsError to avoid circular imports at module load."""
    from scpn_fusion.core.integrated_transport_solver import PhysicsError

    return PhysicsError


def _eped_fallback_exceptions() -> tuple[type[BaseException], ...]:
    """Lazily resolve EPED fallback exception tuple from the host module."""
    from scpn_fusion.core.integrated_transport_solver import _EPED_FALLBACK_EXCEPTIONS

    return _EPED_FALLBACK_EXCEPTIONS


class TransportSolverRuntimeMixin(TransportSolverCouplingMixin):
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

    def _rho_volume_element(self) -> np.ndarray:
        """Toroidal volume element per radial cell [m^3].

        The result is cached after the first call since the grid
        geometry (rho, R0, a) does not change during a simulation.
        """
        cached: np.ndarray | None = getattr(self, "_dV_cache", None)
        if cached is not None:
            return cached

        dims = self.cfg["dimensions"]
        r_min = float(dims["R_min"])
        r_max = float(dims["R_max"])
        r0 = 0.5 * (r_min + r_max)
        a_minor = 0.5 * (r_max - r_min)
        if (not np.isfinite(r0)) or r0 <= 0.0:
            raise _physics_error_type()(f"Invalid major radius from config: r0={r0!r}")
        if (not np.isfinite(a_minor)) or a_minor <= 0.0:
            raise _physics_error_type()(f"Invalid minor radius from config: a={a_minor!r}")

        rho = np.clip(
            np.nan_to_num(np.asarray(self.rho, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        d_v = 2.0 * np.pi * r0 * 2.0 * np.pi * rho * a_minor**2 * self.drho
        if (not np.all(np.isfinite(d_v))) or np.any(d_v < 0.0):
            raise _physics_error_type()("Invalid toroidal volume element computed from rho grid")
        self._dV_cache = d_v
        return d_v

    def _compute_aux_heating_sources(self, P_aux_MW: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ion/electron auxiliary-heating sources in keV/s.

        The source is power-normalised against the radial cell volumes to ensure
        that reconstructed injected power matches ``P_aux_MW`` by construction.
        """
        if (not np.isfinite(P_aux_MW)) or P_aux_MW <= 0.0:
            zeros = np.zeros(self.nr, dtype=np.float64)
            self._last_aux_heating_balance = {
                "target_total_MW": float(max(P_aux_MW, 0.0)) if np.isfinite(P_aux_MW) else 0.0,
                "target_ion_MW": 0.0,
                "target_electron_MW": 0.0,
                "reconstructed_ion_MW": 0.0,
                "reconstructed_electron_MW": 0.0,
                "reconstructed_total_MW": 0.0,
            }
            return zeros, zeros

        profile_width = max(self.aux_heating_profile_width, 1e-6)
        shape = np.exp(-(self.rho**2) / profile_width)
        dV = self._rho_volume_element()
        norm = float(np.sum(shape * dV))
        if (not np.isfinite(norm)) or norm <= 0.0:
            shape = np.ones_like(self.rho)
            norm = float(np.sum(shape * dV))
        if (not np.isfinite(norm)) or norm <= 0.0:
            zeros = np.zeros(self.nr, dtype=np.float64)
            self._last_aux_heating_balance = {
                "target_total_MW": float(P_aux_MW),
                "target_ion_MW": 0.0,
                "target_electron_MW": 0.0,
                "reconstructed_ion_MW": 0.0,
                "reconstructed_electron_MW": 0.0,
                "reconstructed_total_MW": 0.0,
            }
            return zeros, zeros

        e_keV_J = 1.602176634e-16
        ne_safe = np.maximum(self.ne, 0.1) * 1e19  # m^-3

        electron_frac = (
            float(np.clip(self.aux_heating_electron_fraction, 0.0, 1.0))
            if self.multi_ion
            else 0.0
        )
        ion_frac = 1.0 - electron_frac
        p_aux_w = float(P_aux_MW) * 1e6

        p_i_wm3 = ion_frac * p_aux_w * shape / norm
        p_e_wm3 = electron_frac * p_aux_w * shape / norm

        # (3/2) n dT/dt = P  => dT/dt = (2/3) * P / (n e_keV)
        s_heat_i = (2.0 / 3.0) * p_i_wm3 / (ne_safe * e_keV_J)
        s_heat_e = (2.0 / 3.0) * p_e_wm3 / (ne_safe * e_keV_J)

        rec_i_w = 1.5 * np.sum(ne_safe * s_heat_i * e_keV_J * dV)
        rec_e_w = 1.5 * np.sum(ne_safe * s_heat_e * e_keV_J * dV)
        self._last_aux_heating_balance = {
            "target_total_MW": float(P_aux_MW),
            "target_ion_MW": ion_frac * float(P_aux_MW),
            "target_electron_MW": electron_frac * float(P_aux_MW),
            "reconstructed_ion_MW": float(rec_i_w / 1e6),
            "reconstructed_electron_MW": float(rec_e_w / 1e6),
            "reconstructed_total_MW": float((rec_i_w + rec_e_w) / 1e6),
        }

        return s_heat_i, s_heat_e


    @staticmethod
    def _bosch_hale_sigmav(T_keV: np.ndarray) -> np.ndarray:
        """D-T <sigma*v> [m^3/s] using NRL Plasma Formulary fit (Bosch & Hale 1992).

        Valid for 0.2 < T < 100 keV.
        """
        T = np.maximum(T_keV, 0.2)
        return 3.68e-18 / (T ** (2.0 / 3.0)) * np.exp(-19.94 / (T ** (1.0 / 3.0)))

    @staticmethod
    def _tungsten_radiation_rate(Te_keV: np.ndarray) -> np.ndarray:
        r"""Coronal equilibrium radiation rate coefficient L_z(Te) for tungsten [W·m^3].

        Piecewise power-law fit to Pütterich et al. (2010) / ADAS data:
          - Te < 1 keV: L_z ~ 5e-31 * Te^0.5   (line radiation dominant)
          - 1 <= Te < 5: L_z ~ 5e-31             (plateau, shell opening)
          - 5 <= Te < 20: L_z ~ 2e-31 * Te^0.3  (rising continuum)
          - Te >= 20:     L_z ~ 8e-31            (fully ionised Bremsstrahlung)
        """
        Te = np.maximum(Te_keV, 0.01)
        Lz = np.where(
            Te < 1.0,
            5.0e-31 * np.sqrt(Te),
            np.where(
                Te < 5.0,
                5.0e-31 * np.ones_like(Te),
                np.where(
                    Te < 20.0,
                    2.0e-31 * Te ** 0.3,
                    8.0e-31 * np.ones_like(Te),
                ),
            ),
        )
        return Lz

    @staticmethod
    def _bremsstrahlung_power_density(ne_1e19: np.ndarray, Te_keV: np.ndarray, Z_eff: float) -> np.ndarray:
        """Bremsstrahlung power density [W/m^3].

        P_brem = 5.35e-37 * Z_eff * ne^2 * sqrt(Te)
        with ne in m^-3 and Te in keV.
        """
        ne_m3 = ne_1e19 * 1e19
        Te = np.maximum(Te_keV, 0.01)
        return 5.35e-37 * Z_eff * ne_m3 ** 2 * np.sqrt(Te)

    def _evolve_species(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Evolve D, T, He-ash densities for one time-step using Crank-Nicolson.
        Implicit evolution ensures stability without CFL constraints.
        """
        if not self.multi_ion or self.n_D is None:
            return np.zeros(self.nr), np.zeros(self.nr)

        # Fusion source: S_fus = n_D * n_T * <sigma_v> (reactions per m^3 per s)
        sigmav = self._bosch_hale_sigmav(self.Ti)
        S_fus = (self.n_D * 1e19) * (self.n_T * 1e19) * sigmav  # reactions/m^3/s
        S_fuel = S_fus / 1e19
        S_He = S_fus / 1e19

        # He-ash sink: pumping
        tau_E = self.compute_confinement_time(1.0)
        tau_He = max(self.tau_He_factor * tau_E, 0.5)
        S_He_pump_rate = 1.0 / tau_He

        # Use unified CN solver for all species
        L_D_exp = self._explicit_diffusion_rhs(self.n_D, self.D_species * np.ones(self.nr))
        rhs_D = self.n_D + 0.5 * dt * L_D_exp - dt * S_fuel
        a, b, c = self._build_cn_tridiag(self.D_species * np.ones(self.nr), dt)
        self.n_D = self._thomas_solve(a, b, c, rhs_D)
        self.n_D[0] = self.n_D[1]
        self.n_D[-1] = 0.01
        self.n_D = np.maximum(0.001, self.n_D)

        L_T_exp = self._explicit_diffusion_rhs(self.n_T, self.D_species * np.ones(self.nr))
        rhs_T = self.n_T + 0.5 * dt * L_T_exp - dt * S_fuel
        self.n_T = self._thomas_solve(a, b, c, rhs_T)
        self.n_T[0] = self.n_T[1]
        self.n_T[-1] = 0.01
        self.n_T = np.maximum(0.001, self.n_T)

        L_He_exp = self._explicit_diffusion_rhs(self.n_He, self.D_species * np.ones(self.nr))
        # He has linear sink (pump): - n_He / tau_He
        # We can handle this by adding S_He_pump_rate to the 'b' diagonal if needed,
        # but for simplicity we treat it as an explicit sink here or half-implicit.
        rhs_He = self.n_He + 0.5 * dt * L_He_exp + dt * (S_He - S_He_pump_rate * self.n_He)
        self.n_He = self._thomas_solve(a, b, c, rhs_He)
        self.n_He[0] = self.n_He[1]
        self.n_He[-1] = 0.0
        self.n_He = np.maximum(0.0, self.n_He)

        # Recompute ne from quasineutrality: ne = n_D + n_T + 2*n_He + Z_imp*n_imp
        # Z_W (effective charge state for tungsten) - Harden with Te-dependence
        # Coronal equilibrium polynomial fit for W
        # Z_W ~ 15 + 10 * log10(Te_keV) for Te > 0.1
        log_te = np.log10(np.maximum(self.Te, 0.1))
        Z_W = np.clip(15.0 + 12.0 * log_te, 10.0, 50.0)
        
        self.ne = self.n_D + self.n_T + 2.0 * self.n_He + Z_W * np.maximum(self.n_impurity, 0.0)
        self.ne = np.maximum(self.ne, 0.1)

        # Z_eff
        ne_m3 = self.ne * 1e19
        ne_safe = np.maximum(ne_m3, 1e10)
        sum_nZ2 = (self.n_D * 1e19 * 1.0 + self.n_T * 1e19 * 1.0
                   + self.n_He * 1e19 * 4.0
                   + np.maximum(self.n_impurity, 0.0) * 1e19 * Z_W ** 2)
        self._Z_eff = float(np.clip(np.mean(sum_nZ2 / ne_safe), 1.0, 10.0))

        # Tungsten line radiation [W/m^3]
        Lz = self._tungsten_radiation_rate(self.Te)
        n_W_m3 = np.maximum(self.n_impurity, 0.0) * 1e19
        P_rad_line = ne_m3 * n_W_m3 * Lz  # W/m^3

        return S_He, P_rad_line

    def update_pedestal_bc(self):
        """Update T_edge_keV using EPED model if active."""
        contract: dict[str, Any] = {
            "used": self.pedestal_model is not None,
            "updated": False,
            "fallback_used": False,
            "in_domain": None,
            "extrapolation_penalty": None,
            "n_ped_1e19": None,
            "t_edge_keV_before": float(self.T_edge_keV),
            "t_edge_keV_after": float(self.T_edge_keV),
            "error": None,
        }
        if self.pedestal_model is None:
            self._last_pedestal_bc_contract = contract
            return

        # Use density at rho ~ 0.95 as proxy for pedestal density
        n_ped_idx = int(0.95 * self.nr)
        n_ped = float(self.ne[n_ped_idx])
        contract["n_ped_1e19"] = n_ped
        if (not np.isfinite(n_ped)) or n_ped <= 0.0:
            contract["fallback_used"] = True
            contract["error"] = f"invalid_n_ped:{n_ped!r}"
            contract["t_edge_keV_after"] = float(self.T_edge_keV)
            self._last_pedestal_bc_contract = contract
            return

        # Predict
        try:
            result = self.pedestal_model.predict(n_ped, T_ped_guess_keV=self.T_edge_keV)
            in_domain = bool(getattr(result, "in_domain", False))
            penalty = float(getattr(result, "extrapolation_penalty", 0.0))
            contract["in_domain"] = in_domain
            contract["extrapolation_penalty"] = penalty
            if in_domain or penalty > 0.5:
                # Relax towards new value to avoid shocks
                candidate = 0.8 * self.T_edge_keV + 0.2 * float(result.T_ped_keV)
                if (not np.isfinite(candidate)) or candidate <= 0.0:
                    raise ValueError(
                        f"Invalid pedestal boundary candidate {candidate!r}"
                    )
                self.T_edge_keV = float(candidate)
                contract["updated"] = True
        except _eped_fallback_exceptions() as exc:
            contract["fallback_used"] = True
            contract["error"] = f"{exc.__class__.__name__}:{exc}"
            _logger.debug(
                "update_pedestal_bc: retaining previous T_edge_keV=%.6f due to %s",
                float(self.T_edge_keV),
                exc,
            )
        contract["t_edge_keV_after"] = float(self.T_edge_keV)
        self._last_pedestal_bc_contract = contract

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
        self._last_numerical_recovery_count += self._sanitize_runtime_state(
            label_prefix="pre"
        )
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

        new_Ti[0] = new_Ti[1]    # Neumann at core
        new_Ti[-1] = 0.1         # Dirichlet at edge
        self.Ti, n_ti_new = self._sanitize_with_fallback(
            new_Ti, Ti_old, floor=0.01, ceil=1e3
        )
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
            self.Te, n_te_new = self._sanitize_with_fallback(
                new_Te, Te_old, floor=0.01, ceil=1e3
            )
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
        self._last_conservation_error = (
            abs(dW_actual - dW_source) / max(abs(W_before), 1e-10)
        )
        if not np.isfinite(self._last_conservation_error):
            self._last_conservation_error = float("inf")

        if enforce_conservation and self._last_conservation_error > 0.01:
            raise _physics_error_type()(
                f"Energy conservation violated: relative error "
                f"{self._last_conservation_error:.4e} > 1% threshold. "
                f"W_before={W_before:.4e} J, W_after={W_after:.4e} J, "
                f"dW_source={dW_source:.4e} J."
            )

        self._last_numerical_recovery_count += self._sanitize_runtime_state(
            label_prefix="post"
        )
        self._enforce_recovery_budget(
            enforce_numerical_recovery=enforce_numerical_recovery,
            max_numerical_recoveries=max_numerical_recoveries,
        )
        avg_ti: float = np.mean(self.Ti).item()
        core_ti: float = self.Ti[0].item()
        return avg_ti, core_ti
