# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport GS Coupling Mixins
"""GS-coupling and steady-state loop mixins extracted from transport runtime."""

from __future__ import annotations

import logging

import numpy as np

from scpn_fusion.core.integrated_transport_solver_adaptive import AdaptiveTimeController

_logger = logging.getLogger(__name__)


def _physics_error_type() -> type[RuntimeError]:
    """Lazily resolve PhysicsError to avoid circular imports."""
    from scpn_fusion.core.integrated_transport_solver import PhysicsError

    return PhysicsError


class TransportSolverCouplingMixin:
    def map_profiles_to_2d(self) -> None:
        """Project 1D transport profiles onto the 2D Grad-Shafranov grid."""
        idx_max = np.argmax(self.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
        psi_axis = self.Psi[iz_ax, ir_ax]
        _, psi_x = self.find_x_point(self.Psi)
        psi_edge = psi_x
        if abs(psi_edge - psi_axis) < 1.0:
            psi_edge = np.min(self.Psi)

        denom = psi_edge - psi_axis
        if abs(denom) < 1e-9:
            denom = 1e-9
        psi_norm = (self.Psi - psi_axis) / denom
        psi_norm = np.clip(psi_norm, 0, 1)
        rho_2d = np.sqrt(psi_norm)

        r0 = (self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0
        i_target = self.cfg["physics"]["plasma_current_target"]
        b_pol_est = (1.256e-6 * i_target) / (
            2 * np.pi * 0.5 * (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"])
        )
        j_bs_1d = self.calculate_bootstrap_current(r0, b_pol_est)

        self.Pressure_2D = np.interp(rho_2d.flatten(), self.rho, self.ne * (self.Ti + self.Te))
        self.Pressure_2D = self.Pressure_2D.reshape(self.Psi.shape)

        j_bs_2d = np.interp(rho_2d.flatten(), self.rho, j_bs_1d)
        j_bs_2d = j_bs_2d.reshape(self.Psi.shape)

        self.J_phi = (self.Pressure_2D * self.RR) + j_bs_2d

        i_curr = np.sum(self.J_phi) * self.dR * self.dZ
        if abs(i_curr) > 1e-9:
            self.J_phi *= i_target / i_curr

    def compute_confinement_time(self, P_loss_MW: float) -> float:
        """Compute energy confinement time from stored thermal energy."""
        p_loss = float(P_loss_MW)
        if not np.isfinite(p_loss):
            raise ValueError(f"P_loss_MW must be finite, got {P_loss_MW!r}")
        if p_loss <= 0:
            return float("inf")

        e_kev = 1.602176634e-16
        dV = self._rho_volume_element()
        ne_safe = np.clip(
            np.nan_to_num(np.asarray(self.ne, dtype=np.float64), nan=0.0, posinf=1e3, neginf=0.0),
            0.0,
            1e3,
        )
        ti_safe = np.clip(
            np.nan_to_num(np.asarray(self.Ti, dtype=np.float64), nan=0.0, posinf=1e3, neginf=0.0),
            0.0,
            1e3,
        )
        te_safe = np.clip(
            np.nan_to_num(np.asarray(self.Te, dtype=np.float64), nan=0.0, posinf=1e3, neginf=0.0),
            0.0,
            1e3,
        )
        if ne_safe.shape != dV.shape or ti_safe.shape != dV.shape or te_safe.shape != dV.shape:
            raise _physics_error_type()(
                "Profile and geometry shape mismatch in confinement-time estimate"
            )

        energy_density = 1.5 * (ne_safe * 1e19) * (ti_safe + te_safe) * e_kev
        w_stored_j = float(np.sum(energy_density * dV))
        if (not np.isfinite(w_stored_j)) or w_stored_j < 0.0:
            return float("inf")
        w_stored_mw = w_stored_j / 1e6

        tau = w_stored_mw / p_loss
        return float(tau) if np.isfinite(tau) and tau >= 0.0 else float("inf")

    def run_self_consistent(
        self,
        P_aux: float,
        n_inner: int = 100,
        n_outer: int = 10,
        dt: float = 0.01,
        psi_tol: float = 1e-3,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> dict:
        """Run self-consistent GS <-> transport iteration."""
        psi_residuals: list[float] = []
        converged = False
        n_outer_converged = 0

        for outer in range(n_outer):
            psi_old = self.Psi.copy()
            psi_old_norm = float(np.linalg.norm(psi_old))
            if psi_old_norm < 1e-30:
                psi_old_norm = 1.0

            for _ in range(n_inner):
                self.update_transport_model(P_aux)
                self.evolve_profiles(
                    dt,
                    P_aux,
                    enforce_numerical_recovery=enforce_numerical_recovery,
                    max_numerical_recoveries=max_numerical_recoveries,
                )

            self.map_profiles_to_2d()
            self.solve_equilibrium()

            psi_residual = float(np.linalg.norm(self.Psi - psi_old) / psi_old_norm)
            psi_residuals.append(psi_residual)
            n_outer_converged = outer + 1

            _logger.info(
                "GS-transport outer iteration progress",
                extra={
                    "physics_context": {
                        "outer_iteration": outer + 1,
                        "max_outer": n_outer,
                        "psi_residual": psi_residual,
                    }
                },
            )

            if psi_residual < psi_tol:
                converged = True
                _logger.info(
                    "GS-transport converged",
                    extra={
                        "physics_context": {
                            "iterations": outer + 1,
                            "final_residual": psi_residual,
                            "tolerance": psi_tol,
                        }
                    },
                )
                break

        t_avg = float(np.mean(self.Ti))
        t_core = float(self.Ti[0])
        tau_e = self.compute_confinement_time(P_aux)

        return {
            "T_avg": t_avg,
            "T_core": t_core,
            "tau_e": tau_e,
            "n_outer_converged": n_outer_converged,
            "psi_residuals": psi_residuals,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
            "converged": converged,
        }

    def run_to_steady_state(
        self,
        P_aux: float,
        n_steps: int = 500,
        dt: float = 0.01,
        adaptive: bool = False,
        tol: float = 1e-3,
        self_consistent: bool = False,
        sc_n_inner: int = 100,
        sc_n_outer: int = 10,
        sc_psi_tol: float = 1e-3,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> dict:
        """Run transport evolution until approximate steady state."""
        if self_consistent:
            return self.run_self_consistent(
                P_aux=P_aux,
                n_inner=sc_n_inner,
                n_outer=sc_n_outer,
                dt=dt,
                psi_tol=sc_psi_tol,
                enforce_numerical_recovery=enforce_numerical_recovery,
                max_numerical_recoveries=max_numerical_recoveries,
            )

        if not adaptive:
            for _ in range(n_steps):
                self.update_transport_model(P_aux)
                t_avg, t_core = self.evolve_profiles(
                    dt,
                    P_aux,
                    enforce_numerical_recovery=enforce_numerical_recovery,
                    max_numerical_recoveries=max_numerical_recoveries,
                )

            tau_e = self.compute_confinement_time(P_aux)
            return {
                "T_avg": float(t_avg),
                "T_core": float(t_core),
                "tau_e": tau_e,
                "n_steps": n_steps,
                "Ti_profile": self.Ti.copy(),
                "ne_profile": self.ne.copy(),
            }

        atc = AdaptiveTimeController(dt_init=dt, tol=tol)

        for _step in range(n_steps):
            self.update_transport_model(P_aux)
            error = atc.estimate_error(
                self,
                P_aux,
                enforce_numerical_recovery=enforce_numerical_recovery,
                max_numerical_recoveries=max_numerical_recoveries,
            )
            atc.adapt_dt(error)
            t_avg = float(np.mean(self.Ti))
            t_core = float(self.Ti[0])

        tau_e = self.compute_confinement_time(P_aux)
        return {
            "T_avg": float(t_avg),
            "T_core": float(t_core),
            "tau_e": tau_e,
            "n_steps": n_steps,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
            "dt_final": atc.dt,
            "dt_history": atc.dt_history.copy(),
            "error_history": atc.error_history.copy(),
        }
