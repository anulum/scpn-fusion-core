# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver Model Mixins
# ──────────────────────────────────────────────────────────────────────
"""Model/configuration mixins extracted from integrated transport monolith."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.fallback_telemetry import record_fallback_event


def _solver_module() -> Any:
    """Resolve host integrated_transport_solver module lazily."""
    import scpn_fusion.core.integrated_transport_solver as solver_mod

    return solver_mod


class TransportSolverModelMixin:
    def set_numerical_recovery_limit(self, max_recoveries: int | None) -> None:
        """Set optional per-step numerical-recovery cap."""
        if max_recoveries is None:
            self.max_numerical_recoveries_per_step = None
            return
        if isinstance(max_recoveries, bool) or int(max_recoveries) < 0:
            raise ValueError("max_recoveries must be a non-negative integer or None.")
        self.max_numerical_recoveries_per_step = int(max_recoveries)

    def _record_recovery(self, label: str, count: int) -> None:
        """Track recoveries by category for per-step diagnostics."""
        if count <= 0:
            return
        self._last_numerical_recovery_breakdown[label] = (
            self._last_numerical_recovery_breakdown.get(label, 0) + int(count)
        )

    def _resolve_recovery_limit(self, override: int | None) -> int | None:
        """Resolve and validate an optional per-step recovery cap override."""
        if override is None:
            return self.max_numerical_recoveries_per_step
        if isinstance(override, bool) or int(override) < 0:
            raise ValueError("max_numerical_recoveries must be a non-negative integer or None.")
        return int(override)

    def _enforce_recovery_budget(
        self,
        *,
        enforce_numerical_recovery: bool,
        max_numerical_recoveries: int | None,
    ) -> None:
        """Fail fast when recovery volume exceeds configured hardening budget."""
        solver_mod = _solver_module()
        limit = self._resolve_recovery_limit(max_numerical_recoveries)
        self._last_numerical_recovery_limit = limit
        if not enforce_numerical_recovery or limit is None:
            return
        if self._last_numerical_recovery_count <= limit:
            return

        details = ", ".join(
            f"{name}={count}" for name, count in sorted(self._last_numerical_recovery_breakdown.items())
        ) or "no breakdown"
        raise solver_mod.PhysicsError(
            "Numerical recovery budget exceeded: "
            f"{self._last_numerical_recovery_count} > {limit}. "
            f"Breakdown: {details}"
        )

    def set_neoclassical(
        self,
        R0: float,
        a: float,
        B0: float,
        A_ion: float = 2.0,
        Z_eff: float = 1.5,
        q0: float = 1.0,
        q_edge: float = 3.0,
    ) -> None:
        """Configure Chang-Hinton neoclassical transport model."""
        solver_mod = _solver_module()
        r0 = solver_mod._require_positive_finite_scalar("R0", R0)
        a_minor = solver_mod._require_positive_finite_scalar("a", a)
        b0 = solver_mod._require_positive_finite_scalar("B0", B0)
        a_ion = solver_mod._require_positive_finite_scalar("A_ion", A_ion)
        z_eff = solver_mod._require_positive_finite_scalar("Z_eff", Z_eff)
        q0_f = solver_mod._require_positive_finite_scalar("q0", q0)
        q_edge_f = solver_mod._require_positive_finite_scalar("q_edge", q_edge)

        q_profile = q0_f + (q_edge_f - q0_f) * self.rho**2
        if (not np.all(np.isfinite(q_profile))) or np.any(q_profile <= 0.0):
            raise ValueError("Generated q_profile contains invalid values")
        self.neoclassical_params = {
            "R0": r0,
            "a": a_minor,
            "B0": b0,
            "A_ion": a_ion,
            "Z_eff": z_eff,
            "q_profile": q_profile,
        }

    def chang_hinton_chi_profile(self) -> np.ndarray:
        """Backward-compatible Chang-Hinton profile helper."""
        solver_mod = _solver_module()
        rho = np.asarray(getattr(self, "rho"), dtype=np.float64)

        t_i_raw = getattr(self, "t_i", None)
        if t_i_raw is None:
            t_i_raw = getattr(self, "Ti")
        t_i = np.asarray(t_i_raw, dtype=np.float64)

        n_e_raw = getattr(self, "n_e", None)
        if n_e_raw is None:
            n_e_raw = getattr(self, "ne")
        n_e = np.asarray(n_e_raw, dtype=np.float64)
        q_profile = np.asarray(
            getattr(self, "q_profile", np.linspace(1.0, 3.0, len(rho))),
            dtype=np.float64,
        )

        params = getattr(self, "neoclassical_params", None)
        if not isinstance(params, dict):
            params = {}
        R0 = float(params.get("R0", 6.2))
        a = float(params.get("a", 2.0))
        B0 = float(params.get("B0", 5.3))
        A_ion = float(params.get("A_ion", 2.0))
        Z_eff = float(params.get("Z_eff", 1.5))

        if q_profile.shape != rho.shape:
            q_profile = np.linspace(1.0, 3.0, len(rho), dtype=np.float64)

        return solver_mod.chang_hinton_chi_profile(
            rho, t_i, n_e, q_profile, R0, a, B0, A_ion=A_ion, Z_eff=Z_eff
        )

    def inject_impurities(self, flux_from_wall_per_sec: float, dt: float) -> None:
        """Models impurity influx from PWI erosion with explicit diffusion."""
        d_n_edge = (flux_from_wall_per_sec * dt) / 20.0 * 1e-18
        self.n_impurity[-1] += d_n_edge

        d_imp = 1.0
        new_imp = self.n_impurity.copy()
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -d_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        new_imp += (-div) * dt
        new_imp[0] = new_imp[1]
        self.n_impurity = np.maximum(0, new_imp)

    def _evolve_impurity(self, dt: float) -> None:
        """Autonomous impurity evolution with edge source and diffusion."""
        edge_source_rate = 0.01
        self.n_impurity[-1] += edge_source_rate * dt

        d_imp = 1.0
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -d_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        self.n_impurity += (-div) * dt

        self.n_impurity[0] = self.n_impurity[1]
        self.n_impurity = np.maximum(0.0, self.n_impurity)

        z_imp = 6.0
        ne_safe = np.maximum(self.ne, 0.1) * 1e19
        n_imp_m3 = self.n_impurity * 1e19
        sum_nz2 = ne_safe + n_imp_m3 * z_imp**2
        sum_nz = ne_safe + n_imp_m3 * z_imp
        self._Z_eff = float(np.clip(np.mean(sum_nz2 / np.maximum(sum_nz, 1e10)), 1.0, 10.0))

    def calculate_bootstrap_current_simple(self, R0: float, B_pol: np.ndarray) -> np.ndarray:
        """Calibrated-heuristic Sauter bootstrap current density [A/m^2]."""
        a = (self.cfg["dimensions"]["R_max"] - self.cfg["dimensions"]["R_min"]) / 2.0
        r = self.rho * a
        epsilon = r / R0

        f_trapped = 1.46 * np.sqrt(epsilon)
        e_charge = 1.602e-16
        n_e = self.ne * 1e19
        dn_dr = np.gradient(n_e, self.drho * a)
        dte_dr = np.gradient(self.Te * e_charge, self.drho * a)
        dti_dr = np.gradient(self.Ti * e_charge, self.drho * a)

        b_pol = np.maximum(B_pol, 0.1)
        l31 = f_trapped / (1.0 + 0.3 * np.sqrt(epsilon))
        l32 = 0.5 * l31
        zeff_eff = float(np.clip(getattr(self, "_Z_eff", 1.5), 1.0, 5.0))
        l34 = -0.1 * l31 * (1.0 + 0.08 * (zeff_eff - 1.0))

        j_bs = -(1.0 / b_pol) * (
            l31 * (self.Te + self.Ti) * e_charge * dn_dr
            + l32 * n_e * dte_dr
            + l34 * n_e * dti_dr
        )
        j_bs *= 1.4
        j_bs[0] = 0
        j_bs[-1] = 0
        return j_bs

    def calculate_bootstrap_current(self, R0: float, B_pol: np.ndarray) -> np.ndarray:
        """Calculate bootstrap current; uses full Sauter when configured."""
        solver_mod = _solver_module()
        if hasattr(self, "neoclassical_params") and self.neoclassical_params is not None:
            return solver_mod.calculate_sauter_bootstrap_current_full(
                self.rho,
                self.Te,
                self.Ti,
                self.ne,
                self.neoclassical_params.get("q_profile", np.linspace(1, 4, len(self.rho))),
                R0,
                self.neoclassical_params.get("a", 2.0),
                self.neoclassical_params.get("B0", 5.3),
                self.neoclassical_params.get("Z_eff", 1.5),
            )
        return self.calculate_bootstrap_current_simple(R0, B_pol)

    def _gyro_bohm_chi(self) -> np.ndarray:
        """Gyro-Bohm anomalous transport diffusivity [m^2/s]."""
        solver_mod = _solver_module()
        if self.neoclassical_params is None:
            self._last_gyro_bohm_contract = {
                "used": False,
                "source": "neoclassical_disabled",
                "path": str(solver_mod._GYRO_BOHM_COEFF_PATH),
                "c_gB": float(solver_mod._GYRO_BOHM_DEFAULT),
                "fallback_used": True,
                "error": "neoclassical_params_missing",
            }
            return np.full_like(self.rho, 0.5)

        p = self.neoclassical_params
        R0 = p["R0"]
        a = p["a"]
        B0 = p["B0"]
        A_ion = p.get("A_ion", 2.0)
        q = p["q_profile"]

        if "c_gB" in p:
            try:
                c_gB = float(p["c_gB"])
                if (not np.isfinite(c_gB)) or c_gB <= 0.0:
                    raise ValueError(f"Invalid c_gB={p['c_gB']!r}")
                self._last_gyro_bohm_contract = {
                    "used": True,
                    "source": "neoclassical_params",
                    "path": None,
                    "c_gB": float(c_gB),
                    "fallback_used": False,
                    "error": None,
                }
            except (TypeError, ValueError) as exc:
                c_gB, fallback_contract = solver_mod._load_gyro_bohm_coefficient_cached_with_contract()
                self._last_gyro_bohm_contract = {
                    "used": True,
                    "source": "neoclassical_params_invalid_fallback",
                    "path": fallback_contract.get("path"),
                    "c_gB": float(c_gB),
                    "fallback_used": True,
                    "error": f"{exc.__class__.__name__}:{exc}",
                    "requested_c_gB": p.get("c_gB"),
                    "fallback_source": fallback_contract.get("source"),
                }
                record_fallback_event(
                    "integrated_transport_solver",
                    "gyro_bohm_invalid_param_fallback",
                    context={
                        "source": self._last_gyro_bohm_contract["source"],
                        "fallback_source": self._last_gyro_bohm_contract.get("fallback_source"),
                    },
                )
        else:
            c_gB, loader_contract = solver_mod._load_gyro_bohm_coefficient_cached_with_contract()
            self._last_gyro_bohm_contract = {
                "used": True,
                "source": loader_contract.get("source", "json_file"),
                "path": loader_contract.get("path"),
                "c_gB": float(c_gB),
                "fallback_used": bool(loader_contract.get("fallback_used", False)),
                "error": loader_contract.get("error"),
            }
            if self._last_gyro_bohm_contract["fallback_used"]:
                record_fallback_event(
                    "integrated_transport_solver",
                    "gyro_bohm_loader_fallback",
                    context={
                        "source": self._last_gyro_bohm_contract["source"],
                        "path": self._last_gyro_bohm_contract.get("path"),
                    },
                )

        e_charge = 1.602176634e-19
        m_i = A_ion * 1.672621924e-27

        ti_kev = np.maximum(self.Ti, 0.01)
        te_kev = np.maximum(self.Te, 0.01)
        qi = np.maximum(q, 0.5)

        t_i_j = ti_kev * 1e3 * e_charge
        t_e_j = te_kev * 1e3 * e_charge

        rho_s = np.sqrt(t_i_j * m_i) / (e_charge * B0)
        c_s = np.sqrt(t_e_j / m_i)

        denom = np.maximum(a * qi * R0, 1e-6)
        chi_gb = c_gB * rho_s**2 * c_s / denom
        chi_gb = np.where(np.isfinite(chi_gb), np.maximum(chi_gb, 0.01), 0.01)

        return chi_gb

    def update_transport_model(self, P_aux: float) -> None:
        """Gyro-Bohm + neoclassical transport model with EPED-like pedestal."""
        solver_mod = _solver_module()
        grad_t = np.gradient(self.Ti, self.drho)
        threshold = 2.0

        if self.neoclassical_params is not None:
            p = self.neoclassical_params
            chi_nc = solver_mod.chang_hinton_chi_profile(
                self.rho,
                self.Ti,
                self.ne,
                p["q_profile"],
                p["R0"],
                p["a"],
                p["B0"],
                p["A_ion"],
                p["Z_eff"],
            )
            chi_gb = self._gyro_bohm_chi()
            chi_base = chi_nc + chi_gb
        else:
            chi_base = np.full_like(self.rho, 0.5)

        chi_turb = 5.0 * np.maximum(0, -grad_t - threshold)
        is_h_mode = P_aux > 30.0
        self._last_pedestal_contract = {
            "used": False,
            "in_domain": True,
            "extrapolation_score": 0.0,
            "extrapolation_penalty": 1.0,
            "domain_violations": [],
            "fallback_used": False,
        }

        if is_h_mode and self.neoclassical_params is not None:
            try:
                p = self.neoclassical_params
                if self.pedestal_model is None or getattr(
                    self.pedestal_model, "_neo_params_hash", None
                ) != id(self.neoclassical_params):
                    eped = solver_mod.EpedPedestalModel(
                        R0=p["R0"],
                        a=p["a"],
                        B0=p["B0"],
                        Ip_MA=p.get("Ip_MA", 15.0),
                        kappa=p.get("kappa", 1.7),
                        A_ion=p.get("A_ion", 2.0),
                        Z_eff=p.get("Z_eff", 1.5),
                    )
                    eped._neo_params_hash = id(self.neoclassical_params)  # type: ignore[attr-defined]
                    self.pedestal_model = eped
                else:
                    eped = self.pedestal_model
                n_ped = max(float(self.ne[-5]), 1.0)
                ped = eped.predict(n_ped)
                self._last_pedestal_contract = {
                    "used": True,
                    "in_domain": bool(ped.in_domain),
                    "extrapolation_score": float(ped.extrapolation_score),
                    "extrapolation_penalty": float(ped.extrapolation_penalty),
                    "domain_violations": list(ped.domain_violations),
                    "fallback_used": False,
                    "n_ped_1e19": float(n_ped),
                }

                ped_start = 1.0 - ped.Delta_ped
                edge_mask = self.rho > ped_start
                chi_turb[edge_mask] *= 0.05

                ped_idx = np.searchsorted(self.rho, ped_start)
                if ped_idx < len(self.Te):
                    self.Te[ped_idx:] = np.minimum(
                        self.Te[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Te[ped_idx:])),
                    )
                    self.Ti[ped_idx:] = np.minimum(
                        self.Ti[ped_idx:],
                        ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Ti[ped_idx:])),
                    )
            except solver_mod._EPED_FALLBACK_EXCEPTIONS as exc:
                edge_mask = self.rho > 0.9
                chi_turb[edge_mask] *= 0.1
                self._last_pedestal_contract = {
                    "used": False,
                    "in_domain": False,
                    "extrapolation_score": 0.0,
                    "extrapolation_penalty": 1.0,
                    "domain_violations": [f"eped_failure:{exc}"],
                    "fallback_used": True,
                }
                record_fallback_event(
                    "integrated_transport_solver",
                    "eped_model_fallback",
                    context={"error": exc.__class__.__name__},
                )
        elif is_h_mode:
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1
            self._last_pedestal_contract = {
                "used": False,
                "in_domain": False,
                "extrapolation_score": 0.0,
                "extrapolation_penalty": 1.0,
                "domain_violations": ["neoclassical_params_missing"],
                "fallback_used": True,
            }
            record_fallback_event(
                "integrated_transport_solver",
                "eped_neoclassical_missing_fallback",
                context={"mode": "h_mode"},
            )

        self.chi_e = chi_base + chi_turb
        self.chi_i = chi_base + chi_turb
        self.D_n = 0.1 * self.chi_e

