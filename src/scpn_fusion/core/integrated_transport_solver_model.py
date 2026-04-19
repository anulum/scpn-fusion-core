# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Model Mixins
"""Model/configuration mixins extracted from integrated transport monolith."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.fallback_telemetry import record_fallback_event
from scpn_fusion.core.neural_transport import reduced_gyrokinetic_profile_model


def _solver_module() -> Any:
    """Resolve host integrated_transport_solver module lazily."""
    import scpn_fusion.core.integrated_transport_solver as solver_mod

    return solver_mod


class TransportSolverModelMixin:
    neoclassical_params: dict[str, Any] | None
    D_n: np.ndarray
    chi_e: np.ndarray
    chi_i: np.ndarray
    n_impurity: np.ndarray
    _neural_transport_model: Any

    def _get_neural_transport_model(self) -> Any:
        """Resolve the cached neural transport surrogate lazily."""
        from scpn_fusion.core.neural_transport import NeuralTransportModel

        weights_path = getattr(self, "neural_transport_weights_path", None)
        cached = getattr(self, "_neural_transport_model", None)
        cached_weights_path = getattr(self, "_neural_transport_model_weights_path", None)
        if cached is None or cached_weights_path != weights_path:
            cached = NeuralTransportModel(weights_path=weights_path)
            self._neural_transport_model = cached
            self._neural_transport_model_weights_path = weights_path
        return cached

    def _summarize_coarse_transport_channels(
        self,
        chi_e_profile: np.ndarray,
        chi_i_profile: np.ndarray,
    ) -> tuple[str, dict[str, int], dict[str, float]]:
        """Build an honest coarse channel summary from aggregate transport profiles."""
        chi_e_profile = np.asarray(chi_e_profile, dtype=np.float64)
        chi_i_profile = np.asarray(chi_i_profile, dtype=np.float64)
        stable_mask = (chi_e_profile <= 1e-6) & (chi_i_profile <= 1e-6)
        itg_mask = (~stable_mask) & (chi_i_profile >= chi_e_profile)
        electron_dominant_mask = (~stable_mask) & (~itg_mask)
        channel_counts = {
            "ITG": int(np.count_nonzero(itg_mask)),
            "TEM": int(np.count_nonzero(electron_dominant_mask)),
            "ETG": 0,
            "stable": int(np.count_nonzero(stable_mask)),
        }
        channel_energy = {
            "ITG": float(np.sum(chi_i_profile[itg_mask] + chi_e_profile[itg_mask])),
            "TEM": float(np.sum(chi_e_profile[electron_dominant_mask])),
            "ETG": 0.0,
        }
        dominant_channel = max(channel_energy.items(), key=lambda item: item[1])[0]
        if channel_energy[dominant_channel] <= 0.0:
            dominant_channel = "stable"
        return dominant_channel, channel_counts, channel_energy

    def _select_neural_ood_indices(
        self,
        max_abs_z_profile: np.ndarray,
        *,
        sigma_threshold: float,
        max_points: int,
    ) -> tuple[np.ndarray, list[int]]:
        """Pick the highest-severity interior OOD points for TGLF escalation."""
        sigma = float(sigma_threshold)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("neural_transport_tglf_ood_sigma must be finite and > 0.")
        if isinstance(max_points, bool) or int(max_points) <= 0:
            raise ValueError("neural_transport_tglf_max_points must be a positive integer.")

        z_profile = np.asarray(max_abs_z_profile, dtype=np.float64)
        if z_profile.shape != self.rho.shape:
            raise ValueError("Neural OOD profile shape must match transport rho grid.")

        ood_mask = z_profile > sigma
        interior_indices = np.nonzero(
            ood_mask
            & (np.arange(self.rho.size) > 0)
            & (np.arange(self.rho.size) < self.rho.size - 1)
        )[0]
        if interior_indices.size == 0:
            return ood_mask, []

        ranked = interior_indices[np.argsort(z_profile[interior_indices])[::-1]]
        selected = sorted(int(idx) for idx in ranked[: int(max_points)])
        return ood_mask, selected

    def _resolve_transport_closure_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, str]:
        """Resolve geometry and magnetic profiles for the reduced transport closure."""
        if self.neoclassical_params is not None:
            q_profile = np.asarray(
                self.neoclassical_params.get("q_profile", 1.0 + 2.0 * self.rho**2),
                dtype=np.float64,
            )
            q_profile_source = "neoclassical_params"
        else:
            q_profile = 1.0 + 2.0 * self.rho**2
            q_profile_source = "default_parabolic"

        if (
            q_profile.shape != self.rho.shape
            or (not np.all(np.isfinite(q_profile)))
            or np.any(q_profile <= 0.0)
        ):
            q_profile = 1.0 + 2.0 * self.rho**2
            q_profile_source = "fallback_parabolic"

        dims = self.cfg["dimensions"]
        default_r_major = 0.5 * (dims["R_min"] + dims["R_max"])
        default_a_minor = 0.5 * (dims["R_max"] - dims["R_min"])
        default_b0 = float(self.cfg.get("physics", {}).get("B0", self.cfg.get("B0", 5.3)))

        params = self.neoclassical_params or {}
        r_major = float(params.get("R0", default_r_major))
        a_minor = float(params.get("a", default_a_minor))
        b_toroidal = float(params.get("B0", default_b0))

        dq_drho = np.gradient(q_profile, self.rho)
        s_hat_profile = self.rho * dq_drho / np.maximum(q_profile, 0.2)
        s_hat_profile = np.clip(
            np.nan_to_num(s_hat_profile, nan=0.0, posinf=10.0, neginf=0.0),
            0.0,
            10.0,
        )
        return q_profile, s_hat_profile, r_major, a_minor, b_toroidal, q_profile_source

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

        details = (
            ", ".join(
                f"{name}={count}"
                for name, count in sorted(self._last_numerical_recovery_breakdown.items())
            )
            or "no breakdown"
        )
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
        rho = np.asarray(self.rho, dtype=np.float64)

        t_i_raw = getattr(self, "t_i", None)
        if t_i_raw is None:
            t_i_raw = self.Ti
        t_i = np.asarray(t_i_raw, dtype=np.float64)

        n_e_raw = getattr(self, "n_e", None)
        if n_e_raw is None:
            n_e_raw = self.ne
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
            l31 * (self.Te + self.Ti) * e_charge * dn_dr + l32 * n_e * dte_dr + l34 * n_e * dti_dr
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
                c_gB, fallback_contract = (
                    solver_mod._load_gyro_bohm_coefficient_cached_with_contract()
                )
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
        """Reduced multichannel transport model with optional EPED pedestal."""
        solver_mod = _solver_module()
        q_profile, s_hat_profile, r_major, a_minor, b_toroidal, q_profile_source = (
            self._resolve_transport_closure_inputs()
        )
        self.q_profile = q_profile

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
            chi_gb_reference = self._gyro_bohm_chi()
            chi_base = chi_nc
            chi_base_source = "chang_hinton"
        else:
            chi_base = np.full_like(self.rho, 0.5)
            chi_gb_reference = None
            chi_base_source = "constant_fallback"

        transport_backend = str(getattr(self, "transport_backend", "reduced_multichannel"))
        transport_backend_key = transport_backend.strip().lower()

        try:
            if transport_backend_key == "tglf_live":
                from scpn_fusion.core.tglf_interface import run_tglf_profile_scan

                tglf_binary_path = getattr(self, "tglf_binary_path", None)
                if not tglf_binary_path:
                    raise ValueError("tglf_live backend requires physics.tglf_binary_path.")

                scan = run_tglf_profile_scan(
                    self,
                    tglf_binary_path,
                    timeout_s=float(getattr(self, "tglf_timeout_s", 120.0)),
                    max_retries=int(getattr(self, "tglf_max_retries", 2)),
                )
                chi_turb_i = np.asarray(scan.chi_i_profile, dtype=np.float64)
                chi_turb_e = np.asarray(scan.chi_e_profile, dtype=np.float64)
                d_turb = np.maximum(0.15 * chi_turb_e, 0.05 * chi_turb_i)
                dominant_channel = (
                    "ETG" if float(np.mean(chi_turb_e)) > float(np.mean(chi_turb_i)) else "ITG"
                )
                self._last_transport_closure_contract = {
                    "used": True,
                    "model": "tglf_live_profile",
                    "requested_backend": transport_backend,
                    "fallback_used": False,
                    "base_source": chi_base_source,
                    "q_profile_source": q_profile_source,
                    "dominant_channel": dominant_channel,
                    "channel_counts": {"ITG": 0, "TEM": 0, "ETG": 0, "stable": 0},
                    "channel_energy": {
                        "ITG": float(np.sum(chi_turb_i)),
                        "TEM": 0.0,
                        "ETG": float(np.sum(chi_turb_e)),
                    },
                    "gradient_clip_counts": {"grad_te": 0, "grad_ti": 0, "grad_ne": 0},
                    "profile_contract": {
                        "n_points": int(self.rho.size),
                        "rho_min": float(self.rho[0]),
                        "rho_max": float(self.rho[-1]),
                        "r_major": r_major,
                        "a_minor": a_minor,
                        "b_toroidal": b_toroidal,
                        "rho_samples": list(scan.rho_samples),
                    },
                    "chi_base_mean": float(np.mean(chi_base)),
                    "chi_e_turb_mean": float(np.mean(chi_turb_e)),
                    "chi_i_turb_mean": float(np.mean(chi_turb_i)),
                    "d_turb_mean": float(np.mean(d_turb)),
                    "chi_gb_reference_mean": (
                        None if chi_gb_reference is None else float(np.mean(chi_gb_reference))
                    ),
                    "weights_loaded": False,
                    "weights_path": None,
                    "weights_checksum": None,
                    "classification_mode": "external_tglf_scan",
                    "ood_fraction_3sigma": 0.0,
                    "ood_fraction_5sigma": 0.0,
                    "ood_point_count": 0,
                    "max_abs_z": 0.0,
                    "tglf_sample_count": int(len(scan.rho_samples)),
                    "tglf_sample_indices": [],
                    "gamma_profile_mean": float(
                        np.mean(np.asarray(scan.gamma_profile, dtype=np.float64))
                    ),
                    "error": None,
                }
            elif transport_backend_key in {"neural_transport_hybrid", "qlknn_tglf_hybrid"}:
                from scpn_fusion.core.tglf_interface import run_tglf_profile_scan

                tglf_binary_path = getattr(self, "tglf_binary_path", None)
                if not tglf_binary_path:
                    raise ValueError(
                        "neural_transport_hybrid backend requires physics.tglf_binary_path."
                    )

                neural_model = self._get_neural_transport_model()
                if not getattr(neural_model, "is_neural", False):
                    weights_path = getattr(neural_model, "weights_path", None)
                    resolved_path = None if weights_path is None else Path(weights_path)
                    if resolved_path is not None and not resolved_path.exists():
                        raise FileNotFoundError(
                            f"Neural transport weights not found at {resolved_path}."
                        )
                    raise RuntimeError(
                        "Neural transport backend requested but no valid weights were loaded."
                    )

                chi_turb_e, chi_turb_i, d_turb = neural_model.predict_profile(
                    self.rho,
                    self.Te,
                    self.Ti,
                    self.ne,
                    q_profile,
                    s_hat_profile,
                    r_major=r_major,
                    a_minor=a_minor,
                    b_toroidal=b_toroidal,
                )
                surrogate_meta = dict(getattr(neural_model, "_last_surrogate_contract", {}))
                max_abs_z_profile = np.asarray(
                    getattr(neural_model, "_last_max_abs_z_profile", np.zeros_like(self.rho)),
                    dtype=np.float64,
                )
                ood_mask, selected_indices = self._select_neural_ood_indices(
                    max_abs_z_profile,
                    sigma_threshold=float(getattr(self, "neural_transport_tglf_ood_sigma", 5.0)),
                    max_points=int(getattr(self, "neural_transport_tglf_max_points", 7)),
                )
                gamma_profile_mean = 0.0
                if selected_indices:
                    scan = run_tglf_profile_scan(
                        self,
                        tglf_binary_path,
                        rho_indices=selected_indices,
                        timeout_s=float(getattr(self, "tglf_timeout_s", 120.0)),
                        max_retries=int(getattr(self, "tglf_max_retries", 2)),
                    )
                    scan_chi_i = np.asarray(scan.chi_i_profile, dtype=np.float64)
                    scan_chi_e = np.asarray(scan.chi_e_profile, dtype=np.float64)
                    scan_gamma = np.asarray(scan.gamma_profile, dtype=np.float64)
                    chi_turb_i = np.asarray(chi_turb_i, dtype=np.float64)
                    chi_turb_e = np.asarray(chi_turb_e, dtype=np.float64)
                    d_turb = np.asarray(d_turb, dtype=np.float64)
                    chi_turb_i[ood_mask] = scan_chi_i[ood_mask]
                    chi_turb_e[ood_mask] = scan_chi_e[ood_mask]
                    d_turb[ood_mask] = np.maximum(
                        0.15 * chi_turb_e[ood_mask],
                        0.05 * chi_turb_i[ood_mask],
                    )
                    gamma_profile_mean = float(np.mean(scan_gamma))

                dominant_channel, channel_counts, channel_energy = (
                    self._summarize_coarse_transport_channels(chi_turb_e, chi_turb_i)
                )
                gradient_clip_counts = dict(
                    surrogate_meta.get(
                        "gradient_clip_counts",
                        {"grad_te": 0, "grad_ti": 0, "grad_ne": 0},
                    )
                )
                profile_contract = dict(
                    surrogate_meta.get(
                        "profile_contract",
                        {
                            "n_points": int(self.rho.size),
                            "rho_min": float(self.rho[0]),
                            "rho_max": float(self.rho[-1]),
                            "r_major": r_major,
                            "a_minor": a_minor,
                            "b_toroidal": b_toroidal,
                        },
                    )
                )
                profile_contract.setdefault("n_points", int(self.rho.size))
                profile_contract.setdefault("rho_min", float(self.rho[0]))
                profile_contract.setdefault("rho_max", float(self.rho[-1]))
                profile_contract.setdefault("r_major", r_major)
                profile_contract.setdefault("a_minor", a_minor)
                profile_contract.setdefault("b_toroidal", b_toroidal)
                profile_contract["ood_sigma_threshold"] = float(
                    getattr(self, "neural_transport_tglf_ood_sigma", 5.0)
                )
                self._last_transport_closure_contract = {
                    "used": True,
                    "model": "qlknn_tglf_hybrid",
                    "requested_backend": transport_backend,
                    "fallback_used": False,
                    "base_source": chi_base_source,
                    "q_profile_source": q_profile_source,
                    "dominant_channel": dominant_channel,
                    "channel_counts": channel_counts,
                    "channel_energy": channel_energy,
                    "gradient_clip_counts": gradient_clip_counts,
                    "profile_contract": profile_contract,
                    "chi_base_mean": float(np.mean(chi_base)),
                    "chi_e_turb_mean": float(np.mean(chi_turb_e)),
                    "chi_i_turb_mean": float(np.mean(chi_turb_i)),
                    "d_turb_mean": float(np.mean(d_turb)),
                    "chi_gb_reference_mean": (
                        None if chi_gb_reference is None else float(np.mean(chi_gb_reference))
                    ),
                    "weights_loaded": bool(surrogate_meta.get("weights_loaded", True)),
                    "weights_path": surrogate_meta.get("weights_path"),
                    "weights_checksum": surrogate_meta.get("weights_checksum"),
                    "classification_mode": "hybrid_neural_tglf_ood_escalation",
                    "ood_fraction_3sigma": float(surrogate_meta.get("ood_fraction_3sigma", 0.0)),
                    "ood_fraction_5sigma": float(surrogate_meta.get("ood_fraction_5sigma", 0.0)),
                    "ood_point_count": int(np.count_nonzero(ood_mask)),
                    "max_abs_z": float(surrogate_meta.get("max_abs_z", 0.0)),
                    "tglf_sample_count": int(len(selected_indices)),
                    "tglf_sample_indices": list(selected_indices),
                    "gamma_profile_mean": float(gamma_profile_mean),
                    "error": None,
                }
            elif transport_backend_key in {"neural_transport", "qlknn"}:
                neural_model = self._get_neural_transport_model()
                if not getattr(neural_model, "is_neural", False):
                    weights_path = getattr(neural_model, "weights_path", None)
                    resolved_path = None if weights_path is None else Path(weights_path)
                    if resolved_path is not None and not resolved_path.exists():
                        raise FileNotFoundError(
                            f"Neural transport weights not found at {resolved_path}."
                        )
                    raise RuntimeError(
                        "Neural transport backend requested but no valid weights were loaded."
                    )

                chi_turb_e, chi_turb_i, d_turb = neural_model.predict_profile(
                    self.rho,
                    self.Te,
                    self.Ti,
                    self.ne,
                    q_profile,
                    s_hat_profile,
                    r_major=r_major,
                    a_minor=a_minor,
                    b_toroidal=b_toroidal,
                )
                surrogate_meta = dict(getattr(neural_model, "_last_surrogate_contract", {}))
                gradient_clip_counts = dict(
                    surrogate_meta.get(
                        "gradient_clip_counts",
                        {"grad_te": 0, "grad_ti": 0, "grad_ne": 0},
                    )
                )
                channel_counts = dict(
                    surrogate_meta.get(
                        "channel_counts",
                        {"ITG": 0, "TEM": 0, "ETG": 0, "stable": 0},
                    )
                )
                channel_energy = dict(
                    surrogate_meta.get(
                        "channel_energy",
                        {"ITG": 0.0, "TEM": 0.0, "ETG": 0.0},
                    )
                )
                profile_contract = dict(
                    surrogate_meta.get(
                        "profile_contract",
                        {
                            "n_points": int(self.rho.size),
                            "rho_min": float(self.rho[0]),
                            "rho_max": float(self.rho[-1]),
                            "r_major": r_major,
                            "a_minor": a_minor,
                            "b_toroidal": b_toroidal,
                        },
                    )
                )
                profile_contract.setdefault("n_points", int(self.rho.size))
                profile_contract.setdefault("rho_min", float(self.rho[0]))
                profile_contract.setdefault("rho_max", float(self.rho[-1]))
                profile_contract.setdefault("r_major", r_major)
                profile_contract.setdefault("a_minor", a_minor)
                profile_contract.setdefault("b_toroidal", b_toroidal)
                self._last_transport_closure_contract = {
                    "used": True,
                    "model": str(surrogate_meta.get("model", "qlknn_profile_surrogate")),
                    "requested_backend": transport_backend,
                    "fallback_used": False,
                    "base_source": chi_base_source,
                    "q_profile_source": q_profile_source,
                    "dominant_channel": str(surrogate_meta.get("dominant_channel", "stable")),
                    "channel_counts": channel_counts,
                    "channel_energy": channel_energy,
                    "gradient_clip_counts": gradient_clip_counts,
                    "profile_contract": profile_contract,
                    "chi_base_mean": float(np.mean(chi_base)),
                    "chi_e_turb_mean": float(np.mean(chi_turb_e)),
                    "chi_i_turb_mean": float(np.mean(chi_turb_i)),
                    "d_turb_mean": float(np.mean(d_turb)),
                    "chi_gb_reference_mean": (
                        None if chi_gb_reference is None else float(np.mean(chi_gb_reference))
                    ),
                    "weights_loaded": bool(surrogate_meta.get("weights_loaded", True)),
                    "weights_path": surrogate_meta.get("weights_path"),
                    "weights_checksum": surrogate_meta.get("weights_checksum"),
                    "classification_mode": str(
                        surrogate_meta.get("classification_mode", "coarse_ion_vs_electron_dominant")
                    ),
                    "ood_fraction_3sigma": float(surrogate_meta.get("ood_fraction_3sigma", 0.0)),
                    "ood_fraction_5sigma": float(surrogate_meta.get("ood_fraction_5sigma", 0.0)),
                    "ood_point_count": int(surrogate_meta.get("ood_point_count_5sigma", 0)),
                    "max_abs_z": float(surrogate_meta.get("max_abs_z", 0.0)),
                    "tglf_sample_count": 0,
                    "tglf_sample_indices": [],
                    "error": None,
                }
            else:
                chi_turb_e, chi_turb_i, d_turb, closure_meta = reduced_gyrokinetic_profile_model(
                    self.rho,
                    self.Te,
                    self.Ti,
                    self.ne,
                    q_profile,
                    s_hat_profile,
                    r_major=r_major,
                    a_minor=a_minor,
                    b_toroidal=b_toroidal,
                )
                self._last_transport_closure_contract = {
                    "used": True,
                    "model": str(closure_meta["model"]),
                    "requested_backend": transport_backend,
                    "fallback_used": False,
                    "base_source": chi_base_source,
                    "q_profile_source": q_profile_source,
                    "dominant_channel": str(closure_meta["dominant_channel"]),
                    "channel_counts": dict(closure_meta["channel_counts"]),
                    "channel_energy": dict(closure_meta["channel_energy"]),
                    "gradient_clip_counts": dict(closure_meta["gradient_clip_counts"]),
                    "profile_contract": dict(closure_meta["profile_contract"]),
                    "chi_base_mean": float(np.mean(chi_base)),
                    "chi_e_turb_mean": float(np.mean(chi_turb_e)),
                    "chi_i_turb_mean": float(np.mean(chi_turb_i)),
                    "d_turb_mean": float(np.mean(d_turb)),
                    "chi_gb_reference_mean": (
                        None if chi_gb_reference is None else float(np.mean(chi_gb_reference))
                    ),
                    "weights_loaded": False,
                    "weights_path": None,
                    "weights_checksum": None,
                    "classification_mode": "multichannel_reduced",
                    "ood_fraction_3sigma": 0.0,
                    "ood_fraction_5sigma": 0.0,
                    "ood_point_count": 0,
                    "max_abs_z": 0.0,
                    "tglf_sample_count": 0,
                    "tglf_sample_indices": [],
                    "error": None,
                }
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            grad_t = np.gradient(self.Ti, self.drho)
            chi_turb_i = 5.0 * np.maximum(0.0, -grad_t - 2.0)
            chi_turb_e = 0.35 * chi_turb_i
            d_turb = 0.1 * chi_turb_e
            dominant_channel = "ITG" if float(np.sum(chi_turb_i)) > 0.0 else "stable"
            self._last_transport_closure_contract = {
                "used": False,
                "model": "legacy_ti_threshold_fallback",
                "requested_backend": transport_backend,
                "fallback_used": True,
                "base_source": chi_base_source,
                "q_profile_source": q_profile_source,
                "dominant_channel": dominant_channel,
                "channel_counts": {
                    "ITG": int(np.count_nonzero(chi_turb_i > 0.0)),
                    "TEM": 0,
                    "ETG": 0,
                    "stable": int(np.count_nonzero(chi_turb_i <= 0.0)),
                },
                "channel_energy": {
                    "ITG": float(np.sum(chi_turb_i + chi_turb_e)),
                    "TEM": 0.0,
                    "ETG": 0.0,
                },
                "gradient_clip_counts": {"grad_te": 0, "grad_ti": 0, "grad_ne": 0},
                "profile_contract": {
                    "n_points": int(self.rho.size),
                    "rho_min": float(self.rho[0]),
                    "rho_max": float(self.rho[-1]),
                    "r_major": r_major,
                    "a_minor": a_minor,
                    "b_toroidal": b_toroidal,
                },
                "chi_base_mean": float(np.mean(chi_base)),
                "chi_e_turb_mean": float(np.mean(chi_turb_e)),
                "chi_i_turb_mean": float(np.mean(chi_turb_i)),
                "d_turb_mean": float(np.mean(d_turb)),
                "chi_gb_reference_mean": (
                    None if chi_gb_reference is None else float(np.mean(chi_gb_reference))
                ),
                "weights_loaded": False,
                "weights_path": None,
                "weights_checksum": None,
                "classification_mode": "legacy_itg_threshold",
                "ood_fraction_3sigma": 0.0,
                "ood_fraction_5sigma": 0.0,
                "ood_point_count": 0,
                "max_abs_z": 0.0,
                "tglf_sample_count": 0,
                "tglf_sample_indices": [],
                "error": f"{exc.__class__.__name__}:{exc}",
            }
            record_fallback_event(
                "integrated_transport_solver",
                "transport_closure_fallback",
                context={"error": exc.__class__.__name__},
            )

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
                chi_turb_e[edge_mask] *= 0.05
                chi_turb_i[edge_mask] *= 0.05
                d_turb[edge_mask] *= 0.05

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
                chi_turb_e[edge_mask] *= 0.1
                chi_turb_i[edge_mask] *= 0.1
                d_turb[edge_mask] *= 0.1
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
            chi_turb_e[edge_mask] *= 0.1
            chi_turb_i[edge_mask] *= 0.1
            d_turb[edge_mask] *= 0.1
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

        self.chi_e = np.maximum(chi_base + chi_turb_e, 1e-6)
        self.chi_i = np.maximum(chi_base + chi_turb_i, 1e-6)
        self.D_n = np.maximum(d_turb, 0.1 * chi_base)
