# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Model Mixins
from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_fusion.core._integrated_transport_solver_model_common import _solver_module
from scpn_fusion.core.neural_transport import reduced_gyrokinetic_profile_model
from scpn_fusion.fallback_telemetry import record_fallback_event


class TransportSolverBackendMixin:
    def _compute_transport_backend_closure(
        self,
        *,
        transport_backend: str,
        chi_base: np.ndarray,
        chi_base_source: str,
        chi_gb_reference: np.ndarray | None,
        q_profile: np.ndarray,
        s_hat_profile: np.ndarray,
        r_major: float,
        a_minor: float,
        b_toroidal: float,
        q_profile_source: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        solver_mod = _solver_module()
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

        return chi_turb_e, chi_turb_i, d_turb
