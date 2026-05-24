# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Runtime model loader and profile inference for neural transport."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scpn_fusion.io.safe_loaders import checked_np_load
from numpy.typing import NDArray

from ._neural_transport_analytic import (
    _CRIT_ITG,
    _CRIT_TEM,
    _TRANSPORT_FLOOR,
    critical_gradient_model,
    reduced_gyrokinetic_profile_model,
)
from ._neural_transport_types import (
    FloatArray,
    MLPWeights,
    TransportFluxes,
    TransportInputs,
    _MAX_WEIGHTS_FILE_BYTES,
    _WEIGHTS_FORMAT_VERSION,
)
from .neural_transport_math import _compute_nustar, _mlp_forward


logger = logging.getLogger(__name__)


def _append_derived(x: FloatArray, inp: TransportInputs, expected_dim: int) -> FloatArray:
    """Append Ti/Te, collisionality, excess gradients, and log chi_GB features."""
    if expected_dim <= 10:
        return x
    if expected_dim >= 12:
        ti_te = inp.ti_kev / max(inp.te_kev, 1e-6)
        nustar = _compute_nustar(inp.te_kev, inp.ne_19, inp.q, inp.rho)
        x = np.append(x, [ti_te, nustar])
    if expected_dim >= 14:
        itg_excess = max(0.0, inp.grad_ti - _CRIT_ITG)
        tem_excess = max(0.0, inp.grad_te - _CRIT_TEM)
        x = np.append(x, [itg_excess, tem_excess])
    if expected_dim >= 15:
        te_j = inp.te_kev * 1e3 * 1.602e-19
        cs = np.sqrt(te_j / 3.344e-27)
        rho_s = np.sqrt(3.344e-27 * te_j) / (1.602e-19 * 5.3)
        chi_gb = rho_s**2 * cs / 6.2
        x = np.append(x, [np.log(max(chi_gb, 1e-10))])
    return x


class NeuralTransportModel:
    """Neural transport surrogate with analytic fallback.

    On construction, attempts to load MLP weights from *weights_path*.
    If loading fails (file missing, wrong format), the model
    transparently falls back to :func:`critical_gradient_model`.
    """

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        self._weights: Optional[MLPWeights] = None
        self.is_neural: bool = False

        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parents[3] / "weights" / "neural_transport_qlknn.npz"
            )

        self.weights_path: Optional[Path] = None
        self.weights_checksum: Optional[str] = None
        self._last_gradient_clip_counts: dict[str, int] = {"grad_te": 0, "grad_ti": 0, "grad_ne": 0}
        self._last_profile_contract: dict[str, float | int] = {"n_points": 0}
        self._last_max_abs_z_profile: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._last_ood_mask_3sigma: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._last_ood_mask_5sigma: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._last_surrogate_contract: dict[str, Any] = {
            "model": "reduced_multichannel_analytic",
            "backend": "analytic_fallback",
            "weights_loaded": False,
            "weights_path": str(weights_path) if weights_path is not None else None,
            "weights_checksum": None,
            "classification_mode": "multichannel_reduced",
            "dominant_channel": "stable",
            "channel_counts": {"ITG": 0, "TEM": 0, "ETG": 0, "stable": 0},
            "channel_energy": {"ITG": 0.0, "TEM": 0.0, "ETG": 0.0},
            "gradient_clip_counts": dict(self._last_gradient_clip_counts),
            "profile_contract": dict(self._last_profile_contract),
            "ood_fraction_3sigma": 0.0,
            "ood_fraction_5sigma": 0.0,
            "ood_point_count_3sigma": 0,
            "ood_point_count_5sigma": 0,
            "max_abs_z": 0.0,
            "input_dim": 0,
            "n_layers": 0,
            "gated": False,
            "gb_scale": False,
            "log_transform": False,
        }

        if weights_path is not None:
            self.weights_path = Path(weights_path)
            self._try_load_weights()

    def _try_load_weights(self) -> None:
        """Attempt to load MLP weights from disk."""
        if self.weights_path is None or not self.weights_path.exists():
            logger.info(
                "Neural transport weights not found at %s - using "
                "critical-gradient compatibility model",
                self.weights_path,
            )
            return

        if self.weights_path.suffix.lower() != ".npz":
            logger.warning(
                "Neural transport weights must be a .npz file (got %s) - falling back",
                self.weights_path,
            )
            return

        try:
            file_size = int(self.weights_path.stat().st_size)
        except OSError:
            logger.exception("Failed to stat neural transport weights file")
            return
        if file_size <= 0 or file_size > _MAX_WEIGHTS_FILE_BYTES:
            logger.warning(
                "Neural transport weights size %d bytes outside allowed range (1..%d) - falling back",
                file_size,
                _MAX_WEIGHTS_FILE_BYTES,
            )
            return

        try:
            with checked_np_load(self.weights_path, allow_pickle=False) as data:
                n_layers = 0
                while f"w{n_layers + 1}" in data and f"b{n_layers + 1}" in data:
                    n_layers += 1

                if n_layers < 2:
                    logger.warning(
                        "Weight file has only %d layer(s) (need >=2) - falling back",
                        n_layers,
                    )
                    return

                for key in ("input_mean", "input_std", "output_scale"):
                    if key not in data:
                        logger.warning("Weight file missing key '%s' - falling back", key)
                        return

                version = int(data["version"]) if "version" in data else 1
                if version != _WEIGHTS_FORMAT_VERSION:
                    logger.warning(
                        "Weight file version %d != expected %d - falling back",
                        version,
                        _WEIGHTS_FORMAT_VERSION,
                    )
                    return

                layers_w = [data[f"w{i + 1}"] for i in range(n_layers)]
                layers_b = [data[f"b{i + 1}"] for i in range(n_layers)]
                log_transform = (
                    bool(int(data["log_transform"])) if "log_transform" in data else False
                )
                gb_scale = bool(int(data["gb_scale"])) if "gb_scale" in data else False
                gated = bool(int(data["gated"])) if "gated" in data else False

                self._weights = MLPWeights(
                    layers_w=layers_w,
                    layers_b=layers_b,
                    input_mean=data["input_mean"],
                    input_std=data["input_std"],
                    output_scale=data["output_scale"],
                    log_transform=log_transform,
                    gb_scale=gb_scale,
                    gated=gated,
                )
                self.is_neural = True

                raw = b"".join(data[k].tobytes() for k in sorted(data.files) if k != "version")
                self.weights_checksum = hashlib.sha256(raw).hexdigest()[:16]

                dims = [str(layers_w[0].shape[0])]
                for w in layers_w:
                    dims.append(str(w.shape[1]))
                arch_str = "->".join(dims)

                logger.info(
                    "Loaded neural transport weights from %s "
                    "(architecture: %s, %d layers, version=%d, sha256=%s)",
                    self.weights_path,
                    arch_str,
                    n_layers,
                    version,
                    self.weights_checksum,
                )
        except Exception:
            logger.exception("Failed to load neural transport weights")

    def predict(self, inp: TransportInputs) -> TransportFluxes:
        """Predict turbulent transport fluxes for given local parameters."""
        if not self.is_neural or self._weights is None:
            return critical_gradient_model(inp)

        x = np.array(
            [
                inp.rho,
                inp.te_kev,
                inp.ti_kev,
                inp.ne_19,
                inp.grad_te,
                inp.grad_ti,
                inp.grad_ne,
                inp.q,
                inp.s_hat,
                inp.beta_e,
            ]
        )
        expected_dim = self._weights.layers_w[0].shape[0]
        x = _append_derived(x, inp, expected_dim)
        out = _mlp_forward(x, self._weights)

        chi_e = float(out[0])
        chi_i = float(out[1])
        d_e = float(out[2])

        if chi_i > chi_e and chi_i > 0:
            channel = "ITG"
        elif chi_e > 0:
            channel = "TEM"
        else:
            channel = "stable"

        return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)

    def predict_profile(
        self,
        rho: FloatArray,
        te: FloatArray,
        ti: FloatArray,
        ne: FloatArray,
        q_profile: FloatArray,
        s_hat_profile: FloatArray,
        r_major: float = 6.2,
        a_minor: float = 2.0,
        b_toroidal: float = 5.3,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict transport coefficients on the full radial profile."""
        rho = np.asarray(rho, dtype=np.float64)
        te = np.asarray(te, dtype=np.float64)
        ti = np.asarray(ti, dtype=np.float64)
        ne = np.asarray(ne, dtype=np.float64)
        q_profile = np.asarray(q_profile, dtype=np.float64)
        s_hat_profile = np.asarray(s_hat_profile, dtype=np.float64)

        if self.is_neural and self._weights is not None:
            if any(arr.ndim != 1 for arr in (rho, te, ti, ne, q_profile, s_hat_profile)):
                raise ValueError("rho/te/ti/ne/q_profile/s_hat_profile must all be 1D arrays.")
            n = int(rho.size)
            if n < 3:
                raise ValueError("profile arrays must contain at least 3 points.")
            if any(arr.size != n for arr in (te, ti, ne, q_profile, s_hat_profile)):
                raise ValueError("profile arrays must all have identical length.")
            if not np.all(np.isfinite(rho)):
                raise ValueError("rho must contain finite values.")
            if not np.all(np.isfinite(te)):
                raise ValueError("te must contain finite values.")
            if not np.all(np.isfinite(ti)):
                raise ValueError("ti must contain finite values.")
            if not np.all(np.isfinite(ne)):
                raise ValueError("ne must contain finite values.")
            if not np.all(np.isfinite(q_profile)):
                raise ValueError("q_profile must contain finite values.")
            if not np.all(np.isfinite(s_hat_profile)):
                raise ValueError("s_hat_profile must contain finite values.")
            if not np.all(np.diff(rho) > 0.0):
                raise ValueError("rho must be strictly increasing.")
            if rho[0] < 0.0 or rho[-1] > 1.2:
                raise ValueError("rho must satisfy 0 <= rho <= 1.2.")
            r_major = float(r_major)
            if (not np.isfinite(r_major)) or r_major <= 0.0:
                raise ValueError("r_major must be finite and > 0.")
            if (not np.isfinite(a_minor)) or a_minor <= 0.0:
                raise ValueError("a_minor must be finite and > 0.")
            if (not np.isfinite(b_toroidal)) or b_toroidal <= 0.0:
                raise ValueError("b_toroidal must be finite and > 0.")

            def norm_grad(x: FloatArray) -> FloatArray:
                dx = np.gradient(x, rho)
                safe_x = np.maximum(np.abs(x), 1e-6)
                return -r_major * dx / safe_x

            grad_te_raw = norm_grad(te)
            grad_ti_raw = norm_grad(ti)
            grad_ne_raw = norm_grad(ne)
            grad_te = np.clip(grad_te_raw, 0, 50)
            grad_ti = np.clip(grad_ti_raw, 0, 50)
            grad_ne = np.clip(grad_ne_raw, -10, 30)
            beta_e = 4.03e-3 * ne * te

            self._last_gradient_clip_counts = {
                "grad_te": int(np.count_nonzero((grad_te_raw < 0.0) | (grad_te_raw > 50.0))),
                "grad_ti": int(np.count_nonzero((grad_ti_raw < 0.0) | (grad_ti_raw > 50.0))),
                "grad_ne": int(np.count_nonzero((grad_ne_raw < -10.0) | (grad_ne_raw > 30.0))),
            }
            self._last_profile_contract = {
                "n_points": int(n),
                "rho_min": float(rho[0]),
                "rho_max": float(rho[-1]),
                "r_major": float(r_major),
            }
            x_batch = np.column_stack(
                [
                    rho,
                    te,
                    ti,
                    ne,
                    grad_te,
                    grad_ti,
                    grad_ne,
                    q_profile,
                    s_hat_profile,
                    beta_e,
                ]
            )
            expected_dim = self._weights.layers_w[0].shape[0]
            if expected_dim >= 12:
                ti_te = ti / np.maximum(te, 1e-6)
                nustar = np.array(
                    [
                        _compute_nustar(te[i], ne[i], q_profile[i], rho[i], r_major, a_minor)
                        for i in range(n)
                    ]
                )
                x_batch = np.column_stack([x_batch, ti_te, nustar])
            if expected_dim >= 14:
                itg_excess = np.maximum(0.0, grad_ti - _CRIT_ITG)
                tem_excess = np.maximum(0.0, grad_te - _CRIT_TEM)
                x_batch = np.column_stack([x_batch, itg_excess, tem_excess])
            if expected_dim >= 15:
                te_j = te * 1e3 * 1.602e-19
                cs = np.sqrt(te_j / 3.344e-27)
                rho_s = np.sqrt(3.344e-27 * te_j) / (1.602e-19 * b_toroidal)
                chi_gb = rho_s**2 * cs / r_major
                log_chi_gb = np.log(np.maximum(chi_gb, 1e-10))
                x_batch = np.column_stack([x_batch, log_chi_gb])
            if self._weights.input_mean.shape[0] != expected_dim:
                raise ValueError(
                    "Neural transport input_mean dimension does not match the "
                    f"expected feature width ({self._weights.input_mean.shape[0]} != {expected_dim})."
                )
            if self._weights.input_std.shape[0] != expected_dim:
                raise ValueError(
                    "Neural transport input_std dimension does not match the "
                    f"expected feature width ({self._weights.input_std.shape[0]} != {expected_dim})."
                )
            safe_input_std = np.where(
                np.abs(self._weights.input_std) < 1e-8, 1.0, self._weights.input_std
            )
            z_batch = np.abs(
                (x_batch - self._weights.input_mean[None, :]) / safe_input_std[None, :]
            )
            ood_mask_3sigma = np.any(z_batch > 3.0, axis=1)
            ood_mask_5sigma = np.any(z_batch > 5.0, axis=1)
            self._last_max_abs_z_profile = np.max(z_batch, axis=1)
            self._last_ood_mask_3sigma = np.asarray(ood_mask_3sigma, dtype=bool)
            self._last_ood_mask_5sigma = np.asarray(ood_mask_5sigma, dtype=bool)
            out = _mlp_forward(x_batch, self._weights)
            chi_e_out = out[:, 0]
            chi_i_out = out[:, 1]
            d_e_out = out[:, 2]
            stable_mask = (chi_e_out <= _TRANSPORT_FLOOR) & (chi_i_out <= _TRANSPORT_FLOOR)
            itg_mask = (~stable_mask) & (chi_i_out >= chi_e_out)
            electron_dominant_mask = (~stable_mask) & (~itg_mask)
            channel_counts = {
                "ITG": int(np.count_nonzero(itg_mask)),
                "TEM": int(np.count_nonzero(electron_dominant_mask)),
                "ETG": 0,
                "stable": int(np.count_nonzero(stable_mask)),
            }
            channel_energy = {
                "ITG": float(np.sum(chi_i_out[itg_mask] + chi_e_out[itg_mask])),
                "TEM": float(np.sum(chi_e_out[electron_dominant_mask])),
                "ETG": 0.0,
            }
            dominant_channel = max(channel_energy.items(), key=lambda item: item[1])[0]
            if channel_energy[dominant_channel] <= 0.0:
                dominant_channel = "stable"
            profile_contract = {
                "n_points": int(n),
                "rho_min": float(rho[0]),
                "rho_max": float(rho[-1]),
                "r_major": float(r_major),
                "a_minor": float(a_minor),
                "b_toroidal": float(b_toroidal),
            }
            self._last_surrogate_contract = {
                "model": "qlknn_profile_surrogate",
                "backend": "qlknn_profile_mlp",
                "weights_loaded": True,
                "weights_path": None if self.weights_path is None else str(self.weights_path),
                "weights_checksum": self.weights_checksum,
                "classification_mode": "coarse_ion_vs_electron_dominant",
                "dominant_channel": dominant_channel,
                "channel_counts": channel_counts,
                "channel_energy": channel_energy,
                "gradient_clip_counts": dict(self._last_gradient_clip_counts),
                "profile_contract": profile_contract,
                "ood_fraction_3sigma": float(np.mean(ood_mask_3sigma)),
                "ood_fraction_5sigma": float(np.mean(ood_mask_5sigma)),
                "ood_point_count_3sigma": int(np.count_nonzero(ood_mask_3sigma)),
                "ood_point_count_5sigma": int(np.count_nonzero(ood_mask_5sigma)),
                "max_abs_z": float(np.max(self._last_max_abs_z_profile)),
                "input_dim": int(expected_dim),
                "n_layers": int(self._weights.n_layers),
                "gated": bool(self._weights.gated),
                "gb_scale": bool(self._weights.gb_scale),
                "log_transform": bool(self._weights.log_transform),
            }
            self._last_profile_contract = dict(profile_contract)
            return chi_e_out, chi_i_out, d_e_out

        chi_e_out, chi_i_out, d_e_out, metadata = reduced_gyrokinetic_profile_model(
            rho,
            te,
            ti,
            ne,
            q_profile,
            s_hat_profile,
            r_major=r_major,
            a_minor=a_minor,
            b_toroidal=b_toroidal,
        )
        self._last_gradient_clip_counts = dict(metadata["gradient_clip_counts"])
        self._last_profile_contract = dict(metadata["profile_contract"])
        self._last_max_abs_z_profile = np.zeros_like(rho, dtype=np.float64)
        self._last_ood_mask_3sigma = np.zeros_like(rho, dtype=bool)
        self._last_ood_mask_5sigma = np.zeros_like(rho, dtype=bool)
        self._last_surrogate_contract = {
            "model": str(metadata["model"]),
            "backend": "analytic_fallback",
            "weights_loaded": False,
            "weights_path": None if self.weights_path is None else str(self.weights_path),
            "weights_checksum": self.weights_checksum,
            "classification_mode": "multichannel_reduced",
            "dominant_channel": str(metadata["dominant_channel"]),
            "channel_counts": dict(metadata["channel_counts"]),
            "channel_energy": dict(metadata["channel_energy"]),
            "gradient_clip_counts": dict(metadata["gradient_clip_counts"]),
            "profile_contract": dict(metadata["profile_contract"]),
            "ood_fraction_3sigma": 0.0,
            "ood_fraction_5sigma": 0.0,
            "ood_point_count_3sigma": 0,
            "ood_point_count_5sigma": 0,
            "max_abs_z": 0.0,
            "input_dim": 0 if self._weights is None else int(self._weights.layers_w[0].shape[0]),
            "n_layers": 0 if self._weights is None else int(self._weights.n_layers),
            "gated": False if self._weights is None else bool(self._weights.gated),
            "gb_scale": False if self._weights is None else bool(self._weights.gb_scale),
            "log_transform": False if self._weights is None else bool(self._weights.log_transform),
        }
        return chi_e_out, chi_i_out, d_e_out


__all__ = ["NeuralTransportModel", "_append_derived"]
