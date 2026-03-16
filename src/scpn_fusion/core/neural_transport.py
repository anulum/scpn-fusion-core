# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural Transport Surrogate
"""
Neural-network surrogate for turbulent transport coefficients.

Replaces the simple critical-gradient transport model with a trained
MLP that reproduces gyrokinetic-level predictions at millisecond
inference speeds.  When no trained weights are available the module
falls back to an analytic critical-gradient model, so existing code
keeps working without any neural network dependency.

The architecture follows the QLKNN paradigm (van de Plassche et al.,
*Phys. Plasmas* 27, 022310, 2020): a small MLP maps local plasma
parameters to turbulent fluxes (heat, particle) across ITG/TEM/ETG
channels.

Training data
-------------
The module is designed for the QLKNN-10D public dataset:

    https://doi.org/10.5281/zenodo.3700755

Download the dataset and run the training recipe in
``docs/NEURAL_TRANSPORT_TRAINING.md`` to produce an ``.npz`` weight
file that this module loads at construction time.

References
----------
.. [1] van de Plassche, K.L. et al. (2020). "Fast modeling of
       turbulent transport in fusion plasmas using neural networks."
       *Phys. Plasmas* 27, 022310. doi:10.1063/1.5134126
.. [2] Citrin, J. et al. (2015). "Real-time capable first-principles
       based modelling of tokamak turbulent transport." *Nucl. Fusion*
       55, 092001.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .neural_transport_math import _compute_nustar, _mlp_forward
from .neural_transport_math import _relu as _relu  # noqa: F401
from .neural_transport_math import _softplus as _softplus  # noqa: F401

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Weight file format version expected by this loader.
_WEIGHTS_FORMAT_VERSION = 1
_MAX_WEIGHTS_FILE_BYTES = 128 * 1024 * 1024


@dataclass
class TransportInputs:
    """Local plasma parameters at a single radial location.

    All quantities are in SI / conventional tokamak units.

    Parameters
    ----------
    rho : float
        Normalised toroidal flux coordinate (0 = axis, 1 = edge).
    te_kev : float
        Electron temperature [keV].
    ti_kev : float
        Ion temperature [keV].
    ne_19 : float
        Electron density [10^19 m^-3].
    grad_te : float
        Normalised electron temperature gradient R/L_Te.
    grad_ti : float
        Normalised ion temperature gradient R/L_Ti.
    grad_ne : float
        Normalised electron density gradient R/L_ne.
    q : float
        Safety factor.
    s_hat : float
        Magnetic shear s = (r/q)(dq/dr).
    beta_e : float
        Electron beta (kinetic pressure / magnetic pressure).
    r_major_m : float
        Major radius used in reduced gyro-Bohm normalization [m].
    a_minor_m : float
        Minor radius used for inverse-aspect-ratio effects [m].
    b_tesla : float
        Toroidal magnetic field used in reduced gyro-Bohm normalization [T].
    z_eff : float
        Effective charge used for collisionality estimate.
    """

    rho: float = 0.5
    te_kev: float = 5.0
    ti_kev: float = 5.0
    ne_19: float = 5.0
    grad_te: float = 6.0
    grad_ti: float = 6.0
    grad_ne: float = 2.0
    q: float = 1.5
    s_hat: float = 0.8
    beta_e: float = 0.01
    r_major_m: float = 6.2
    a_minor_m: float = 2.0
    b_tesla: float = 5.3
    z_eff: float = 1.0


@dataclass
class TransportFluxes:
    """Predicted turbulent transport fluxes.

    Parameters
    ----------
    chi_e : float
        Electron thermal diffusivity [m^2/s].
    chi_i : float
        Ion thermal diffusivity [m^2/s].
    d_e : float
        Particle diffusivity [m^2/s].
    channel : str
        Dominant instability channel ("ITG", "TEM", "ETG", or "stable").
    chi_e_itg, chi_e_tem, chi_e_etg : float
        Electron transport contributions from the reduced ITG/TEM/ETG channels.
    chi_i_itg : float
        Ion transport contribution from the reduced ITG channel.
    """

    chi_e: float = 0.0
    chi_i: float = 0.0
    d_e: float = 0.0
    channel: str = "stable"
    chi_e_itg: float = 0.0
    chi_e_tem: float = 0.0
    chi_e_etg: float = 0.0
    chi_i_itg: float = 0.0


# Critical gradient thresholds (Dimits shift included)
_CRIT_ITG = 4.0  # R/L_Ti threshold for ITG
_CRIT_TEM = 5.0  # R/L_Te threshold for TEM (reduced-order closure)
_CRIT_ETG = 12.0  # R/L_Te threshold for ETG branch in reduced closure
_CHI_GB = 1.0  # Gyro-Bohm normalisation [m^2/s]

# Transport stiffness exponent.  Physical range 1.5–4.0 (Dimits PoP 2000,
# Citrin NF 2015); values outside [1.0, 6.0] are non-physical.
_STIFFNESS = 2.0
_STIFFNESS_MIN = 1.0
_STIFFNESS_MAX = 6.0
_TRANSPORT_FLOOR = 1e-6


def _gyro_bohm_diffusivity(inp: TransportInputs) -> float:
    """Estimate local gyro-Bohm diffusivity from reduced geometry inputs."""
    e_charge = 1.602176634e-19
    m_i = 2.0 * 1.672621924e-27  # deuterium reference mass
    te_kev = max(float(inp.te_kev), 0.01)
    b_t = max(float(inp.b_tesla), 0.1)
    r_major = max(float(inp.r_major_m), 0.1)

    te_j = te_kev * 1e3 * e_charge
    cs = np.sqrt(te_j / m_i)
    rho_s = np.sqrt(m_i * te_j) / (e_charge * b_t)
    chi_gb = rho_s**2 * cs / r_major
    if not np.isfinite(chi_gb):
        return _TRANSPORT_FLOOR
    return float(max(chi_gb, _TRANSPORT_FLOOR))


def _dominant_channel(
    *,
    chi_i_itg: float,
    chi_e_itg: float,
    chi_e_tem: float,
    chi_e_etg: float,
) -> str:
    """Return the dominant reduced transport channel."""
    strengths = {
        "ITG": float(chi_i_itg + chi_e_itg),
        "TEM": float(chi_e_tem),
        "ETG": float(chi_e_etg),
    }
    name, value = max(strengths.items(), key=lambda item: item[1])
    return name if value > 0.0 else "stable"


def critical_gradient_model(
    inp: TransportInputs,
    *,
    stiffness: float = _STIFFNESS,
) -> TransportFluxes:
    """Reduced multichannel gyrokinetic closure used as analytic fallback.

    This is still a reduced-order model, but it separates three dominant
    turbulence branches:

    - ITG: ion-driven, mostly ion heat transport with a smaller electron tail
    - TEM: trapped-electron-mode, electron-dominant with trapped-particle drive
    - ETG: electron-temperature-gradient branch, high-k electron heat channel
    """
    if not (_STIFFNESS_MIN <= stiffness <= _STIFFNESS_MAX):
        raise ValueError(
            f"stiffness={stiffness} outside physical range " f"[{_STIFFNESS_MIN}, {_STIFFNESS_MAX}]"
        )
    eps = float(np.clip(inp.rho * inp.a_minor_m / max(inp.r_major_m, 1e-6), 0.0, 0.8))
    trapped_frac = float(np.clip(1.46 * np.sqrt(max(eps, 0.0)), 0.0, 1.0))
    nustar = float(
        _compute_nustar(
            inp.te_kev,
            inp.ne_19,
            inp.q,
            inp.rho,
            inp.r_major_m,
            inp.a_minor_m,
            inp.z_eff,
        )
    )
    chi_gb = _gyro_bohm_diffusivity(inp)
    shear_supp = 1.0 / (1.0 + 0.35 * max(inp.s_hat, 0.0) ** 2)
    beta_supp = 1.0 / (1.0 + max(inp.beta_e, 0.0) / 0.03)
    electron_ratio = float(np.clip(inp.te_kev / max(inp.ti_kev, 0.05), 0.5, 4.0))

    crit_itg = _CRIT_ITG + 0.4 * max(inp.s_hat, 0.0) + 8.0 * max(inp.beta_e, 0.0)
    density_excess = max(inp.grad_ne - 2.5, 0.0)
    crit_tem = max(
        2.5,
        5.0 + 1.1 * eps + 0.12 * min(max(nustar, 0.0), 10.0) - 0.35 * density_excess,
    )
    crit_etg = 10.5 + 1.0 * eps + 0.3 * max(inp.s_hat, 0.0) + 0.2 * max(nustar, 0.0)

    excess_itg = max(0.0, inp.grad_ti - crit_itg)
    excess_tem = max(0.0, inp.grad_te - crit_tem)
    excess_etg = max(0.0, inp.grad_te - crit_etg)

    chi_i_itg = chi_gb * excess_itg**stiffness * shear_supp * beta_supp
    chi_e_itg = 0.35 * chi_i_itg

    collisional_tem = 1.0 / (1.0 + 0.8 * max(nustar, 0.0))
    density_drive = 0.15 + 0.35 * density_excess
    chi_e_tem = (
        chi_gb * excess_tem**stiffness * trapped_frac * collisional_tem * beta_supp * density_drive
    )

    collisional_etg = 1.0 / (1.0 + 1.5 * max(nustar, 0.0))
    etg_shear = 1.0 / (1.0 + 0.2 * max(inp.s_hat, 0.0) ** 2)
    electron_gradient_split = 1.0 + 0.18 * max(inp.grad_te - inp.grad_ti, 0.0)
    chi_e_etg = (
        0.85
        * chi_gb
        * excess_etg ** (0.9 * stiffness)
        * collisional_etg
        * etg_shear
        * electron_ratio
        * electron_gradient_split
    )

    chi_i = max(chi_i_itg, 0.0)
    chi_e = max(chi_e_itg + chi_e_tem + chi_e_etg, 0.0)
    d_e = chi_e * (0.1 + 0.5 * np.sqrt(max(eps, 0.0)))
    channel = _dominant_channel(
        chi_i_itg=chi_i_itg,
        chi_e_itg=chi_e_itg,
        chi_e_tem=chi_e_tem,
        chi_e_etg=chi_e_etg,
    )

    return TransportFluxes(
        chi_e=chi_e,
        chi_i=chi_i,
        d_e=d_e,
        channel=channel,
        chi_e_itg=max(chi_e_itg, 0.0),
        chi_e_tem=max(chi_e_tem, 0.0),
        chi_e_etg=max(chi_e_etg, 0.0),
        chi_i_itg=max(chi_i_itg, 0.0),
    )


def reduced_gyrokinetic_profile_model(
    rho: FloatArray,
    te: FloatArray,
    ti: FloatArray,
    ne: FloatArray,
    q_profile: FloatArray,
    s_hat_profile: FloatArray,
    *,
    r_major: float = 6.2,
    a_minor: float = 2.0,
    b_toroidal: float = 5.3,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    """Evaluate the reduced ITG/TEM/ETG closure across a radial profile."""
    rho = np.asarray(rho, dtype=np.float64)
    te = np.asarray(te, dtype=np.float64)
    ti = np.asarray(ti, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    q_profile = np.asarray(q_profile, dtype=np.float64)
    s_hat_profile = np.asarray(s_hat_profile, dtype=np.float64)

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
    a_minor = float(a_minor)
    b_toroidal = float(b_toroidal)
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
    grad_te = np.clip(grad_te_raw, 0.0, 50.0)
    grad_ti = np.clip(grad_ti_raw, 0.0, 50.0)
    grad_ne = np.clip(grad_ne_raw, -10.0, 30.0)
    beta_e = 4.03e-3 * ne * te

    fluxes = [
        critical_gradient_model(
            TransportInputs(
                rho=float(rho[i]),
                te_kev=float(te[i]),
                ti_kev=float(ti[i]),
                ne_19=float(ne[i]),
                grad_te=float(grad_te[i]),
                grad_ti=float(grad_ti[i]),
                grad_ne=float(grad_ne[i]),
                q=float(q_profile[i]),
                s_hat=float(s_hat_profile[i]),
                beta_e=float(beta_e[i]),
                r_major_m=r_major,
                a_minor_m=a_minor,
                b_tesla=b_toroidal,
            )
        )
        for i in range(n)
    ]

    chi_e = np.array([f.chi_e for f in fluxes], dtype=np.float64)
    chi_i = np.array([f.chi_i for f in fluxes], dtype=np.float64)
    d_e = np.array([f.d_e for f in fluxes], dtype=np.float64)
    dominant_channels = [f.channel for f in fluxes]

    channel_energy = {
        "ITG": float(np.sum([f.chi_i_itg + f.chi_e_itg for f in fluxes])),
        "TEM": float(np.sum([f.chi_e_tem for f in fluxes])),
        "ETG": float(np.sum([f.chi_e_etg for f in fluxes])),
    }
    dominant_channel = max(channel_energy.items(), key=lambda item: item[1])[0]
    if channel_energy[dominant_channel] <= 0.0:
        dominant_channel = "stable"

    metadata: dict[str, Any] = {
        "model": "reduced_multichannel_analytic",
        "dominant_channel": dominant_channel,
        "channel_counts": {
            name: int(sum(ch == name for ch in dominant_channels))
            for name in ("ITG", "TEM", "ETG", "stable")
        },
        "channel_energy": channel_energy,
        "gradient_clip_counts": {
            "grad_te": int(np.count_nonzero((grad_te_raw < 0.0) | (grad_te_raw > 50.0))),
            "grad_ti": int(np.count_nonzero((grad_ti_raw < 0.0) | (grad_ti_raw > 50.0))),
            "grad_ne": int(np.count_nonzero((grad_ne_raw < -10.0) | (grad_ne_raw > 30.0))),
        },
        "profile_contract": {
            "n_points": int(n),
            "rho_min": float(rho[0]),
            "rho_max": float(rho[-1]),
            "r_major": r_major,
            "a_minor": a_minor,
            "b_toroidal": b_toroidal,
        },
        "edge_etg_fraction": (
            float(
                np.mean(
                    [
                        1.0 if ch == "ETG" else 0.0
                        for ch, r in zip(dominant_channels, rho)
                        if r >= 0.8
                    ]
                )
            )
            if np.any(rho >= 0.8)
            else 0.0
        ),
    }
    return chi_e, chi_i, d_e, metadata


@dataclass
class MLPWeights:
    """Stored weights for a variable-depth feedforward MLP.

    Architecture: input(10) → hidden1 → [hidden2 → ...] → output(3)
    Activation: GELU on hidden layers, softplus on output (ensures chi > 0).

    Supports 2+ layer architectures.  The number of layers is
    auto-detected from the ``.npz`` keys at load time (w1/b1, w2/b2, ...).
    Legacy 3-layer weights (w1→w2→w3) load without modification.
    """

    layers_w: list[FloatArray] = field(default_factory=list)
    layers_b: list[FloatArray] = field(default_factory=list)
    input_mean: FloatArray = field(default_factory=lambda: np.zeros(10))
    input_std: FloatArray = field(default_factory=lambda: np.ones(10))
    output_scale: FloatArray = field(default_factory=lambda: np.ones(3))
    log_transform: bool = False
    gb_scale: bool = False
    gated: bool = False

    @property
    def w1(self) -> FloatArray:
        return self.layers_w[0] if self.layers_w else np.zeros((0, 0))

    @property
    def b1(self) -> FloatArray:
        return self.layers_b[0] if self.layers_b else np.zeros(0)

    @property
    def w2(self) -> FloatArray:
        return self.layers_w[1] if len(self.layers_w) > 1 else np.zeros((0, 0))

    @property
    def b2(self) -> FloatArray:
        return self.layers_b[1] if len(self.layers_b) > 1 else np.zeros(0)

    @property
    def w3(self) -> FloatArray:
        return self.layers_w[2] if len(self.layers_w) > 2 else np.zeros((0, 0))

    @property
    def b3(self) -> FloatArray:
        return self.layers_b[2] if len(self.layers_b) > 2 else np.zeros(0)

    @property
    def n_layers(self) -> int:
        return len(self.layers_w)


def _append_derived(x: FloatArray, inp, expected_dim: int) -> FloatArray:
    """Append Ti_Te, Nustar, ITG/TEM excess, log(chi_gb) as needed."""
    if expected_dim <= 10:
        return x
    # 12D base: append Ti_Te and Nustar first
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

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to a ``.npz`` file containing MLP weights.  The file must
        contain arrays ``w1, b1, ..., wN, bN, input_mean, input_std,
        output_scale``.  The number of layers N is auto-detected.
    """

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        self._weights: Optional[MLPWeights] = None
        self.is_neural: bool = False

        # Default to bundled QLKNN-10D weights if not specified
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
                "Neural transport weights not found at %s — using "
                "critical-gradient compatibility model",
                self.weights_path,
            )
            return

        if self.weights_path.suffix.lower() != ".npz":
            logger.warning(
                "Neural transport weights must be a .npz file (got %s) — falling back",
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
                "Neural transport weights size %d bytes outside allowed range (1..%d) — falling back",
                file_size,
                _MAX_WEIGHTS_FILE_BYTES,
            )
            return

        try:
            with np.load(self.weights_path, allow_pickle=False) as data:
                # Auto-detect MLP depth from keys: w1/b1, w2/b2, ...
                n_layers = 0
                while f"w{n_layers + 1}" in data and f"b{n_layers + 1}" in data:
                    n_layers += 1

                if n_layers < 2:
                    logger.warning(
                        "Weight file has only %d layer(s) (need >=2) — falling back",
                        n_layers,
                    )
                    return

                for key in ("input_mean", "input_std", "output_scale"):
                    if key not in data:
                        logger.warning("Weight file missing key '%s' — falling back", key)
                        return

                # Version check (optional key, defaults to 1)
                version = int(data["version"]) if "version" in data else 1
                if version != _WEIGHTS_FORMAT_VERSION:
                    logger.warning(
                        "Weight file version %d != expected %d — falling back",
                        version,
                        _WEIGHTS_FORMAT_VERSION,
                    )
                    return

                layers_w = [data[f"w{i+1}"] for i in range(n_layers)]
                layers_b = [data[f"b{i+1}"] for i in range(n_layers)]

                # Check for log-space transform flag
                log_transform = (
                    bool(int(data["log_transform"])) if "log_transform" in data else False
                )
                # Check for gyro-Bohm skip connection flag
                gb_scale = bool(int(data["gb_scale"])) if "gb_scale" in data else False
                # Check for gated output architecture flag
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

                # Compute checksum for reproducibility tracking
                raw = b"".join(data[k].tobytes() for k in sorted(data.files) if k != "version")
                self.weights_checksum = hashlib.sha256(raw).hexdigest()[:16]

                # Build architecture description string
                dims = [str(layers_w[0].shape[0])]
                for w in layers_w:
                    dims.append(str(w.shape[1]))
                arch_str = "→".join(dims)

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
        """Predict turbulent transport fluxes for given local parameters.

        Uses the neural MLP if weights are loaded, otherwise falls back
        to the analytic critical-gradient model.

        Parameters
        ----------
        inp : TransportInputs
            Local plasma parameters at a single radial point.

        Returns
        -------
        TransportFluxes
            Predicted heat and particle diffusivities.
        """
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
        """Predict transport coefficients on the full radial profile.

        Computes normalised gradients from the profile arrays via
        central finite differences, then evaluates the surrogate at
        each radial point.  When the MLP is loaded the entire profile
        is evaluated in a single batched forward pass (no Python loop).

        Parameters
        ----------
        rho : FloatArray
            Normalised radius grid (0 to 1), shape ``(N,)``.
        te, ti : FloatArray
            Electron/ion temperature profiles [keV], shape ``(N,)``.
        ne : FloatArray
            Electron density profile [10^19 m^-3], shape ``(N,)``.
        q_profile : FloatArray
            Safety factor profile, shape ``(N,)``.
        s_hat_profile : FloatArray
            Magnetic shear profile, shape ``(N,)``.
        r_major : float
            Major radius [m] for gradient normalisation.
        a_minor : float
            Minor radius [m].

        Returns
        -------
        chi_e, chi_i, d_e : FloatArray
            Transport coefficient profiles, each shape ``(N,)``.
        """
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
            )  # (N, 10)
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
            out = _mlp_forward(x_batch, self._weights)  # (N, 3)
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


# Backward-compatible class name used by older interop/parity tests.
NeuralTransportSurrogate = NeuralTransportModel
