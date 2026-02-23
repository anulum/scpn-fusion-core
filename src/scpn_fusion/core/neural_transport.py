# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Surrogate
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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
``docs/NEURAL_TRANSPORT_TRAINING.md`` (to be created) to produce an
``.npz`` weight file that this module loads at construction time.

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
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Weight file format version expected by this loader.
_WEIGHTS_FORMAT_VERSION = 1


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
    """

    chi_e: float = 0.0
    chi_i: float = 0.0
    d_e: float = 0.0
    channel: str = "stable"


# Critical gradient thresholds (Dimits shift included)
_CRIT_ITG = 4.0   # R/L_Ti threshold for ITG
_CRIT_TEM = 5.0   # R/L_Te threshold for TEM (simplified)
_CHI_GB = 1.0     # Gyro-Bohm normalisation [m^2/s]

# Transport stiffness exponent.  Physical range 1.5–4.0 (Dimits PoP 2000,
# Citrin NF 2015); values outside [1.0, 6.0] are non-physical.
_STIFFNESS = 2.0
_STIFFNESS_MIN = 1.0
_STIFFNESS_MAX = 6.0


def critical_gradient_model(
    inp: TransportInputs, *, stiffness: float = _STIFFNESS,
) -> TransportFluxes:
    """Analytic critical-gradient transport model (fallback)."""
    if not (_STIFFNESS_MIN <= stiffness <= _STIFFNESS_MAX):
        raise ValueError(
            f"stiffness={stiffness} outside physical range "
            f"[{_STIFFNESS_MIN}, {_STIFFNESS_MAX}]"
        )
    eps = inp.rho / 3.1
    
    # TEM threshold increases with epsilon (trapped fraction)
    # R/L_Te threshold ~ 4.0 * (1 + 2*eps)
    crit_tem = 4.0 * (1.0 + 2.0 * eps)
    
    excess_itg = max(0.0, inp.grad_ti - _CRIT_ITG)
    excess_tem = max(0.0, inp.grad_te - crit_tem)

    chi_i = _CHI_GB * excess_itg ** stiffness
    chi_e = _CHI_GB * excess_tem ** stiffness

    d_e = chi_e * (0.1 + 0.5 * np.sqrt(eps))

    if chi_i > chi_e and chi_i > 0:
        channel = "ITG"
    elif chi_e > 0:
        channel = "TEM"
    else:
        channel = "stable"

    return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)


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


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(0.0, x)


def _softplus(x: FloatArray) -> FloatArray:
    return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))


def _gelu(x: FloatArray) -> FloatArray:
    """Gaussian Error Linear Unit (matches JAX/PyTorch training)."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _mlp_forward(x: FloatArray, weights: MLPWeights) -> FloatArray:
    """Forward pass through a variable-depth MLP.

    Parameters
    ----------
    x : FloatArray
        Input vector of shape ``(10,)`` or ``(batch, 10)``.
    weights : MLPWeights
        Network parameters (auto-detected depth).

    Returns
    -------
    FloatArray
        Output vector of shape ``(3,)`` or ``(batch, 3)``
        representing ``[chi_e, chi_i, D_e]``.
    """
    # Normalise inputs
    h = (x - weights.input_mean) / np.maximum(weights.input_std, 1e-8)

    # Hidden layers (all except last): GELU activation
    for i in range(weights.n_layers - 1):
        h = _gelu(h @ weights.layers_w[i] + weights.layers_b[i])

    # Output layer
    raw = h @ weights.layers_w[-1] + weights.layers_b[-1]
    if weights.gated:
        # Gated: first 3 outputs are flux logits, last 3 are gate logits
        flux = _softplus(raw[..., :3]) * weights.output_scale
        gate = 1.0 / (1.0 + np.exp(-np.clip(raw[..., 3:], -20.0, 20.0)))  # sigmoid
        out = gate * flux
    else:
        out = _softplus(raw) * weights.output_scale

    # If trained in log-space, convert: output = exp(log(1+Y)) - 1 = Y
    if weights.log_transform:
        out = np.expm1(np.clip(out, 0.0, 20.0))

    # Gyro-Bohm skip connection: multiply by chi_gb(Te)
    if weights.gb_scale:
        te = x[..., 1]  # Te_keV is column 1
        te_j = te * 1e3 * 1.602e-19
        cs = np.sqrt(te_j / 3.344e-27)
        rho_s = np.sqrt(3.344e-27 * te_j) / (1.602e-19 * 5.3)
        chi_gb = rho_s ** 2 * cs / 6.2
        if chi_gb.ndim == 0:
            out = out * float(chi_gb)
        else:
            out = out * chi_gb[..., np.newaxis]

    return out


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
            weights_path = Path(__file__).resolve().parents[3] / "weights" / "neural_transport_qlknn.npz"
            
        self.weights_path: Optional[Path] = None
        self.weights_checksum: Optional[str] = None
        self._last_gradient_clip_counts: dict[str, int] = {"grad_te": 0, "grad_ti": 0, "grad_ne": 0}
        self._last_profile_contract: dict[str, float | int] = {"n_points": 0}

        if weights_path is not None:
            self.weights_path = Path(weights_path)
            self._try_load_weights()

    def _try_load_weights(self) -> None:
        """Attempt to load MLP weights from disk."""
        if self.weights_path is None or not self.weights_path.exists():
            logger.info(
                "Neural transport weights not found at %s — using "
                "critical-gradient fallback",
                self.weights_path,
            )
            return

        try:
            data = np.load(self.weights_path)

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
                    logger.warning(
                        "Weight file missing key '%s' — falling back", key
                    )
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
            log_transform = bool(int(data["log_transform"])) if "log_transform" in data else False
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
            raw = b"".join(
                data[k].tobytes() for k in sorted(data.files)
                if k != "version"
            )
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

        x = np.array([
            inp.rho, inp.te_kev, inp.ti_kev, inp.ne_19,
            inp.grad_te, inp.grad_ti, inp.grad_ne,
            inp.q, inp.s_hat, inp.beta_e,
        ])
        # Append derived features if model expects >10D input
        expected_dim = self._weights.layers_w[0].shape[0]
        if expected_dim >= 12:
            itg_excess = max(0.0, inp.grad_ti - _CRIT_ITG)
            tem_excess = max(0.0, inp.grad_te - _CRIT_TEM)
            x = np.append(x, [itg_excess, tem_excess])
        if expected_dim >= 13:
            te_j = inp.te_kev * 1e3 * 1.602e-19
            cs = np.sqrt(te_j / 3.344e-27)
            rho_s = np.sqrt(3.344e-27 * te_j) / (1.602e-19 * 5.3)
            chi_gb = rho_s ** 2 * cs / 6.2
            x = np.append(x, [np.log(max(chi_gb, 1e-10))])
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

        # Normalised gradients: R/L_X = -R * (1/X) * dX/dr
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

        if self.is_neural and self._weights is not None:
            x_batch = np.column_stack([
                rho, te, ti, ne,
                grad_te, grad_ti, grad_ne,
                q_profile, s_hat_profile, beta_e,
            ])  # (N, 10)
            # Append derived features if model expects >10D input
            expected_dim = self._weights.layers_w[0].shape[0]
            if expected_dim >= 12:
                itg_excess = np.maximum(0.0, grad_ti - _CRIT_ITG)
                tem_excess = np.maximum(0.0, grad_te - _CRIT_TEM)
                x_batch = np.column_stack([x_batch, itg_excess, tem_excess])
            if expected_dim >= 13:
                te_j = te * 1e3 * 1.602e-19
                cs = np.sqrt(te_j / 3.344e-27)
                rho_s = np.sqrt(3.344e-27 * te_j) / (1.602e-19 * 5.3)
                chi_gb = rho_s ** 2 * cs / 6.2
                log_chi_gb = np.log(np.maximum(chi_gb, 1e-10))
                x_batch = np.column_stack([x_batch, log_chi_gb])
            out = _mlp_forward(x_batch, self._weights)  # (N, 3)
            chi_e_out = out[:, 0]
            chi_i_out = out[:, 1]
            d_e_out = out[:, 2]
            return chi_e_out, chi_i_out, d_e_out

        # Inverse aspect ratio eps(rho)
        eps = rho / (r_major / 2.0) # a approx R/2
        eps = np.clip(eps, 0.0, 0.5)
        
        crit_tem = 4.0 * (1.0 + 2.0 * eps)
        
        excess_itg = np.maximum(0.0, grad_ti - _CRIT_ITG)
        excess_tem = np.maximum(0.0, grad_te - crit_tem)

        chi_i_out = _CHI_GB * excess_itg ** _STIFFNESS
        chi_e_out = _CHI_GB * excess_tem ** _STIFFNESS
        
        d_e_out = chi_e_out * (0.1 + 0.5 * np.sqrt(eps))

        return chi_e_out, chi_i_out, d_e_out


# Backward-compatible class name used by older interop/parity tests.
NeuralTransportSurrogate = NeuralTransportModel
