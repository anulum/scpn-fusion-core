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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


# ── Data containers ───────────────────────────────────────────────────

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


# ── Analytic fallback (critical-gradient model) ──────────────────────

# Critical gradient thresholds (Dimits shift included)
_CRIT_ITG = 4.0   # R/L_Ti threshold for ITG
_CRIT_TEM = 5.0   # R/L_Te threshold for TEM (simplified)
_CHI_GB = 1.0     # Gyro-Bohm normalisation [m^2/s]
_STIFFNESS = 2.0  # Transport stiffness exponent


def critical_gradient_model(inp: TransportInputs) -> TransportFluxes:
    """Analytic critical-gradient transport model (fallback).

    Implements a stiff critical-gradient model:

        chi_i = chi_GB * max(0, R/L_Ti - crit_ITG)^stiffness
        chi_e = chi_GB * max(0, R/L_Te - crit_TEM)^stiffness
        D_e   = chi_e / 3  (simplified Ware pinch)

    This is the same physics as the Rust ``TransportSolver`` but
    parameterised in terms of normalised gradients rather than raw
    temperature differences.

    Parameters
    ----------
    inp : TransportInputs
        Local plasma parameters.

    Returns
    -------
    TransportFluxes
        Predicted fluxes with dominant channel identification.
    """
    excess_itg = max(0.0, inp.grad_ti - _CRIT_ITG)
    excess_tem = max(0.0, inp.grad_te - _CRIT_TEM)

    chi_i = _CHI_GB * excess_itg ** _STIFFNESS
    chi_e = _CHI_GB * excess_tem ** _STIFFNESS
    d_e = chi_e / 3.0

    if chi_i > chi_e and chi_i > 0:
        channel = "ITG"
    elif chi_e > 0:
        channel = "TEM"
    else:
        channel = "stable"

    return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)


# ── MLP inference engine ─────────────────────────────────────────────

@dataclass
class MLPWeights:
    """Stored weights for a simple feedforward MLP.

    Architecture: input(10) → hidden1 → hidden2 → output(3)
    Activation: ReLU on hidden layers, softplus on output (ensures chi > 0).
    """

    w1: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b1: FloatArray = field(default_factory=lambda: np.zeros(0))
    w2: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b2: FloatArray = field(default_factory=lambda: np.zeros(0))
    w3: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b3: FloatArray = field(default_factory=lambda: np.zeros(0))
    input_mean: FloatArray = field(default_factory=lambda: np.zeros(10))
    input_std: FloatArray = field(default_factory=lambda: np.ones(10))
    output_scale: FloatArray = field(default_factory=lambda: np.ones(3))


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(0.0, x)


def _softplus(x: FloatArray) -> FloatArray:
    return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))


def _mlp_forward(x: FloatArray, weights: MLPWeights) -> FloatArray:
    """Forward pass through the 3-layer MLP.

    Parameters
    ----------
    x : FloatArray
        Input vector of shape ``(10,)`` or ``(batch, 10)``.
    weights : MLPWeights
        Network parameters.

    Returns
    -------
    FloatArray
        Output vector of shape ``(3,)`` or ``(batch, 3)``
        representing ``[chi_e, chi_i, D_e]``.
    """
    # Normalise inputs
    x_norm = (x - weights.input_mean) / np.maximum(weights.input_std, 1e-8)

    h1 = _relu(x_norm @ weights.w1 + weights.b1)
    h2 = _relu(h1 @ weights.w2 + weights.b2)
    out = _softplus(h2 @ weights.w3 + weights.b3)

    return out * weights.output_scale


# ── Main transport surrogate ─────────────────────────────────────────

class NeuralTransportModel:
    """Neural transport surrogate with analytic fallback.

    On construction, attempts to load MLP weights from *weights_path*.
    If loading fails (file missing, wrong format), the model
    transparently falls back to :func:`critical_gradient_model`.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to a ``.npz`` file containing MLP weights.  The file must
        contain arrays: ``w1, b1, w2, b2, w3, b3, input_mean,
        input_std, output_scale``.

    Examples
    --------
    >>> model = NeuralTransportModel()          # fallback mode
    >>> inp = TransportInputs(grad_ti=8.0)
    >>> fluxes = model.predict(inp)
    >>> fluxes.chi_i > 0
    True
    >>> model.is_neural
    False
    """

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        self._weights: Optional[MLPWeights] = None
        self.is_neural: bool = False
        self.weights_path: Optional[Path] = None

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
            required = ["w1", "b1", "w2", "b2", "w3", "b3",
                        "input_mean", "input_std", "output_scale"]
            for key in required:
                if key not in data:
                    logger.warning(
                        "Weight file missing key '%s' — falling back", key
                    )
                    return

            self._weights = MLPWeights(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                w3=data["w3"],
                b3=data["b3"],
                input_mean=data["input_mean"],
                input_std=data["input_std"],
                output_scale=data["output_scale"],
            )
            self.is_neural = True
            logger.info(
                "Loaded neural transport weights from %s "
                "(layers: %s→%s→%s→3)",
                self.weights_path,
                self._weights.w1.shape[0],
                self._weights.w1.shape[1],
                self._weights.w2.shape[1],
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
        each radial point.

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
        n = len(rho)
        dr = np.gradient(rho) * r_major  # convert to metres
        dr = np.maximum(dr, 1e-6)

        # Normalised gradients: R/L_X = -R * (1/X) * dX/dr
        def norm_grad(x: FloatArray) -> FloatArray:
            dx = np.gradient(x, rho)
            safe_x = np.maximum(np.abs(x), 1e-6)
            return -r_major * dx / safe_x

        grad_te = norm_grad(te)
        grad_ti = norm_grad(ti)
        grad_ne = norm_grad(ne)

        chi_e_out = np.zeros(n)
        chi_i_out = np.zeros(n)
        d_e_out = np.zeros(n)

        for i in range(n):
            beta_e = 4.03e-3 * ne[i] * te[i]  # approximate beta_e
            inp = TransportInputs(
                rho=float(rho[i]),
                te_kev=float(te[i]),
                ti_kev=float(ti[i]),
                ne_19=float(ne[i]),
                grad_te=float(np.clip(grad_te[i], 0, 50)),
                grad_ti=float(np.clip(grad_ti[i], 0, 50)),
                grad_ne=float(np.clip(grad_ne[i], -10, 30)),
                q=float(q_profile[i]),
                s_hat=float(s_hat_profile[i]),
                beta_e=float(beta_e),
            )
            fluxes = self.predict(inp)
            chi_e_out[i] = fluxes.chi_e
            chi_i_out[i] = fluxes.chi_i
            d_e_out[i] = fluxes.d_e

        return chi_e_out, chi_i_out, d_e_out
