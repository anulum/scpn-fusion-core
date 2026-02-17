# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Pure-NumPy training for a multi-layer Fourier Neural Operator turbulence model.

.. deprecated:: 2.1.0

    **DEPRECATED / EXPERIMENTAL** — Trained on 60 synthetic Hasegawa-Wakatani
    samples.  Not validated against production gyrokinetic codes (GENE, GS2,
    QuaLiKiz).  Current relative L2 error is ~0.79 (explains ~21% of variance).
    Removed from default pipeline in v2.1.0; will be retired in v3.0.0 unless
    real gyrokinetic training data becomes available.  The trained FNO can be
    hot-swapped when real simulation data becomes available without changing
    the model architecture.

Supports two data generation modes:

1. **Single-regime Hasegawa-Wakatani** (legacy) — fixed adiabaticity, one
   damping parameter.  Uses ``_generate_training_pairs()``.

2. **Multi-regime SPARC-parameterized** — samples from ITG / TEM / ETG
   regimes with SPARC-relevant adiabaticity α, gradient drive κ, viscosity ν,
   and spectral cutoff k_c.  Uses ``_generate_multi_regime_pairs()``.

The multi-regime generator produces modified Hasegawa-Wakatani-like drift-wave
fields whose spectral character matches the three dominant tokamak micro-
instability channels:

    ITG — long-wavelength, low adiabaticity, strong ∇T_i drive
    TEM — intermediate wavelength, moderate-to-high adiabaticity
    ETG — short-wavelength, high adiabaticity, strong ∇T_e drive

This is a first-principles proxy for real gyrokinetic data (GENE, GS2,
QuaLiKiz).  The trained FNO can be hot-swapped when real simulation data
becomes available without changing the model architecture.

References
----------
.. [1] Hasegawa, A. & Wakatani, M. (1983). Phys. Rev. Lett. 50, 682.
.. [2] Li, Z. et al. (2021). "Fourier Neural Operator for Parametric PDEs."
       ICLR 2021.
.. [3] Howard, N.T. et al. (2021). "Multi-scale gyrokinetic simulation of
       tokamak plasmas." Phys. Plasmas 28, 072505.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import logging

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "weights" / "fno_turbulence.npz"
DEFAULT_SPARC_WEIGHTS_PATH = REPO_ROOT / "weights" / "fno_turbulence_sparc.npz"

# ── SPARC-relevant turbulence regime parameters ──────────────────────
#
# Each regime maps to a range of physical parameters used to construct
# the modified Hasegawa-Wakatani spectral time-stepper.  These ranges
# are derived from SPARC scenario modelling (Howard et al., 2021;
# Rodriguez-Fernandez et al., 2022).
#
# Keys:
#   alpha  — adiabaticity parameter α = k‖²v_th_e² / (ω_e·ν_ei)
#   kappa  — gradient drive strength κ ∝ R/L_T
#   nu     — collisional viscosity (normalised)
#   damp   — nonlinear damping coefficient
#   k_cut  — spectral cutoff wavenumber (controls dominant wavelength)

SPARC_REGIMES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "itg": {
        "alpha": (0.1, 0.5),
        "kappa": (5.0, 15.0),
        "nu": (0.001, 0.01),
        "damp": (0.05, 0.15),
        "k_cut": (4.0, 8.0),
    },
    "tem": {
        "alpha": (0.5, 2.0),
        "kappa": (2.0, 8.0),
        "nu": (0.005, 0.05),
        "damp": (0.10, 0.25),
        "k_cut": (6.0, 12.0),
    },
    "etg": {
        "alpha": (1.0, 3.0),
        "kappa": (3.0, 12.0),
        "nu": (0.01, 0.1),
        "damp": (0.15, 0.30),
        "k_cut": (10.0, 20.0),
    },
}


def gelu(x: np.ndarray) -> np.ndarray:
    """Fast GeLU approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


class AdamOptimizer:
    """Minimal Adam optimizer for NumPy arrays."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
        self.t += 1
        for key, param in params.items():
            grad = grads[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MultiLayerFNO:
    """
    Multi-layer FNO model:
    Input [N,N] -> Lift (1->width) -> 4x FNO layers -> Project (width->1) -> [N,N].

    Training routine updates the project head with Adam while keeping the spectral
    backbone fixed. This keeps the implementation NumPy-only and fast enough for
    iterative dataset generation.
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        seed: int = 42,
    ) -> None:
        self.modes = int(modes)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.rng = np.random.default_rng(seed)

        self.lift_w = self.rng.normal(0.0, 0.1, size=(self.width,))
        self.lift_b = np.zeros((self.width,), dtype=np.float64)
        self.project_w = self.rng.normal(0.0, 0.1, size=(self.width,))
        self.project_b = 0.0

        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(self.n_layers):
            self.layers.append(
                {
                    "wr": self.rng.normal(0.0, 0.03, size=(self.width, self.modes, self.modes)),
                    "wi": self.rng.normal(0.0, 0.03, size=(self.width, self.modes, self.modes)),
                    "skip_w": np.eye(self.width) + self.rng.normal(0.0, 0.01, size=(self.width, self.width)),
                    "skip_b": np.zeros((self.width,), dtype=np.float64),
                }
            )

    def _spectral_convolution(self, h: np.ndarray, layer: Dict[str, np.ndarray]) -> np.ndarray:
        n = h.shape[0]
        modes = min(self.modes, n)
        out = np.zeros_like(h)

        for c in range(self.width):
            h_k = np.fft.fft2(h[:, :, c])
            out_k = np.zeros_like(h_k)
            w = layer["wr"][c, :modes, :modes] + 1j * layer["wi"][c, :modes, :modes]
            out_k[:modes, :modes] = h_k[:modes, :modes] * w
            out[:, :, c] = np.fft.ifft2(out_k).real

        return out

    def _forward_hidden(self, x_field: np.ndarray) -> np.ndarray:
        h = x_field[:, :, None] * self.lift_w[None, None, :] + self.lift_b[None, None, :]
        for layer in self.layers:
            spectral = self._spectral_convolution(h, layer)
            pointwise = np.tensordot(h, layer["skip_w"], axes=([2], [0])) + layer["skip_b"][None, None, :]
            h = gelu(spectral + pointwise)
        return h

    def forward_with_hidden(self, x_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self._forward_hidden(x_field)
        y = np.tensordot(h, self.project_w, axes=([2], [0])) + self.project_b
        return y, h

    def forward(self, x_field: np.ndarray) -> np.ndarray:
        y, _ = self.forward_with_hidden(x_field)
        return y

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, np.ndarray] = {
            "version": np.array([2], dtype=np.int32),
            "modes": np.array([self.modes], dtype=np.int32),
            "width": np.array([self.width], dtype=np.int32),
            "n_layers": np.array([self.n_layers], dtype=np.int32),
            "lift_w": self.lift_w.astype(np.float64),
            "lift_b": self.lift_b.astype(np.float64),
            "project_w": self.project_w.astype(np.float64),
            "project_b": np.array([self.project_b], dtype=np.float64),
        }
        for i, layer in enumerate(self.layers):
            payload[f"layer{i}_wr"] = layer["wr"].astype(np.float64)
            payload[f"layer{i}_wi"] = layer["wi"].astype(np.float64)
            payload[f"layer{i}_skip_w"] = layer["skip_w"].astype(np.float64)
            payload[f"layer{i}_skip_b"] = layer["skip_b"].astype(np.float64)

        np.savez(path, **payload)

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.modes = int(data["modes"][0])
            self.width = int(data["width"][0])
            self.n_layers = int(data["n_layers"][0])
            self.lift_w = np.array(data["lift_w"], dtype=np.float64)
            self.lift_b = np.array(data["lift_b"], dtype=np.float64)
            self.project_w = np.array(data["project_w"], dtype=np.float64)
            self.project_b = float(np.array(data["project_b"], dtype=np.float64).reshape(-1)[0])

            self.layers = []
            for i in range(self.n_layers):
                self.layers.append(
                    {
                        "wr": np.array(data[f"layer{i}_wr"], dtype=np.float64),
                        "wi": np.array(data[f"layer{i}_wi"], dtype=np.float64),
                        "skip_w": np.array(data[f"layer{i}_skip_w"], dtype=np.float64),
                        "skip_b": np.array(data[f"layer{i}_skip_b"], dtype=np.float64),
                    }
                )


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target) + 1e-8
    return float(np.linalg.norm(pred - target) / denom)


def _generate_training_pairs(
    n_samples: int,
    grid_size: int,
    seed: int,
    damping: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.empty((n_samples, grid_size, grid_size), dtype=np.float64)
    y = np.empty_like(x)

    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    k2[0, 0] = 1.0
    mask_low_k = (k2 < 25.0).astype(np.float64)

    dt = 0.01
    omega = ky_grid / (1.0 + k2)
    phase_shift = np.exp(-1j * omega * dt)
    viscous = np.exp(-0.001 * k2 * dt) * (1.0 - damping)

    for i in range(n_samples):
        field = rng.standard_normal((grid_size, grid_size)) * 0.1
        field_k = np.fft.fft2(field)

        forcing = rng.standard_normal((grid_size, grid_size)) + 1j * rng.standard_normal((grid_size, grid_size))
        forcing_k = np.fft.fft2(forcing) * mask_low_k * 5.0

        next_k = (field_k * phase_shift) + forcing_k * dt
        next_k = next_k * viscous

        x[i] = field
        y[i] = np.fft.ifft2(next_k).real

    return x, y


def _sample_regime_params(
    rng: np.random.Generator,
    regime: str,
) -> Dict[str, float]:
    """Sample a random parameter vector from a given turbulence regime."""
    bounds = SPARC_REGIMES[regime]
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _generate_multi_regime_pairs(
    n_samples: int,
    grid_size: int,
    seed: int,
    regime_weights: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Generate training data spanning ITG / TEM / ETG regimes.

    Each sample uses a randomly sampled parameter set from one of the
    three turbulence regimes.  The modified Hasegawa-Wakatani dispersion
    relation is parameterised by (α, κ, ν, damping, k_cut):

        ω(k) = α · k_y / (α + k²)
        γ(k) = κ · k_y · k² / (α + k²)² − ν · k⁴

    This produces drift-wave fields whose spectral character changes with
    the regime: ITG fields are dominated by long wavelengths, ETG fields
    by short wavelengths.

    Parameters
    ----------
    n_samples : int
        Total number of (x, y) pairs.
    grid_size : int
        Grid resolution (NxN).
    seed : int
        RNG seed for reproducibility.
    regime_weights : dict, optional
        Regime sampling probabilities.  Default: equal weight.

    Returns
    -------
    x : ndarray, shape (n_samples, grid_size, grid_size)
    y : ndarray, shape (n_samples, grid_size, grid_size)
    metadata : list of dicts
        Per-sample regime name and sampled parameters.
    """
    rng = np.random.default_rng(seed)
    regimes = list(SPARC_REGIMES.keys())

    if regime_weights is None:
        probs = np.ones(len(regimes)) / len(regimes)
    else:
        probs = np.array([regime_weights.get(r, 1.0) for r in regimes])
        probs /= probs.sum()

    x = np.empty((n_samples, grid_size, grid_size), dtype=np.float64)
    y = np.empty_like(x)
    metadata: List[Dict[str, object]] = []

    # Pre-compute wavenumber grids
    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    k4 = k2**2
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0  # avoid div-by-zero

    dt = 0.01

    for i in range(n_samples):
        regime = rng.choice(regimes, p=probs)
        params = _sample_regime_params(rng, regime)

        alpha = params["alpha"]
        kappa = params["kappa"]
        nu = params["nu"]
        damp = params["damp"]
        k_cut = params["k_cut"]

        # Modified H-W dispersion: ω = α·ky/(α+k²)
        denom = alpha + k2_safe
        omega = alpha * ky_grid / denom

        # Linear growth rate: γ = κ·ky·k²/(α+k²)² - ν·k⁴
        growth = kappa * ky_grid * k2 / (denom**2) - nu * k4

        # Spectral cutoff: exponential damping above k_cut
        spectral_filter = np.exp(-((k2 / k_cut**2) ** 2))

        # Phase rotation and growth/damping over dt
        phase_shift = np.exp(-1j * omega * dt)
        amplitude = np.exp(growth * dt) * spectral_filter * (1.0 - damp)

        # Low-k forcing mask (drives the instability)
        mask_low_k = (k2 < (k_cut * 0.5)**2).astype(np.float64)

        # Initial field: filtered noise with regime-dependent spectrum
        field = rng.standard_normal((grid_size, grid_size)) * 0.1
        field_k = np.fft.fft2(field) * spectral_filter

        # Forcing: random low-k injection (simulates gradient drive)
        forcing_r = rng.standard_normal((grid_size, grid_size))
        forcing_i = rng.standard_normal((grid_size, grid_size))
        forcing_k = np.fft.fft2(forcing_r + 1j * forcing_i) * mask_low_k
        forcing_k *= kappa * 0.5  # Scale forcing by gradient drive

        # Time step
        next_k = (field_k * phase_shift * amplitude) + forcing_k * dt
        # Ensure reality (conjugate symmetry — already handled by ifft2)

        x[i] = np.fft.ifft2(field_k).real
        y[i] = np.fft.ifft2(next_k).real

        metadata.append({
            "regime": regime,
            "alpha": alpha,
            "kappa": kappa,
            "nu": nu,
            "damp": damp,
            "k_cut": k_cut,
        })

    return x, y, metadata


def _evaluate_loss(model: MultiLayerFNO, x: np.ndarray, y: np.ndarray, max_samples: int = 16) -> float:
    n = min(max_samples, len(x))
    if n == 0:
        return 0.0
    idx = np.arange(n)
    losses = []
    for i in idx:
        pred = model.forward(x[i])
        losses.append(_relative_l2(pred, y[i]))
    return float(np.mean(losses))


def train_fno(
    n_samples: int = 10_000,
    epochs: int = 500,
    lr: float = 1e-3,
    modes: int = 12,
    width: int = 32,
    save_path: str | Path = DEFAULT_WEIGHTS_PATH,
    batch_size: int = 8,
    seed: int = 42,
    patience: int = 50,
) -> Dict[str, object]:
    """
    Train MultiLayerFNO with pure NumPy.

    Returns a history dictionary with loss curves and saved model metadata.
    """

    x, y = _generate_training_pairs(n_samples=n_samples, grid_size=64, seed=seed)
    split = max(1, int(0.9 * n_samples))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = MultiLayerFNO(modes=modes, width=width, n_layers=4, seed=seed)
    optimizer = AdamOptimizer()
    rng = np.random.default_rng(seed + 123)

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "trained_parameters": "project_head_only",
        "samples": n_samples,
        "epochs_requested": epochs,
    }

    best_project_w = model.project_w.copy()
    best_project_b = model.project_b
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            grad_w = np.zeros_like(model.project_w)
            grad_b = 0.0

            for i in batch_idx:
                pred, hidden = model.forward_with_hidden(x_train[i])
                target = y_train[i]
                target_energy = float(np.mean(target * target) + 1e-8)
                error = pred - target

                grad_y = (2.0 / error.size) * error / target_energy
                grad_w += np.tensordot(hidden, grad_y, axes=([0, 1], [0, 1]))
                grad_b += float(np.sum(grad_y))

            if len(batch_idx) == 0:
                continue

            grad_w /= len(batch_idx)
            grad_b /= len(batch_idx)

            params = {
                "project_w": model.project_w,
                "project_b": np.array([model.project_b], dtype=np.float64),
            }
            grads = {
                "project_w": grad_w,
                "project_b": np.array([grad_b], dtype=np.float64),
            }
            optimizer.step(params, grads, lr=lr)
            model.project_b = float(params["project_b"][0])

        train_loss = _evaluate_loss(model, x_train, y_train)
        val_loss = _evaluate_loss(model, x_val, y_val)
        history["train_loss"].append(train_loss)  # type: ignore[attr-defined]
        history["val_loss"].append(val_loss)  # type: ignore[attr-defined]

        if val_loss < best_val:
            best_val = val_loss
            best_project_w = model.project_w.copy()
            best_project_b = model.project_b
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.project_w = best_project_w
    model.project_b = best_project_b
    model.save_weights(save_path)

    history["saved_path"] = str(Path(save_path))
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    history["final_train_loss"] = float(history["train_loss"][-1]) if history["train_loss"] else None
    history["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else None
    return history


def train_fno_multi_regime(
    n_samples: int = 10_000,
    epochs: int = 500,
    lr: float = 1e-3,
    modes: int = 12,
    width: int = 32,
    save_path: str | Path = DEFAULT_SPARC_WEIGHTS_PATH,
    batch_size: int = 8,
    seed: int = 42,
    patience: int = 50,
    regime_weights: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """
    Train FNO on multi-regime SPARC-parameterized turbulence data.

    Generates data from ITG / TEM / ETG regimes with SPARC-relevant parameter
    ranges, then trains using the same project-head-only strategy as
    :func:`train_fno`.

    Parameters
    ----------
    n_samples : int
        Total training + validation samples.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam.
    modes : int
        Number of Fourier modes retained per layer.
    width : int
        Hidden channel width.
    save_path : str or Path
        Where to save the trained .npz weights.
    batch_size : int
        Mini-batch size.
    seed : int
        RNG seed.
    patience : int
        Early stopping patience (epochs without improvement).
    regime_weights : dict, optional
        Sampling probabilities per regime.  Default: uniform.

    Returns
    -------
    dict
        Training history with per-regime validation breakdown.
    """
    logger.info(
        "Generating %d multi-regime samples (ITG/TEM/ETG)...", n_samples
    )
    x, y, meta = _generate_multi_regime_pairs(
        n_samples=n_samples,
        grid_size=64,
        seed=seed,
        regime_weights=regime_weights,
    )

    split = max(1, int(0.9 * n_samples))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    meta_val = meta[split:]

    # Count regimes
    regime_counts: Dict[str, int] = {}
    for m in meta:
        r = m["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1
    logger.info("Regime distribution: %s", regime_counts)

    model = MultiLayerFNO(modes=modes, width=width, n_layers=4, seed=seed)
    optimizer = AdamOptimizer()
    rng = np.random.default_rng(seed + 456)

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "trained_parameters": "project_head_only",
        "samples": n_samples,
        "epochs_requested": epochs,
        "regime_counts": regime_counts,
        "data_mode": "multi_regime_sparc",
    }

    best_project_w = model.project_w.copy()
    best_project_b = model.project_b
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            grad_w = np.zeros_like(model.project_w)
            grad_b = 0.0

            for i in batch_idx:
                pred, hidden = model.forward_with_hidden(x_train[i])
                target = y_train[i]
                target_energy = float(np.mean(target * target) + 1e-8)
                error = pred - target

                grad_y = (2.0 / error.size) * error / target_energy
                grad_w += np.tensordot(hidden, grad_y, axes=([0, 1], [0, 1]))
                grad_b += float(np.sum(grad_y))

            if len(batch_idx) == 0:
                continue

            grad_w /= len(batch_idx)
            grad_b /= len(batch_idx)

            params = {
                "project_w": model.project_w,
                "project_b": np.array([model.project_b], dtype=np.float64),
            }
            grads = {
                "project_w": grad_w,
                "project_b": np.array([grad_b], dtype=np.float64),
            }
            optimizer.step(params, grads, lr=lr)
            model.project_b = float(params["project_b"][0])

        train_loss = _evaluate_loss(model, x_train, y_train)
        val_loss = _evaluate_loss(model, x_val, y_val)
        history["train_loss"].append(train_loss)  # type: ignore[attr-defined]
        history["val_loss"].append(val_loss)  # type: ignore[attr-defined]

        if val_loss < best_val:
            best_val = val_loss
            best_project_w = model.project_w.copy()
            best_project_b = model.project_b
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if epoch % 50 == 0:
            logger.info("Epoch %d: train=%.4f val=%.4f", epoch, train_loss, val_loss)

    model.project_w = best_project_w
    model.project_b = best_project_b
    model.save_weights(save_path)

    # Per-regime validation breakdown
    regime_val_losses: Dict[str, List[float]] = {}
    for j in range(len(x_val)):
        r = meta_val[j]["regime"]
        pred = model.forward(x_val[j])
        loss_j = _relative_l2(pred, y_val[j])
        regime_val_losses.setdefault(r, []).append(loss_j)

    regime_summary = {
        r: {"mean": float(np.mean(v)), "n": len(v)}
        for r, v in regime_val_losses.items()
    }
    history["regime_val_losses"] = regime_summary

    history["saved_path"] = str(Path(save_path))
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    history["final_train_loss"] = float(history["train_loss"][-1]) if history["train_loss"] else None  # type: ignore[index]
    history["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else None  # type: ignore[index]

    logger.info(
        "Multi-regime training complete: %d epochs, best_val=%.4f",
        history["epochs_completed"],
        best_val,
    )
    for r, s in regime_summary.items():
        logger.info("  %s: mean_val_loss=%.4f (n=%d)", r, s["mean"], s["n"])

    return history


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    mode = sys.argv[1] if len(sys.argv) > 1 else "multi"

    if mode == "legacy":
        summary = train_fno(
            n_samples=128, epochs=5, lr=1e-3,
            save_path=DEFAULT_WEIGHTS_PATH, patience=5,
        )
        print("FNO legacy smoke training complete.")
        print(f"Saved: {summary['saved_path']}")
        print(f"Best val loss: {summary['best_val_loss']}")
    else:
        summary = train_fno_multi_regime(
            n_samples=256, epochs=10, lr=1e-3,
            save_path=DEFAULT_SPARC_WEIGHTS_PATH, patience=5,
        )
        print("\nFNO multi-regime SPARC training complete.")
        print(f"Saved: {summary['saved_path']}")
        print(f"Best val loss: {summary['best_val_loss']}")
        print(f"Regime distribution: {summary['regime_counts']}")
        if "regime_val_losses" in summary:
            print("Per-regime validation:")
            for r, s in summary["regime_val_losses"].items():  # type: ignore[union-attr]
                print(f"  {r}: {s['mean']:.4f} (n={s['n']})")
