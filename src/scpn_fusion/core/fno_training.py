# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Pure-NumPy training for a multi-layer Fourier Neural Operator turbulence model (LEGACY).

.. note::
    As of v3.6.0, this module is superseded by the JAX-accelerated version
    in ``fno_jax_training.py``, which provides 100x faster training and
    higher accuracy (~0.001 loss).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import logging

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "weights" / "fno_turbulence.npz"
DEFAULT_SPARC_WEIGHTS_PATH = REPO_ROOT / "weights" / "fno_turbulence_sparc.npz"
DEFAULT_GS_TRANSPORT_WEIGHTS_PATH = REPO_ROOT / "weights" / "gs_transport_surrogate.npz"

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


# ── GS-Transport oracle data generator ───────────────────────────────

# MOCK_CONFIG template matching FusionKernel / TransportSolver expectations.
_GS_TRANSPORT_MOCK_CONFIG_TEMPLATE: Dict[str, object] = {
    "reactor_name": "GS-Transport-Surrogate",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


def _generate_gs_transport_pairs(
    n_samples: int = 5000,
    grid_size: int = 50,
    seed: int = 20260218,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Generate training data using TransportSolver as a physics oracle.

    For each sample, random reactor parameters are drawn from ITER/SPARC/DIII-D
    ranges.  A :class:`TransportSolver` is instantiated, given a parabolic Ti
    initial profile, evolved for 5 time-steps, and the initial/final Ti profiles
    are recorded as an (x, y) training pair.

    Parameters
    ----------
    n_samples : int
        Number of (x, y) pairs to attempt.  Failed runs are skipped.
    grid_size : int
        Radial grid resolution (must match ``TransportSolver.nr``).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    x : ndarray, shape (n_valid, grid_size)
        Initial Ti profiles.
    y : ndarray, shape (n_valid, grid_size)
        Final Ti profiles after 5 evolution steps.
    metadata : list of dicts
        Per-sample reactor parameters (Ip, BT, kappa, n_e20, P_aux, T0).
    """
    # Lazy import to avoid circular dependency
    from scpn_fusion.core.integrated_transport_solver import TransportSolver

    rng = np.random.default_rng(seed)

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    for i in range(n_samples):
        if i > 0 and i % 100 == 0:
            logger.info(
                "GS-transport pair generation: %d / %d (collected %d)",
                i, n_samples, len(x_list),
            )

        # Sample reactor parameters from ITER / SPARC / DIII-D ranges
        Ip = float(rng.uniform(1.0, 15.0))       # MA
        BT = float(rng.uniform(2.0, 12.5))       # T
        kappa = float(rng.uniform(1.5, 2.1))      # elongation
        n_e20 = float(rng.uniform(0.3, 1.2))      # 10^20 m^-3
        P_aux = float(rng.uniform(10.0, 120.0))   # MW
        T0 = float(rng.uniform(3.0, 25.0))        # keV (peak temperature)

        try:
            # Build a minimal config dict for this sample
            config = dict(_GS_TRANSPORT_MOCK_CONFIG_TEMPLATE)
            config["physics"] = {
                "plasma_current_target": Ip,
                "vacuum_permeability": 1.0,
            }

            # Write temporary config JSON (TransportSolver expects a file path)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8",
            ) as f:
                json.dump(config, f)
                tmp_path = f.name

            try:
                solver = TransportSolver(tmp_path)
            finally:
                # Clean up temp file immediately
                Path(tmp_path).unlink(missing_ok=True)

            # Ensure grid_size matches
            if solver.nr != grid_size:
                solver.nr = grid_size
                solver.rho = np.linspace(0, 1, grid_size)
                solver.drho = 1.0 / (grid_size - 1)
                solver.Te = T0 * (1 - solver.rho ** 2)
                solver.Ti = T0 * (1 - solver.rho ** 2)
                solver.ne = n_e20 * 10.0 * (1 - solver.rho ** 2) ** 0.5
                solver.chi_e = np.ones(grid_size)
                solver.chi_i = np.ones(grid_size)
                solver.D_n = np.ones(grid_size)
                solver.n_impurity = np.zeros(grid_size)
            else:
                # Set initial parabolic profiles
                solver.Ti = T0 * (1 - solver.rho ** 2)
                solver.Te = T0 * (1 - solver.rho ** 2)
                solver.ne = n_e20 * 10.0 * (1 - solver.rho ** 2) ** 0.5

            # Record initial Ti profile
            Ti_initial = solver.Ti.copy()

            # Evolve for 5 steps
            for _ in range(5):
                solver.update_transport_model(P_aux)
                solver.evolve_profiles(dt=0.1, P_aux=P_aux)

            Ti_final = solver.Ti.copy()

            # Sanity checks
            if not (np.all(np.isfinite(Ti_initial)) and np.all(np.isfinite(Ti_final))):
                continue

            x_list.append(Ti_initial)
            y_list.append(Ti_final)
            metadata.append({
                "Ip": Ip,
                "BT": BT,
                "kappa": kappa,
                "n_e20": n_e20,
                "P_aux": P_aux,
                "T0": T0,
            })

        except Exception as exc:
            logger.debug("Sample %d failed: %s", i, exc)
            continue

    if len(x_list) == 0:
        raise RuntimeError(
            "GS-transport pair generation produced 0 valid samples "
            f"out of {n_samples} attempts."
        )

    logger.info(
        "GS-transport pair generation complete: %d / %d valid samples",
        len(x_list), n_samples,
    )

    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)
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


# ── MLP Surrogate for 1D GS-Transport profiles ──────────────────────


class MLPSurrogate:
    """Simple MLP surrogate for 1D transport profile prediction.

    Architecture:
        Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, hidden_dim)
        -> GELU -> Linear(hidden_dim, input_dim)

    Uses Xavier (Glorot) uniform initialisation for weights and zero-init
    for biases.

    Parameters
    ----------
    input_dim : int
        Size of the input and output vectors (radial grid points).
    hidden_dim : int
        Width of the two hidden layers.
    seed : int
        RNG seed for reproducible initialisation.
    """

    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # Xavier uniform: U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out))
        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)

        self.W1 = _xavier(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = _xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float64)
        self.W3 = _xavier(hidden_dim, input_dim)
        self.b3 = np.zeros(input_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the MLP.

        Parameters
        ----------
        x : ndarray, shape (..., input_dim)
            Input profiles.  Supports single vectors and batches.

        Returns
        -------
        ndarray, same shape as *x*
            Predicted output profiles.
        """
        h = gelu(x @ self.W1 + self.b1)
        h = gelu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def _params_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def save_weights(self, path: str | Path) -> None:
        """Save MLP weights to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._params_dict())

    def load_weights(self, path: str | Path) -> None:
        """Load MLP weights from a ``.npz`` file."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.W1 = np.array(data["W1"], dtype=np.float64)
            self.b1 = np.array(data["b1"], dtype=np.float64)
            self.W2 = np.array(data["W2"], dtype=np.float64)
            self.b2 = np.array(data["b2"], dtype=np.float64)
            self.W3 = np.array(data["W3"], dtype=np.float64)
            self.b3 = np.array(data["b3"], dtype=np.float64)
            self.input_dim = self.W1.shape[0]
            self.hidden_dim = self.W1.shape[1]


def train_gs_transport_surrogate(
    n_samples: int = 5000,
    epochs: int = 200,
    lr: float = 1e-3,
    save_path: str | Path | None = None,
    seed: int = 42,
    patience: int = 30,
) -> Dict[str, object]:
    """Train an MLP surrogate on GS-transport oracle data.

    Generates (initial Ti profile, final Ti profile) pairs by running the
    full :class:`TransportSolver`, then fits a lightweight MLP to predict
    the evolved profile from the initial one.

    Parameters
    ----------
    n_samples : int
        Number of training samples to generate.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam.
    save_path : str, Path, or None
        Where to save the trained ``.npz`` weights.  Defaults to
        ``weights/gs_transport_surrogate.npz``.
    seed : int
        RNG seed.
    patience : int
        Early stopping patience (epochs without improvement).

    Returns
    -------
    dict
        Training history with train/val/test losses and metadata.
    """
    if save_path is None:
        save_path = DEFAULT_GS_TRANSPORT_WEIGHTS_PATH
    save_path = Path(save_path)

    logger.info(
        "Generating %d GS-transport oracle samples...", n_samples,
    )
    x, y, meta = _generate_gs_transport_pairs(
        n_samples=n_samples, grid_size=50, seed=seed,
    )
    n_valid = len(x)
    logger.info("Collected %d valid samples", n_valid)

    # Split 80 / 10 / 10
    rng = np.random.default_rng(seed + 789)
    perm = rng.permutation(n_valid)
    n_train = max(1, int(0.8 * n_valid))
    n_val = max(1, int(0.1 * n_valid))

    x_train, y_train = x[perm[:n_train]], y[perm[:n_train]]
    x_val, y_val = x[perm[n_train : n_train + n_val]], y[perm[n_train : n_train + n_val]]
    x_test, y_test = x[perm[n_train + n_val :]], y[perm[n_train + n_val :]]

    grid_size = x.shape[1]
    model = MLPSurrogate(input_dim=grid_size, hidden_dim=128, seed=seed)
    optimizer = AdamOptimizer()

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "n_samples_generated": n_valid,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(x_test),
        "epochs_requested": epochs,
        "data_mode": "gs_transport_oracle",
    }

    # Track best weights
    best_params = {k: v.copy() for k, v in model._params_dict().items()}
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        # ── Forward pass on full training set (batch = all) ──
        pred_train = model.forward(x_train)
        error_train = pred_train - y_train
        mse_train = float(np.mean(error_train ** 2))

        # ── Backprop through the 3-layer MLP (manual) ──
        # Notation: z1 = x @ W1 + b1, h1 = gelu(z1)
        #           z2 = h1 @ W2 + b2, h2 = gelu(z2)
        #           out = h2 @ W3 + b3
        z1 = x_train @ model.W1 + model.b1
        h1 = gelu(z1)
        z2 = h1 @ model.W2 + model.b2
        h2 = gelu(z2)

        # d_loss/d_out = 2/N * (pred - target) where N = n_train * grid_size
        N = float(x_train.shape[0] * grid_size)
        d_out = (2.0 / N) * error_train  # (n_train, grid_size)

        # Layer 3: out = h2 @ W3 + b3
        grad_W3 = h2.T @ d_out                  # (hidden, grid_size)
        grad_b3 = np.sum(d_out, axis=0)          # (grid_size,)
        d_h2 = d_out @ model.W3.T               # (n_train, hidden)

        # GELU derivative: gelu'(z) ≈ 0.5*(1+tanh(a)) + 0.5*z*sech^2(a)*a'
        # where a = sqrt(2/pi)*(z + 0.044715*z^3), a' = sqrt(2/pi)*(1+3*0.044715*z^2)
        # For simplicity use finite-difference approximation
        eps_fd = 1e-5
        gelu_deriv_z2 = (gelu(z2 + eps_fd) - gelu(z2 - eps_fd)) / (2.0 * eps_fd)
        d_z2 = d_h2 * gelu_deriv_z2             # (n_train, hidden)

        # Layer 2: z2 = h1 @ W2 + b2
        grad_W2 = h1.T @ d_z2                   # (hidden, hidden)
        grad_b2 = np.sum(d_z2, axis=0)           # (hidden,)
        d_h1 = d_z2 @ model.W2.T                # (n_train, hidden)

        gelu_deriv_z1 = (gelu(z1 + eps_fd) - gelu(z1 - eps_fd)) / (2.0 * eps_fd)
        d_z1 = d_h1 * gelu_deriv_z1             # (n_train, hidden)

        # Layer 1: z1 = x @ W1 + b1
        grad_W1 = x_train.T @ d_z1              # (grid_size, hidden)
        grad_b1 = np.sum(d_z1, axis=0)           # (hidden,)

        # Adam update
        params = {
            "W1": model.W1, "b1": model.b1,
            "W2": model.W2, "b2": model.b2,
            "W3": model.W3, "b3": model.b3,
        }
        grads = {
            "W1": grad_W1, "b1": grad_b1,
            "W2": grad_W2, "b2": grad_b2,
            "W3": grad_W3, "b3": grad_b3,
        }
        optimizer.step(params, grads, lr=lr)

        # ── Validation loss ──
        pred_val = model.forward(x_val)
        mse_val = float(np.mean((pred_val - y_val) ** 2))

        history["train_loss"].append(mse_train)  # type: ignore[attr-defined]
        history["val_loss"].append(mse_val)  # type: ignore[attr-defined]

        if mse_val < best_val:
            best_val = mse_val
            best_params = {k: v.copy() for k, v in model._params_dict().items()}
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = mse_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if epoch % 50 == 0:
            logger.info(
                "Epoch %d: train_mse=%.6f val_mse=%.6f", epoch, mse_train, mse_val,
            )

    # Restore best weights
    for k, v in best_params.items():
        setattr(model, k, v)

    # ── Test set evaluation ──
    if len(x_test) > 0:
        pred_test = model.forward(x_test)
        test_mse = float(np.mean((pred_test - y_test) ** 2))
        test_rel_l2 = _relative_l2(pred_test.ravel(), y_test.ravel())
    else:
        test_mse = float("nan")
        test_rel_l2 = float("nan")

    model.save_weights(save_path)

    history["saved_path"] = str(save_path)
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    history["final_train_loss"] = float(history["train_loss"][-1]) if history["train_loss"] else None  # type: ignore[index]
    history["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else None  # type: ignore[index]
    history["test_mse"] = test_mse
    history["test_rel_l2"] = test_rel_l2

    # Per-sample metadata summary (regime breakdown by machine class)
    machine_counts: Dict[str, int] = {"ITER": 0, "SPARC": 0, "DIII-D": 0, "other": 0}
    for m in meta:
        ip = float(m["Ip"])  # type: ignore[arg-type]
        bt = float(m["BT"])  # type: ignore[arg-type]
        if ip > 10 and bt > 5:
            machine_counts["ITER"] += 1
        elif bt > 10:
            machine_counts["SPARC"] += 1
        elif ip < 3:
            machine_counts["DIII-D"] += 1
        else:
            machine_counts["other"] += 1
    history["machine_class_counts"] = machine_counts

    logger.info(
        "GS-transport surrogate training complete: %d epochs, "
        "best_val_mse=%.6f, test_rel_l2=%.4f",
        history["epochs_completed"],
        best_val,
        test_rel_l2,
    )
    logger.info("Machine-class distribution: %s", machine_counts)

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
    elif mode == "gs_transport":
        summary = train_gs_transport_surrogate(
            n_samples=50, epochs=10, lr=1e-3,
            save_path=DEFAULT_GS_TRANSPORT_WEIGHTS_PATH, patience=5,
        )
        print("\nGS-transport surrogate training complete.")
        print(f"Saved: {summary['saved_path']}")
        print(f"Best val MSE: {summary['best_val_loss']}")
        print(f"Test rel L2: {summary['test_rel_l2']}")
        print(f"Machine-class distribution: {summary['machine_class_counts']}")
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
