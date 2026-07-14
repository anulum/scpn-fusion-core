# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO Training
"""
Pure-NumPy training for a multi-layer Fourier Neural Operator turbulence model (LEGACY).

.. note::
    As of v3.6.0, this module is superseded by the JAX-accelerated version
    in ``fno_jax_training.py``, which provides 100x faster training and
    higher accuracy (~0.001 loss).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

import logging

import numpy as np
from numpy.typing import NDArray

from scpn_fusion._data_paths import default_artifact_path
from scpn_fusion.io.safe_loaders import checked_np_load
from ._surrogate_utils import AdamOptimizer, gelu, relative_l2
from scpn_fusion.core.fno_training_multi_regime import (
    SPARC_REGIMES,  # noqa: F401 - re-exported compatibility surface
    _generate_multi_regime_pairs,  # noqa: F401 - re-exported compatibility surface
    _sample_regime_params,  # noqa: F401 - re-exported compatibility surface
    train_fno_multi_regime as _train_fno_multi_regime_impl,
)
from scpn_fusion.core.gs_transport_surrogate_training import (
    MLPSurrogate,  # noqa: F401 - re-exported compatibility surface
    _generate_gs_transport_pairs,  # noqa: F401 - re-exported compatibility surface
    train_gs_transport_surrogate,
)

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = default_artifact_path("weights", "fno_turbulence.npz")
DEFAULT_SPARC_WEIGHTS_PATH = default_artifact_path("weights", "fno_turbulence_sparc.npz")
DEFAULT_GS_TRANSPORT_WEIGHTS_PATH = default_artifact_path("weights", "gs_transport_surrogate.npz")


class MultiLayerFNO:
    """
    Multi-layer FNO model.

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

        self.lift_w: FloatArray = self.rng.normal(0.0, 0.1, size=(self.width,))
        self.lift_b: FloatArray = np.zeros((self.width,), dtype=np.float64)
        self.project_w: FloatArray = self.rng.normal(0.0, 0.1, size=(self.width,))
        self.project_b = 0.0

        self.layers: List[Dict[str, FloatArray]] = []
        for _ in range(self.n_layers):
            self.layers.append(
                {
                    "wr": self.rng.normal(0.0, 0.03, size=(self.width, self.modes, self.modes)),
                    "wi": self.rng.normal(0.0, 0.03, size=(self.width, self.modes, self.modes)),
                    "skip_w": np.eye(self.width)
                    + self.rng.normal(0.0, 0.01, size=(self.width, self.width)),
                    "skip_b": np.zeros((self.width,), dtype=np.float64),
                }
            )

    def _spectral_convolution(self, h: FloatArray, layer: Dict[str, FloatArray]) -> FloatArray:
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

    def _forward_hidden(self, x_field: FloatArray) -> FloatArray:
        h = x_field[:, :, None] * self.lift_w[None, None, :] + self.lift_b[None, None, :]
        for layer in self.layers:
            spectral = self._spectral_convolution(h, layer)
            pointwise = (
                np.tensordot(h, layer["skip_w"], axes=([2], [0])) + layer["skip_b"][None, None, :]
            )
            h = gelu(spectral + pointwise)
        return np.asarray(h, dtype=np.float64)

    def forward_with_hidden(self, x_field: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Return the projected field and final hidden representation."""
        h = self._forward_hidden(x_field)
        y = np.asarray(
            np.tensordot(h, self.project_w, axes=([2], [0])) + self.project_b, dtype=np.float64
        )
        return y, h

    def forward(self, x_field: FloatArray) -> FloatArray:
        """Evaluate the FNO field-to-field surrogate for one input field."""
        y, _ = self.forward_with_hidden(x_field)
        return y

    def save_weights(self, path: str | Path) -> None:
        """Serialise FNO architecture metadata and NumPy weights to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, FloatArray] = {
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

        # numpy's savez stub types **kwds against its keyword-only allow_pickle: bool
        # parameter, so a dynamically-keyed payload mapping cannot be expressed without
        # this suppression; the runtime call is the documented dict-unpacking form.
        np.savez(path, **payload)  # type: ignore[arg-type, unused-ignore]

    def load_weights(self, path: str | Path) -> None:
        """Load FNO architecture metadata and NumPy weights from ``path``."""
        path = Path(path)
        with checked_np_load(path, allow_pickle=False) as data:
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


def _generate_training_pairs(
    n_samples: int,
    grid_size: int,
    seed: int,
    damping: float = 0.18,
) -> Tuple[FloatArray, FloatArray]:
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

        forcing = rng.standard_normal((grid_size, grid_size)) + 1j * rng.standard_normal(
            (grid_size, grid_size)
        )
        forcing_k = np.fft.fft2(forcing) * mask_low_k * 5.0

        next_k = (field_k * phase_shift) + forcing_k * dt
        next_k = next_k * viscous

        x[i] = field
        y[i] = np.fft.ifft2(next_k).real

    return x, y


def _evaluate_loss(
    model: MultiLayerFNO, x: FloatArray, y: FloatArray, max_samples: int = 16
) -> float:
    n = min(max_samples, len(x))
    if n == 0:
        return 0.0
    idx = np.arange(n)
    losses = []
    for i in idx:
        pred = model.forward(x[i])
        losses.append(relative_l2(pred, y[i]))
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

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    history: Dict[str, object] = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
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
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

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
    history["epochs_completed"] = len(train_loss_hist)
    history["final_train_loss"] = float(train_loss_hist[-1]) if train_loss_hist else None
    history["final_val_loss"] = float(val_loss_hist[-1]) if val_loss_hist else None
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
    """Compatibility wrapper over extracted multi-regime training runtime."""
    return _train_fno_multi_regime_impl(
        n_samples=n_samples,
        epochs=epochs,
        lr=lr,
        modes=modes,
        width=width,
        save_path=save_path,
        batch_size=batch_size,
        seed=seed,
        patience=patience,
        regime_weights=regime_weights,
    )


def _run_training_smoke_cli(argv: Sequence[str]) -> Dict[str, object]:
    """Run the lightweight standalone training entrypoint for one mode."""
    mode = argv[0] if argv else "multi"
    if mode == "legacy":
        summary = train_fno(
            n_samples=128,
            epochs=5,
            lr=1e-3,
            save_path=DEFAULT_WEIGHTS_PATH,
            patience=5,
        )
        logger.info("FNO legacy smoke training complete")
        logger.info("Saved: %s", summary["saved_path"])
        logger.info("Best val loss: %s", summary["best_val_loss"])
    elif mode == "gs_transport":
        summary = train_gs_transport_surrogate(
            n_samples=50,
            epochs=10,
            lr=1e-3,
            save_path=DEFAULT_GS_TRANSPORT_WEIGHTS_PATH,
            patience=5,
        )
        logger.info("GS-transport surrogate training complete")
        logger.info("Saved: %s", summary["saved_path"])
        logger.info("Best val MSE: %s", summary["best_val_loss"])
        logger.info("Test rel L2: %s", summary["test_rel_l2"])
        logger.info("Machine-class distribution: %s", summary["machine_class_counts"])
    else:
        summary = train_fno_multi_regime(
            n_samples=256,
            epochs=10,
            lr=1e-3,
            save_path=DEFAULT_SPARC_WEIGHTS_PATH,
            patience=5,
        )
        logger.info("FNO multi-regime SPARC training complete")
        logger.info("Saved: %s", summary["saved_path"])
        logger.info("Best val loss: %s", summary["best_val_loss"])
        logger.info("Regime distribution: %s", summary["regime_counts"])
        if "regime_val_losses" in summary:
            logger.info("Per-regime validation")
            regime_val_losses = cast("dict[str, dict[str, float]]", summary["regime_val_losses"])
            for r, s in regime_val_losses.items():
                logger.info("Regime validation: regime=%s mean=%.4f n=%s", r, s["mean"], s["n"])
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    _run_training_smoke_cli(sys.argv[1:])
