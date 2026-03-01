# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Turbulence Suppressor
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Legacy turbulence suppression compatibility module.

Default execution uses a deterministic reduced-order compatibility backend.
The historical JAX-FNO backend is available only via explicit opt-in:
``allow_legacy=True`` or ``SCPN_ENABLE_LEGACY_FNO=1``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    try:
        from .fno_jax_training import fno_layer, model_forward
    except ImportError:
        from fno_jax_training import fno_layer, model_forward
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

DEFAULT_JAX_WEIGHTS = Path("weights/fno_turbulence_jax.npz")
LEGACY_ENABLE_ENV = "SCPN_ENABLE_LEGACY_FNO"


MODES = 12
WIDTH = 32
GRID_SIZE = 64
TIME_STEPS = 200


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _compat_suppression_from_field(field: np.ndarray) -> float:
    """Deterministic reduced-order suppression estimate from field statistics."""
    finite = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(finite**2)))
    grad_r = np.gradient(finite, axis=0)
    grad_z = np.gradient(finite, axis=1)
    grad_rms = float(np.sqrt(np.mean(grad_r**2 + grad_z**2)))
    signal = 0.9 * rms + 0.35 * grad_rms
    return float(np.clip(np.tanh(1.6 * signal), 0.0, 0.98))


class SpectralTurbulenceGenerator:
    """
    Generates synthetic ITG turbulence with Fourier-space drift-wave dynamics.
    """

    def __init__(
        self,
        size: int = GRID_SIZE,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")
        self.size = size
        self._rng = (
            rng
            if rng is not None
            else (np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng())
        )
        self.field = self._rng.standard_normal((size, size)) * 0.1
        self.field_k = np.fft.fft2(self.field)
        self.zonal_flow = 0.0 # Predator state

    def step(self, dt: float = 0.01, damping: float = 0.0) -> np.ndarray:
        kx = np.fft.fftfreq(self.size) * self.size
        ky = np.fft.fftfreq(self.size) * self.size
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k2 = kx_grid**2 + ky_grid**2
        k2[0, 0] = 1.0

        # Zonal Flow (ZF) Coupling (Predator-Prey)
        # 1. Update ZF based on turbulence intensity (Reynolds stress proxy)
        turb_intensity = np.mean(self.field**2)
        dzf_dt = 5.0 * turb_intensity - 0.5 * self.zonal_flow
        self.zonal_flow += dzf_dt * dt
        self.zonal_flow = max(0.0, self.zonal_flow)
        
        # 2. Add ZF shearing to damping
        total_damping = damping + 0.2 * self.zonal_flow

        omega = ky_grid / (1.0 + k2)
        phase_shift = np.exp(-1j * omega * dt)

        forcing = self._rng.standard_normal((self.size, self.size)) + 1j * self._rng.standard_normal((self.size, self.size))
        forcing_k = np.fft.fft2(forcing)
        forcing_k *= (k2 < 25.0) * 5.0

        self.field_k = (self.field_k * phase_shift) + (forcing_k * dt)
        self.field_k *= np.exp(-0.001 * k2 * dt)
        self.field_k *= 1.0 - np.clip(total_damping, 0.0, 1.0)

        self.field = np.fft.ifft2(self.field_k).real
        return self.field


class FNO_Controller:
    """
    Predict turbulence suppression with explicit legacy opt-in semantics.

    Default backend is deterministic compatibility mode. Legacy JAX-FNO can be
    enabled with ``allow_legacy=True`` or ``SCPN_ENABLE_LEGACY_FNO=1``.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        *,
        allow_legacy: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_JAX_WEIGHTS
        self.params = {}
        self.loaded_weights = False
        self.legacy_enabled = bool(allow_legacy) or _env_enabled(LEGACY_ENABLE_ENV)
        self.backend = "compat_reduced_order"

        if self.legacy_enabled and _HAS_JAX and self.weights_path.exists():
            self.load_weights(str(self.weights_path))
            self.backend = "legacy_jax_fno"
        elif self.legacy_enabled and not _HAS_JAX:
            logger.warning("Legacy FNO requested but JAX is unavailable; using compatibility backend.")
        elif self.legacy_enabled and not self.weights_path.exists():
            logger.warning(
                "Legacy FNO requested but weights not found at %s; using compatibility backend.",
                self.weights_path,
            )
        elif not self.legacy_enabled:
            logger.info(
                "Legacy FNO disabled by default; using compatibility backend. "
                "Set %s=1 or allow_legacy=True to enable historical lane.",
                LEGACY_ENABLE_ENV,
            )

    def load_weights(self, path: str) -> None:
        if not _HAS_JAX:
            raise RuntimeError("JAX backend not available; cannot load legacy FNO weights.")
        with np.load(path, allow_pickle=False) as data:
            self.params = {k: jnp.array(data[k]) for k in data.files}
        self.loaded_weights = True

    @staticmethod
    def _postprocess_prediction(field: np.ndarray, suppression: float) -> np.ndarray:
        pred_field = np.asarray(field, dtype=np.float64) * (1.0 - float(np.clip(suppression, 0.0, 0.98)))
        field_k = np.fft.fft2(pred_field)
        field_k[0, 0] = 0.0  # remove non-physical DC drift
        out = np.fft.ifft2(field_k).real
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_and_suppress(self, field: np.ndarray) -> Tuple[float, np.ndarray]:
        field_np = np.asarray(field, dtype=np.float64)
        if field_np.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"field must have shape {(GRID_SIZE, GRID_SIZE)}, got {field_np.shape}.")

        # Compatibility mode is the safe default lane.
        if not (self.legacy_enabled and _HAS_JAX and self.loaded_weights):
            suppression = _compat_suppression_from_field(field_np)
            return suppression, self._postprocess_prediction(field_np, suppression)

        x_jax = jnp.asarray(field_np).reshape(GRID_SIZE, GRID_SIZE, 1)
        prediction_val = model_forward(self.params, x_jax)
        suppression = float(np.clip(jnp.tanh(prediction_val * 2.0), 0.0, 0.98))
        return suppression, self._postprocess_prediction(field_np, suppression)


def run_fno_simulation(
    time_steps: int = TIME_STEPS,
    weights_path: Optional[str] = None,
    *,
    seed: int = 42,
    allow_legacy: bool = False,
    save_plot: bool = True,
    output_path: str = "FNO_Turbulence_Result.png",
    verbose: bool = True,
) -> dict[str, Any]:
    if verbose:
        print("--- SCPN FNO Compatibility: Spectral Turbulence Suppression ---")

    sim = SpectralTurbulenceGenerator(seed=int(seed))
    ai = FNO_Controller(weights_path=weights_path, allow_legacy=allow_legacy)

    if verbose:
        if ai.backend == "legacy_jax_fno":
            print(f"Loaded trained FNO weights: {ai.weights_path}")
        else:
            print(f"Using backend={ai.backend} (legacy_enabled={ai.legacy_enabled}).")

    history_energy = []
    last_control = 0.0

    if verbose:
        print(f"Running {time_steps} steps of Gyro-Fluid dynamics...")

    for t in range(time_steps):
        control = 0.0
        prediction = np.zeros_like(sim.field)

        if t > 50:
            control, prediction = ai.predict_and_suppress(sim.field)
        last_control = float(control)

        field = sim.step(damping=control)

        turb_energy = float(np.mean(field**2))
        pred_energy = float(np.mean(prediction**2))
        history_energy.append(turb_energy)

        if t % 20 == 0:
            if verbose:
                print(
                    f"Step {t}: Energy={turb_energy:.4f} | "
                    f"PredE={pred_energy:.4f} | Suppression={control:.2f}"
                )

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(sim.field, cmap="RdBu", vmin=-0.5, vmax=0.5)
            ax1.set_title(f"Turbulence Density (t={time_steps})")

            ax2.plot(history_energy, "k-", label="Turbulence Energy")
            ax2.axvline(50, color="r", linestyle="--", label="AI ON")
            ax2.set_title("Suppression Performance")
            ax2.set_xlabel("Time Step")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            plot_saved = True
            if verbose:
                print(f"Analysis saved: {output_path}")
        except Exception as exc:
            plot_error = str(exc)
            if verbose:
                print(f"Simulation completed without plot artifact: {exc}")

    hist = np.asarray(history_energy, dtype=np.float64)
    return {
        "seed": int(seed),
        "steps": int(time_steps),
        "loaded_weights": bool(ai.loaded_weights),
        "backend": str(ai.backend),
        "legacy_enabled": bool(ai.legacy_enabled),
        "final_energy": float(hist[-1]) if hist.size else 0.0,
        "mean_energy_last_20": float(np.mean(hist[-20:])) if hist.size else 0.0,
        "max_energy": float(np.max(hist)) if hist.size else 0.0,
        "final_suppression": float(last_control),
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
        "output_path": str(output_path) if plot_saved else None,
    }


if __name__ == "__main__":
    run_fno_simulation()
