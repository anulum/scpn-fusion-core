# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Run Diagnostics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.diagnostics.synthetic_sensors import SensorSuite
from scpn_fusion.diagnostics.tomography import PlasmaTomography

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "validation" / "iter_validated_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "diagnostics_demo"


def run_diag_demo(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = 42,
    save_figures: bool = True,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
    sensor_factory: Callable[[Any], Any] = SensorSuite,
    tomography_factory: Callable[[Any], Any] = PlasmaTomography,
) -> Dict[str, Any]:
    """
    Run synthetic diagnostics + tomography and return deterministic summary.
    """
    cfg = Path(config_path)
    out_dir = Path(output_dir)
    np.random.seed(int(seed))

    if verbose:
        print("--- SCPN SYNTHETIC DIAGNOSTICS & TOMOGRAPHY ---")

    kernel = kernel_factory(str(cfg))
    if hasattr(kernel, "solve_equilibrium"):
        kernel.solve_equilibrium()

    psi = np.asarray(kernel.Psi, dtype=np.float64)
    if psi.size == 0:
        raise ValueError("Kernel Psi grid is empty.")
    idx_max = int(np.argmax(psi))
    iz_ax, ir_ax = np.unravel_index(idx_max, psi.shape)
    psi_ax = float(psi[iz_ax, ir_ax])
    if abs(psi_ax) < 1e-12:
        psi_ax = 1.0

    phantom = np.clip((psi / psi_ax) ** 2, 0.0, None)
    hot_iz = int(np.clip(int(0.65 * (phantom.shape[0] - 1)), 0, phantom.shape[0] - 1))
    hot_ir = int(np.clip(int(0.35 * (phantom.shape[1] - 1)), 0, phantom.shape[1] - 1))
    phantom[hot_iz, hot_ir] += 0.5

    sensors = sensor_factory(kernel)
    if verbose:
        print("Measuring Signals...")
    mag_signals = np.asarray(sensors.measure_magnetics(), dtype=np.float64)
    bolo_signals = np.asarray(sensors.measure_bolometer(phantom), dtype=np.float64)

    if verbose:
        print(f"  Magnetic Probes: {int(mag_signals.size)} channels")
        print(f"  Bolometer Cameras: {int(bolo_signals.size)} channels")
        print("Running Tomographic Inversion...")

    tomo = tomography_factory(sensors)
    reconstruction = np.asarray(tomo.reconstruct(bolo_signals), dtype=np.float64)

    flat_phantom = phantom.reshape(-1)
    flat_recon = reconstruction.reshape(-1)
    n = min(flat_phantom.size, flat_recon.size)
    rmse = (
        float(np.sqrt(np.mean((flat_phantom[:n] - flat_recon[:n]) ** 2)))
        if n > 0
        else 0.0
    )

    plot_saved = False
    plot_error: Optional[str] = None
    tomo_path: Optional[str] = None
    geom_path: Optional[str] = None
    if save_figures:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = tomo.plot_reconstruction(phantom, reconstruction)
            tomo_out = out_dir / "Tomography_Result.png"
            fig.savefig(str(tomo_out))
            plt.close(fig)

            fig2 = sensors.visualize_setup()
            geom_out = out_dir / "Sensor_Geometry.png"
            fig2.savefig(str(geom_out))
            plt.close(fig2)

            tomo_path = str(tomo_out)
            geom_path = str(geom_out)
            plot_saved = bool(tomo_out.exists() and geom_out.exists())
            if verbose:
                print(f"Saved: {tomo_out}")
                print(f"Saved: {geom_out}")
        except Exception as exc:
            plot_error = f"{exc.__class__.__name__}: {exc}"

    return {
        "seed": int(seed),
        "config_path": str(cfg),
        "mag_channels": int(mag_signals.size),
        "bolo_channels": int(bolo_signals.size),
        "phantom_sum": float(np.sum(phantom)),
        "reconstruction_sum": float(np.sum(reconstruction)),
        "reconstruction_rmse": rmse,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
        "tomography_path": tomo_path,
        "sensor_geometry_path": geom_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synthetic diagnostics and tomography demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to reactor configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated diagnostic figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sensor noise/reconstruction runs.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Run diagnostics without writing PNG outputs.",
    )
    args = parser.parse_args()
    run_diag_demo(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=int(args.seed),
        save_figures=not bool(args.no_figures),
    )
