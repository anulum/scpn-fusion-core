# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Run Diagnostics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    print("--- SCPN SYNTHETIC DIAGNOSTICS & TOMOGRAPHY ---")
    
    # 1. Physics Ground Truth
    kernel = FusionKernel(str(config_path))
    kernel.solve_equilibrium()
    
    # Create Phantom (Emission Profile)
    # Emission ~ Density^2 * Te^0.5 (Bremsstrahlung)
    # We use Psi to shape it
    idx_max = np.argmax(kernel.Psi)
    iz_ax, ir_ax = np.unravel_index(idx_max, kernel.Psi.shape)
    Psi_ax = kernel.Psi[iz_ax, ir_ax]
    
    Phantom = (kernel.Psi / Psi_ax)**2
    Phantom = np.clip(Phantom, 0, None)
    # Add a "Hot Spot" (Instability) to see if Tomography catches it
    Phantom[40, 20] += 0.5 
    
    # 2. Sensors
    sensors = SensorSuite(kernel)
    
    # Measure
    print("Measuring Signals...")
    mag_signals = sensors.measure_magnetics()
    bolo_signals = sensors.measure_bolometer(Phantom)
    
    print(f"  Magnetic Probes: {len(mag_signals)} channels")
    print(f"  Bolometer Cameras: {len(bolo_signals)} channels")
    
    # 3. Tomography
    print("Running Tomographic Inversion...")
    tomo = PlasmaTomography(sensors)
    reconstruction = tomo.reconstruct(bolo_signals)
    
    # 4. Visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = tomo.plot_reconstruction(Phantom, reconstruction)
    tomo_path = output_dir / "Tomography_Result.png"
    plt.savefig(str(tomo_path))
    print(f"Saved: {tomo_path}")
    
    # Save Sensors Geometry
    fig2 = sensors.visualize_setup()
    geom_path = output_dir / "Sensor_Geometry.png"
    plt.savefig(str(geom_path))
    print(f"Saved: {geom_path}")

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
    args = parser.parse_args()
    run_diag_demo(config_path=args.config, output_dir=args.output_dir)
