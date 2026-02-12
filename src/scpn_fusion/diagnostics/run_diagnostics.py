import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.diagnostics.synthetic_sensors import SensorSuite
from scpn_fusion.diagnostics.tomography import PlasmaTomography

def run_diag_demo():
    print("--- SCPN SYNTHETIC DIAGNOSTICS & TOMOGRAPHY ---")
    
    # 1. Physics Ground Truth
    config_path = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    kernel = FusionKernel(config_path)
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
    fig = tomo.plot_reconstruction(Phantom, reconstruction)
    plt.savefig("Tomography_Result.png")
    print("Saved: Tomography_Result.png")
    
    # Save Sensors Geometry
    fig2 = sensors.visualize_setup()
    plt.savefig("Sensor_Geometry.png")

if __name__ == "__main__":
    run_diag_demo()
