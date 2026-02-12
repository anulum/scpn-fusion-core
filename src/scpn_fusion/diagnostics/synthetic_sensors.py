import numpy as np
import matplotlib.pyplot as plt

class SensorSuite:
    """
    Simulates physical diagnostics installed on the tokamak wall.
    1. Magnetic Loops (Measure Flux Psi).
    2. Bolometer Cameras (Measure Line-Integrated Radiation).
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.wall_R, self.wall_Z = self._generate_sensor_positions()
        
        # Bolometer Chords (Lines of Sight)
        # Fan geometry looking from top port
        self.bolo_chords = self._generate_bolo_chords()
        
    def _generate_sensor_positions(self):
        # Place 20 magnetic probes around the wall
        theta = np.linspace(0, 2*np.pi, 20)
        R0, a, kappa = 6.0, 3.0, 1.8
        R_s = R0 + (a+0.5) * np.cos(theta)
        Z_s = (a+0.5) * kappa * np.sin(theta)
        return R_s, Z_s

    def _generate_bolo_chords(self):
        # 16 Chords fanning out from a top port (R=6, Z=5)
        # Watching the Divertor region (R=4..8, Z=-4)
        origin = np.array([6.0, 5.0])
        targets_R = np.linspace(3.0, 9.0, 16)
        targets_Z = np.full(16, -4.0)
        
        chords = []
        for i in range(16):
            target = np.array([targets_R[i], targets_Z[i]])
            chords.append((origin, target))
        return chords

    def measure_magnetics(self):
        """
        Returns Flux Psi at sensor locations.
        Interpolates from Kernel grid.
        """
        # Map R,Z to grid indices
        measurements = []
        for i in range(len(self.wall_R)):
            r, z = self.wall_R[i], self.wall_Z[i]
            
            # Simple Nearest Neighbor (for speed) or Bilinear
            ir = int((r - self.kernel.R[0]) / self.kernel.dR)
            iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)
            
            # Bounds check
            ir = np.clip(ir, 0, self.kernel.NR-1)
            iz = np.clip(iz, 0, self.kernel.NZ-1)
            
            val = self.kernel.Psi[iz, ir]
            # Add Sensor Noise
            val += np.random.normal(0, 0.01) 
            measurements.append(val)
            
        return np.array(measurements)

    def measure_bolometer(self, emission_profile):
        """
        Integrates emission along chords.
        Signal = Integral( E(l) * dl )
        """
        signals = []
        
        # Grid coordinates
        RR = self.kernel.RR
        ZZ = self.kernel.ZZ
        
        for start, end in self.bolo_chords:
            # Ray marching along the chord
            num_samples = 50
            r_samples = np.linspace(start[0], end[0], num_samples)
            z_samples = np.linspace(start[1], end[1], num_samples)
            
            integral = 0.0
            dl = np.linalg.norm(end - start) / num_samples
            
            for k in range(num_samples):
                r, z = r_samples[k], z_samples[k]
                
                # Nearest Grid Point
                ir = int((r - self.kernel.R[0]) / self.kernel.dR)
                iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)
                
                if 0 <= ir < self.kernel.NR and 0 <= iz < self.kernel.NZ:
                    val = emission_profile[iz, ir]
                    integral += val * dl
            
            # Add Noise (Photon shot noise)
            integral += np.random.normal(0, 0.05 * integral if integral > 0 else 0.001)
            signals.append(integral)
            
        return np.array(signals)

    def visualize_setup(self):
        fig, ax = plt.subplots()
        ax.set_title("Diagnostics Geometry")
        # Plasma contour
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, colors='gray', alpha=0.3)
        # Magnetic Probes
        ax.plot(self.wall_R, self.wall_Z, 'ro', label='B-Probes')
        # Bolo Chords
        for i, (start, end) in enumerate(self.bolo_chords):
            ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', alpha=0.5, 
                    label='Bolometer' if i==0 else "")
            
        ax.set_aspect('equal')
        ax.legend()
        return fig
