# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Synthetic Sensors
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from scpn_fusion.diagnostics.forward import ForwardDiagnosticChannels, generate_forward_channels

class SensorSuite:
    """
    Simulates physical diagnostics installed on the tokamak wall.
    1. Magnetic Loops (Measure Flux Psi).
    2. Bolometer Cameras (Measure Line-Integrated Radiation).
    """
    def __init__(
        self,
        kernel: Any,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")
        self.kernel = kernel
        self._rng = (
            rng
            if rng is not None
            else (np.random.default_rng(int(seed)) if seed is not None else None)
        )
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

    def _noise(self, scale: float) -> float:
        sigma = float(max(scale, 0.0))
        if self._rng is not None:
            return float(self._rng.normal(0.0, sigma))
        return float(np.random.normal(0.0, sigma))

    def measure_magnetics(self):
        """
        Returns Flux Psi at sensor locations.
        Interpolates from Kernel grid.
        """
        # Map R,Z to grid indices
        measurements = []
        for i in range(len(self.wall_R)):
            r, z = self.wall_R[i], self.wall_Z[i]
            
            # Bilinear Interpolation for higher accuracy
            ir = int((r - self.kernel.R[0]) / self.kernel.dR)
            iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)
            
            if 0 <= ir < self.kernel.NR-1 and 0 <= iz < self.kernel.NZ-1:
                # Interpolation weights
                wr = (r - self.kernel.R[ir]) / self.kernel.dR
                wz = (z - self.kernel.Z[iz]) / self.kernel.dZ
                
                v00 = self.kernel.Psi[iz, ir]
                v10 = self.kernel.Psi[iz, ir+1]
                v01 = self.kernel.Psi[iz+1, ir]
                v11 = self.kernel.Psi[iz+1, ir+1]
                
                val = (1-wr)*(1-wz)*v00 + wr*(1-wz)*v10 + (1-wr)*wz*v01 + wr*wz*v11
            else:
                val = self.kernel.Psi[np.clip(iz, 0, self.kernel.NZ-1), np.clip(ir, 0, self.kernel.NR-1)]
                
            # Add Sensor Noise
            val += self._noise(0.01)
            measurements.append(val)
            
        return np.array(measurements)

    def measure_b_field(self):
        """
        Calculates local B_field (Br, Bz) at sensor locations using Biot-Savart.
        Harden with toroidal filament integration.
        """
        mu0 = 4.0 * np.pi * 1e-7
        Br = np.zeros(len(self.wall_R))
        Bz = np.zeros(len(self.wall_R))
        
        # Plasma filaments from J map
        # Optimization: Downsample J for BS-integration
        step = 4
        J_sub = self.kernel.J[::step, ::step]
        R_sub = self.kernel.RR[::step, ::step]
        Z_sub = self.kernel.ZZ[::step, ::step]
        dA = (self.kernel.dR * step) * (self.kernel.dZ * step)
        
        fil_r = R_sub.flatten()
        fil_z = Z_sub.flatten()
        fil_I = J_sub.flatten() * dA
        
        for i in range(len(self.wall_R)):
            wr, wz = self.wall_R[i], self.wall_Z[i]
            
            # Simple 2D Green's function for toroidal current filaments (Infinite line approx or elliptic)
            # Br = -mu0/2pi * sum( I_j * (wz - fil_z) / dist^2 )
            # Bz =  mu0/2pi * sum( I_j * (wr - fil_r) / dist^2 )
            dr = wr - fil_r
            dz = wz - fil_z
            dist_sq = dr**2 + dz**2 + 1e-6
            
            Br[i] = - (mu0 / (2*np.pi)) * np.sum(fil_I * dz / dist_sq)
            Bz[i] = (mu0 / (2*np.pi)) * np.sum(fil_I * dr / dist_sq)
            
            # Add Noise
            Br[i] += self._noise(0.005)
            Bz[i] += self._noise(0.005)
            
        return Br, Bz

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
            integral += self._noise(0.05 * integral if integral > 0 else 0.001)
            signals.append(integral)
            
        return np.array(signals)

    def measure_forward_channels(
        self,
        electron_density_m3,
        neutron_source_m3_s,
        *,
        detector_efficiency=0.12,
        solid_angle_fraction=1.0e-4,
        laser_wavelength_m=1.064e-6,
    ) -> ForwardDiagnosticChannels:
        """
        Generate deterministic forward-model raw channels.
        """
        chords = [(tuple(start), tuple(end)) for start, end in self.bolo_chords]
        return generate_forward_channels(
            electron_density_m3=np.asarray(electron_density_m3, dtype=np.float64),
            neutron_source_m3_s=np.asarray(neutron_source_m3_s, dtype=np.float64),
            r_grid=np.asarray(self.kernel.R, dtype=np.float64),
            z_grid=np.asarray(self.kernel.Z, dtype=np.float64),
            interferometer_chords=chords,
            volume_element_m3=float(self.kernel.dR * self.kernel.dZ),
            detector_efficiency=float(detector_efficiency),
            solid_angle_fraction=float(solid_angle_fraction),
            laser_wavelength_m=float(laser_wavelength_m),
        )

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
