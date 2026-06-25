# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Synthetic Sensors
"""Synthetic diagnostics surface generating magnetics, bolometer and interferometry signals."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from scpn_fusion.diagnostics.forward import ForwardDiagnosticChannels, generate_forward_channels

FloatArray = NDArray[np.float64]
Chord = tuple[FloatArray, FloatArray]

# Magnetic-probe wall geometry (shared with the Rust tier's SensorSuite constants).
N_MAGNETIC_PROBES = 20
WALL_MAJOR_RADIUS = 6.0
WALL_MINOR_RADIUS = 3.0
WALL_ELONGATION = 1.8
WALL_OFFSET = 0.5


def magnetic_probe_positions() -> tuple[FloatArray, FloatArray]:
    """Return the (R, Z) positions of the magnetic probes on the D-shaped wall.

    Returns
    -------
    wall_R, wall_Z : numpy.ndarray
        Probe coordinates [m], ``N_MAGNETIC_PROBES`` points spread over the wall
        ellipse (``theta`` endpoint-inclusive ``linspace(0, 2*pi, N)``).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, N_MAGNETIC_PROBES)
    wall_radius = WALL_MINOR_RADIUS + WALL_OFFSET
    wall_r = WALL_MAJOR_RADIUS + wall_radius * np.cos(theta)
    wall_z = wall_radius * WALL_ELONGATION * np.sin(theta)
    return np.asarray(wall_r, dtype=np.float64), np.asarray(wall_z, dtype=np.float64)


def measure_magnetics(
    psi: FloatArray,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> FloatArray:
    """Bilinearly interpolate the poloidal flux at the magnetic-probe positions.

    Canonical free-function reference for the ``measure_magnetics`` dispatch
    kernel (:mod:`scpn_fusion.core._multi_compat`). It is the deterministic,
    noise-free measurement; sensor noise is an additive simulation concern added
    by :meth:`SensorSuite.measure_magnetics`. The Rust tier
    (``scpn_fusion_rs.measure_magnetics``) evaluates the same bilinear stencil at
    the same probe positions, so the two tiers agree to a tight tolerance.

    Parameters
    ----------
    psi : FloatArray
        Poloidal flux on the grid, shape ``(nz, nr)``.
    nr, nz : int
        Grid dimensions.
    r_min, r_max, z_min, z_max : float
        Grid extent [m].

    Returns
    -------
    FloatArray
        Flux at each probe, length ``N_MAGNETIC_PROBES``.

    Raises
    ------
    ValueError
        If ``psi`` does not have shape ``(nz, nr)``.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.shape != (nz, nr):
        raise ValueError(f"psi must have shape (nz, nr) = ({nz}, {nr}); got {psi_arr.shape}.")
    dr = (r_max - r_min) / (nr - 1) if nr > 1 else 1.0
    dz = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0

    wall_r, wall_z = magnetic_probe_positions()
    measurements = np.empty(wall_r.size, dtype=np.float64)
    for i in range(wall_r.size):
        r = float(wall_r[i])
        z = float(wall_z[i])
        ir = int((r - r_min) / dr)
        iz = int((z - z_min) / dz)
        if 0 <= ir < nr - 1 and 0 <= iz < nz - 1:
            wr = (r - (r_min + ir * dr)) / dr
            wz = (z - (z_min + iz * dz)) / dz
            v00 = psi_arr[iz, ir]
            v10 = psi_arr[iz, ir + 1]
            v01 = psi_arr[iz + 1, ir]
            v11 = psi_arr[iz + 1, ir + 1]
            measurements[i] = (
                (1.0 - wr) * (1.0 - wz) * v00
                + wr * (1.0 - wz) * v10
                + (1.0 - wr) * wz * v01
                + wr * wz * v11
            )
        else:
            measurements[i] = psi_arr[int(np.clip(iz, 0, nz - 1)), int(np.clip(ir, 0, nr - 1))]
    return measurements


class SensorSuite:
    """Simulate physical diagnostics installed on the tokamak wall.

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

    def _generate_sensor_positions(self) -> tuple[FloatArray, FloatArray]:
        # Place the magnetic probes around the D-shaped wall.
        return magnetic_probe_positions()

    def _generate_bolo_chords(self) -> list[Chord]:
        # 16 Chords fanning out from a top port (R=6, Z=5)
        # Watching the Divertor region (R=4..8, Z=-4)
        origin = np.array([6.0, 5.0])
        targets_R = np.linspace(3.0, 9.0, 16)
        targets_Z = np.full(16, -4.0)

        chords: list[Chord] = []
        for i in range(16):
            target = np.array([targets_R[i], targets_Z[i]])
            chords.append((origin, target))
        return chords

    def _noise(self, scale: float) -> float:
        sigma = float(max(scale, 0.0))
        if self._rng is not None:
            return float(self._rng.normal(0.0, sigma))
        return float(np.random.normal(0.0, sigma))

    def measure_magnetics(self) -> FloatArray:
        """Return flux Psi at the probe locations (bilinear interp + sensor noise).

        The deterministic bilinear measurement is delegated to the free function
        :func:`measure_magnetics`; this method adds the simulated Gaussian sensor
        noise on top.
        """
        clean = measure_magnetics(
            np.asarray(self.kernel.Psi, dtype=np.float64),
            int(self.kernel.NR),
            int(self.kernel.NZ),
            float(self.kernel.R[0]),
            float(self.kernel.R[-1]),
            float(self.kernel.Z[0]),
            float(self.kernel.Z[-1]),
        )
        return np.asarray(
            [float(v) + self._noise(0.01) for v in clean],
            dtype=np.float64,
        )

    def measure_b_field(self) -> tuple[FloatArray, FloatArray]:
        """Compute local B_field (Br, Bz) at sensor locations using Biot-Savart.

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

            Br[i] = -(mu0 / (2 * np.pi)) * np.sum(fil_I * dz / dist_sq)
            Bz[i] = (mu0 / (2 * np.pi)) * np.sum(fil_I * dr / dist_sq)

            # Add Noise
            Br[i] += self._noise(0.005)
            Bz[i] += self._noise(0.005)

        return Br, Bz

    def measure_bolometer(self, emission_profile: FloatArray) -> FloatArray:
        """Integrate emission along chords.

        Signal = Integral( E(l) * dl ).
        """
        signals: list[float] = []

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
                    integral += float(val) * float(dl)

            # Add Noise (Photon shot noise)
            integral += self._noise(0.05 * integral if integral > 0 else 0.001)
            signals.append(integral)

        return np.asarray(signals, dtype=np.float64)

    def measure_interferometer(self, density_profile_19: FloatArray) -> FloatArray:
        """Simulate Multi-Chord Interferometer.

        Measures Phase Shift phi = lambda * r_e * Integral(n_e dl).
        Harden with Phase Wrapping (modulo 2pi) and Refraction noise.
        """
        signals_phase: list[float] = []
        lambda_laser = 10.6e-6  # CO2 laser (10.6 um)
        r_e = 2.817e-15  # Classical electron radius

        # Grid coordinates
        RR = self.kernel.RR
        ZZ = self.kernel.ZZ

        for start, end in self.bolo_chords:  # Reuse geometry for now
            # Ray marching
            num_samples = 100
            dl = np.linalg.norm(end - start) / num_samples

            integral_ne = 0.0

            # Refraction "walk-off" accumulator
            # Density gradients bend the beam, missing the detector slightly
            # Modeled as signal attenuation
            refraction_loss = 1.0

            for k in range(num_samples):
                alpha = k / num_samples
                r = (1 - alpha) * start[0] + alpha * end[0]
                z = (1 - alpha) * start[1] + alpha * end[1]

                ir = int((r - self.kernel.R[0]) / self.kernel.dR)
                iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)

                if 0 <= ir < self.kernel.NR - 1 and 0 <= iz < self.kernel.NZ - 1:
                    ne_val = density_profile_19[iz, ir] * 1e19
                    integral_ne += float(ne_val) * float(dl)

                    # Simple refraction heuristic: proportional to density
                    if ne_val > 1e20:
                        refraction_loss *= 0.995  # 0.5% loss per high-density step

            # Phase Shift (radians)
            phi = lambda_laser * r_e * integral_ne

            # Apply refraction loss (simulated fringe jump risk)
            phi_measured = phi * refraction_loss

            # Add Noise
            phi_measured += self._noise(0.1)  # 0.1 rad noise

            # Phase Wrapping
            phi_wrapped = phi_measured % (2 * np.pi)

            signals_phase.append(phi_wrapped)

        return np.asarray(signals_phase, dtype=np.float64)

    def measure_forward_channels(
        self,
        electron_density_m3: FloatArray,
        neutron_source_m3_s: FloatArray,
        *,
        detector_efficiency: float = 0.12,
        solid_angle_fraction: float = 1.0e-4,
        laser_wavelength_m: float = 1.064e-6,
    ) -> ForwardDiagnosticChannels:
        """Generate deterministic forward-model raw channels."""
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

    def visualize_setup(self) -> Figure:
        """Plot diagnostic geometry and return a Matplotlib figure."""
        fig, ax = plt.subplots()
        ax.set_title("Diagnostics Geometry")
        # Plasma contour
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, colors="gray", alpha=0.3)
        # Magnetic Probes
        ax.plot(self.wall_R, self.wall_Z, "ro", label="B-Probes")
        # Bolo Chords
        for i, (start, end) in enumerate(self.bolo_chords):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                "g-",
                alpha=0.5,
                label="Bolometer" if i == 0 else "",
            )

        ax.set_aspect("equal")
        ax.legend()
        return fig
