# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Forward Diagnostics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Forward-model diagnostics for raw-observable comparison lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ELECTRON_RADIUS_M = 2.8179403262e-15


@dataclass(frozen=True)
class ForwardDiagnosticChannels:
    interferometer_phase_rad: FloatArray
    neutron_count_rate_hz: float


def _nearest_index(axis: FloatArray, value: float) -> int:
    idx = int(np.argmin(np.abs(axis - float(value))))
    return int(np.clip(idx, 0, axis.size - 1))


def _line_integral_nearest(
    field: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    start: tuple[float, float],
    end: tuple[float, float],
    samples: int = 96,
) -> float:
    samples = max(int(samples), 8)
    r_vals = np.linspace(float(start[0]), float(end[0]), samples)
    z_vals = np.linspace(float(start[1]), float(end[1]), samples)
    dl = float(np.hypot(end[0] - start[0], end[1] - start[1]) / samples)

    accum = 0.0
    for r, z in zip(r_vals, z_vals):
        ir = _nearest_index(r_grid, float(r))
        iz = _nearest_index(z_grid, float(z))
        accum += float(field[iz, ir]) * dl
    return float(accum)


def interferometer_phase_shift(
    electron_density_m3: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    laser_wavelength_m: float = 1.064e-6,
    samples: int = 96,
) -> FloatArray:
    """Predict line-integrated interferometer phase shift [rad]."""
    wavelength = max(float(laser_wavelength_m), 1e-12)
    coeff = ELECTRON_RADIUS_M * wavelength
    phases = np.zeros(len(chords), dtype=np.float64)
    for i, (start, end) in enumerate(chords):
        line_ne = _line_integral_nearest(
            np.asarray(electron_density_m3, dtype=np.float64),
            np.asarray(r_grid, dtype=np.float64),
            np.asarray(z_grid, dtype=np.float64),
            start,
            end,
            samples=samples,
        )
        phases[i] = coeff * line_ne
    return phases


def neutron_count_rate(
    neutron_source_m3_s: FloatArray,
    *,
    volume_element_m3: float,
    detector_efficiency: float = 0.12,
    solid_angle_fraction: float = 1.0e-4,
) -> float:
    """Predict detector neutron count rate [Hz] from volumetric source."""
    source = np.asarray(neutron_source_m3_s, dtype=np.float64)
    vol = max(float(volume_element_m3), 1e-12)
    eff = float(np.clip(detector_efficiency, 0.0, 1.0))
    omega = max(float(solid_angle_fraction), 0.0)
    emitted_per_s = float(np.sum(np.clip(source, 0.0, None)) * vol)
    return float(emitted_per_s * eff * omega)


def generate_forward_channels(
    *,
    electron_density_m3: FloatArray,
    neutron_source_m3_s: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    interferometer_chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    volume_element_m3: float,
    detector_efficiency: float = 0.12,
    solid_angle_fraction: float = 1.0e-4,
    laser_wavelength_m: float = 1.064e-6,
) -> ForwardDiagnosticChannels:
    """Generate synthetic raw diagnostic channels from plasma state maps."""
    phases = interferometer_phase_shift(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        interferometer_chords,
        laser_wavelength_m=laser_wavelength_m,
    )
    rate = neutron_count_rate(
        np.asarray(neutron_source_m3_s, dtype=np.float64),
        volume_element_m3=volume_element_m3,
        detector_efficiency=detector_efficiency,
        solid_angle_fraction=solid_angle_fraction,
    )
    return ForwardDiagnosticChannels(
        interferometer_phase_rad=phases,
        neutron_count_rate_hz=rate,
    )
