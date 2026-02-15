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
from typing import Sequence

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


def _validate_field_grid(
    field: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    *,
    name: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    arr = np.asarray(field, dtype=np.float64)
    r = np.asarray(r_grid, dtype=np.float64).reshape(-1)
    z = np.asarray(z_grid, dtype=np.float64).reshape(-1)

    if r.ndim != 1 or z.ndim != 1 or r.size < 2 or z.size < 2:
        raise ValueError(f"{name}: r_grid and z_grid must be 1D with at least 2 points.")
    if arr.ndim != 2 or arr.shape != (z.size, r.size):
        raise ValueError(
            f"{name}: field shape {arr.shape} must match (len(z_grid), len(r_grid)) "
            f"= ({z.size}, {r.size})."
        )
    if not (np.all(np.isfinite(arr)) and np.all(np.isfinite(r)) and np.all(np.isfinite(z))):
        raise ValueError(f"{name}: field and grids must be finite.")
    if not (np.all(np.diff(r) > 0.0) and np.all(np.diff(z) > 0.0)):
        raise ValueError(f"{name}: r_grid and z_grid must be strictly increasing.")
    return arr, r, z


def _validate_chords(
    chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> Sequence[tuple[tuple[float, float], tuple[float, float]]]:
    for i, chord in enumerate(chords):
        if len(chord) != 2:
            raise ValueError(f"chords[{i}] must be a 2-point chord.")
        start, end = chord
        if len(start) != 2 or len(end) != 2:
            raise ValueError(f"chords[{i}] points must be 2D (r, z) tuples.")
        vals = [start[0], start[1], end[0], end[1]]
        if not np.all(np.isfinite(np.asarray(vals, dtype=np.float64))):
            raise ValueError(f"chords[{i}] has non-finite coordinates.")
    return chords


def _line_integral_nearest(
    field: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    start: tuple[float, float],
    end: tuple[float, float],
    samples: int = 96,
) -> float:
    samples = int(samples)
    if samples < 8:
        raise ValueError("samples must be >= 8.")
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
    wavelength = float(laser_wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("laser_wavelength_m must be finite and > 0.")
    coeff = ELECTRON_RADIUS_M * wavelength
    ne, r, z = _validate_field_grid(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="interferometer_phase_shift",
    )
    _validate_chords(chords)
    phases = np.zeros(len(chords), dtype=np.float64)
    for i, (start, end) in enumerate(chords):
        line_ne = _line_integral_nearest(
            ne,
            r,
            z,
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
    if source.size == 0 or source.ndim != 2:
        raise ValueError("neutron_source_m3_s must be a non-empty 2D array.")
    if not np.all(np.isfinite(source)):
        raise ValueError("neutron_source_m3_s must be finite.")
    vol = float(volume_element_m3)
    eff = float(detector_efficiency)
    omega = float(solid_angle_fraction)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("volume_element_m3 must be finite and > 0.")
    if not np.isfinite(eff) or eff < 0.0 or eff > 1.0:
        raise ValueError("detector_efficiency must be finite and within [0, 1].")
    if not np.isfinite(omega) or omega < 0.0:
        raise ValueError("solid_angle_fraction must be finite and >= 0.")
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
