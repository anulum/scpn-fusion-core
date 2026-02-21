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
    thomson_scattering_voltage_v: FloatArray
    ece_temperature_kev: FloatArray | None = None
    sxr_brightness_w_m2: FloatArray | None = None
    bolometer_power_w_m2_sr: FloatArray | None = None
    cxrs_ti_kev: FloatArray | None = None
    cxrs_rotation_km_s: FloatArray | None = None


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
    *,
    r_min: float | None = None,
    r_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    enforce_domain_bounds: bool = False,
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
        if enforce_domain_bounds:
            assert r_min is not None and r_max is not None
            assert z_min is not None and z_max is not None
            if (
                float(start[0]) < float(r_min)
                or float(start[0]) > float(r_max)
                or float(end[0]) < float(r_min)
                or float(end[0]) > float(r_max)
                or float(start[1]) < float(z_min)
                or float(start[1]) > float(z_max)
                or float(end[1]) < float(z_min)
                or float(end[1]) > float(z_max)
            ):
                raise ValueError(
                    f"chords[{i}] lies outside diagnostic domain bounds."
                )
    return chords


def _validate_points(
    points: Sequence[tuple[float, float]],
    *,
    r_min: float | None = None,
    r_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    enforce_domain_bounds: bool = False,
) -> Sequence[tuple[float, float]]:
    if len(points) == 0:
        raise ValueError("sample_points must contain at least one point.")
    for i, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"sample_points[{i}] must be a 2D (r, z) tuple.")
        r_point = float(point[0])
        z_point = float(point[1])
        if not np.isfinite(r_point) or not np.isfinite(z_point):
            raise ValueError(f"sample_points[{i}] has non-finite coordinates.")
        if enforce_domain_bounds:
            assert r_min is not None and r_max is not None
            assert z_min is not None and z_max is not None
            if (
                r_point < float(r_min)
                or r_point > float(r_max)
                or z_point < float(z_min)
                or z_point > float(z_max)
            ):
                raise ValueError(
                    f"sample_points[{i}] lies outside diagnostic domain bounds."
                )
    return points


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
    enforce_domain_bounds: bool = False,
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
    _validate_chords(
        chords,
        r_min=float(np.min(r)),
        r_max=float(np.max(r)),
        z_min=float(np.min(z)),
        z_max=float(np.max(z)),
        enforce_domain_bounds=bool(enforce_domain_bounds),
    )
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


def thomson_scattering_voltage(
    electron_density_m3: FloatArray,
    electron_temp_keV: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    sample_points: Sequence[tuple[float, float]],
    *,
    gain_v_per_m3: float = 2.5e-24,
    temp_sensitivity_per_kev: float = 0.08,
    baseline_voltage_v: float = 0.0,
    enforce_domain_bounds: bool = False,
) -> FloatArray:
    """Predict Thomson-scattering detector voltage [V] at sample points."""
    gain = float(gain_v_per_m3)
    temp_sensitivity = float(temp_sensitivity_per_kev)
    baseline = float(baseline_voltage_v)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain_v_per_m3 must be finite and > 0.")
    if not np.isfinite(temp_sensitivity) or temp_sensitivity < 0.0:
        raise ValueError("temp_sensitivity_per_kev must be finite and >= 0.")
    if not np.isfinite(baseline):
        raise ValueError("baseline_voltage_v must be finite.")

    ne, r, z = _validate_field_grid(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="thomson_scattering_voltage.electron_density_m3",
    )
    te, _, _ = _validate_field_grid(
        np.asarray(electron_temp_keV, dtype=np.float64),
        r,
        z,
        name="thomson_scattering_voltage.electron_temp_keV",
    )
    _validate_points(
        sample_points,
        r_min=float(np.min(r)),
        r_max=float(np.max(r)),
        z_min=float(np.min(z)),
        z_max=float(np.max(z)),
        enforce_domain_bounds=bool(enforce_domain_bounds),
    )

    out = np.zeros(len(sample_points), dtype=np.float64)
    for i, point in enumerate(sample_points):
        ir = _nearest_index(r, float(point[0]))
        iz = _nearest_index(z, float(point[1]))
        ne_local = max(float(ne[iz, ir]), 0.0)
        te_local = max(float(te[iz, ir]), 0.0)
        out[i] = gain * ne_local * (1.0 + temp_sensitivity * te_local) + baseline
    if not np.all(np.isfinite(out)):
        raise ValueError("thomson_scattering_voltage produced non-finite values.")
    return out


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
    enforce_chord_domain_bounds: bool = False,
    electron_temp_keV: FloatArray | None = None,
    thomson_sample_points: Sequence[tuple[float, float]] | None = None,
    thomson_gain_v_per_m3: float = 2.5e-24,
    thomson_temp_sensitivity_per_kev: float = 0.08,
    thomson_baseline_voltage_v: float = 0.0,
    enforce_thomson_domain_bounds: bool = False,
) -> ForwardDiagnosticChannels:
    """Generate synthetic raw diagnostic channels from plasma state maps."""
    ne = np.asarray(electron_density_m3, dtype=np.float64)
    r = np.asarray(r_grid, dtype=np.float64)
    z = np.asarray(z_grid, dtype=np.float64)
    temp_map = (
        np.asarray(electron_temp_keV, dtype=np.float64)
        if electron_temp_keV is not None
        else np.clip(ne / 1.0e19, 0.0, None)
    )
    if thomson_sample_points is None:
        r_min = float(np.min(r))
        r_max = float(np.max(r))
        z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
        span = r_max - r_min
        thomson_sample_points = (
            (r_min + 0.25 * span, z_mid),
            (r_min + 0.50 * span, z_mid),
            (r_min + 0.75 * span, z_mid),
        )
    phases = interferometer_phase_shift(
        ne,
        r,
        z,
        interferometer_chords,
        laser_wavelength_m=laser_wavelength_m,
        enforce_domain_bounds=bool(enforce_chord_domain_bounds),
    )
    rate = neutron_count_rate(
        np.asarray(neutron_source_m3_s, dtype=np.float64),
        volume_element_m3=volume_element_m3,
        detector_efficiency=detector_efficiency,
        solid_angle_fraction=solid_angle_fraction,
    )
    thomson = thomson_scattering_voltage(
        ne,
        temp_map,
        r,
        z,
        thomson_sample_points,
        gain_v_per_m3=thomson_gain_v_per_m3,
        temp_sensitivity_per_kev=thomson_temp_sensitivity_per_kev,
        baseline_voltage_v=thomson_baseline_voltage_v,
        enforce_domain_bounds=bool(enforce_thomson_domain_bounds),
    )
    return ForwardDiagnosticChannels(
        interferometer_phase_rad=phases,
        neutron_count_rate_hz=rate,
        thomson_scattering_voltage_v=thomson,
    )


# ── Extended diagnostics: ECE, SXR, Bolometry ────────────────────────


def ece_radiometer_temperature(
    electron_temp_keV: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    channel_r_positions: Sequence[float],
    *,
    z_observation: float = 0.0,
    optical_depth_factor: float = 1.0,
) -> FloatArray:
    """Predict ECE radiometer Te [keV] at specified R channel positions.

    Assumes optically thick plasma: signal = Te(R_ch, z_obs) * optical_depth_factor.
    """
    if not channel_r_positions:
        raise ValueError("channel_r_positions must be non-empty.")
    odf = float(optical_depth_factor)
    if not np.isfinite(odf) or odf <= 0.0:
        raise ValueError("optical_depth_factor must be finite and > 0.")
    z_obs = float(z_observation)
    if not np.isfinite(z_obs):
        raise ValueError("z_observation must be finite.")

    te, r, z = _validate_field_grid(
        np.asarray(electron_temp_keV, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="ece_radiometer_temperature",
    )

    iz = _nearest_index(z, z_obs)
    out = np.zeros(len(channel_r_positions), dtype=np.float64)
    for i, r_ch in enumerate(channel_r_positions):
        r_val = float(r_ch)
        if not np.isfinite(r_val):
            raise ValueError(f"channel_r_positions[{i}] must be finite.")
        ir = _nearest_index(r, r_val)
        out[i] = max(float(te[iz, ir]), 0.0) * odf
    return out


def soft_xray_brightness(
    electron_density_m3: FloatArray,
    electron_temp_keV: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    z_eff: float = 1.5,
    filter_energy_kev: float = 1.0,
    samples: int = 96,
    enforce_domain_bounds: bool = False,
) -> FloatArray:
    """Predict soft X-ray line-integrated brightness [W/m^2].

    Emissivity model: epsilon = ne^2 * sqrt(Te) * Z_eff * exp(-E_filter / Te).
    """
    z_eff_val = float(z_eff)
    e_filter = float(filter_energy_kev)
    if not np.isfinite(z_eff_val) or z_eff_val < 1.0:
        raise ValueError("z_eff must be finite and >= 1.0.")
    if not np.isfinite(e_filter) or e_filter <= 0.0:
        raise ValueError("filter_energy_kev must be finite and > 0.")

    ne, r, z = _validate_field_grid(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="soft_xray_brightness.ne",
    )
    te, _, _ = _validate_field_grid(
        np.asarray(electron_temp_keV, dtype=np.float64),
        r,
        z,
        name="soft_xray_brightness.te",
    )
    _validate_chords(
        chords,
        r_min=float(np.min(r)),
        r_max=float(np.max(r)),
        z_min=float(np.min(z)),
        z_max=float(np.max(z)),
        enforce_domain_bounds=bool(enforce_domain_bounds),
    )

    # Compute 2D emissivity field
    te_safe = np.clip(te, 0.01, None)  # avoid log(0) / div-by-zero
    emissivity = ne**2 * np.sqrt(te_safe) * z_eff_val * np.exp(-e_filter / te_safe)

    out = np.zeros(len(chords), dtype=np.float64)
    for i, (start, end) in enumerate(chords):
        out[i] = _line_integral_nearest(emissivity, r, z, start, end, samples=samples)
    return out


def bolometer_power_density(
    electron_density_m3: FloatArray,
    electron_temp_keV: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    z_eff: float = 1.5,
    impurity_fraction: float = 0.02,
    samples: int = 96,
    enforce_domain_bounds: bool = False,
) -> FloatArray:
    """Predict bolometer line-integrated radiated power [W/m^2/sr].

    Radiation model: P_rad = ne^2 * L_z(Te) where L_z = C_rad * Z_eff^2 * sqrt(Te).
    C_rad ~ 1e-31 W m^3 (coronal equilibrium approximation).
    """
    z_eff_val = float(z_eff)
    imp_frac = float(impurity_fraction)
    if not np.isfinite(z_eff_val) or z_eff_val < 1.0:
        raise ValueError("z_eff must be finite and >= 1.0.")
    if not np.isfinite(imp_frac) or imp_frac < 0.0:
        raise ValueError("impurity_fraction must be finite and >= 0.")

    ne, r, z = _validate_field_grid(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="bolometer_power_density.ne",
    )
    te, _, _ = _validate_field_grid(
        np.asarray(electron_temp_keV, dtype=np.float64),
        r,
        z,
        name="bolometer_power_density.te",
    )
    _validate_chords(
        chords,
        r_min=float(np.min(r)),
        r_max=float(np.max(r)),
        z_min=float(np.min(z)),
        z_max=float(np.max(z)),
        enforce_domain_bounds=bool(enforce_domain_bounds),
    )

    C_rad = 1.0e-31  # W m^3 (coronal equilibrium)
    te_safe = np.clip(te, 0.01, None)
    # P_rad = ne^2 * C_rad * Z_eff^2 * sqrt(Te) * (1 + impurity_fraction)
    p_rad = ne**2 * C_rad * z_eff_val**2 * np.sqrt(te_safe) * (1.0 + imp_frac)

    out = np.zeros(len(chords), dtype=np.float64)
    for i, (start, end) in enumerate(chords):
        out[i] = _line_integral_nearest(p_rad, r, z, start, end, samples=samples)
    return out


def cxrs_ion_diagnostics(
    ion_temp_keV: FloatArray,
    rotation_km_s: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    chords: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    beam_r_center: float = 6.2,
    beam_width: float = 0.1,
    samples: int = 96,
) -> tuple[FloatArray, FloatArray]:
    """Predict CXRS ion temperature and toroidal rotation.
    
    Weights signals by a Gaussian beam-emission profile centered at beam_r_center.
    """
    ti, r, z = _validate_field_grid(
        np.asarray(ion_temp_keV, dtype=np.float64),
        np.asarray(r_grid, dtype=np.float64),
        np.asarray(z_grid, dtype=np.float64),
        name="cxrs.ti",
    )
    vphi, _, _ = _validate_field_grid(
        np.asarray(rotation_km_s, dtype=np.float64),
        r,
        z,
        name="cxrs.vphi",
    )
    
    # Emission weight: exp(-(R - R_beam)^2 / w^2)
    rr_mesh, _ = np.meshgrid(r, z)
    weight_map = np.exp(-((rr_mesh - beam_r_center)**2) / (beam_width**2))
    
    ti_out = np.zeros(len(chords), dtype=np.float64)
    vphi_out = np.zeros(len(chords), dtype=np.float64)
    
    for i, (start, end) in enumerate(chords):
        # Weighted integrals
        sum_w = _line_integral_nearest(weight_map, r, z, start, end, samples=samples)
        if sum_w > 1e-9:
            ti_out[i] = _line_integral_nearest(ti * weight_map, r, z, start, end, samples=samples) / sum_w
            vphi_out[i] = _line_integral_nearest(vphi * weight_map, r, z, start, end, samples=samples) / sum_w
            
    return ti_out, vphi_out
