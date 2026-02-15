# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Forward Diagnostics Guard Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Input-validation guard tests for forward diagnostics module."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.diagnostics.forward import (
    generate_forward_channels,
    interferometer_phase_shift,
    neutron_count_rate,
    thomson_scattering_voltage,
)


def _make_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(4.0, 8.0, 33)
    z = np.linspace(-2.0, 2.0, 33)
    rr, zz = np.meshgrid(r, z)
    ne = 4.5e19 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.9)
    sn = 8.0e15 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.6)
    return r, z, ne, sn


def test_interferometer_rejects_grid_shape_mismatch() -> None:
    r, z, ne, _ = _make_fields()
    chords = [((4.2, 0.0), (7.8, 0.0))]
    with pytest.raises(ValueError, match="field shape"):
        interferometer_phase_shift(ne[:, :-1], r, z, chords)


def test_neutron_count_rate_rejects_invalid_scalar_inputs() -> None:
    _, _, _, sn = _make_fields()
    with pytest.raises(ValueError, match="volume_element_m3"):
        neutron_count_rate(sn, volume_element_m3=0.0)
    with pytest.raises(ValueError, match="detector_efficiency"):
        neutron_count_rate(sn, volume_element_m3=0.02, detector_efficiency=1.5)
    with pytest.raises(ValueError, match="solid_angle_fraction"):
        neutron_count_rate(sn, volume_element_m3=0.02, solid_angle_fraction=-0.1)


def test_generate_forward_channels_rejects_nonfinite_fields() -> None:
    r, z, ne, sn = _make_fields()
    ne[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        generate_forward_channels(
            electron_density_m3=ne,
            neutron_source_m3_s=sn,
            r_grid=r,
            z_grid=z,
            interferometer_chords=[((4.2, 0.0), (7.8, 0.0))],
            volume_element_m3=0.02,
        )


@pytest.mark.parametrize("axis_name", ["r", "z"])
def test_interferometer_rejects_non_monotonic_grids(axis_name: str) -> None:
    r, z, ne, _ = _make_fields()
    chords = [((4.2, 0.0), (7.8, 0.0))]
    if axis_name == "r":
        r_bad = np.array(r, copy=True)
        r_bad[[7, 8]] = r_bad[[8, 7]]
        with pytest.raises(ValueError, match="strictly increasing"):
            interferometer_phase_shift(ne, r_bad, z, chords)
    else:
        z_bad = np.array(z, copy=True)
        z_bad[[10, 11]] = z_bad[[11, 10]]
        with pytest.raises(ValueError, match="strictly increasing"):
            interferometer_phase_shift(ne, r, z_bad, chords)


def test_interferometer_rejects_duplicate_grid_points() -> None:
    r, z, ne, _ = _make_fields()
    chords = [((4.2, 0.0), (7.8, 0.0))]
    r_bad = np.array(r, copy=True)
    r_bad[12] = r_bad[11]
    with pytest.raises(ValueError, match="strictly increasing"):
        interferometer_phase_shift(ne, r_bad, z, chords)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples": 4}, "samples"),
        ({"laser_wavelength_m": 0.0}, "laser_wavelength_m"),
        ({"laser_wavelength_m": float("nan")}, "laser_wavelength_m"),
    ],
)
def test_interferometer_rejects_invalid_runtime_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    r, z, ne, _ = _make_fields()
    chords = [((4.2, 0.0), (7.8, 0.0))]
    params: dict[str, float | int] = {
        "laser_wavelength_m": 1.064e-6,
        "samples": 96,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        interferometer_phase_shift(ne, r, z, chords, **params)


@pytest.mark.parametrize(
    ("chords", "match"),
    [
        ([((4.2, 0.0),)], "2-point chord"),
        ([((4.2, 0.0, 1.0), (7.8, 0.0))], "2D"),
        ([((4.2, 0.0), (7.8, 0.0, 1.0))], "2D"),
    ],
)
def test_interferometer_rejects_malformed_chord_geometry(
    chords: object,
    match: str,
) -> None:
    r, z, ne, _ = _make_fields()
    with pytest.raises(ValueError, match=match):
        interferometer_phase_shift(ne, r, z, chords)  # type: ignore[arg-type]


def test_interferometer_enforce_domain_bounds_rejects_out_of_domain_chords() -> None:
    r, z, ne, _ = _make_fields()
    chords = [((3.0, 0.0), (9.0, 0.0))]
    with pytest.raises(ValueError, match="outside diagnostic domain"):
        interferometer_phase_shift(
            ne,
            r,
            z,
            chords,
            enforce_domain_bounds=True,
        )


def test_interferometer_default_allows_out_of_domain_chords_via_clamping() -> None:
    r, z, ne, _ = _make_fields()
    chords = [((3.0, 0.0), (9.0, 0.0))]
    phases = interferometer_phase_shift(ne, r, z, chords)
    assert phases.shape == (1,)
    assert np.isfinite(phases[0])
    assert phases[0] > 0.0


def test_generate_forward_channels_enforce_domain_bounds_rejects_out_of_domain_chords() -> None:
    r, z, ne, sn = _make_fields()
    with pytest.raises(ValueError, match="outside diagnostic domain"):
        generate_forward_channels(
            electron_density_m3=ne,
            neutron_source_m3_s=sn,
            r_grid=r,
            z_grid=z,
            interferometer_chords=[((3.0, 0.0), (9.0, 0.0))],
            volume_element_m3=0.02,
            enforce_chord_domain_bounds=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gain_v_per_m3": 0.0}, "gain_v_per_m3"),
        ({"temp_sensitivity_per_kev": -0.1}, "temp_sensitivity_per_kev"),
        ({"baseline_voltage_v": float("nan")}, "baseline_voltage_v"),
    ],
)
def test_thomson_rejects_invalid_scalar_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    r, z, ne, _ = _make_fields()
    rr, zz = np.meshgrid(r, z)
    te = 12.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.1)
    params = {
        "gain_v_per_m3": 2.5e-24,
        "temp_sensitivity_per_kev": 0.08,
        "baseline_voltage_v": 0.0,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        thomson_scattering_voltage(
            ne,
            te,
            r,
            z,
            [(6.0, 0.0)],
            **params,
        )


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([], "at least one point"),
        ([(6.0,)], "2D"),
        ([(6.0, 0.0), (float("nan"), 0.0)], "non-finite"),
    ],
)
def test_thomson_rejects_invalid_sampling_points(points: object, match: str) -> None:
    r, z, ne, _ = _make_fields()
    rr, zz = np.meshgrid(r, z)
    te = 12.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.1)
    with pytest.raises(ValueError, match=match):
        thomson_scattering_voltage(ne, te, r, z, points)  # type: ignore[arg-type]


def test_thomson_rejects_field_shape_mismatch() -> None:
    r, z, ne, _ = _make_fields()
    rr, zz = np.meshgrid(r, z)
    te = 12.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.1)
    with pytest.raises(ValueError, match="field shape"):
        thomson_scattering_voltage(ne[:, :-1], te, r, z, [(6.0, 0.0)])
    with pytest.raises(ValueError, match="field shape"):
        thomson_scattering_voltage(ne, te[:, :-1], r, z, [(6.0, 0.0)])


def test_generate_forward_channels_rejects_invalid_thomson_temp_map() -> None:
    r, z, ne, sn = _make_fields()
    with pytest.raises(ValueError, match="field shape"):
        generate_forward_channels(
            electron_density_m3=ne,
            electron_temp_keV=ne[:, :-1],
            neutron_source_m3_s=sn,
            r_grid=r,
            z_grid=z,
            interferometer_chords=[((4.2, 0.0), (7.8, 0.0))],
            volume_element_m3=0.02,
        )
