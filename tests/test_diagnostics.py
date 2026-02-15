# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Forward Diagnostics Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for forward-model diagnostics channels."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.diagnostics.forward import (
    generate_forward_channels,
    interferometer_phase_shift,
    neutron_count_rate,
    thomson_scattering_voltage,
)
from scpn_fusion.diagnostics.synthetic_sensors import SensorSuite


class _KernelStub:
    def __init__(self) -> None:
        self.NR = 33
        self.NZ = 33
        self.R = np.linspace(4.0, 8.0, self.NR)
        self.Z = np.linspace(-2.0, 2.0, self.NZ)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.exp(-((self.RR - 6.0) ** 2 + self.ZZ**2))


def _make_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.linspace(4.0, 8.0, 33)
    z = np.linspace(-2.0, 2.0, 33)
    rr, zz = np.meshgrid(r, z)
    ne = 4.5e19 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.9)
    sn = 8.0e15 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 0.6)
    return r, z, ne, sn


def test_interferometer_phase_scales_with_density() -> None:
    r, z, ne, _ = _make_fields()
    chords = [((4.2, 0.0), (7.8, 0.0))]
    phase_lo = interferometer_phase_shift(ne * 0.5, r, z, chords)
    phase_hi = interferometer_phase_shift(ne * 1.2, r, z, chords)
    assert phase_hi.shape == (1,)
    assert float(phase_hi[0]) > float(phase_lo[0]) > 0.0


def test_neutron_count_rate_positive_and_linear() -> None:
    _, _, _, sn = _make_fields()
    a = neutron_count_rate(sn, volume_element_m3=0.02, detector_efficiency=0.1)
    b = neutron_count_rate(sn * 2.0, volume_element_m3=0.02, detector_efficiency=0.1)
    assert a > 0.0
    assert b == pytest.approx(2.0 * a, rel=1e-12, abs=0.0)


def test_generate_forward_channels_and_sensor_suite_bridge() -> None:
    r, z, ne, sn = _make_fields()
    rr, zz = np.meshgrid(r, z)
    te = 12.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.1)
    chords = [
        ((4.2, 0.0), (7.8, 0.0)),
        ((4.2, 0.8), (7.8, -0.8)),
    ]
    out = generate_forward_channels(
        electron_density_m3=ne,
        electron_temp_keV=te,
        neutron_source_m3_s=sn,
        r_grid=r,
        z_grid=z,
        interferometer_chords=chords,
        volume_element_m3=0.02,
    )
    assert out.interferometer_phase_rad.shape == (2,)
    assert np.all(out.interferometer_phase_rad > 0.0)
    assert out.neutron_count_rate_hz > 0.0
    assert out.thomson_scattering_voltage_v.shape == (3,)
    assert np.all(out.thomson_scattering_voltage_v > 0.0)

    suite = SensorSuite(_KernelStub())
    from_suite = suite.measure_forward_channels(ne, sn)
    assert from_suite.interferometer_phase_rad.shape[0] == len(suite.bolo_chords)
    assert from_suite.neutron_count_rate_hz > 0.0
    assert from_suite.thomson_scattering_voltage_v.shape == (3,)
    assert np.all(from_suite.thomson_scattering_voltage_v > 0.0)


def test_thomson_scattering_voltage_scales_with_density_and_temperature() -> None:
    r, z, ne, _ = _make_fields()
    rr, zz = np.meshgrid(r, z)
    te = 10.0 * np.exp(-((rr - 6.0) ** 2 + zz**2) / 1.2)
    points = [(5.0, 0.0), (6.0, 0.0), (7.0, 0.0)]

    v_lo = thomson_scattering_voltage(
        ne * 0.8,
        te,
        r,
        z,
        points,
    )
    v_hi = thomson_scattering_voltage(
        ne * 1.2,
        te * 1.3,
        r,
        z,
        points,
    )
    assert v_lo.shape == (3,)
    assert np.all(v_lo > 0.0)
    assert np.all(v_hi > v_lo)
