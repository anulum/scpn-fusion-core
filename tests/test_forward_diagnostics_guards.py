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
