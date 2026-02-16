# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Extended Forward Diagnostics Tests (ECE, SXR, Bolo)
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.diagnostics.forward import (
    ForwardDiagnosticChannels,
    bolometer_power_density,
    ece_radiometer_temperature,
    soft_xray_brightness,
)


def _make_grid(nr: int = 33, nz: int = 33):
    """Create test grids and plasma fields."""
    r = np.linspace(1.0, 3.0, nr)
    z = np.linspace(-1.5, 1.5, nz)
    R, Z = np.meshgrid(r, z, indexing="ij")
    # Peaked Gaussian profiles centred at (2.0, 0.0)
    te = 5.0 * np.exp(-((R - 2.0) ** 2 + Z**2) / 0.5)
    ne = 3e19 * np.exp(-((R - 2.0) ** 2 + Z**2) / 0.6)
    # Transpose to (nz, nr) convention used by forward.py
    return r, z, te.T, ne.T


def _midplane_chords(r, z, n=3):
    """Create horizontal midplane chords."""
    r_min, r_max = float(r[0]), float(r[-1])
    z_mid = 0.0
    span = (r_max - r_min) / (n + 1)
    return [
        ((r_min + span * (i + 0.5), z_mid), (r_min + span * (i + 1.5), z_mid))
        for i in range(n)
    ]


# ── ECE Radiometer Tests ────────────────────────────────────────────


def test_ece_scales_with_temperature() -> None:
    r, z, te, ne = _make_grid()
    channels = [1.5, 2.0, 2.5]
    result = ece_radiometer_temperature(te, r, z, channels)
    # Centre (R=2.0) should be hottest
    assert result[1] > result[0]
    assert result[1] > result[2]
    assert np.all(result > 0.0)


def test_ece_channel_positions_correct() -> None:
    r, z, te, _ = _make_grid()
    # Single channel at R=2.0 (peak), z=0 (midplane)
    result = ece_radiometer_temperature(te, r, z, [2.0], z_observation=0.0)
    # Should be close to peak Te = 5.0
    assert result[0] > 4.0


def test_ece_optical_depth_reduces_signal() -> None:
    r, z, te, _ = _make_grid()
    channels = [2.0]
    full = ece_radiometer_temperature(te, r, z, channels, optical_depth_factor=1.0)
    reduced = ece_radiometer_temperature(te, r, z, channels, optical_depth_factor=0.5)
    assert reduced[0] < full[0]
    np.testing.assert_allclose(reduced[0], full[0] * 0.5, atol=1e-10)


def test_ece_rejects_invalid_inputs() -> None:
    r, z, te, _ = _make_grid()
    with pytest.raises(ValueError, match="non-empty"):
        ece_radiometer_temperature(te, r, z, [])
    with pytest.raises(ValueError, match="optical_depth_factor"):
        ece_radiometer_temperature(te, r, z, [2.0], optical_depth_factor=-1.0)


# ── Soft X-Ray Tests ────────────────────────────────────────────────


def test_sxr_positive_for_hot_dense_plasma() -> None:
    r, z, te, ne = _make_grid()
    chords = _midplane_chords(r, z)
    result = soft_xray_brightness(ne, te, r, z, chords)
    assert np.all(result > 0.0)


def test_sxr_scales_with_density_squared() -> None:
    r, z, te, ne = _make_grid()
    chords = _midplane_chords(r, z, n=1)
    result_1x = soft_xray_brightness(ne, te, r, z, chords)
    result_2x = soft_xray_brightness(2.0 * ne, te, r, z, chords)
    # Emissivity ~ ne^2, so doubling ne should ~4x the signal
    ratio = float(result_2x[0] / result_1x[0])
    assert 3.5 < ratio < 4.5


def test_sxr_filter_suppresses_cold_edge() -> None:
    r, z, te, ne = _make_grid()
    # Use chords that sample the cold edge
    cold_chord = [((float(r[0]), 0.0), (float(r[2]), 0.0))]
    hot_chord = [((1.8, 0.0), (2.2, 0.0))]
    cold = soft_xray_brightness(ne, te, r, z, cold_chord, filter_energy_kev=2.0)
    hot = soft_xray_brightness(ne, te, r, z, hot_chord, filter_energy_kev=2.0)
    # Hot region should have much stronger SXR signal
    assert hot[0] > cold[0]


def test_sxr_rejects_invalid_chords() -> None:
    r, z, te, ne = _make_grid()
    nan_chord = [((float("nan"), 0.0), (2.0, 0.0))]
    with pytest.raises(ValueError, match="non-finite"):
        soft_xray_brightness(ne, te, r, z, nan_chord)


# ── Bolometer Tests ──────────────────────────────────────────────────


def test_bolo_positive_for_radiating_plasma() -> None:
    r, z, te, ne = _make_grid()
    chords = _midplane_chords(r, z)
    result = bolometer_power_density(ne, te, r, z, chords)
    assert np.all(result > 0.0)


def test_bolo_scales_with_impurity() -> None:
    r, z, te, ne = _make_grid()
    chords = _midplane_chords(r, z, n=1)
    result_low = bolometer_power_density(ne, te, r, z, chords, impurity_fraction=0.01)
    result_high = bolometer_power_density(ne, te, r, z, chords, impurity_fraction=0.10)
    assert result_high[0] > result_low[0]


def test_bolo_rejects_negative_zeff() -> None:
    r, z, te, ne = _make_grid()
    chords = _midplane_chords(r, z)
    with pytest.raises(ValueError, match="z_eff"):
        bolometer_power_density(ne, te, r, z, chords, z_eff=0.5)


def test_forward_channels_optional_fields_default_none() -> None:
    """Verify backward compat: new optional fields default to None."""
    phases = np.array([0.1])
    thomson = np.array([0.5])
    ch = ForwardDiagnosticChannels(
        interferometer_phase_rad=phases,
        neutron_count_rate_hz=100.0,
        thomson_scattering_voltage_v=thomson,
    )
    assert ch.ece_temperature_kev is None
    assert ch.sxr_brightness_w_m2 is None
    assert ch.bolometer_power_w_m2_sr is None


def test_sxr_bolo_domain_bounds() -> None:
    r, z, te, ne = _make_grid()
    out_of_bounds_chord = [((0.0, 0.0), (2.0, 0.0))]
    # Without enforce_domain_bounds, should work (nearest-index clipping)
    result = soft_xray_brightness(ne, te, r, z, out_of_bounds_chord)
    assert result.size == 1
    # With enforce_domain_bounds, should raise
    with pytest.raises(ValueError, match="outside"):
        soft_xray_brightness(
            ne, te, r, z, out_of_bounds_chord, enforce_domain_bounds=True
        )
