# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PyO3 Nuclear Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for PyO3 bindings: fusion-diagnostics + fusion-nuclear → Python.

Covers: Tomography (WP-PY4), Breeding Blanket (WP-PY5).
"""

import numpy as np
import pytest

try:
    import scpn_fusion_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_fusion_rs not compiled")


# ── WP-PY4: Tomography ──────────────────────────────────────────────


class TestPyTomography:
    """Tests for PlasmaTomography binding (fusion-diagnostics/tomography.rs)."""

    @pytest.fixture
    def tomo(self):
        """Create a simple 20-chord tomography setup."""
        chords = []
        for i in range(20):
            angle = np.pi * i / 20
            start = (1.0 + 4.0 * np.cos(angle), 5.0 * np.sin(angle))
            end = (9.0 - 4.0 * np.cos(angle), -5.0 * np.sin(angle))
            chords.append((start, end))
        r_range = (1.0, 9.0)
        z_range = (-5.0, 5.0)
        return scpn_fusion_rs.PyTomography(chords, r_range, z_range, 32)

    def test_reconstruct_shape(self, tomo):
        signals = [1.0] * 20
        result = tomo.reconstruct(signals)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32)

    def test_reconstruct_non_negative(self, tomo):
        signals = [abs(np.sin(i * 0.3)) for i in range(20)]
        result = tomo.reconstruct(signals)
        # Tomographic reconstruction should be approximately non-negative
        assert np.mean(result >= -0.1) > 0.9  # 90% of pixels near-non-negative

    def test_zero_signals_zero_reconstruction(self, tomo):
        signals = [0.0] * 20
        result = tomo.reconstruct(signals)
        assert np.allclose(result, 0.0, atol=1e-10)


# ── WP-PY5: Breeding Blanket ────────────────────────────────────────


class TestPyBreedingBlanket:
    """Tests for BreedingBlanket binding (fusion-nuclear/neutronics.rs)."""

    def test_construction_default(self):
        blanket = scpn_fusion_rs.PyBreedingBlanket()
        assert blanket is not None

    def test_construction_custom(self):
        blanket = scpn_fusion_rs.PyBreedingBlanket(100.0, 0.9)
        assert blanket is not None

    def test_solve_transport(self):
        blanket = scpn_fusion_rs.PyBreedingBlanket(80.0, 0.6)
        tbr, heat, attenuation, tritium_rate = blanket.solve_transport(1e14)
        assert isinstance(tbr, float) and np.isfinite(tbr)
        assert isinstance(heat, float) and heat >= 0
        assert 0.0 < attenuation <= 1.0
        assert tritium_rate >= 0

    def test_tbr_range(self):
        """Standard blanket should have TBR in physically reasonable range."""
        blanket = scpn_fusion_rs.PyBreedingBlanket(80.0, 0.6)
        tbr, _, _, _ = blanket.solve_transport(1e14)
        assert 0.5 <= tbr <= 2.0  # physically plausible range

    def test_thicker_blanket_higher_tbr(self):
        thin = scpn_fusion_rs.PyBreedingBlanket(40.0, 0.6)
        thick = scpn_fusion_rs.PyBreedingBlanket(120.0, 0.6)
        tbr_thin, _, _, _ = thin.solve_transport(1e14)
        tbr_thick, _, _, _ = thick.solve_transport(1e14)
        assert tbr_thick > tbr_thin

    def test_higher_enrichment_higher_tbr(self):
        low = scpn_fusion_rs.PyBreedingBlanket(80.0, 0.3)
        high = scpn_fusion_rs.PyBreedingBlanket(80.0, 0.9)
        tbr_low, _, _, _ = low.solve_transport(1e14)
        tbr_high, _, _, _ = high.solve_transport(1e14)
        assert tbr_high > tbr_low
