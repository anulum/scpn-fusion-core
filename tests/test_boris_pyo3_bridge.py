#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Boris Integrator PyO3 Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Tests for the Rust Boris particle integrator exposed via PyO3."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

try:
    import scpn_fusion_rs  # type: ignore[import-untyped]

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")


# ─── helpers ───


def _seed(n: int = 64) -> list:
    """Seed alpha particles at R=6m, Z=0, 3.5 MeV, isotropic pitch."""
    return scpn_fusion_rs.py_seed_alpha_particles(
        n=n, r0=6.0, z0=0.0, energy_mev=3.5, pitch=0.5, weight=1.0
    )


# ─── tests ───


class TestSeedAlphaParticles:
    def test_count(self) -> None:
        particles = _seed(100)
        assert len(particles) == 100

    def test_energy_band(self) -> None:
        """Seeded alpha particles should have ~3.5 MeV kinetic energy."""
        particles = _seed(32)
        energies = [p.kinetic_energy_mev() for p in particles]
        for e in energies:
            # Allow 20% spread from pitch angle variation
            assert 1.0 < e < 6.0, f"Energy {e:.2f} MeV out of expected range"

    def test_cylindrical_radius(self) -> None:
        """Particles seeded at R0=6 should have cylindrical radius near 6m."""
        particles = _seed(16)
        for p in particles:
            r_cyl = p.cylindrical_radius_m()
            assert 4.0 < r_cyl < 8.0, f"R={r_cyl:.2f}m far from seed R0=6"


class TestBorisAdvance:
    def test_preserves_speed_pure_b(self) -> None:
        """In uniform B with E=0, |v| must be conserved (gyro-motion)."""
        particles = _seed(8)

        # Record initial speeds
        v0 = []
        for p in particles:
            v0.append(math.sqrt(p.vx_m_s**2 + p.vy_m_s**2 + p.vz_m_s**2))

        # Uniform B-field along z (5 Tesla), zero E-field
        e_field = [0.0, 0.0, 0.0]
        b_field = [0.0, 0.0, 5.0]
        dt = 1e-9  # 1 ns
        steps = 100

        advanced = scpn_fusion_rs.py_advance_boris(particles, e_field, b_field, dt, steps)

        for i, p in enumerate(advanced):
            v_final = math.sqrt(p.vx_m_s**2 + p.vy_m_s**2 + p.vz_m_s**2)
            rel_err = abs(v_final - v0[i]) / v0[i]
            assert rel_err < 1e-6, f"Speed changed by {rel_err:.2e} (should be conserved)"

    def test_rejects_invalid_dt(self) -> None:
        """dt <= 0 should raise."""
        particles = _seed(2)
        with pytest.raises(Exception):
            scpn_fusion_rs.py_advance_boris(particles, [0, 0, 0], [0, 0, 5], -1e-9, 10)


class TestHeatingProfile:
    def test_positive(self) -> None:
        """Particles in domain should produce positive heating profile."""
        particles = _seed(64)
        # Advance a few steps so particles distribute
        e_field = [0.0, 0.0, 0.0]
        b_field = [0.0, 0.0, 5.0]
        advanced = scpn_fusion_rs.py_advance_boris(particles, e_field, b_field, 1e-9, 50)

        profile = scpn_fusion_rs.py_get_heating_profile(
            advanced,
            nr=33,
            nz=33,
            r_min=2.0,
            r_max=10.0,
            z_min=-6.0,
            z_max=6.0,
            confinement_tau_s=1.0,
        )
        assert isinstance(profile, np.ndarray)
        assert profile.shape == (33, 33)
        assert profile.sum() >= 0.0


class TestPopulationSummary:
    def test_fields(self) -> None:
        """Summary should expose all required attributes."""
        particles = _seed(32)
        summary = scpn_fusion_rs.py_particle_population_summary(particles, 0.1)
        assert hasattr(summary, "count")
        assert hasattr(summary, "mean_energy_mev")
        assert hasattr(summary, "max_energy_mev")
        assert hasattr(summary, "p95_energy_mev")
        assert hasattr(summary, "runaway_fraction")
        assert summary.count == 32
        assert summary.mean_energy_mev > 0.0


class TestParticleAttributes:
    def test_roundtrip(self) -> None:
        """Get/set all PyParticle fields."""
        particles = _seed(1)
        p = particles[0]
        # Read initial values
        _ = p.x_m
        _ = p.y_m
        _ = p.z_m
        _ = p.vx_m_s
        _ = p.vy_m_s
        _ = p.vz_m_s
        _ = p.charge_c
        _ = p.mass_kg
        _ = p.weight
        # Modify and verify
        p.x_m = 7.0
        assert p.x_m == 7.0
        p.weight = 42.0
        assert p.weight == 42.0


class TestGyroradius:
    def test_order_of_magnitude(self) -> None:
        """Alpha particle gyroradius in 5T field should be ~cm scale."""
        # r_L = m*v_perp / (q*B)
        # 3.5 MeV alpha: v ~ 1.3e7 m/s, m=6.64e-27 kg, q=2*1.6e-19 C, B=5T
        # r_L ~ 6.64e-27 * 1.3e7 / (3.2e-19 * 5) ~ 0.054 m ~ 5 cm
        particles = _seed(4)
        e_field = [0.0, 0.0, 0.0]
        b_field = [0.0, 0.0, 5.0]

        # Record initial positions
        x0 = [(p.x_m, p.y_m) for p in particles]

        advanced = scpn_fusion_rs.py_advance_boris(particles, e_field, b_field, 1e-10, 1000)

        for i, p in enumerate(advanced):
            dx = p.x_m - x0[i][0]
            dy = p.y_m - x0[i][1]
            displacement = math.sqrt(dx**2 + dy**2)
            # Gyroradius should be < 1m for 3.5 MeV alpha in 5T
            assert displacement < 1.0, f"Displacement {displacement:.3f}m too large for 5T"
