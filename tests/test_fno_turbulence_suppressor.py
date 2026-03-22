# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — FNO Turbulence Suppressor Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.fno_turbulence_suppressor import SpectralTurbulenceGenerator


class TestSpectralTurbulenceGeneratorInit:
    def test_default(self):
        gen = SpectralTurbulenceGenerator(size=16, seed=42)
        assert gen.size == 16
        assert gen.field.shape == (16, 16)
        assert gen.zonal_flow == 0.0

    def test_seed_rng_exclusive(self):
        with pytest.raises(ValueError, match="seed or rng"):
            SpectralTurbulenceGenerator(seed=1, rng=np.random.default_rng(1))


class TestStep:
    def test_returns_field(self):
        gen = SpectralTurbulenceGenerator(size=16, seed=42)
        field = gen.step(dt=0.01)
        assert field.shape == (16, 16)
        assert np.all(np.isfinite(field))

    def test_zonal_flow_grows(self):
        gen = SpectralTurbulenceGenerator(size=16, seed=42)
        for _ in range(20):
            gen.step(dt=0.01)
        assert gen.zonal_flow >= 0.0

    def test_damping_reduces_energy(self):
        gen1 = SpectralTurbulenceGenerator(size=16, seed=42)
        gen2 = SpectralTurbulenceGenerator(size=16, seed=42)
        for _ in range(20):
            gen1.step(dt=0.01, damping=0.0)
            gen2.step(dt=0.01, damping=0.9)
        e1 = np.mean(gen1.field**2)
        e2 = np.mean(gen2.field**2)
        assert e2 <= e1

    def test_deterministic_with_seed(self):
        g1 = SpectralTurbulenceGenerator(size=16, seed=42)
        g2 = SpectralTurbulenceGenerator(size=16, seed=42)
        for _ in range(10):
            f1 = g1.step(dt=0.01)
            f2 = g2.step(dt=0.01)
        np.testing.assert_array_equal(f1, f2)

    def test_multiple_steps_stable(self):
        gen = SpectralTurbulenceGenerator(size=16, seed=42)
        for _ in range(100):
            field = gen.step(dt=0.01)
        assert np.all(np.isfinite(field))
