# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Synthetic Sensors Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for synthetic sensor noise/control paths."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_sensor_suite_seeded_noise_is_deterministic() -> None:
    k1 = _KernelStub()
    k2 = _KernelStub()
    s1 = SensorSuite(k1, seed=123)
    s2 = SensorSuite(k2, seed=123)

    m1 = s1.measure_magnetics()
    m2 = s2.measure_magnetics()
    np.testing.assert_allclose(m1, m2, rtol=0.0, atol=0.0)

    emission = np.exp(-((k1.RR - 6.0) ** 2 + k1.ZZ**2))
    b1 = s1.measure_bolometer(emission)
    b2 = s2.measure_bolometer(emission)
    np.testing.assert_allclose(b1, b2, rtol=0.0, atol=0.0)


def test_sensor_suite_different_seeds_change_noise_realization() -> None:
    k1 = _KernelStub()
    k2 = _KernelStub()
    s1 = SensorSuite(k1, seed=11)
    s2 = SensorSuite(k2, seed=12)
    m1 = s1.measure_magnetics()
    m2 = s2.measure_magnetics()
    assert not np.allclose(m1, m2)


def test_sensor_suite_rejects_seed_and_rng_together() -> None:
    with pytest.raises(ValueError, match="either seed or rng"):
        SensorSuite(_KernelStub(), seed=7, rng=np.random.default_rng(7))
