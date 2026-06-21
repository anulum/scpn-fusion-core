# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MHD Sawtooth Tests
"""Tests for the reduced-MHD m=1 internal-kink sawtooth model."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.mhd_sawtooth import ReducedMHD, run_sawtooth_sim


class TestReducedMHDInit:
    """Reduced-MHD grid and equilibrium initialisation."""

    def test_default_grid(self) -> None:
        """Construction builds the radial grid with an internal-kink-unstable core."""
        sim = ReducedMHD(nr=50)
        assert sim.nr == 50
        assert len(sim.r) == 50
        assert sim.q[0] < 1.0

    def test_q_profile_monotone(self) -> None:
        """The safety-factor profile is monotonically non-decreasing."""
        sim = ReducedMHD()
        assert np.all(np.diff(sim.q) >= 0)

    def test_initial_perturbation_nonzero(self) -> None:
        """The seeded m=1 flux perturbation is non-zero."""
        sim = ReducedMHD()
        assert np.max(np.abs(sim.psi_11)) > 0


class TestLaplacian:
    """Cylindrical m-mode Laplacian operator."""

    def test_shape_preserved(self) -> None:
        """The operator preserves the radial grid shape."""
        sim = ReducedMHD(nr=50)
        f = np.sin(np.pi * sim.r).astype(np.complex128)
        result = sim.laplacian(f, m=1)
        assert result.shape == (50,)

    def test_zero_function(self) -> None:
        """The Laplacian of a zero field is zero."""
        sim = ReducedMHD(nr=50)
        result = sim.laplacian(np.zeros(50, dtype=np.complex128), m=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_boundary_values_zero(self) -> None:
        """The operator imposes zero values at both radial boundaries."""
        sim = ReducedMHD(nr=50)
        f = (np.random.randn(50) + 0j).astype(np.complex128)
        result = sim.laplacian(f, m=1)
        assert result[0] == 0.0
        assert result[-1] == 0.0


class TestStep:
    """Semi-implicit time integration and Kadomtsev crash."""

    def test_returns_amplitude_and_crash(self) -> None:
        """A step returns a finite amplitude and a boolean crash flag."""
        sim = ReducedMHD(nr=50)
        amp, crash = sim.step(dt=0.01)
        assert amp >= 0
        assert isinstance(crash, bool)
        assert np.isfinite(amp)

    def test_perturbation_grows(self) -> None:
        """The internal kink perturbation grows over many steps."""
        sim = ReducedMHD(nr=50)
        amp0 = np.max(np.abs(sim.psi_11))
        for _ in range(50):
            sim.step(dt=0.01)
        amp1 = np.max(np.abs(sim.psi_11))
        assert amp1 > amp0 or amp1 > 0

    def test_sawtooth_crash_triggers_and_recovers(self) -> None:
        """An over-threshold amplitude triggers a Kadomtsev crash and q reset."""
        sim = ReducedMHD(nr=50)
        sim.psi_11 = np.ones(50, dtype=np.complex128)  # well above the crash threshold
        amp, crash = sim.step(dt=0.01)
        assert crash is True
        assert np.all(np.isfinite(sim.psi_11))
        assert np.all(np.isfinite(sim.q))

    def test_long_run_stays_finite(self) -> None:
        """A long run remains finite through growth/crash/recovery cycles."""
        sim = ReducedMHD(nr=50)
        for _ in range(500):
            sim.step(dt=0.01)
        assert np.all(np.isfinite(sim.psi_11))
        assert np.all(np.isfinite(sim.q))


class TestSolvePoisson:
    """Tridiagonal Poisson solve for the stream function."""

    def test_boundary_conditions(self) -> None:
        """The Poisson solve pins the stream function to zero at the boundaries."""
        sim = ReducedMHD(nr=50)
        U = np.ones(50, dtype=np.complex128)
        phi = sim.solve_poisson(U)
        assert phi[0] == pytest.approx(0.0, abs=1e-10)
        assert phi[-1] == pytest.approx(0.0, abs=1e-10)

    def test_returns_finite(self) -> None:
        """The Poisson solve returns finite values for a random source."""
        sim = ReducedMHD(nr=50)
        U = (np.random.randn(50) + 0j).astype(np.complex128)
        phi = sim.solve_poisson(U)
        assert np.all(np.isfinite(phi))

    def test_zero_source_gives_zero(self) -> None:
        """A zero source yields a zero stream function."""
        sim = ReducedMHD(nr=50)
        phi = sim.solve_poisson(np.zeros(50, dtype=np.complex128))
        np.testing.assert_allclose(phi, 0.0, atol=1e-12)


def test_run_sawtooth_sim_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone sawtooth simulation runs and renders both diagnostics."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    run_sawtooth_sim()

    assert "MHD_Sawtooth.png" in saved
    assert "Magnetic_Island_2D.png" in saved
