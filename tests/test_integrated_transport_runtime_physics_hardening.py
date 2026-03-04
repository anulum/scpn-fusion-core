# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Runtime Physics Hardening Tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Numerical-hardening regressions for runtime transport physics closures."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.integrated_transport_solver_runtime_physics import (
    TransportSolverRuntimePhysicsMixin,
)


def test_bosch_hale_sigmav_sanitizes_nonfinite_inputs() -> None:
    t = np.array([np.nan, -5.0, 0.0, 0.2, 1.0, 10.0, np.inf], dtype=np.float64)
    out = TransportSolverRuntimePhysicsMixin._bosch_hale_sigmav(t)

    assert out.shape == t.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_bosch_hale_sigmav_increases_in_fusion_relevant_band() -> None:
    # D-T reactivity should increase strongly between ~1 and ~20 keV.
    t = np.array([1.0, 5.0, 10.0, 20.0], dtype=np.float64)
    out = TransportSolverRuntimePhysicsMixin._bosch_hale_sigmav(t)
    assert np.all(np.diff(out) > 0.0), out


def test_bremsstrahlung_power_density_sanitizes_nonfinite_inputs() -> None:
    ne = np.array([np.nan, -1.0, 0.0, 5.0, np.inf], dtype=np.float64)
    te = np.array([np.nan, -2.0, 0.0, 8.0, np.inf], dtype=np.float64)
    out = TransportSolverRuntimePhysicsMixin._bremsstrahlung_power_density(ne, te, Z_eff=np.nan)

    assert out.shape == ne.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_bremsstrahlung_power_density_monotonic_in_density() -> None:
    ne = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    te = np.full_like(ne, 5.0)
    out = TransportSolverRuntimePhysicsMixin._bremsstrahlung_power_density(ne, te, Z_eff=1.5)
    assert np.all(np.diff(out) > 0.0), out
