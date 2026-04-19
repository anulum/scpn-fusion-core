# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runtime Physics Hardening Tests
"""Numerical-hardening regressions for runtime transport physics closures."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.integrated_transport_solver_runtime_physics import (
    TransportSolverRuntimePhysicsMixin,
)


class _AuxHeatingDummy(TransportSolverRuntimePhysicsMixin):
    def __init__(self) -> None:
        self.nr = 8
        self.rho = np.linspace(0.0, 1.0, self.nr, dtype=np.float64)
        self.drho = float(self.rho[1] - self.rho[0])
        self.cfg = {"dimensions": {"R_min": 4.2, "R_max": 8.2}}
        self.aux_heating_profile_width = 0.35
        self.aux_heating_electron_fraction = 0.4
        self.multi_ion = True
        self.ne = np.full(self.nr, 5.0, dtype=np.float64)
        self._last_aux_heating_balance: dict[str, float] = {}


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


def test_tungsten_radiation_rate_sanitizes_nonfinite_inputs() -> None:
    te = np.array([np.nan, -3.0, 0.0, 1.0, 10.0, np.inf], dtype=np.float64)
    out = TransportSolverRuntimePhysicsMixin._tungsten_radiation_rate(te)

    assert out.shape == te.shape
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_aux_heating_sources_sanitize_nonfinite_density() -> None:
    dummy = _AuxHeatingDummy()
    dummy.ne = np.array([np.nan, -1.0, 0.0, 2.0, 5.0, np.inf, 1.0, 3.0], dtype=np.float64)
    s_i, s_e = dummy._compute_aux_heating_sources(8.0)

    assert s_i.shape == (dummy.nr,)
    assert s_e.shape == (dummy.nr,)
    assert np.all(np.isfinite(s_i))
    assert np.all(np.isfinite(s_e))
    assert dummy._last_aux_heating_balance["target_total_MW"] == 8.0


def test_aux_heating_sources_sanitize_nonfinite_rho_grid() -> None:
    dummy = _AuxHeatingDummy()
    dummy.rho[2] = float("nan")
    dummy.rho[5] = float("inf")
    s_i, s_e = dummy._compute_aux_heating_sources(6.0)

    assert np.all(np.isfinite(s_i))
    assert np.all(np.isfinite(s_e))
    assert dummy._last_aux_heating_balance["target_total_MW"] == 6.0
