# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import numpy as np

from scpn_fusion.core.integrated_transport_solver_runtime_physics import (
    TransportSolverRuntimePhysicsMixin,
)


class _DummyRuntime(TransportSolverRuntimePhysicsMixin):
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


def test_rho_volume_element_positive_and_finite() -> None:
    d = _DummyRuntime()
    d_v = d._rho_volume_element()
    assert d_v.shape == (d.nr,)
    assert np.all(np.isfinite(d_v))
    assert np.all(d_v >= 0.0)


def test_compute_aux_heating_sources_returns_finite_profiles() -> None:
    d = _DummyRuntime()
    s_i, s_e = d._compute_aux_heating_sources(12.0)
    assert s_i.shape == (d.nr,)
    assert s_e.shape == (d.nr,)
    assert np.all(np.isfinite(s_i))
    assert np.all(np.isfinite(s_e))
    assert d._last_aux_heating_balance["target_total_MW"] == 12.0


def test_static_radiation_helpers_are_well_behaved() -> None:
    te = np.array([0.5, 2.0, 10.0, 25.0], dtype=np.float64)
    sigmav = TransportSolverRuntimePhysicsMixin._bosch_hale_sigmav(te)
    lz = TransportSolverRuntimePhysicsMixin._tungsten_radiation_rate(te)
    pb = TransportSolverRuntimePhysicsMixin._bremsstrahlung_power_density(
        np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
        te,
        2.0,
    )
    assert np.all(np.isfinite(sigmav))
    assert np.all(np.isfinite(lz))
    assert np.all(np.isfinite(pb))
    assert np.all(sigmav > 0.0)
    assert np.all(lz > 0.0)
    assert np.all(pb > 0.0)
