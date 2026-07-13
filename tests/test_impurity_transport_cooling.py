# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Cooling and Radiated Power Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport_cooling import CoolingCurve, total_radiated_power


def test_cooling_curves() -> None:
    c_W = CoolingCurve("W")
    L_W_core = c_W.L_z(np.array([1500.0]))[0]
    L_W_edge = c_W.L_z(np.array([10.0]))[0]

    assert L_W_core > L_W_edge
    assert L_W_core > 1e-32


def test_tungsten_cooling_curve_peak_order_of_magnitude() -> None:
    """Pin the absolute peak of the tungsten cooling curve.

    Putterich et al. 2010: tungsten coronal Lz peaks at ~1e-31 W m^3 near
    ~1.5 keV. The other ``test_cooling_curves`` checks only an ordering, so a
    wrong peak magnitude would pass it.
    """
    te_eV = np.logspace(1.0, 4.0, 400)  # 10 eV - 10 keV
    cooling = CoolingCurve("W").L_z(te_eV)
    peak = float(np.max(cooling))
    assert 3e-32 < peak < 3e-31
    assert 800.0 < te_eV[int(np.argmax(cooling))] < 2500.0


def test_light_impurity_cooling_curve_peaks_match_verified_data() -> None:
    """Pin the C/Ne/Ar cooling-curve peaks to their verified source values.

    Carbon, neon, and argon are the OpenADAS adf11 coronal cooling rates computed
    by tools/compute_coronal_lz_from_adas.py (carbon 5.84e-32 near 7 eV, neon
    5.74e-32 near 30 eV, argon 1.98e-31 near 20 eV). All sit in the coronal
    1e-31 band; guards against the prefactor drifting an order of magnitude low
    (the prior bug) or to the unverified Post & Jensen ~2e-31.
    """
    te_eV = np.logspace(0.0, 3.5, 400)  # 1 eV - ~3 keV
    for element in ("C", "Ne", "Ar"):
        peak = float(np.max(CoolingCurve(element).L_z(te_eV)))
        assert 3e-32 < peak < 3e-31, f"{element} peak {peak:.2e} outside coronal band"

    # Peak magnitudes (evaluated at the peak temperature) and locations.
    assert float(CoolingCurve("C").L_z(np.array([7.0]))[0]) == pytest.approx(5.84e-32, rel=1e-9)
    assert float(CoolingCurve("Ne").L_z(np.array([30.0]))[0]) == pytest.approx(5.74e-32, rel=1e-9)
    assert float(CoolingCurve("Ar").L_z(np.array([20.0]))[0]) == pytest.approx(1.98e-31, rel=1e-9)
    assert 5.0 < te_eV[int(np.argmax(CoolingCurve("C").L_z(te_eV)))] < 10.0
    assert 20.0 < te_eV[int(np.argmax(CoolingCurve("Ne").L_z(te_eV)))] < 45.0
    assert 15.0 < te_eV[int(np.argmax(CoolingCurve("Ar").L_z(te_eV)))] < 30.0


def test_cooling_curve_returns_zero_for_nonpositive_temperatures_without_warning() -> None:
    curve = CoolingCurve("W")

    with np.errstate(all="raise"):
        values = curve.L_z(np.array([-10.0, 0.0, 1500.0]))

    assert values[0] == 0.0
    assert values[1] == 0.0
    assert np.isfinite(values[2])
    assert values[2] > 0.0


def test_total_radiated_power() -> None:
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1e20
    Te = np.ones(50) * 1500.0

    # Concentration 1e-4 W
    nW = ne * 1e-4
    n_imp = {"W": nW}

    P_rad = total_radiated_power(ne, n_imp, Te, rho, 6.2, 2.0)
    assert P_rad > 10.0  # Should be substantial (tens of MW)


class TestCoolingCurveBranches:
    def test_all_invalid_temperatures_return_zeros(self) -> None:
        out = CoolingCurve("W").L_z(np.array([-1.0, 0.0, np.nan]))
        assert np.allclose(out, 0.0)

    def test_unknown_element_returns_zeros(self) -> None:
        out = CoolingCurve("Xe").L_z(np.array([10.0, 100.0]))
        assert np.allclose(out, 0.0)
