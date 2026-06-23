# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MARFE Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.marfe import (
    DensityLimitPredictor,
    MARFEFrontModel,
    MARFEStabilityDiagram,
    RadiationCondensation,
)


def test_radiation_condensation_stability():
    # Low density
    rc = RadiationCondensation("W", ne_20=0.1, f_imp=1e-4)
    # T=50 eV is on the unstable (high-T) side of the W cooling peak (peak is ~1.5 keV and ~50 eV, actually the low-T peak is at 50, so let's use 100 eV where dL/dT < 0)
    # W cooling curve: 1500 eV peak, 50 eV peak.
    # At 100 eV, it's between the peaks, might be stable or unstable depending on the curve details.
    # Let's test at 500 eV which is definitely on the rising slope to 1500 eV -> dL/dT > 0 -> stable

    assert not rc.is_unstable(Te_eV=500.0, k_par=0.1, kappa_par=2000.0)

    # At 2000 eV (above main peak) -> dL/dT < 0 -> unstable if density is high enough
    rc_high = RadiationCondensation("W", ne_20=10.0, f_imp=1e-4)
    assert rc_high.is_unstable(Te_eV=2000.0, k_par=0.01, kappa_par=2000.0)


def test_marfe_front_model():
    model = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)

    # High density -> MARFE
    T_prof = model.equilibrium(ne_20=5.0)
    # The low-conduction high-density setup should collapse T to the floor (1.0 eV) in the interior
    assert np.mean(T_prof) < 20.0

    # Low density -> Attached/Hot
    model2 = MARFEFrontModel(L_par=100.0, kappa_par=2000.0, q_perp=1e5, impurity="W", f_imp=1e-4)
    T_prof2 = model2.equilibrium(ne_20=0.1)
    assert np.mean(T_prof2) > 100.0


def test_density_limit_predictor():
    n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=2.0)
    # 15 / (pi * 4) = 15 / 12.56 ~ 1.19
    assert 1.0 < n_gw < 1.5

    # MARFE limit with clean plasma
    n_marfe = DensityLimitPredictor.marfe_limit(15.0, 2.0, P_SOL_MW=100.0, impurity="W", f_imp=1e-5)
    # factor = sqrt(100) / (10 * sqrt(1e-5)) = 10 / (10 * 0.00316) = 1 / 0.00316 = 316.
    # With f_imp=1e-5, MARFE limit is very high (higher than Greenwald)
    assert n_marfe > n_gw

    # High impurity
    n_marfe_dirty = DensityLimitPredictor.marfe_limit(
        15.0, 2.0, P_SOL_MW=100.0, impurity="W", f_imp=1e-2
    )
    assert n_marfe_dirty < n_marfe


def test_density_limit_predictor_can_use_condensation_threshold() -> None:
    rc = RadiationCondensation("W", ne_20=1.0, f_imp=1e-4)
    expected = rc.critical_density(Te_eV=2000.0, k_par=0.01, kappa_par=2000.0)

    actual = DensityLimitPredictor.marfe_limit(
        15.0,
        2.0,
        P_SOL_MW=100.0,
        impurity="W",
        f_imp=1e-4,
        Te_eV=2000.0,
        k_par=0.01,
        kappa_par=2000.0,
    )

    assert actual == pytest.approx(expected)


def test_marfe_stability_diagram():
    diag = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W")

    # Push density range much higher so it definitely crosses the limit
    ne_range = np.linspace(0.1, 500.0, 10)
    P_SOL_range = np.linspace(10.0, 100.0, 10)

    res = diag.scan_density_power(ne_range, P_SOL_range)

    assert res.shape == (10, 10)
    # Low density, high power should be stable (+1)
    assert res[0, -1] == 1
    # High density, low power should be unstable (-1)
    assert res[-1, 0] == -1


def test_marfe_stability_diagram_current_sensitivity():
    ne_range = np.linspace(0.1, 50.0, 10)
    p_sol_range = np.linspace(10.0, 100.0, 10)

    low_ip = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", Ip_MA=8.0)
    high_ip = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", Ip_MA=20.0)

    low_map = low_ip.scan_density_power(ne_range, p_sol_range)
    high_map = high_ip.scan_density_power(ne_range, p_sol_range)
    # Higher plasma current should not reduce the stable region under this model.
    assert np.sum(high_map == 1) >= np.sum(low_map == 1)


def test_marfe_stability_diagram_impurity_fraction_sensitivity():
    ne_range = np.linspace(0.1, 20.0, 10)
    p_sol_range = np.linspace(10.0, 100.0, 10)

    clean = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", f_imp=1e-5)
    dirty = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", f_imp=1e-3)

    clean_map = clean.scan_density_power(ne_range, p_sol_range)
    dirty_map = dirty.scan_density_power(ne_range, p_sol_range)
    # Dirtier plasma should not increase the stable region.
    assert np.sum(dirty_map == 1) <= np.sum(clean_map == 1)


def test_marfe_stability_diagram_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Ip_MA"):
        MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", Ip_MA=0.0)
    with pytest.raises(ValueError, match="f_imp"):
        MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W", f_imp=0.0)

    diag = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W")
    with pytest.raises(ValueError, match="one-dimensional"):
        diag.scan_density_power(np.zeros((2, 2)), np.array([10.0, 20.0]))


def test_greenwald_limit_inf_for_nonpositive_minor_radius() -> None:
    assert DensityLimitPredictor.greenwald_limit(15.0, 0.0) == float("inf")


def test_radiation_condensation_critical_density_inf_on_rising_cooling_curve() -> None:
    rc = RadiationCondensation("Ne", 1.0, 1e-4)
    vals = [rc.critical_density(te, 1.0, 1.0) for te in (2.0, 5.0, 10.0, 20.0)]
    assert any(v == float("inf") for v in vals)


def test_marfe_stability_diagram_rejects_nonpositive_geometry() -> None:
    for kw, match in (
        ({"R0": 0.0, "a": 2.0, "q95": 3.0, "impurity": "Ne"}, "R0"),
        ({"R0": 6.2, "a": 0.0, "q95": 3.0, "impurity": "Ne"}, "a must be"),
        ({"R0": 6.2, "a": 2.0, "q95": 0.0, "impurity": "Ne"}, "q95"),
        ({"R0": 6.2, "a": 2.0, "q95": 3.0, "impurity": "Ne", "Ip_MA": 0.0}, "Ip_MA"),
        ({"R0": 6.2, "a": 2.0, "q95": 3.0, "impurity": "Ne", "f_imp": 0.0}, "f_imp"),
    ):
        with pytest.raises(ValueError, match=match):
            MARFEStabilityDiagram(**kw)


def test_marfe_limit_requires_all_or_none_optional_args() -> None:
    pred = DensityLimitPredictor()
    import inspect
    sig = inspect.signature(pred.marfe_limit)
    req = [p for p in list(sig.parameters) if sig.parameters[p].default is inspect._empty]
    base = {p: 1.0 for p in req}
    with pytest.raises(ValueError, match="must be supplied together"):
        pred.marfe_limit(**base, Te_eV=100.0)


def test_marfe_front_is_marfe_evaluates_state() -> None:
    front = MARFEFrontModel(L_par=20.0, kappa_par=2000.0, q_perp=5.0, impurity="Ne", f_imp=1e-3)
    front.equilibrium(ne_20=0.5)
    assert isinstance(front.is_marfe(), bool)


def test_marfe_scan_density_power_validates_inputs() -> None:
    diag = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="Ne")
    psol = np.linspace(1.0, 10.0, 4)
    with pytest.raises(ValueError, match="one-dimensional"):
        diag.scan_density_power(np.ones((2, 2)), psol)
    with pytest.raises(ValueError, match="must not be empty"):
        diag.scan_density_power(np.array([]), psol)
    with pytest.raises(ValueError, match="must be finite"):
        diag.scan_density_power(np.array([0.1, np.inf, 0.5]), psol)
    with pytest.raises(ValueError, match=r">= 0"):
        diag.scan_density_power(np.array([-0.1, 0.5]), psol)
