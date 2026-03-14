# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MARFE Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

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
    n_marfe_dirty = DensityLimitPredictor.marfe_limit(15.0, 2.0, P_SOL_MW=100.0, impurity="W", f_imp=1e-2)
    assert n_marfe_dirty < n_marfe


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
