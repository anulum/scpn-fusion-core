# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — PB-KBM Miller Second-Stability Geometry Tests
"""Tests for the opt-in Miller second-stability ballooning geometry of PB-KBM.

The shaped ``alpha_crit`` reference is injected (a schema-valid synthetic
payload), so these exercise the wiring without the committed pyrokinetics
artifact or a heavy eigenvalue solve.
"""

from __future__ import annotations

import pytest

from scpn_fusion.core.ballooning_second_stability import (
    ARTIFACT_SCHEMA,
    BallooningSecondStabilityReference,
)
from scpn_fusion.core.eped_pb_kbm import PBKBMPedestalModel, predict_pedestal

DIIID_LIKE = dict(R0=1.67, a=0.67, B0=2.1, Ip_MA=1.0, kappa=1.74, delta=0.3)


def _reference(alpha_crit: float, *, second_stability: bool = False):
    """Flat synthetic alpha_crit(shear) reference over the pedestal shear range."""
    rows = [
        {
            "shat": shat,
            "alpha_crit": None if second_stability else alpha_crit,
            "second_stability_access": second_stability,
            "alpha_max": alpha_crit if second_stability else 8.0,
        }
        for shat in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    return BallooningSecondStabilityReference.from_payload(
        {"schema": ARTIFACT_SCHEMA, "diiid_alpha_crit": rows, "diiid_shaping": {}, "provenance": {}}
    )


def test_miller_geometry_uses_reference_alpha_crit():
    ref = _reference(2.5)
    model = PBKBMPedestalModel(
        **DIIID_LIKE, geometry="miller_second_stability", ballooning_reference=ref
    )
    assert model.geometry == "miller_second_stability"
    # flat reference -> alpha_crit at the pedestal shear is the reference value
    assert model.ballooning_alpha_crit() == pytest.approx(2.5, abs=1e-9)


def test_miller_geometry_lifts_cap_under_second_stability_access():
    salpha = PBKBMPedestalModel(**DIIID_LIKE)
    miller = PBKBMPedestalModel(
        **DIIID_LIKE,
        geometry="miller_second_stability",
        ballooning_reference=_reference(8.0, second_stability=True),
    )
    # shaped second-stability lifts alpha_crit far above the collapsing s-alpha value
    assert miller.ballooning_alpha_crit() == pytest.approx(8.0, abs=1e-9)
    assert miller.ballooning_alpha_crit() > salpha.ballooning_alpha_crit()


def test_miller_geometry_tags_method_and_records_alpha_crit():
    ref = _reference(3.0)
    model = PBKBMPedestalModel(
        **DIIID_LIKE, geometry="miller_second_stability", ballooning_reference=ref
    )
    result = model.predict(n_ped_1e19=6.0, coarse_points=8, refine_iterations=4)
    assert result.method == "pb_kbm_miller_second_stability"
    assert result.alpha_crit == pytest.approx(3.0, abs=1e-9)


def test_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="geometry must be one of"):
        PBKBMPedestalModel(**DIIID_LIKE, geometry="stellarator")


def test_rejects_reference_supplied_to_salpha_geometry():
    with pytest.raises(ValueError, match="only used by"):
        PBKBMPedestalModel(**DIIID_LIKE, ballooning_reference=_reference(2.0))


def test_miller_geometry_without_reference_requires_committed_artifact(monkeypatch):
    # No reference injected -> loads the committed artifact, absent until the
    # heavy pyrokinetics run is committed.
    def _missing():
        raise FileNotFoundError("ballooning reference artifact not found: <deferred>")

    monkeypatch.setattr(BallooningSecondStabilityReference, "from_artifact", staticmethod(_missing))
    with pytest.raises(FileNotFoundError, match="not found"):
        PBKBMPedestalModel(**DIIID_LIKE, geometry="miller_second_stability")


def test_predict_pedestal_routes_miller_geometry_and_reference():
    ref = _reference(2.5)
    result = predict_pedestal(
        n_ped_1e19=6.0,
        tier="pb_kbm",
        geometry="miller_second_stability",
        ballooning_reference=ref,
        **DIIID_LIKE,
    )
    assert result.method == "pb_kbm_miller_second_stability"
    assert result.alpha_crit == pytest.approx(2.5, abs=1e-9)


def test_validation_rejects_non_numeric_field():
    # float() raising on a non-numeric field is re-wrapped as a ValueError.
    with pytest.raises(ValueError, match="finite and > 0"):
        PBKBMPedestalModel(**{**DIIID_LIKE, "R0": "not-a-number"})


def test_full_second_stability_can_saturate_the_scan():
    # High current (large peeling-current limit) plus a very high alpha_crit and
    # a low, capped temperature scan keeps the whole KBM family
    # peeling-ballooning stable, so the prediction saturates at T_max rather than
    # crossing a boundary (the "entire scan stable" branch: converged is False
    # with the pedestal pinned at T_max).
    ref = _reference(50.0, second_stability=True)
    model = PBKBMPedestalModel(
        R0=1.67,
        a=0.67,
        B0=2.1,
        Ip_MA=5.0,
        kappa=1.74,
        delta=0.3,
        geometry="miller_second_stability",
        ballooning_reference=ref,
    )
    result = model.predict(n_ped_1e19=1.0, T_max_keV=1.0, coarse_points=6, refine_iterations=4)
    assert result.converged is False
    assert result.T_ped_keV == pytest.approx(1.0, abs=1e-6)  # pinned at T_max (saturated)
    assert result.method == "pb_kbm_miller_second_stability"
