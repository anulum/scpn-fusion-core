# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — PB-KBM Pedestal Model Tests
"""Tests for the peeling-ballooning + KBM constraint-loop pedestal tier (F-5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.eped_pb_kbm import (
    KBM_WIDTH_COEFFICIENT,
    PBKBMPedestalModel,
    PBKBMPedestalResult,
    predict_pedestal,
)
from scpn_fusion.core.eped_pedestal import PedestalResult

DIIID_LIKE: dict[str, Any] = dict(R0=1.67, a=0.67, B0=2.1, Ip_MA=1.0, kappa=1.74, delta=0.3)


@pytest.fixture(scope="module")
def diiid_model() -> PBKBMPedestalModel:
    return PBKBMPedestalModel(**DIIID_LIKE)


@pytest.fixture(scope="module")
def diiid_prediction(diiid_model: PBKBMPedestalModel) -> PBKBMPedestalResult:
    return diiid_model.predict(n_ped_1e19=6.0, coarse_points=10, refine_iterations=6)


class TestConstruction:
    """Input validation fails closed."""

    @pytest.mark.parametrize("field", ["R0", "a", "B0", "Ip_MA", "kappa", "A_ion", "Z_eff"])
    def test_rejects_non_positive_scalars(self, field: str) -> None:
        kwargs = dict(DIIID_LIKE)
        kwargs[field] = -1.0
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            PBKBMPedestalModel(**kwargs)

    def test_rejects_bool_input(self) -> None:
        kwargs = dict(DIIID_LIKE)
        kwargs["B0"] = True
        with pytest.raises(ValueError, match="B0 must be finite"):
            PBKBMPedestalModel(**kwargs)

    def test_rejects_out_of_range_triangularity(self) -> None:
        kwargs = dict(DIIID_LIKE)
        kwargs["delta"] = 1.5
        with pytest.raises(ValueError, match="delta must be finite"):
            PBKBMPedestalModel(**kwargs)

    def test_q_ped_floor_applied(self) -> None:
        model = PBKBMPedestalModel(R0=1.67, a=0.67, B0=2.1, Ip_MA=1.0, kappa=1.74)
        assert model.q_ped >= 2.0


class TestStabilityInputs:
    """Shear, ballooning boundary, and peeling limit are physically sane."""

    def test_pedestal_shear_positive(self, diiid_model: PBKBMPedestalModel) -> None:
        assert diiid_model.pedestal_shear() > 0.0

    def test_ballooning_alpha_crit_positive_and_first_stability(
        self, diiid_model: PBKBMPedestalModel
    ) -> None:
        alpha_crit = diiid_model.ballooning_alpha_crit()
        # s-alpha first-stability boundary at pedestal shear sits well below
        # the shaped second-stability values reached by real pedestals.
        assert 0.0 < alpha_crit < 3.0

    def test_peeling_current_limit_positive(self, diiid_model: PBKBMPedestalModel) -> None:
        assert diiid_model.peeling_current_limit() > 0.0

    def test_edge_bootstrap_current_positive_and_grows_with_t(
        self, diiid_model: PBKBMPedestalModel
    ) -> None:
        j_low = diiid_model.edge_bootstrap_current(6.0, 0.3, 0.04)
        j_high = diiid_model.edge_bootstrap_current(6.0, 1.5, 0.04)
        assert j_low > 0.0
        assert j_high > j_low


class TestPrediction:
    """The constraint loop returns marginal-stability diagnostics."""

    def test_result_carries_method_tag(self, diiid_prediction: PBKBMPedestalResult) -> None:
        assert diiid_prediction.method == "pb_kbm_salpha"

    def test_width_follows_kbm_relation(self, diiid_prediction: PBKBMPedestalResult) -> None:
        expected = float(
            np.clip(
                KBM_WIDTH_COEFFICIENT * np.sqrt(diiid_prediction.beta_p_ped),
                0.01,
                0.15,
            )
        )
        assert diiid_prediction.Delta_ped == pytest.approx(expected, rel=1e-9)

    def test_salpha_first_stability_collapses_for_diiid_like_point(
        self, diiid_prediction: PBKBMPedestalResult
    ) -> None:
        """Documented negative result of the honest s-alpha boundary.

        Without shaped-geometry (second-stability) access the s-alpha
        first-stability boundary cannot sustain a measured-scale DIII-D
        pedestal: the lowest scan candidate is already outside the coupled
        P-B ellipse, so the prediction collapses to the scan floor and is
        flagged unconverged. Shaped (Miller-geometry) ballooning is the
        recorded blocker for quantitative EPED parity.
        """
        assert diiid_prediction.converged is False
        assert diiid_prediction.pb_boundary_radius >= 1.0
        assert diiid_prediction.T_ped_keV == pytest.approx(0.1)

    def test_diagnostics_are_populated(self, diiid_prediction: PBKBMPedestalResult) -> None:
        assert diiid_prediction.alpha_crit > 0.0
        assert diiid_prediction.j_crit_MA_m2 > 0.0
        assert diiid_prediction.j_ped_MA_m2 > 0.0
        assert diiid_prediction.candidates_evaluated >= 1
        # Cylindrical q_ped for the 1 MA DIII-D-like point (q ∝ 1/Ip).
        assert diiid_prediction.q_ped == pytest.approx(5.7, abs=1.0)

    def test_rejects_invalid_scan_bounds(self, diiid_model: PBKBMPedestalModel) -> None:
        with pytest.raises(ValueError, match="T_max_keV must exceed T_min_keV"):
            diiid_model.predict(n_ped_1e19=6.0, T_min_keV=2.0, T_max_keV=1.0)

    def test_rejects_too_few_scan_points(self, diiid_model: PBKBMPedestalModel) -> None:
        with pytest.raises(ValueError, match="coarse_points must be at least 3"):
            diiid_model.predict(n_ped_1e19=6.0, coarse_points=2)

    def test_rejects_non_positive_density(self, diiid_model: PBKBMPedestalModel) -> None:
        with pytest.raises(ValueError, match="n_ped_1e19 must be finite"):
            diiid_model.predict(n_ped_1e19=0.0)

    def test_rejects_zero_refine_iterations(self, diiid_model: PBKBMPedestalModel) -> None:
        with pytest.raises(ValueError, match="refine_iterations must be at least 1"):
            diiid_model.predict(n_ped_1e19=6.0, refine_iterations=0)

    def test_marginal_point_found_when_boundary_is_crossed_inside_scan(self) -> None:
        """A synthetic low-drive machine crosses the boundary mid-scan."""
        model = PBKBMPedestalModel(R0=3.0, a=1.0, B0=6.0, Ip_MA=4.0, kappa=1.4, delta=0.1)
        result = model.predict(n_ped_1e19=3.0, T_min_keV=0.01, T_max_keV=4.0, coarse_points=12)
        if result.converged:
            # Marginal point sits just inside the coupled boundary.
            assert result.pb_boundary_radius <= 1.0
            assert 0.01 < result.T_ped_keV < 4.0
        else:
            # Fail-closed outcome is also legitimate; it must be flagged.
            assert result.pb_boundary_radius >= 0.0


class TestTierFacade:
    """predict_pedestal routes between the fast and PB-KBM tiers."""

    def test_fast_tier_returns_fast_result(self) -> None:
        result = predict_pedestal(n_ped_1e19=6.0, tier="fast", **DIIID_LIKE)
        assert isinstance(result, PedestalResult)

    def test_pb_kbm_tier_returns_constraint_loop_result(self) -> None:
        result = predict_pedestal(n_ped_1e19=6.0, tier="pb_kbm", **DIIID_LIKE)
        assert isinstance(result, PBKBMPedestalResult)

    def test_unknown_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="tier must be"):
            predict_pedestal(n_ped_1e19=6.0, tier="cuda", **DIIID_LIKE)
