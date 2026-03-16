# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GK Corrector Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_corrector import CorrectionRecord, CorrectorConfig, GKCorrector


@pytest.fixture
def rho50():
    return np.linspace(0, 1, 50)


def _make_record(
    idx: int, rho: float, surr_i: float, gk_i: float, surr_e: float = 1.0, gk_e: float = 1.0
) -> CorrectionRecord:
    return CorrectionRecord(
        rho_idx=idx,
        rho=rho,
        chi_i_surrogate=surr_i,
        chi_i_gk=gk_i,
        chi_e_surrogate=surr_e,
        chi_e_gk=gk_e,
        D_e_surrogate=0.1,
        D_e_gk=0.1,
    )


def test_no_correction_without_update(rho50):
    corr = GKCorrector(nr=50)
    chi_i = np.ones(50) * 2.0
    chi_e = np.ones(50) * 1.5
    D_e = np.ones(50) * 0.3
    out_i, out_e, out_d = corr.correct(chi_i, chi_e, D_e)
    np.testing.assert_array_equal(out_i, chi_i)
    np.testing.assert_array_equal(out_e, chi_e)


def test_multiplicative_correction(rho50):
    corr = GKCorrector(nr=50, config=CorrectorConfig(mode="multiplicative", smoothing_alpha=1.0))
    records = [_make_record(25, 0.5, surr_i=2.0, gk_i=3.0)]
    corr.update(records, rho50)

    chi_i = np.ones(50) * 2.0
    chi_e = np.ones(50)
    D_e = np.ones(50) * 0.1
    out_i, _, _ = corr.correct(chi_i, chi_e, D_e)

    # At rho=0.5, correction = 3.0/2.0 = 1.5, so chi_i should be 3.0
    assert out_i[25] == pytest.approx(3.0, rel=0.01)


def test_temporal_smoothing(rho50):
    corr = GKCorrector(nr=50, config=CorrectorConfig(smoothing_alpha=0.3))
    records = [_make_record(25, 0.5, surr_i=1.0, gk_i=2.0)]

    # First update: alpha_new = 0.3 * 2.0 + 0.7 * 1.0 = 1.3
    corr.update(records, rho50)
    chi_i = np.ones(50)
    out_i, _, _ = corr.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    assert out_i[25] == pytest.approx(1.3, rel=0.05)

    # Second update: alpha_new = 0.3 * 2.0 + 0.7 * 1.3 = 1.51
    corr.update(records, rho50)
    out_i2, _, _ = corr.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    assert out_i2[25] > out_i[25]


def test_additive_mode(rho50):
    corr = GKCorrector(nr=50, config=CorrectorConfig(mode="additive", smoothing_alpha=1.0))
    records = [_make_record(25, 0.5, surr_i=2.0, gk_i=3.0)]
    corr.update(records, rho50)

    chi_i = np.ones(50) * 2.0
    out_i, _, _ = corr.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    # Additive equivalent to multiplicative: 2.0 + 2.0*(1.5-1) = 3.0
    assert out_i[25] == pytest.approx(3.0, rel=0.01)


def test_replace_mode(rho50):
    corr = GKCorrector(
        nr=50, config=CorrectorConfig(mode="replace", smoothing_alpha=1.0, replace_threshold=0.3)
    )
    # Small error: factor = 1.1, |1.1-1| = 0.1 < 0.3 → no replacement
    records_small = [_make_record(25, 0.5, surr_i=1.0, gk_i=1.1)]
    corr.update(records_small, rho50)
    chi_i = np.ones(50)
    out_i, _, _ = corr.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    assert out_i[25] == pytest.approx(1.0)  # no replacement

    # Reset and large error: factor = 2.0, |2.0-1| = 1.0 > 0.3 → replace
    corr2 = GKCorrector(
        nr=50, config=CorrectorConfig(mode="replace", smoothing_alpha=1.0, replace_threshold=0.3)
    )
    records_big = [_make_record(25, 0.5, surr_i=1.0, gk_i=2.0)]
    corr2.update(records_big, rho50)
    out_i2, _, _ = corr2.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    assert out_i2[25] == pytest.approx(2.0, rel=0.01)


def test_interpolation_across_grid(rho50):
    corr = GKCorrector(nr=50, config=CorrectorConfig(smoothing_alpha=1.0))
    # Two spot-checks at different radii
    records = [
        _make_record(15, 0.3, surr_i=1.0, gk_i=1.5),
        _make_record(40, 0.8, surr_i=1.0, gk_i=2.0),
    ]
    corr.update(records, rho50)
    chi_i = np.ones(50)
    out_i, _, _ = corr.correct(chi_i, np.ones(50), np.ones(50) * 0.1)
    # Midpoint should be interpolated between 1.5 and 2.0
    assert 1.4 < out_i[25] < 2.1


def test_correction_record_rel_error():
    r = _make_record(0, 0.0, surr_i=2.0, gk_i=1.0)
    assert r.rel_error_chi_i == pytest.approx(1.0)  # (2-1)/1

    r2 = _make_record(0, 0.0, surr_i=0.5, gk_i=1.0)
    assert r2.rel_error_chi_i == pytest.approx(-0.5)


def test_max_and_mean_correction():
    corr = GKCorrector(nr=10, config=CorrectorConfig(smoothing_alpha=1.0))
    rho = np.linspace(0, 1, 10)
    records = [_make_record(5, 0.5, surr_i=1.0, gk_i=2.0)]
    corr.update(records, rho)
    assert corr.max_correction_factor > 0
    assert corr.mean_correction_factor > 0


def test_history_tracking(rho50):
    corr = GKCorrector(nr=50)
    records = [_make_record(25, 0.5, surr_i=1.0, gk_i=1.5)]
    corr.update(records, rho50)
    corr.update(records, rho50)
    assert len(corr.history) == 2
