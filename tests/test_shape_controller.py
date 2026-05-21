# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Shape Controller Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.shape_controller import (
    CoilSet,
    PlasmaShapeController,
    iter_lower_single_null_target,
)


def test_iter_target_generation():
    target = iter_lower_single_null_target()
    assert len(target.isoflux_points) == 30
    assert len(target.gap_points) == 3
    assert target.xpoint_target is not None


def test_jacobian_full_rank():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    J = ctrl.jacobian.compute()
    rank = np.linalg.matrix_rank(J)
    assert rank == coils.n_coils


def test_jacobian_is_deterministic_for_same_target_and_coils():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    c1 = PlasmaShapeController(target, coils, kernel=None)
    c2 = PlasmaShapeController(target, coils, kernel=None)
    np.testing.assert_allclose(c1.jacobian.compute(), c2.jacobian.compute())


def test_shape_controller_step():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    psi = np.ones((33, 33))
    currents = np.zeros(10)

    delta_I = ctrl.step(psi, currents)

    assert np.any(np.abs(delta_I) > 0.0)
    assert np.all(np.abs(delta_I) <= 1000.0)


def test_shape_controller_limits():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    coils.max_currents = np.ones(10) * 100.0
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    currents = np.ones(10) * 99.0

    ctrl.K_shape = -np.ones_like(ctrl.K_shape) * 1e6
    psi = np.ones((33, 33))

    delta_I = ctrl.step(psi, currents)

    I_next = currents + delta_I
    assert np.all(I_next <= 100.0)
    assert np.all(I_next >= -100.0)


def test_shape_performance_metrics():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    psi = np.ones((33, 33))
    res = ctrl.evaluate_performance(psi)

    assert res.isoflux_error > 0.0
    assert len(res.gap_errors) == 3
    assert res.min_gap > 0.0
    assert res.xpoint_error > 0.0


def test_shape_jacobian_update_accepts_explicit_matrix():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    new_j = np.ones((ctrl.jacobian.n_errors, coils.n_coils)) * 2.5e-4
    ctrl.jacobian.update({"jacobian": new_j})

    got = ctrl.jacobian.compute()
    assert np.allclose(got, new_j)
    assert got is not new_j


def test_shape_jacobian_update_scales_with_operating_point_state():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    j_ref = ctrl.jacobian.compute()
    ctrl.jacobian.update(
        {
            "plasma_current_ma": 30.0,
            "beta_p": 1.2,
            "coil_coupling": np.ones(coils.n_coils) * 1.1,
            "error_coupling": np.ones(ctrl.jacobian.n_errors) * 0.9,
        }
    )
    j_upd = ctrl.jacobian.compute()

    expected_scale = (30.0 / 15.0) * (1.0 + 0.25 * (1.2 - 1.0)) * 1.1 * 0.9
    assert np.allclose(j_upd, j_ref * expected_scale)


def test_shape_jacobian_update_rejects_missing_or_invalid_payload():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    with pytest.raises(ValueError, match="must provide either 'jacobian'"):
        ctrl.jacobian.update({})

    with pytest.raises(ValueError, match="must have shape"):
        ctrl.jacobian.update({"jacobian": np.ones((2, 2))})
