# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Shape Controller Tests
from __future__ import annotations

import numpy as np

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
