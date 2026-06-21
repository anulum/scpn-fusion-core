# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nonlinear GK Shared State Interface Tests
"""Tests for the nonlinear-GK shared state typing base."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core._gk_nonlinear_base import NonlinearGKSolverState


def test_state_methods_are_unimplemented_on_the_bare_base() -> None:
    """The base only declares the cross-mixin surface; its methods must not run."""
    base = NonlinearGKSolverState()
    dummy = np.zeros((1, 1, 1), dtype=np.complex128)
    for call in (
        lambda: base.field_solve(dummy),
        lambda: base.ampere_solve(dummy),
        lambda: base.magnetic_compression_solve(dummy),
        lambda: base.exb_bracket(dummy, dummy),
        lambda: base._roll_ballooning(dummy, 1),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_solver_inherits_the_shared_state_interface() -> None:
    """NonlinearGKSolver composes the mixins on top of the shared state base."""
    from scpn_fusion.core.gk_nonlinear import NonlinearGKSolver

    assert issubclass(NonlinearGKSolver, NonlinearGKSolverState)


def test_shared_attribute_surface_is_declared() -> None:
    """The shared grid/geometry/species attributes are declared for the mixins."""
    annotations = NonlinearGKSolverState.__annotations__
    for name in ("cfg", "kx", "ky", "vpar", "mu", "dealias_mask", "geom", "ion", "elec"):
        assert name in annotations
