# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Shared State Interface Tests
"""Tests for the integrated transport solver shared state typing base."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core._integrated_transport_solver_base import TransportSolverState


def test_transport_state_methods_are_unimplemented_on_the_bare_base() -> None:
    """The base only declares the cross-mixin surface; its methods must not run."""
    base = TransportSolverState()
    arr = np.zeros(3, dtype=np.float64)
    for call in (
        base._rho_volume_element,
        lambda: base.update_transport_model(1.0),
        lambda: base.evolve_profiles(0.1, 1.0),
        lambda: base.compute_confinement_time(1.0),
        lambda: base._explicit_diffusion_rhs(arr, arr),
        lambda: base._thomas_solve(arr, arr, arr, arr),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_solver_inherits_the_shared_state_interface() -> None:
    """TransportSolver composes its mixins on top of the shared state base."""
    from scpn_fusion.core.integrated_transport_solver import TransportSolver

    assert issubclass(TransportSolver, TransportSolverState)


def test_kernel_members_are_annotation_only_no_runtime_shadow() -> None:
    """Kernel-provided members are declared but not defined on the base.

    A bare class annotation must not create a runtime attribute, otherwise the base
    would shadow the real FusionKernel implementation ahead of it in the MRO.
    """
    for name in ("solve_equilibrium", "find_x_point", "cfg", "Psi"):
        assert name not in TransportSolverState.__dict__
        assert name in TransportSolverState.__annotations__
