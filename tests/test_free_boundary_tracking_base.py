# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Shared State Interface Tests
"""Tests for the free-boundary tracking shared state typing base."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control._free_boundary_tracking_base import _FreeBoundaryTrackingState


def test_state_methods_are_unimplemented_on_the_bare_base() -> None:
    """The base only declares the cross-mixin surface; its methods must not run."""
    base = _FreeBoundaryTrackingState()
    obs = np.zeros(3, dtype=np.float64)
    for call in (
        base._solve_free_boundary_state,
        base._snapshot_actuator_states,
        base._observe_objectives,
        lambda: base.evaluate_objectives(obs),
        lambda: base.compute_correction(obs),
        lambda: base.identify_response_matrix(),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_controller_inherits_the_shared_state_interface() -> None:
    """FreeBoundaryTrackingController composes the mixins on the shared state base."""
    from scpn_fusion.control.free_boundary_tracking import FreeBoundaryTrackingController

    assert issubclass(FreeBoundaryTrackingController, _FreeBoundaryTrackingState)


def test_shared_attribute_surface_is_declared() -> None:
    """The shared coil/objective/observer attributes are declared for the mixins."""
    annotations = _FreeBoundaryTrackingState.__annotations__
    for name in (
        "kernel",
        "coils",
        "n_coils",
        "target_vector",
        "objective_tolerances",
        "response_matrix",
        "coil_current_limits",
    ):
        assert name in annotations
