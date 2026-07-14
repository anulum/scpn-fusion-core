# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Facade contract for :mod:`scpn_fusion.core.fusion_ignition_sim`.

The burn-model behaviour is exercised in ``test_fusion_burn_physics`` and
``test_dynamic_burn_model``; this module pins that the historical facade keeps
re-exporting the same objects downstream importers rely on
(``scpn_fusion.core.__init__``, ``ui.app``, ``nuclear.nuclear_wall_interaction``).
"""

from __future__ import annotations

import scpn_fusion.core.fusion_ignition_sim as fusion_ignition_sim
from scpn_fusion.core.dynamic_burn_model import (
    BurnPhysicsError as _BurnPhysicsError,
    DynamicBurnModel as _DynamicBurnModel,
)
from scpn_fusion.core.fusion_burn_physics import (
    FloatArray as _FloatArray,
    FusionBurnPhysics as _FusionBurnPhysics,
    run_ignition_experiment as _run_ignition_experiment,
)


def test_facade_reexports_are_identical_objects() -> None:
    """Every public name resolves to the object defined in its sibling module."""
    assert fusion_ignition_sim.BurnPhysicsError is _BurnPhysicsError
    assert fusion_ignition_sim.DynamicBurnModel is _DynamicBurnModel
    assert fusion_ignition_sim.FloatArray is _FloatArray
    assert fusion_ignition_sim.FusionBurnPhysics is _FusionBurnPhysics
    assert fusion_ignition_sim.run_ignition_experiment is _run_ignition_experiment


def test_facade_all_matches_public_surface() -> None:
    """``__all__`` enumerates exactly the re-exported public names."""
    assert sorted(fusion_ignition_sim.__all__) == [
        "BurnPhysicsError",
        "DynamicBurnModel",
        "FloatArray",
        "FusionBurnPhysics",
        "run_ignition_experiment",
    ]
    for name in fusion_ignition_sim.__all__:
        assert hasattr(fusion_ignition_sim, name)
