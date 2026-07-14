# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fusion Ignition Sim
"""Zero-dimensional ignition and dynamic burn calculations for equilibrium states.

This module is the stable public facade over two independent burn models:

- :class:`FusionBurnPhysics` / :func:`run_ignition_experiment` — the
  equilibrium-coupled static 0-D burn map (see :mod:`fusion_burn_physics`).
- :class:`DynamicBurnModel` / :class:`BurnPhysicsError` — the self-consistent
  dynamic burn ODE integrator (see :mod:`dynamic_burn_model`).
"""

from __future__ import annotations

from .dynamic_burn_model import BurnPhysicsError, DynamicBurnModel
from .fusion_burn_physics import FloatArray, FusionBurnPhysics, run_ignition_experiment

__all__ = [
    "BurnPhysicsError",
    "DynamicBurnModel",
    "FloatArray",
    "FusionBurnPhysics",
    "run_ignition_experiment",
]


if __name__ == "__main__":
    run_ignition_experiment()
