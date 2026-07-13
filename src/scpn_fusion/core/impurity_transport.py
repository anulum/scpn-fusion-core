# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Species Impurity Transport
"""Multi-species impurity transport, cooling, radiation, and accumulation diagnostics.

This module is a thin facade that re-exports the public impurity-transport API
from its four responsibility submodules, preserving the historical
``scpn_fusion.core.impurity_transport`` import surface:

- :mod:`impurity_transport_contracts` — ``FloatArray``, ``ImpuritySpecies``,
  ``AdasChargeStateCoefficients``, ``AuroraStrahlArtifact``, ``AuroraParityCase``,
  ``_strict_axis``.
- :mod:`impurity_transport_charge_state` — ADAS-style coefficients and
  conservative charge-state collisional-radiative transfer math.
- :mod:`impurity_transport_cooling` — ``CoolingCurve`` and ``total_radiated_power``.
- :mod:`impurity_transport_aurora_parity` — ``AuroraParityImpuritySolver`` and
  ``build_aurora_strahl_charge_state_artifact``.
- :mod:`impurity_transport_diagnostics` — ``neoclassical_impurity_pinch`` and
  ``tungsten_accumulation_diagnostic``.
- :mod:`impurity_transport_solver` — ``ImpurityTransportSolver``.
"""

from __future__ import annotations

from scpn_fusion.core.impurity_transport_aurora_parity import (
    AuroraParityImpuritySolver,
    build_aurora_strahl_charge_state_artifact,
)
from scpn_fusion.core.impurity_transport_charge_state import (
    _source_sink_transfer_matrix,
    adas_style_charge_state_coefficients,
    advance_charge_state_collisional_radiative,
    collisional_radiative_source_sink_matrices,
)
from scpn_fusion.core.impurity_transport_contracts import (
    AdasChargeStateCoefficients,
    AuroraParityCase,
    AuroraStrahlArtifact,
    FloatArray,
    ImpuritySpecies,
    _strict_axis,
)
from scpn_fusion.core.impurity_transport_cooling import CoolingCurve, total_radiated_power
from scpn_fusion.core.impurity_transport_diagnostics import (
    neoclassical_impurity_pinch,
    tungsten_accumulation_diagnostic,
)
from scpn_fusion.core.impurity_transport_solver import ImpurityTransportSolver

__all__ = [
    "AdasChargeStateCoefficients",
    "AuroraParityCase",
    "AuroraParityImpuritySolver",
    "AuroraStrahlArtifact",
    "CoolingCurve",
    "FloatArray",
    "ImpuritySpecies",
    "ImpurityTransportSolver",
    "_source_sink_transfer_matrix",
    "_strict_axis",
    "adas_style_charge_state_coefficients",
    "advance_charge_state_collisional_radiative",
    "build_aurora_strahl_charge_state_artifact",
    "collisional_radiative_source_sink_matrices",
    "neoclassical_impurity_pinch",
    "total_radiated_power",
    "tungsten_accumulation_diagnostic",
]
