# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Supervisory Control
from __future__ import annotations

from scpn_fusion.control._free_boundary_control_geometry import (
    FreeBoundarySupervisoryController,
    extract_free_boundary_state,
)
from scpn_fusion.control._free_boundary_estimator import FreeBoundaryStateEstimator
from scpn_fusion.control._free_boundary_safety_supervisor import FreeBoundarySafetySupervisor
from scpn_fusion.control._free_boundary_simulation import run_free_boundary_supervisory_simulation
from scpn_fusion.control._free_boundary_supervisory_types import (
    DEFAULT_TARGET_VECTOR,
    SUPERVISORY_ALERT_LEVEL_NAMES,
    SUPERVISORY_DISRUPTION_RISK_BIAS,
    FloatArray,
    FreeBoundaryEstimate,
    FreeBoundarySafetyMargins,
    FreeBoundaryTarget,
    SafetyFilterResult,
    estimate_free_boundary_safety_margins,
)

__all__ = [
    "DEFAULT_TARGET_VECTOR",
    "SUPERVISORY_ALERT_LEVEL_NAMES",
    "SUPERVISORY_DISRUPTION_RISK_BIAS",
    "FloatArray",
    "FreeBoundaryEstimate",
    "FreeBoundarySafetyMargins",
    "FreeBoundarySafetySupervisor",
    "FreeBoundaryStateEstimator",
    "FreeBoundarySupervisoryController",
    "FreeBoundaryTarget",
    "SafetyFilterResult",
    "estimate_free_boundary_safety_margins",
    "extract_free_boundary_state",
    "run_free_boundary_supervisory_simulation",
]


if __name__ == "__main__":
    run_free_boundary_supervisory_simulation(verbose=True)
