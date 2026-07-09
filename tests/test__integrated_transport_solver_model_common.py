# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Common Helper Tests
"""Contract test for the lazy integrated-transport-solver module resolver."""

from __future__ import annotations

import scpn_fusion.core.integrated_transport_solver as solver_mod
from scpn_fusion.core._integrated_transport_solver_model_common import _solver_module


def test_solver_module_resolves_host_module() -> None:
    """The lazy resolver returns the integrated transport solver host module."""
    assert _solver_module() is solver_mod
