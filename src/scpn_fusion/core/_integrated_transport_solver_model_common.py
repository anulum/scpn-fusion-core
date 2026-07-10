# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared lazy-import helper for integrated transport solver mixins."""

from __future__ import annotations

from typing import Any


def _solver_module() -> Any:
    """Resolve host integrated_transport_solver module lazily."""
    import scpn_fusion.core.integrated_transport_solver as solver_mod

    return solver_mod
