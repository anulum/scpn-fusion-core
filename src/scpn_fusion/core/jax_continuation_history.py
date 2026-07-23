# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Continuation-aware Anderson-history policy shared by predictive loops."""

from __future__ import annotations

import jax.numpy as jnp


def continuation_history_requires_reset(
    iteration: int | jnp.ndarray,
    *,
    ip_ramp: int,
    use_separatrix_continuation: bool,
    separatrix_start: int,
    separatrix_ramp: int,
) -> jnp.ndarray:
    """Return whether continuation has just reached a fixed endpoint.

    Anderson history must remain active while the map moves because unaccelerated
    Picard continuation is unstable. Once Ip or separatrix refinement reaches
    its final value, the current row belongs to the stationary map and starts a
    fresh history without earlier moving-map differences.
    """
    index = jnp.asarray(iteration)
    ip_reached_endpoint = (index > 0) & (index == ip_ramp - 1)
    separatrix_reached_endpoint = (
        use_separatrix_continuation
        & (index > 0)
        & (index == separatrix_start + separatrix_ramp - 1)
    )
    return ip_reached_endpoint | separatrix_reached_endpoint
