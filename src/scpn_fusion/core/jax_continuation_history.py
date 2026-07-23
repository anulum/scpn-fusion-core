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
    """Return whether this iteration uses a different fixed-point map.

    Anderson differences are valid only while the map ``G`` is fixed. The
    predictive map changes on every non-initial Ip-ramp step and on every
    non-zero separatrix-refinement step. Iteration zero has no prior history to
    invalidate.
    """
    index = jnp.asarray(iteration)
    has_prior_history = index > 0
    ip_changed = index < ip_ramp
    separatrix_changed = (
        use_separatrix_continuation
        & (index >= separatrix_start)
        & (index < separatrix_start + separatrix_ramp)
    )
    return has_prior_history & (ip_changed | separatrix_changed)
