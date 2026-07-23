# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for continuation-aware Anderson-history compatibility."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from scpn_fusion.core.jax_continuation_history import (
    continuation_history_requires_reset,
)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, False),
        (1, True),
        (29, True),
        (30, False),
        (99, False),
        (100, True),
        (119, True),
        (120, False),
        (121, False),
    ],
)
def test_continuation_history_reset_boundaries(
    iteration: int,
    expected: bool,
) -> None:
    """Only iterations that change Ip or refinement invalidate history."""
    reset = continuation_history_requires_reset(
        iteration,
        ip_ramp=30,
        use_separatrix_continuation=True,
        separatrix_start=100,
        separatrix_ramp=20,
    )
    assert bool(reset) is expected


def test_warm_fixed_map_retains_history() -> None:
    """A warm solve at full Ip/refinement never invalidates fixed-map rows."""
    reset = jax.jit(
        lambda iteration: continuation_history_requires_reset(
            iteration,
            ip_ramp=1,
            use_separatrix_continuation=False,
            separatrix_start=100,
            separatrix_ramp=20,
        )
    )(jnp.arange(8))
    assert not bool(jnp.any(reset))
