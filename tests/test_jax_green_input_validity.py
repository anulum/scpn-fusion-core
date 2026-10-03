# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Green function input validity tests
"""Public vacuum APIs must preserve invalid conductor observations as failures."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax import Array

from scpn_fusion.core.jax_equilibrium_solver import vacuum_field
from scpn_fusion.core.jax_free_boundary_gs import vacuum_field_si

Vacuum = Callable[..., Array]


@pytest.mark.parametrize("vacuum", [vacuum_field, vacuum_field_si])
@pytest.mark.parametrize(
    ("radius", "height", "current"),
    [
        (2.3, 0.8, np.nan),
        (2.3, 0.8, np.inf),
        (np.nan, 0.8, 0.2),
        (2.3, np.nan, 0.2),
        (-2.3, 0.8, 0.2),
        (0.0, 0.8, 0.2),
    ],
)
def test_invalid_conductor_cannot_disappear(
    vacuum: Vacuum, radius: float, height: float, current: float
) -> None:
    """A two-conductor request with missing data must not become a one-coil field."""
    r = jnp.linspace(1.0, 2.0, 9)
    z = jnp.linspace(-0.7, 0.7, 9)
    field = vacuum(
        r,
        z,
        jnp.array([2.2, radius]),
        jnp.array([0.9, height]),
        jnp.array([0.1, current]),
    )
    assert np.isnan(np.asarray(field)).all()


@pytest.mark.parametrize("vacuum", [vacuum_field, vacuum_field_si])
def test_invalid_batch_member_does_not_corrupt_valid_members(vacuum: Vacuum) -> None:
    """Batched evaluation preserves valid requests and refuses the missing current."""
    r = jnp.linspace(1.0, 2.0, 9)
    z = jnp.linspace(-0.7, 0.7, 9)
    coil_r, coil_z = jnp.array([2.2, 2.3]), jnp.array([0.9, 0.8])
    currents = jnp.array([[0.1, 0.2], [0.1, jnp.nan], [-0.1, 0.2]])
    evaluate = jax.jit(jax.vmap(lambda value: vacuum(r, z, coil_r, coil_z, value)))
    actual = np.asarray(evaluate(currents))
    assert np.isnan(actual[1]).all()
    for index in (0, 2):
        expected = np.asarray(vacuum(r, z, coil_r, coil_z, currents[index]))
        tolerance = float(16 * np.finfo(expected.dtype).eps)
        np.testing.assert_allclose(actual[index], expected, rtol=tolerance, atol=0.0)
        assert np.isfinite(actual[index]).all()


@pytest.mark.parametrize("vacuum", [vacuum_field, vacuum_field_si])
def test_zero_and_signed_currents_remain_valid(vacuum: Vacuum) -> None:
    """A measured zero is valid and signed current reverses its vacuum field."""
    r, z = jnp.linspace(1.0, 2.0, 9), jnp.linspace(-0.7, 0.7, 9)
    coil_r, coil_z = jnp.array([2.2]), jnp.array([0.9])
    positive = np.asarray(vacuum(r, z, coil_r, coil_z, jnp.array([0.1])))
    negative = np.asarray(vacuum(r, z, coil_r, coil_z, jnp.array([-0.1])))
    zero = np.asarray(vacuum(r, z, coil_r, coil_z, jnp.array([0.0])))
    assert np.isfinite(positive).all()
    np.testing.assert_array_equal(negative, -positive)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
