# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Training Mathematics
"""JAX branch-trunk model, physical objective, and AdamW update."""

from __future__ import annotations

from typing import TypeAlias, cast

import jax
import jax.numpy as jnp
from jax import random, value_and_grad

Layer: TypeAlias = dict[str, jax.Array]
Params: TypeAlias = list[Layer]
OperatorParams: TypeAlias = dict[str, Params]


def init_network(
    key: jax.Array,
    *,
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
) -> Params:
    """Create deterministic float32 He-initialised SiLU parameters.

    Parameters
    ----------
    key : jax.Array
        JAX random key that fixes every generated weight.
    input_dim : int
        Number of input features.
    hidden_sizes : tuple[int, ...]
        Width of each hidden layer.
    output_dim : int
        Width of the final linear layer.

    Returns
    -------
    Params
        Ordered dense layers with ``W`` and ``b`` arrays.
    """
    dimensions = (input_dim, *hidden_sizes, output_dim)
    params: Params = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True):
        key, subkey = random.split(key)
        params.append(
            {
                "W": random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32)
                * jnp.sqrt(2.0 / fan_in),
                "b": jnp.zeros(fan_out, dtype=jnp.float32),
            }
        )
    return params


def network_forward(params: Params, values: jax.Array) -> jax.Array:
    """Evaluate a dense SiLU network over a row batch.

    Parameters
    ----------
    params : Params
        Ordered dense layers returned by :func:`init_network`.
    values : jax.Array
        Input matrix with rows as samples or coordinates.

    Returns
    -------
    jax.Array
        Final linear-layer values for every row.
    """
    activation = values
    for index, layer in enumerate(params):
        activation = activation @ layer["W"] + layer["b"]
        if index + 1 < len(params):
            activation = jax.nn.silu(activation)
    return activation


def operator_forward(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
) -> jax.Array:
    """Evaluate branch coefficients against learned coordinate basis values.

    Parameters
    ----------
    params : OperatorParams
        Branch and trunk network parameters.
    features : jax.Array
        Normalised causal controls with shape ``(shots, n_features)``.
    coordinates : jax.Array
        Normalised ``(R, Z)`` coordinates with shape ``(points, 2)``.

    Returns
    -------
    jax.Array
        Normalised field residuals with shape ``(shots, points)``.

    Notes
    -----
    The scaled branch-trunk inner product follows Lu et al. (2021),
    DOI: 10.1038/s42256-021-00302-5.
    """
    branch = network_forward(params["branch"], features)
    trunk = network_forward(params["trunk"], coordinates)
    return branch @ trunk.T / jnp.sqrt(branch.shape[1])


def relative_field_objective(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
) -> jax.Array:
    """Return field-normalised coordinate-sampled physical error.

    Parameters
    ----------
    params : OperatorParams
        Branch and trunk network parameters.
    features : jax.Array
        Normalised causal controls for the shot minibatch.
    coordinates : jax.Array
        Normalised coordinate minibatch.
    targets : jax.Array
        Normalised physical field targets for each shot-coordinate pair.
    sample_weights : jax.Array
        One training-only relative-field weight per shot.

    Returns
    -------
    jax.Array
        Scalar mean weighted squared error.
    """
    squared_error = jnp.square(operator_forward(params, features, coordinates) - targets)
    return jnp.mean(sample_weights * jnp.mean(squared_error, axis=1))


@jax.jit
def adamw_step(
    params: OperatorParams,
    first_moment: OperatorParams,
    second_moment: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    step: int,
) -> tuple[OperatorParams, OperatorParams, OperatorParams, jax.Array]:
    """Run one deterministic AdamW step with global gradient clipping.

    Parameters
    ----------
    params, first_moment, second_moment : OperatorParams
        Current parameters and AdamW moment trees.
    features, coordinates, targets, sample_weights : jax.Array
        Physical minibatch consumed by :func:`relative_field_objective`.
    learning_rate : float
        Positive AdamW step size.
    weight_decay : float
        Decoupled non-negative parameter decay.
    gradient_clip : float
        Positive global gradient-norm ceiling.
    step : int
        One-based absolute optimiser step used for bias correction.

    Returns
    -------
    tuple[OperatorParams, OperatorParams, OperatorParams, jax.Array]
        Updated parameters, first moments, second moments, and pre-update loss.
    """
    loss, gradients = value_and_grad(relative_field_objective)(
        params, features, coordinates, targets, sample_weights
    )
    gradient_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(gradients))
    )
    gradient_scale = jnp.minimum(1.0, gradient_clip / jnp.maximum(gradient_norm, 1.0e-12))
    gradients = cast(
        OperatorParams,
        jax.tree_util.tree_map(lambda gradient: gradient * gradient_scale, gradients),
    )
    beta_1, beta_2, epsilon = 0.9, 0.999, 1.0e-8
    first_moment = cast(
        OperatorParams,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_1 * moment + (1.0 - beta_1) * gradient,
            first_moment,
            gradients,
        ),
    )
    second_moment = cast(
        OperatorParams,
        jax.tree_util.tree_map(
            lambda moment, gradient: beta_2 * moment + (1.0 - beta_2) * gradient * gradient,
            second_moment,
            gradients,
        ),
    )
    corrected_first = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_1**step), first_moment
    )
    corrected_second = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - beta_2**step), second_moment
    )
    params = cast(
        OperatorParams,
        jax.tree_util.tree_map(
            lambda parameter, moment_1, moment_2: (
                parameter
                - learning_rate
                * (moment_1 / (jnp.sqrt(moment_2) + epsilon) + weight_decay * parameter)
            ),
            params,
            corrected_first,
            corrected_second,
        ),
    )
    return params, first_moment, second_moment, loss


@jax.jit
def validation_objective(
    params: OperatorParams,
    features: jax.Array,
    coordinates: jax.Array,
    targets: jax.Array,
    sample_weights: jax.Array,
) -> jax.Array:
    """Evaluate the frozen validation probe used only for model selection.

    Parameters
    ----------
    params : OperatorParams
        Candidate branch and trunk parameters.
    features, coordinates, targets, sample_weights : jax.Array
        Fixed validation-only arrays matching the physical objective.

    Returns
    -------
    jax.Array
        Scalar validation objective used to select a step.
    """
    return relative_field_objective(params, features, coordinates, targets, sample_weights)


__all__ = [
    "OperatorParams",
    "Params",
    "adamw_step",
    "init_network",
    "network_forward",
    "operator_forward",
    "relative_field_objective",
    "validation_objective",
]
