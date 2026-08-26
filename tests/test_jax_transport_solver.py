# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Transport Solver Tests

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import j0

import scpn_fusion.core as fusion_core
from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.core.jax_solvers import crank_nicolson_step
from scpn_fusion.core.jax_transport_solver import (
    simulate_scenario_jax,
    transport_step_checked,
    transport_step_jax,
)

FloatArray = NDArray[np.float64]
J01 = 2.4048255576957728


def _case(nodes: int = 129) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rho = np.linspace(0.0, 1.0, nodes, dtype=np.float64)
    temperature = np.asarray(0.1 + 0.9 * j0(J01 * rho), dtype=np.float64)
    diffusivity = np.ones(nodes, dtype=np.float64)
    source = np.zeros(nodes, dtype=np.float64)
    return rho, temperature, diffusivity, source


def _numpy_step(
    temperature: FloatArray,
    diffusivity: FloatArray,
    source: FloatArray,
    rho: FloatArray,
    dt: float,
) -> FloatArray:
    return crank_nicolson_step(
        temperature,
        diffusivity,
        source,
        rho,
        float(rho[1] - rho[0]),
        dt,
        T_edge=0.1,
        use_jax=False,
    )


def test_jax_transport_matches_canonical_cn_and_analytic_mode() -> None:
    """The direct differentiable path retains canonical CN accuracy."""
    rho, initial, diffusivity, source = _case()
    te_jax = jnp.asarray(initial)
    ti_jax = jnp.asarray(initial)
    for _ in range(10):
        te_jax, ti_jax = transport_step_jax(
            te_jax,
            ti_jax,
            jnp.asarray(diffusivity),
            jnp.asarray(diffusivity),
            jnp.asarray(source),
            jnp.asarray(source),
            jnp.asarray(rho),
            1.0e-3,
        )
    te_result = np.asarray(te_jax.block_until_ready(), dtype=np.float64)
    ti_result = np.asarray(ti_jax.block_until_ready(), dtype=np.float64)

    reference = initial.copy()
    for _ in range(10):
        reference = _numpy_step(reference, diffusivity, source, rho, 1.0e-3)
    exact = 0.1 + 0.9 * j0(J01 * rho) * math.exp(-(J01**2) * 0.01)

    np.testing.assert_allclose(te_result, reference, rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(ti_result, te_result)
    assert float(np.sqrt(np.mean((te_result - exact) ** 2))) <= 2.2829306504541543e-6
    assert te_result[-1] == 0.1
    assert bool(cast(Any, jax.config).read("jax_enable_x64")) is True


def test_public_scenario_auto_uses_evidence_backed_numpy_preference() -> None:
    """The automatic runtime avoids the measured small-grid JAX slowdown."""
    rho, initial, diffusivity, source = _case(65)
    source_history = np.broadcast_to(source, (3, source.size)).copy()
    result_te, result_ti = fusion_core.simulate_transport_scenario(
        initial,
        initial,
        diffusivity,
        diffusivity,
        source_history,
        source_history,
        rho,
        1.0e-3,
    )
    reference = initial.copy()
    expected = []
    for source_now in source_history:
        reference = _numpy_step(reference, diffusivity, source_now, rho, 1.0e-3)
        expected.append(reference)

    assert multi.dispatch_tier("transport_cn_rollout") == "numpy"
    np.testing.assert_array_equal(result_te, np.asarray(expected))
    np.testing.assert_array_equal(result_ti, result_te)


def test_public_scenario_explicit_jax_uses_registered_provider() -> None:
    """An explicit JAX request traverses the real provider without fallback."""
    rho, initial, diffusivity, source = _case(65)
    source_history = np.broadcast_to(source, (3, source.size)).copy()
    result_te, result_ti = fusion_core.simulate_transport_scenario(
        initial,
        initial,
        diffusivity,
        diffusivity,
        source_history,
        source_history,
        rho,
        1.0e-3,
        backend="jax",
    )
    reference = initial.copy()
    expected = []
    for source_now in source_history:
        reference = _numpy_step(reference, diffusivity, source_now, rho, 1.0e-3)
        expected.append(reference)

    np.testing.assert_allclose(result_te, np.asarray(expected), rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(result_ti, result_te)


def test_explicit_jax_request_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable explicit JAX request never substitutes NumPy silently."""
    rho, initial, diffusivity, source = _case(65)
    source_history = np.broadcast_to(source, (2, source.size)).copy()
    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.JAX, False)
    with multi._registry_lock:
        multi._dispatch_cache.pop("transport_cn_rollout", None)
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            fusion_core.simulate_transport_scenario(
                initial,
                initial,
                diffusivity,
                diffusivity,
                source_history,
                source_history,
                rho,
                1.0e-3,
                backend="jax",
            )
    finally:
        with multi._registry_lock:
            multi._dispatch_cache.pop("transport_cn_rollout", None)


@pytest.mark.parametrize(
    ("history_transform", "backend", "message"),
    [
        (lambda history: history[0], "auto", "source_e_history must have shape"),
        (lambda history: history[:0], "auto", "source_e_history must have shape"),
        (lambda history: history[:, :-1], "auto", "radial profile length"),
        (lambda history: history, "cuda", "backend must be one of"),
    ],
)
def test_public_scenario_rejects_invalid_history_or_backend(
    history_transform: Callable[[FloatArray], FloatArray],
    backend: str,
    message: str,
) -> None:
    """The runtime boundary rejects malformed histories and unknown tiers."""
    rho, initial, diffusivity, source = _case(17)
    source_history = np.broadcast_to(source, (2, source.size)).copy()
    transformed = history_transform(source_history)
    with pytest.raises(ValueError, match=message):
        fusion_core.simulate_transport_scenario(
            initial,
            initial,
            diffusivity,
            diffusivity,
            transformed,
            transformed,
            rho,
            1.0e-3,
            backend=backend,
        )


def test_public_scenario_rejects_mismatched_or_nonfinite_source_histories() -> None:
    """Both source histories must have identical finite shape and values."""
    rho, initial, diffusivity, source = _case(17)
    source_history = np.broadcast_to(source, (2, source.size)).copy()
    with pytest.raises(ValueError, match="source_i_history must match"):
        fusion_core.simulate_transport_scenario(
            initial,
            initial,
            diffusivity,
            diffusivity,
            source_history,
            source_history[:1],
            rho,
            1.0e-3,
        )

    nonfinite = source_history.copy()
    nonfinite[1, 3] = np.inf
    with pytest.raises(ValueError, match="only finite"):
        fusion_core.simulate_transport_scenario(
            initial,
            initial,
            diffusivity,
            diffusivity,
            nonfinite,
            source_history,
            rho,
            1.0e-3,
        )


def test_source_gradient_matches_central_finite_difference() -> None:
    """Autodiff through the canonical JAX solve agrees with NumPy replay."""
    rho, initial, diffusivity, _ = _case(65)
    source_shape = np.asarray(1.0 - rho**2, dtype=np.float64)
    rho_jax = jnp.asarray(rho)
    initial_jax = jnp.asarray(initial)
    diffusivity_jax = jnp.asarray(diffusivity)
    source_shape_jax = jnp.asarray(source_shape)

    def jax_cost(amplitude: jax.Array) -> jax.Array:
        source = amplitude * source_shape_jax
        result, _ = transport_step_jax(
            initial_jax,
            initial_jax,
            diffusivity_jax,
            diffusivity_jax,
            source,
            source,
            rho_jax,
            1.0e-3,
        )
        return jnp.mean(result[:-1])

    def numpy_cost(amplitude: float) -> float:
        result = _numpy_step(
            initial,
            diffusivity,
            amplitude * source_shape,
            rho,
            1.0e-3,
        )
        return float(np.mean(result[:-1]))

    autodiff = float(jax.grad(jax_cost)(jnp.asarray(0.2)).block_until_ready())
    epsilon = 1.0e-5
    finite_difference = (numpy_cost(0.2 + epsilon) - numpy_cost(0.2 - epsilon)) / (2.0 * epsilon)
    relative_error = abs(autodiff - finite_difference) / max(abs(finite_difference), 1.0e-15)

    assert math.isfinite(autodiff)
    assert autodiff > 0.0
    assert relative_error <= 0.01


def test_scan_rollout_matches_repeated_canonical_steps() -> None:
    """The multi-step JAX scan consumes explicit sources without hidden scaling."""
    rho, initial, diffusivity, _ = _case(33)
    source_shape = 1.0 - rho**2
    amplitudes = np.asarray([0.0, 0.1, 0.25, 0.4], dtype=np.float64)
    source_history = amplitudes[:, None] * source_shape[None, :]
    te_history, ti_history = simulate_scenario_jax(
        jnp.asarray(initial),
        jnp.asarray(initial),
        jnp.asarray(diffusivity),
        jnp.asarray(diffusivity),
        jnp.asarray(source_history),
        jnp.asarray(source_history),
        jnp.asarray(rho),
        1.0e-3,
    )
    te_result = np.asarray(te_history.block_until_ready(), dtype=np.float64)
    ti_result = np.asarray(ti_history.block_until_ready(), dtype=np.float64)

    reference = initial.copy()
    expected = []
    for source in source_history:
        reference = _numpy_step(reference, diffusivity, source, rho, 1.0e-3)
        expected.append(reference)

    np.testing.assert_allclose(te_result, np.asarray(expected), rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(ti_result, te_result)


def _valid_checked_arguments() -> list[Any]:
    rho, initial, diffusivity, source = _case(17)
    return [
        initial.copy(),
        initial.copy(),
        diffusivity.copy(),
        diffusivity.copy(),
        source.copy(),
        source.copy(),
        rho.copy(),
        1.0e-3,
    ]


def _mutate_te_rank(arguments: list[Any]) -> None:
    arguments[0] = np.ones((2, 8), dtype=np.float64)


def _mutate_chi_rank(arguments: list[Any]) -> None:
    arguments[2] = np.ones((1, 17), dtype=np.float64)


def _mutate_length(arguments: list[Any]) -> None:
    arguments[1] = np.ones(16, dtype=np.float64)


def _mutate_nonfinite(arguments: list[Any]) -> None:
    source = cast(FloatArray, arguments[4]).copy()
    source[3] = np.nan
    arguments[4] = source


def _mutate_temperature(arguments: list[Any]) -> None:
    temperature = cast(FloatArray, arguments[0]).copy()
    temperature[2] = 0.0
    arguments[0] = temperature


def _mutate_diffusivity(arguments: list[Any]) -> None:
    diffusivity = cast(FloatArray, arguments[2]).copy()
    diffusivity[2] = -0.1
    arguments[2] = diffusivity


def _mutate_monotonicity(arguments: list[Any]) -> None:
    rho = cast(FloatArray, arguments[6]).copy()
    rho[4] = rho[3]
    arguments[6] = rho


def _mutate_spacing(arguments: list[Any]) -> None:
    rho = cast(FloatArray, arguments[6]).copy()
    rho[4] += 1.0e-3
    arguments[6] = rho


def _mutate_dt(arguments: list[Any]) -> None:
    arguments[7] = 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_mutate_te_rank, "te must be one-dimensional"),
        (_mutate_chi_rank, "chi_e must be one-dimensional"),
        (_mutate_length, "ti must have length"),
        (_mutate_nonfinite, "s_heat_e must contain only finite"),
        (_mutate_temperature, "temperature profiles"),
        (_mutate_diffusivity, "diffusivity profiles"),
        (_mutate_monotonicity, "strictly increasing"),
        (_mutate_spacing, "uniformly spaced"),
        (_mutate_dt, "dt must be finite"),
    ],
)
def test_checked_transport_rejects_invalid_profiles(
    mutate: Callable[[list[Any]], None],
    message: str,
) -> None:
    """The checked public boundary rejects malformed physical inputs."""
    arguments = _valid_checked_arguments()
    mutate(arguments)
    with pytest.raises(ValueError, match=message):
        transport_step_checked(*arguments)


@pytest.mark.parametrize(("edge", "message"), [(0.0, "t_edge_e"), (math.nan, "t_edge_i")])
def test_checked_transport_rejects_invalid_edges(edge: float, message: str) -> None:
    """Non-positive or non-finite edge values fail before JAX execution."""
    arguments = _valid_checked_arguments()
    with pytest.raises(ValueError, match=message):
        if message == "t_edge_e":
            transport_step_checked(*arguments, t_edge_e=edge)
        else:
            transport_step_checked(*arguments, t_edge_i=edge)


def test_checked_transport_executes_valid_host_inputs() -> None:
    """The checked JAX boundary validates and advances a real profile."""
    rho, initial, diffusivity, source = _case(17)
    checked_te, checked_ti = transport_step_checked(
        initial,
        initial,
        diffusivity,
        diffusivity,
        source,
        source,
        rho,
        1.0e-3,
    )
    direct_te, direct_ti = transport_step_jax(
        jnp.asarray(initial),
        jnp.asarray(initial),
        jnp.asarray(diffusivity),
        jnp.asarray(diffusivity),
        jnp.asarray(source),
        jnp.asarray(source),
        jnp.asarray(rho),
        1.0e-3,
    )
    np.testing.assert_array_equal(np.asarray(checked_te), np.asarray(direct_te))
    np.testing.assert_array_equal(np.asarray(checked_ti), np.asarray(direct_ti))
