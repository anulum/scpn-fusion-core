# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real runtime-to-compiled-solver tests for neural equilibrium warm starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

cast(Any, jax.config).update("jax_enable_x64", True)

from scpn_fusion.core import (
    DeepONetPredictiveWarmStarter as PublicWarmStarter,
    DeepONetWarmStartResult as PublicWarmStartResult,
)
from scpn_fusion.core.deeponet_solver_warm_start import (
    DeepONetPredictiveWarmStarter,
    DeepONetWarmStartResult,
)
from scpn_fusion.core.jax_free_boundary_predictive import (
    DEFAULT_ANDERSON_DEPTH,
    DEFAULT_MIXING,
    DEFAULT_TOL,
    build_response_matrix,
)
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_compiled,
)

R_GRID = jnp.linspace(1.0, 2.5, 33)
Z_GRID = jnp.linspace(-1.4, 1.4, 33)
COIL_R = jnp.asarray([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
COIL_Z = jnp.asarray([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
COIL_I = jnp.asarray([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
COIL_NAMES = ("C1", "C2", "C3", "C4", "C5", "C6")
PSIN = jnp.linspace(0.0, 1.0, 6)
PPRIME = jnp.asarray([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
FFPRIME = jnp.asarray([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
IP_TARGET = 1.0e6
MANIFEST_SHA256 = "b" * 64
FEATURE_NAMES = (
    "plasma_current_target_a",
    *(f"coil_current_a.{name}" for name in COIL_NAMES),
    *(f"pprime_knot_{index}" for index in range(5)),
    *(f"ffprime_knot_{index}" for index in range(5)),
)


@pytest.fixture(scope="module")
def solver_case() -> dict[str, Any]:
    response_matrix, wall_idx, source_idx = build_response_matrix(R_GRID, Z_GRID)
    solved = solve_predictive_equilibrium_compiled(
        COIL_I,
        PPRIME,
        FFPRIME,
        R_GRID,
        Z_GRID,
        COIL_R,
        COIL_Z,
        PSIN,
        IP_TARGET,
        response_matrix,
        wall_idx,
        source_idx,
        n_iter=150,
        anderson_depth=DEFAULT_ANDERSON_DEPTH,
        mixing=DEFAULT_MIXING,
        return_iterations=True,
    )
    equilibrium, iterations = cast(tuple[jnp.ndarray, int], solved)
    equilibrium.block_until_ready()
    assert iterations < 150
    return {
        "response_matrix": response_matrix,
        "wall_idx": wall_idx,
        "source_idx": source_idx,
        "equilibrium": equilibrium,
        "iterations": iterations,
    }


def _write_seed_artifact(
    path: Path,
    field_mean: np.ndarray[Any, np.dtype[np.float64]],
    *,
    unstable: bool = False,
) -> None:
    grid_r, grid_z = np.meshgrid(np.asarray(R_GRID), np.asarray(Z_GRID), indexing="xy")
    coordinates = np.column_stack((grid_r.ravel(), grid_z.ravel()))
    branch_weight = np.zeros((len(FEATURE_NAMES), 1), dtype=np.float64)
    if unstable:
        branch_weight.fill(np.finfo(np.float64).max)
    np.savez(
        path,
        artifact_schema=np.asarray(["scpn-fusion.equilibrium-deeponet.v1"]),
        branch_n_layers=np.asarray([1], dtype=np.int64),
        branch_0_W=branch_weight,
        branch_0_b=np.zeros(1, dtype=np.float64),
        trunk_n_layers=np.asarray([1], dtype=np.int64),
        trunk_0_W=np.ones((2, 1), dtype=np.float64),
        trunk_0_b=np.zeros(1, dtype=np.float64),
        input_mean=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
        input_std=np.ones(len(FEATURE_NAMES), dtype=np.float64),
        coordinates_rz_m=coordinates,
        coordinate_mean=np.mean(coordinates, axis=0),
        coordinate_std=np.std(coordinates, axis=0),
        field_mean=field_mean.ravel(),
        field_scale=np.asarray([1.0]),
        basis_width=np.asarray([1], dtype=np.int64),
        grid_nh=np.asarray([len(Z_GRID)], dtype=np.int64),
        grid_nw=np.asarray([len(R_GRID)], dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
        dataset_manifest_sha256=np.asarray([MANIFEST_SHA256]),
    )


def _starter(path: Path) -> DeepONetPredictiveWarmStarter:
    return DeepONetPredictiveWarmStarter(path, prefer_rust=False)


def _solve(
    starter: DeepONetPredictiveWarmStarter,
    solver_case: dict[str, Any],
    **overrides: Any,
) -> DeepONetWarmStartResult:
    arguments: dict[str, Any] = {
        "machine_manifest_sha256": MANIFEST_SHA256,
        "coil_names": COIL_NAMES,
        "coil_i": COIL_I,
        "pprime_vals": PPRIME,
        "ffprime_vals": FFPRIME,
        "r_grid": R_GRID,
        "z_grid": Z_GRID,
        "coil_r": COIL_R,
        "coil_z": COIL_Z,
        "psin_knots": PSIN,
        "ip_target": IP_TARGET,
        "response_matrix": solver_case["response_matrix"],
        "wall_idx": solver_case["wall_idx"],
        "source_idx": solver_case["source_idx"],
        "n_iter": 150,
    }
    arguments.update(overrides)
    return starter.solve(**arguments)


def test_neural_seed_reaches_the_mechanistic_fixed_point(
    tmp_path: Path, solver_case: dict[str, Any]
) -> None:
    artifact = tmp_path / "exact_seed.npz"
    _write_seed_artifact(artifact, np.asarray(solver_case["equilibrium"]))
    starter = _starter(artifact)
    result = _solve(starter, solver_case)
    span = float(jnp.ptp(solver_case["equilibrium"]))
    relative_difference = (
        float(jnp.max(jnp.abs(result.equilibrium - solver_case["equilibrium"]))) / span
    )
    assert isinstance(result, DeepONetWarmStartResult)
    assert result.used_neural_seed is True
    assert result.fallback_reason is None
    assert result.iterations < 150
    assert result.solver_tolerance == DEFAULT_TOL
    assert result.neural_seed_weight == 1.0
    assert result.warm_separatrix_start == 100
    assert result.warm_separatrix_ramp == 20
    assert relative_difference < 1.0e-6
    assert starter.artifact_path == artifact
    assert len(starter.artifact_sha256) == 64
    assert PublicWarmStarter is DeepONetPredictiveWarmStarter
    assert PublicWarmStartResult is DeepONetWarmStartResult


def test_inference_failure_uses_real_cold_solver_fallback(
    tmp_path: Path, solver_case: dict[str, Any]
) -> None:
    artifact = tmp_path / "unstable_seed.npz"
    _write_seed_artifact(
        artifact, np.zeros_like(np.asarray(solver_case["equilibrium"])), unstable=True
    )
    result = _solve(_starter(artifact), solver_case)
    assert result.used_neural_seed is False
    assert result.fallback_reason == "neural_inference_failed"
    assert result.iterations == solver_case["iterations"]
    assert float(jnp.max(jnp.abs(result.equilibrium - solver_case["equilibrium"]))) == 0.0


def test_rejected_warm_seed_and_failed_cold_fallback_raise(
    tmp_path: Path, solver_case: dict[str, Any]
) -> None:
    artifact = tmp_path / "poor_seed.npz"
    _write_seed_artifact(
        artifact,
        np.full_like(np.asarray(solver_case["equilibrium"]), 1.0e9),
    )
    with pytest.raises(RuntimeError, match="cold-start fallback did not converge"):
        _solve(_starter(artifact), solver_case, n_iter=10)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"machine_manifest_sha256": "c" * 64}, "manifest digest"),
        ({"coil_names": ("D1", "D2", "D3", "D4", "D5", "D6")}, "feature order"),
        ({"r_grid": R_GRID.reshape(1, -1)}, "finite R/Z"),
        ({"z_grid": Z_GRID.at[0].set(jnp.nan)}, "finite R/Z"),
        ({"r_grid": R_GRID.at[-1].set(2.6)}, "coordinate grid"),
    ],
)
def test_wrong_machine_context_fails_before_inference(
    tmp_path: Path,
    solver_case: dict[str, Any],
    overrides: dict[str, Any],
    message: str,
) -> None:
    artifact = tmp_path / "exact_seed.npz"
    _write_seed_artifact(artifact, np.asarray(solver_case["equilibrium"]))
    with pytest.raises(ValueError, match=message):
        _solve(_starter(artifact), solver_case, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coil_i": jnp.ones((1, 6))}, "one-dimensional"),
        ({"ip_target": float("nan")}, "finite"),
        ({"coil_names": ("C1",) * 6}, "uniquely match"),
        ({"pprime_vals": PPRIME[:-1]}, "match at least two"),
        ({"ffprime_vals": FFPRIME.at[-1].set(1.0)}, "edge profile knots"),
        ({"neural_seed_weight": 0.0}, "neural_seed_weight"),
        ({"neural_seed_weight": 1.1}, "neural_seed_weight"),
        ({"neural_seed_weight": float("nan")}, "neural_seed_weight"),
        ({"warm_separatrix_start": -1}, "warm_separatrix_start"),
        ({"warm_separatrix_ramp": 0}, "warm_separatrix_ramp"),
    ],
)
def test_invalid_neural_control_contract_fails_before_solver(
    tmp_path: Path,
    solver_case: dict[str, Any],
    overrides: dict[str, Any],
    message: str,
) -> None:
    artifact = tmp_path / "exact_seed.npz"
    _write_seed_artifact(artifact, np.asarray(solver_case["equilibrium"]))
    with pytest.raises(ValueError, match=message):
        _solve(_starter(artifact), solver_case, **overrides)
