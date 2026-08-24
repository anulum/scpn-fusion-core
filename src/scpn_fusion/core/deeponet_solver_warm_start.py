# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Predictive-Solver Warm Start
"""Validated neural seeding for the compiled predictive equilibrium solver."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator
from scpn_fusion.core.jax_free_boundary_gs import MU0_SI, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import (
    DEFAULT_ANDERSON_DEPTH,
    DEFAULT_CUTOFF_WIDTH,
    DEFAULT_IP_RAMP,
    DEFAULT_MIXING,
    DEFAULT_N_ITER,
    DEFAULT_SEPARATRIX_RAMP,
    DEFAULT_SEPARATRIX_START,
    DEFAULT_TOL,
)
from scpn_fusion.core.jax_predictive_forward_compiled import solve_predictive_equilibrium_compiled
from scpn_fusion.io.safe_loaders import checked_np_load

FloatArray: TypeAlias = NDArray[np.float64]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeepONetWarmStartResult:
    """One solver-accepted equilibrium and its neural-seed disposition.

    The returned field always comes from the mechanistic solver. A false
    ``used_neural_seed`` means inference or warm convergence failed and the
    solve completed through the genuine vacuum-field cold path.

    Custom topology-continuation settings can converge to a different valid
    fixed-point basin. The recorded start/ramp values make that disposition
    explicit; only the canonical defaults carry the measured branch-equivalence
    evidence documented for this adapter.
    """

    equilibrium: jnp.ndarray
    iterations: int
    used_neural_seed: bool
    neural_backend: str
    fallback_reason: str | None
    solver_tolerance: float
    neural_seed_weight: float
    warm_separatrix_start: int
    warm_separatrix_ramp: int


def _feature_contract(coil_names: Sequence[str], knot_count: int) -> tuple[str, ...]:
    return (
        "plasma_current_target_a",
        *(f"coil_current_a.{name}" for name in coil_names),
        *(f"pprime_knot_{index}" for index in range(knot_count - 1)),
        *(f"ffprime_knot_{index}" for index in range(knot_count - 1)),
    )


def _causal_feature_row(
    *,
    ip_target: float,
    coil_i: jnp.ndarray,
    pprime_vals: jnp.ndarray,
    ffprime_vals: jnp.ndarray,
    psin_knots: jnp.ndarray,
    coil_names: Sequence[str],
) -> tuple[FloatArray, tuple[str, ...]]:
    coil = np.asarray(coil_i, dtype=np.float64)
    pprime = np.asarray(pprime_vals, dtype=np.float64)
    ffprime = np.asarray(ffprime_vals, dtype=np.float64)
    knots = np.asarray(psin_knots, dtype=np.float64)
    arrays = (coil, pprime, ffprime, knots)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("DeepONet warm-start controls must be one-dimensional")
    if not np.isfinite(ip_target) or any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("DeepONet warm-start controls must be finite")
    if len(coil) != len(coil_names) or len(set(coil_names)) != len(coil_names):
        raise ValueError("DeepONet coil names must uniquely match the current vector")
    if len(knots) < 2 or len(pprime) != len(knots) or len(ffprime) != len(knots):
        raise ValueError("DeepONet profile values must match at least two psi-normal knots")
    if pprime[-1] != 0.0 or ffprime[-1] != 0.0:
        raise ValueError("DeepONet omitted edge profile knots must be exactly zero")
    feature_names = _feature_contract(coil_names, len(knots))
    row = np.concatenate(
        (
            np.asarray([ip_target], dtype=np.float64),
            coil,
            pprime[:-1],
            ffprime[:-1],
        )
    )
    return np.asarray(row, dtype=np.float64), feature_names


class DeepONetPredictiveWarmStarter:
    """Bind one authenticated DeepONet artifact to predictive solver calls.

    Parameters
    ----------
    artifact_path : str | Path
        Pickle-free machine-conditioned DeepONet artifact.
    prefer_rust : bool, optional
        Prefer the bit-parity checked Rust inference kernel when available.

    Notes
    -----
    The neural model supplies only the initial poloidal-flux field. The
    compiled mechanistic solver owns fixed-point convergence and the returned
    equilibrium. Machine digest, causal feature order, and physical R/Z grid
    must exactly match the artifact contract before inference begins.
    """

    def __init__(self, artifact_path: str | Path, *, prefer_rust: bool = True) -> None:
        """Load the candidate and its immutable machine-coordinate contract."""
        path = Path(artifact_path)
        runtime = DeepONetEquilibriumAccelerator(prefer_rust=prefer_rust)
        runtime.load_weights(path)
        with checked_np_load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates_rz_m"], dtype=np.float64).copy()
        coordinates.setflags(write=False)
        self._runtime = runtime
        self._coordinates = coordinates
        self.artifact_path = path
        self.artifact_sha256 = _sha256_file(path)

    def _validate_machine_context(
        self,
        *,
        manifest_sha256: str,
        feature_names: Sequence[str],
        grid_r_m: jnp.ndarray,
        grid_z_m: jnp.ndarray,
    ) -> None:
        if manifest_sha256 != self._runtime.machine_manifest_sha256:
            raise ValueError("DeepONet machine manifest digest does not match the solver context")
        if tuple(feature_names) != self._runtime.feature_names:
            raise ValueError("DeepONet feature order does not match the solver context")
        r = np.asarray(grid_r_m, dtype=np.float64)
        z = np.asarray(grid_z_m, dtype=np.float64)
        if r.ndim != 1 or z.ndim != 1 or not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)):
            raise ValueError("DeepONet solver grid must contain finite R/Z vectors")
        grid_r, grid_z = np.meshgrid(r, z, indexing="xy")
        supplied = np.column_stack((grid_r.ravel(), grid_z.ravel()))
        if not np.array_equal(supplied, self._coordinates):
            raise ValueError("DeepONet coordinate grid does not match the solver context")

    def solve(
        self,
        *,
        machine_manifest_sha256: str,
        coil_names: Sequence[str],
        coil_i: jnp.ndarray,
        pprime_vals: jnp.ndarray,
        ffprime_vals: jnp.ndarray,
        r_grid: jnp.ndarray,
        z_grid: jnp.ndarray,
        coil_r: jnp.ndarray,
        coil_z: jnp.ndarray,
        psin_knots: jnp.ndarray,
        ip_target: float,
        response_matrix: jnp.ndarray,
        wall_idx: jnp.ndarray,
        source_idx: jnp.ndarray,
        fixed_support_weights: jnp.ndarray | None = None,
        neural_seed_weight: float = 1.0,
        warm_separatrix_start: int = DEFAULT_SEPARATRIX_START,
        warm_separatrix_ramp: int = DEFAULT_SEPARATRIX_RAMP,
        n_iter: int = DEFAULT_N_ITER,
        anderson_depth: int = DEFAULT_ANDERSON_DEPTH,
        mixing: float = DEFAULT_MIXING,
        ip_ramp: int = DEFAULT_IP_RAMP,
        cutoff_width: float = DEFAULT_CUTOFF_WIDTH,
        tol: float = DEFAULT_TOL,
        mu0: float = MU0_SI,
        use_mg_preconditioner: bool = True,
        inner_solver: str = "bicgstab",
        inner_cycles: int = 3,
        anderson_solver: str = "lstsq",
    ) -> DeepONetWarmStartResult:
        """Solve from a machine-bound neural seed with cold-start fallback.

        Returns
        -------
        DeepONetWarmStartResult
            Finite mechanistic equilibrium, iteration count, backend, and seed
            disposition.

        Raises
        ------
        RuntimeError
            If the cold-start fallback cannot converge to a finite field.
        ValueError
            If machine identity, feature order, grid, controls, or solver
            settings violate their declared contracts.

        Notes
        -----
        The default warm continuation uses the solver's complete canonical
        schedule. Shorter start/ramp settings are research controls: residual
        convergence alone does not prove equivalence to the canonical cold
        branch.
        """
        feature_row, feature_names = _causal_feature_row(
            ip_target=ip_target,
            coil_i=coil_i,
            pprime_vals=pprime_vals,
            ffprime_vals=ffprime_vals,
            psin_knots=psin_knots,
            coil_names=coil_names,
        )
        if not np.isfinite(neural_seed_weight) or not 0.0 < neural_seed_weight <= 1.0:
            raise ValueError("DeepONet neural_seed_weight must be finite and lie in (0, 1]")
        if warm_separatrix_start < 0 or warm_separatrix_ramp < 1:
            raise ValueError(
                "DeepONet warm_separatrix_start must be >= 0 and warm_separatrix_ramp must be >= 1"
            )
        self._validate_machine_context(
            manifest_sha256=machine_manifest_sha256,
            feature_names=feature_names,
            grid_r_m=r_grid,
            grid_z_m=z_grid,
        )

        def solve_mechanistic(
            seed: jnp.ndarray | None,
            *,
            ramp: int,
            continuation: bool | None,
        ) -> tuple[jnp.ndarray, int]:
            result = solve_predictive_equilibrium_compiled(
                coil_i,
                pprime_vals,
                ffprime_vals,
                r_grid,
                z_grid,
                coil_r,
                coil_z,
                psin_knots,
                ip_target,
                response_matrix,
                wall_idx,
                source_idx,
                psi_init=seed,
                n_iter=n_iter,
                anderson_depth=anderson_depth,
                mixing=mixing,
                ip_ramp=ramp,
                cutoff_width=cutoff_width,
                tol=tol,
                mu0=mu0,
                use_mg_preconditioner=use_mg_preconditioner,
                inner_solver=inner_solver,
                inner_cycles=inner_cycles,
                anderson_solver=anderson_solver,
                return_iterations=True,
                fixed_support_weights=fixed_support_weights,
                use_separatrix_continuation=continuation,
                separatrix_start=(
                    warm_separatrix_start if continuation is True else DEFAULT_SEPARATRIX_START
                ),
                separatrix_ramp=(
                    warm_separatrix_ramp if continuation is True else DEFAULT_SEPARATRIX_RAMP
                ),
            )
            return cast(tuple[jnp.ndarray, int], result)

        fallback_reason: str | None = None
        try:
            neural_prediction = jnp.asarray(self._runtime.predict(feature_row))
        except RuntimeError:
            fallback_reason = "neural_inference_failed"
        else:
            vacuum_seed = vacuum_field_si(r_grid, z_grid, coil_r, coil_z, coil_i, mu0)
            neural_seed = vacuum_seed + neural_seed_weight * (neural_prediction - vacuum_seed)
            warm_psi, warm_iterations = solve_mechanistic(
                neural_seed,
                ramp=1,
                continuation=True,
            )
            warm_is_finite = bool(np.all(np.isfinite(np.asarray(warm_psi))))
            if warm_iterations < n_iter and warm_is_finite:
                return DeepONetWarmStartResult(
                    equilibrium=warm_psi,
                    iterations=warm_iterations,
                    used_neural_seed=True,
                    neural_backend=self._runtime.backend,
                    fallback_reason=None,
                    solver_tolerance=tol,
                    neural_seed_weight=neural_seed_weight,
                    warm_separatrix_start=warm_separatrix_start,
                    warm_separatrix_ramp=warm_separatrix_ramp,
                )
            fallback_reason = (
                "warm_iteration_cap" if warm_iterations >= n_iter else "warm_non_finite"
            )

        cold_psi, cold_iterations = solve_mechanistic(None, ramp=ip_ramp, continuation=None)
        cold_is_finite = bool(np.all(np.isfinite(np.asarray(cold_psi))))
        if cold_iterations >= n_iter or not cold_is_finite:
            raise RuntimeError(
                "DeepONet warm start and cold-start fallback did not converge to a finite field"
            )
        return DeepONetWarmStartResult(
            equilibrium=cold_psi,
            iterations=cold_iterations,
            used_neural_seed=False,
            neural_backend=self._runtime.backend,
            fallback_reason=fallback_reason,
            solver_tolerance=tol,
            neural_seed_weight=neural_seed_weight,
            warm_separatrix_start=warm_separatrix_start,
            warm_separatrix_ramp=warm_separatrix_ramp,
        )


__all__ = ["DeepONetPredictiveWarmStarter", "DeepONetWarmStartResult"]
