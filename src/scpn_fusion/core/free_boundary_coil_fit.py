# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — inverse coil-current fit to isoflux and null-point targets
"""Fit coil currents so SCPN's free-boundary equilibrium meets isoflux and null-point targets.

The objective mirrors the upstream FreeGSNKE ``Inverse_optimizer`` targets:

* pairwise differences of total poloidal flux between isoflux points, and
* the poloidal field ``B_R = -(1/R) dpsi/dZ`` and ``B_Z = (1/R) dpsi/dR`` at null points.

Flux at a point is the exact filament Green's-function coil flux plus a Keys bicubic
interpolation of the plasma flux on the solver grid. The fit is a Levenberg-Marquardt
iteration with Marquardt scaling, adaptive damping and a Tikhonov term on the step, and
an optional projection onto current limits. Every forward solve uses the predictive
free-boundary solver at a validated tolerance; a solve that does not converge is never
used: the trial step is rejected, or the fit stops with a stated reason.

The module needs float64 JAX and refuses to run without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.jax_free_boundary_gs import greens_psi_si, vacuum_field_si
from scpn_fusion.core.jax_free_boundary_predictive import solve_predictive_equilibrium

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]
StopReason = Literal[
    "objective_converged",
    "no_descent_step",
    "stalled",
    "max_outer_iterations",
    "start_not_converged",
    "jacobian_solve_not_converged",
]

KEYS_A = -0.5
DEFAULT_INNER_TOLERANCE = 1.0e-9
DEFAULT_INNER_ITERATIONS = 300
_DAMPING_ATTEMPTS = 12


def require_float64() -> None:
    """Raise unless JAX runs in float64, which the 1e-9 inner tolerance relies on."""
    read = cast(Callable[[str], object], jax.config.read)
    if not bool(read("jax_enable_x64")):
        raise RuntimeError(
            "coil-current fitting needs float64 JAX; set JAX_ENABLE_X64=1 or call "
            'jax.config.update("jax_enable_x64", True) before computing'
        )


def keys_cubic_weights(offset: jax.Array) -> jax.Array:
    """Return the four Keys cubic-convolution weights (a = -0.5) for a fractional offset.

    Parameters
    ----------
    offset
        Fractional position in ``[0, 1)`` between the second and third sample.

    Returns
    -------
    jax.Array
        Weights for the samples at relative positions ``-1, 0, 1, 2``.
    """

    def kernel(x: jax.Array) -> jax.Array:
        x = jnp.abs(x)
        near = (KEYS_A + 2.0) * x**3 - (KEYS_A + 3.0) * x**2 + 1.0
        far = KEYS_A * x**3 - 5.0 * KEYS_A * x**2 + 8.0 * KEYS_A * x - 4.0 * KEYS_A
        return jnp.where(x <= 1.0, near, jnp.where(x < 2.0, far, 0.0))

    return jnp.stack(
        [kernel(offset + 1.0), kernel(offset), kernel(offset - 1.0), kernel(offset - 2.0)]
    )


def interpolate_bicubic(
    field_zr: jax.Array, r_grid: jax.Array, z_grid: jax.Array, r: jax.Array, z: jax.Array
) -> jax.Array:
    """Interpolate a ``(nz, nr)`` field at ``(r, z)`` with Keys bicubic convolution.

    The 4x4 stencil is clamped to the grid interior, so points within one cell of the edge
    use the nearest full stencil. Uniform grid spacing is assumed.
    """
    fr = (r - r_grid[0]) / (r_grid[1] - r_grid[0])
    fz = (z - z_grid[0]) / (z_grid[1] - z_grid[0])
    ir = jnp.clip(jnp.floor(fr).astype(int), 1, r_grid.shape[0] - 3)
    iz = jnp.clip(jnp.floor(fz).astype(int), 1, z_grid.shape[0] - 3)
    block = jax.lax.dynamic_slice(field_zr, (iz - 1, ir - 1), (4, 4))
    return keys_cubic_weights(fz - iz) @ block @ keys_cubic_weights(fr - ir)


def _finite_vector(name: str, values: FloatArray, *, minimum_size: int = 1) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with at least {minimum_size} entries")
    return array


@dataclass(frozen=True)
class IsofluxTargets:
    """Isoflux points whose flux must agree pairwise, and points where the field must vanish."""

    isoflux_r: FloatArray
    isoflux_z: FloatArray
    null_r: FloatArray
    null_z: FloatArray

    def __post_init__(self) -> None:
        """Validate the point sets and freeze them as float arrays."""
        for name in ("isoflux_r", "isoflux_z"):
            object.__setattr__(
                self, name, _finite_vector(name, getattr(self, name), minimum_size=2)
            )
        for name in ("null_r", "null_z"):
            object.__setattr__(
                self, name, _finite_vector(name, getattr(self, name), minimum_size=0)
            )
        if self.isoflux_r.size != self.isoflux_z.size or self.null_r.size != self.null_z.size:
            raise ValueError("each point set needs as many R as Z coordinates")
        if np.any(self.isoflux_r <= 0.0) or np.any(self.null_r <= 0.0):
            raise ValueError("target points need R > 0")

    @property
    def pairs(self) -> IndexArray:
        """Every isoflux pair ``(i, j)`` with ``i < j``."""
        n = self.isoflux_r.size
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)

    @property
    def residual_count(self) -> int:
        """Number of residual entries: one per isoflux pair plus two per null point."""
        return int(self.pairs.shape[0] + 2 * self.null_r.size)


@dataclass(frozen=True)
class CoilFitProblem:
    """One inverse problem: the solver set-up, the targets and the controllable circuits.

    ``incidence`` maps circuit currents to filament currents (filaments x circuits).
    ``control`` marks the circuits the fit may change; the others keep ``fixed_currents``.
    ``tikhonov``, ``lower`` and ``upper`` have one entry per controlled circuit.
    """

    r_grid: FloatArray
    z_grid: FloatArray
    filament_r: FloatArray
    filament_z: FloatArray
    incidence: FloatArray
    psin_knots: FloatArray
    pprime: FloatArray
    ffprime: FloatArray
    ip_target: float
    response_matrix: jax.Array
    wall_idx: jax.Array
    source_idx: jax.Array
    targets: IsofluxTargets
    control: BoolArray
    fixed_currents: FloatArray
    tikhonov: FloatArray
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    inner_tolerance: float = DEFAULT_INNER_TOLERANCE
    inner_iterations: int = DEFAULT_INNER_ITERATIONS
    _residual: Callable[[jax.Array, jax.Array], jax.Array] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check every shape against the circuit and control counts."""
        incidence = np.asarray(self.incidence, dtype=np.float64)
        n_filaments, n_circuits = incidence.shape
        control = np.asarray(self.control, dtype=bool)
        if control.shape != (n_circuits,) or not control.any():
            raise ValueError("control must mark at least one of the incidence matrix's circuits")
        if np.asarray(self.filament_r).shape != (n_filaments,) or np.asarray(
            self.filament_z
        ).shape != (n_filaments,):
            raise ValueError("filament coordinates must match the incidence matrix rows")
        n_control = int(control.sum())
        fixed = _finite_vector("fixed_currents", self.fixed_currents)
        tikhonov = _finite_vector("tikhonov", self.tikhonov)
        if fixed.size != n_circuits or tikhonov.size != n_control or np.any(tikhonov < 0.0):
            raise ValueError(
                "fixed_currents needs one entry per circuit; tikhonov one >= 0 per control"
            )
        for name in ("lower", "upper"):
            bound = getattr(self, name)
            if bound is not None and np.asarray(bound).shape != (n_control,):
                raise ValueError(f"{name} needs one entry per controlled circuit")
        if (
            self.lower is not None
            and self.upper is not None
            and np.any(np.asarray(self.lower) > np.asarray(self.upper))
        ):
            raise ValueError("lower bounds must not exceed upper bounds")
        if not (self.inner_tolerance > 0.0 and self.inner_iterations >= 1):
            raise ValueError("inner_tolerance must be > 0 and inner_iterations >= 1")
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "fixed_currents", fixed)
        object.__setattr__(self, "tikhonov", tikhonov)

    @property
    def control_count(self) -> int:
        """Number of circuits the fit may change."""
        return int(self.control.sum())

    def circuit_currents(self, controls: FloatArray) -> FloatArray:
        """Return every circuit current with ``controls`` placed in the controlled circuits."""
        currents = self.fixed_currents.copy()
        currents[self.control] = np.asarray(controls, dtype=np.float64)
        return currents

    def project(self, controls: FloatArray) -> FloatArray:
        """Clip controls onto the current limits, where limits are given."""
        lower = -np.inf if self.lower is None else np.asarray(self.lower)
        upper = np.inf if self.upper is None else np.asarray(self.upper)
        return np.clip(np.asarray(controls, dtype=np.float64), lower, upper)

    def residual(self, psi_total: jax.Array, circuits: FloatArray) -> FloatArray:
        """Evaluate the target residual for a total flux map and circuit currents."""
        evaluate = self._residual
        if evaluate is None:
            evaluate = self._build_residual()
            object.__setattr__(self, "_residual", evaluate)
        return np.asarray(evaluate(psi_total, jnp.asarray(circuits)), dtype=np.float64)

    def _build_residual(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
        r_grid, z_grid = jnp.asarray(self.r_grid), jnp.asarray(self.z_grid)
        fil_r, fil_z = jnp.asarray(self.filament_r), jnp.asarray(self.filament_z)
        incidence = jnp.asarray(self.incidence)
        iso_r, iso_z = jnp.asarray(self.targets.isoflux_r), jnp.asarray(self.targets.isoflux_z)
        null_r, null_z = jnp.asarray(self.targets.null_r), jnp.asarray(self.targets.null_z)
        pairs = jnp.asarray(self.targets.pairs)

        def residual(psi_total: jax.Array, circuits: jax.Array) -> jax.Array:
            filament_currents = incidence @ circuits
            psi_plasma = psi_total - vacuum_field_si(
                r_grid, z_grid, fil_r, fil_z, filament_currents
            )

            def psi_at(rp: jax.Array, zp: jax.Array) -> jax.Array:
                coil = jnp.sum(greens_psi_si(rp, zp, fil_r, fil_z, filament_currents))
                return coil + interpolate_bicubic(psi_plasma, r_grid, z_grid, rp, zp)

            values = jax.vmap(psi_at)(iso_r, iso_z)
            dpsi_dr = jax.vmap(jax.grad(psi_at, argnums=0))(null_r, null_z)
            dpsi_dz = jax.vmap(jax.grad(psi_at, argnums=1))(null_r, null_z)
            return jnp.concatenate(
                [values[pairs[:, 0]] - values[pairs[:, 1]], -dpsi_dz / null_r, dpsi_dr / null_r]
            )

        return cast(Callable[[jax.Array, jax.Array], jax.Array], jax.jit(residual))


@dataclass(frozen=True)
class ForwardResult:
    """One forward solve: the flux map, its target residual and its convergence."""

    psi: jax.Array
    residual: FloatArray
    converged: bool
    iterations: int

    @property
    def objective(self) -> float:
        """Sum of squared residuals."""
        return float(self.residual @ self.residual)


def forward(
    problem: CoilFitProblem, controls: FloatArray, psi_init: jax.Array | None
) -> ForwardResult:
    """Solve the equilibrium for ``controls`` and evaluate the target residual."""
    require_float64()
    circuits = problem.circuit_currents(controls)
    flags: list[bool] = []
    psi = solve_predictive_equilibrium(
        jnp.asarray(problem.incidence @ circuits),
        jnp.asarray(problem.pprime),
        jnp.asarray(problem.ffprime),
        jnp.asarray(problem.r_grid),
        jnp.asarray(problem.z_grid),
        jnp.asarray(problem.filament_r),
        jnp.asarray(problem.filament_z),
        jnp.asarray(problem.psin_knots),
        float(problem.ip_target),
        problem.response_matrix,
        problem.wall_idx,
        problem.source_idx,
        psi_init=psi_init,
        n_iter=int(problem.inner_iterations),
        ip_ramp=1,
        tol=float(problem.inner_tolerance),
        decompose_coil_field=True,
        iteration_observer=lambda state: flags.append(bool(state.converged)),
    )
    return ForwardResult(
        psi=psi,
        residual=problem.residual(psi, circuits),
        converged=bool(flags and flags[-1]),
        iterations=len(flags),
    )


@dataclass(frozen=True)
class CoilFitResult:
    """Outcome of one fit, including why it stopped."""

    controls: FloatArray
    objective: float
    max_abs_residual: float
    objective_history: tuple[float, ...]
    stop_reason: StopReason
    outer_iterations: int
    jacobian_singular_values: tuple[float, ...]
    bounds_active: int
    continuation_steps: int
    psi: jax.Array | None


def _continue_from(
    problem: CoilFitProblem, reference: FloatArray, target: FloatArray, psi_init: jax.Array | None
) -> tuple[ForwardResult | None, FloatArray, int]:
    """Walk from a converged reference to ``target`` in converged steps, halving failed steps."""
    fraction, step, steps = 0.0, 0.1, 0
    state = forward(problem, reference, psi_init)
    if not state.converged:
        return None, reference, steps
    controls = reference
    while fraction < 1.0 and step >= 1.0 / 1024.0:
        trial_fraction = min(1.0, fraction + step)
        trial = reference + trial_fraction * (target - reference)
        result = forward(problem, trial, state.psi)
        steps += 1
        if result.converged:
            fraction, state, controls = trial_fraction, result, trial
        else:
            step /= 2.0
    return (state if fraction >= 1.0 else None), controls, steps


def _jacobian(
    problem: CoilFitProblem, controls: FloatArray, psi: jax.Array, relative_step: float
) -> FloatArray | None:
    columns = []
    for k in range(controls.size):
        h = max(abs(float(controls[k])) * relative_step, 1.0)
        e = np.zeros_like(controls)
        e[k] = h
        plus, minus = forward(problem, controls + e, psi), forward(problem, controls - e, psi)
        if not (plus.converged and minus.converged):
            return None
        columns.append((plus.residual - minus.residual) / (2.0 * h))
    return np.stack(columns, axis=1)


def fit_coil_currents(
    problem: CoilFitProblem,
    start: FloatArray,
    psi_init: jax.Array | None,
    *,
    continuation_reference: FloatArray | None = None,
    max_outer: int = 30,
    finite_difference_relative_step: float = 1.0e-4,
    objective_relative_tolerance: float = 1.0e-24,
    objective_absolute_tolerance: float = 1.0e-30,
    stall_relative_decrease: float = 1.0e-6,
    stall_iterations: int = 3,
) -> CoilFitResult:
    """Fit the controlled circuit currents from ``start`` by Levenberg-Marquardt.

    Parameters
    ----------
    problem
        The inverse problem.
    start
        Starting currents of the controlled circuits; projected onto the limits.
    psi_init
        Warm-start flux map for the first forward solve.
    continuation_reference
        Converged controls to walk from when the start does not converge directly.
    max_outer
        Maximum number of Levenberg-Marquardt iterations.
    finite_difference_relative_step
        Relative central-difference step per control, at least 1 A.
    objective_relative_tolerance, objective_absolute_tolerance
        The fit stops as converged once the objective falls below either the relative
        tolerance times the starting objective or the absolute tolerance.
    stall_relative_decrease, stall_iterations
        The fit stops as stalled after ``stall_iterations`` consecutive accepted steps that
        each lower the objective by at most ``stall_relative_decrease`` of its value.

    Returns
    -------
    CoilFitResult
        Final controls, objective, history and the reason the fit stopped.
    """
    require_float64()
    if max_outer < 1 or stall_iterations < 1:
        raise ValueError("max_outer and stall_iterations must be >= 1")
    controls = problem.project(_finite_vector("start", start))
    if controls.size != problem.control_count:
        raise ValueError("start needs one entry per controlled circuit")
    state = forward(problem, controls, psi_init)
    steps = 0
    if not state.converged and continuation_reference is not None:
        walked, controls, steps = _continue_from(
            problem,
            problem.project(np.asarray(continuation_reference, dtype=np.float64)),
            controls,
            psi_init,
        )
        state = walked if walked is not None else state
    if not state.converged:
        return _result(
            problem, controls, state, (state.objective,), "start_not_converged", 0, None, steps
        )

    history = [state.objective]
    regularisation = np.diag(problem.tikhonov)
    damping, stall, jacobian = 1.0e-3, 0, None
    stop: StopReason = "max_outer_iterations"
    outer = 0
    for outer in range(1, max_outer + 1):
        jacobian = _jacobian(problem, controls, state.psi, finite_difference_relative_step)
        if jacobian is None:
            stop = "jacobian_solve_not_converged"
            break
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ state.residual
        scale = np.diag(np.maximum(np.diag(normal), 1.0e-300))
        previous = state.objective
        accepted = False
        for _attempt in range(_DAMPING_ATTEMPTS):
            step = -np.linalg.solve(normal + damping * scale + regularisation, gradient)
            trial_controls = problem.project(controls + step)
            trial = forward(problem, trial_controls, state.psi)
            if trial.converged and trial.objective < state.objective:
                controls, state = trial_controls, trial
                damping = max(damping / 3.0, 1.0e-12)
                accepted = True
                break
            damping *= 10.0
        history.append(state.objective)
        if not accepted:
            stop = "no_descent_step"
            break
        if (
            state.objective <= objective_relative_tolerance * max(history[0], 1.0e-300)
            or state.objective <= objective_absolute_tolerance
        ):
            stop = "objective_converged"
            break
        stall = stall + 1 if previous - state.objective <= stall_relative_decrease * previous else 0
        if stall >= stall_iterations:
            stop = "stalled"
            break
    return _result(problem, controls, state, tuple(history), stop, outer, jacobian, steps)


def _result(
    problem: CoilFitProblem,
    controls: FloatArray,
    state: ForwardResult,
    history: tuple[float, ...],
    stop: StopReason,
    outer: int,
    jacobian: FloatArray | None,
    steps: int,
) -> CoilFitResult:
    singular = (
        tuple(float(v) for v in np.linalg.svd(jacobian, compute_uv=False))
        if jacobian is not None and np.all(np.isfinite(jacobian))
        else ()
    )
    lower = -np.inf if problem.lower is None else np.asarray(problem.lower)
    upper = np.inf if problem.upper is None else np.asarray(problem.upper)
    return CoilFitResult(
        controls=controls,
        objective=state.objective,
        max_abs_residual=float(np.max(np.abs(state.residual))) if state.residual.size else 0.0,
        objective_history=history,
        stop_reason=stop,
        outer_iterations=outer,
        jacobian_singular_values=singular,
        bounds_active=int(np.sum((controls <= lower) | (controls >= upper))),
        continuation_steps=steps,
        psi=state.psi if state.converged else None,
    )
