# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Traceable Runtime
# ──────────────────────────────────────────────────────────────────────
"""Optional JAX-traceable control-loop utilities with NumPy fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TraceableRuntimeSpec:
    """Configuration for reduced traceable first-order actuator dynamics."""

    dt_s: float = 1.0e-3
    tau_s: float = 5.0e-3
    gain: float = 1.0
    command_limit: float = 1.0


@dataclass(frozen=True)
class TraceableRuntimeResult:
    """Result of a traceable control-loop rollout."""

    state_history: FloatArray
    backend_used: str
    compiled: bool


def _validate_spec(spec: TraceableRuntimeSpec) -> None:
    if not np.isfinite(spec.dt_s) or spec.dt_s <= 0.0:
        raise ValueError("dt_s must be finite and > 0.")
    if not np.isfinite(spec.tau_s) or spec.tau_s <= 0.0:
        raise ValueError("tau_s must be finite and > 0.")
    if not np.isfinite(spec.gain):
        raise ValueError("gain must be finite.")
    if not np.isfinite(spec.command_limit) or spec.command_limit <= 0.0:
        raise ValueError("command_limit must be finite and > 0.")


def _validate_commands(commands: FloatArray) -> None:
    if commands.ndim != 1 or commands.size == 0:
        raise ValueError("commands must be a non-empty 1D array.")
    if not np.all(np.isfinite(commands)):
        raise ValueError("commands must contain only finite values.")


def _simulate_numpy(
    commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec
) -> FloatArray:
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    state = float(initial_state)
    out = np.empty_like(commands, dtype=np.float64)
    for i, cmd in enumerate(commands):
        cmd_clipped = float(np.clip(cmd, -spec.command_limit, spec.command_limit))
        state = state + alpha * ((spec.gain * cmd_clipped) - state)
        out[i] = state
    return out


def _simulate_jax(
    commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec
) -> FloatArray:
    if not _HAS_JAX:
        raise RuntimeError("JAX backend requested but JAX is not installed.")
    assert jnp is not None
    assert jax is not None

    cmd = jnp.asarray(commands, dtype=jnp.float64)
    alpha = jnp.asarray(spec.dt_s / (spec.tau_s + spec.dt_s), dtype=jnp.float64)
    gain = jnp.asarray(spec.gain, dtype=jnp.float64)
    limit = jnp.asarray(spec.command_limit, dtype=jnp.float64)

    def _step(state, u):
        u_clip = jnp.clip(u, -limit, limit)
        next_state = state + alpha * ((gain * u_clip) - state)
        return next_state, next_state

    @jax.jit
    def _rollout(x0, u):
        _, hist = jax.lax.scan(_step, x0, u)
        return hist

    hist = _rollout(jnp.asarray(initial_state, dtype=jnp.float64), cmd)
    return np.asarray(hist, dtype=np.float64)


def run_traceable_control_loop(
    commands: FloatArray,
    *,
    initial_state: float = 0.0,
    spec: TraceableRuntimeSpec | None = None,
    backend: str = "auto",
) -> TraceableRuntimeResult:
    """
    Run a reduced control loop suitable for optional JAX tracing/JIT.

    `backend` can be `auto`, `numpy`, or `jax`.
    """
    cmd_arr = np.asarray(commands, dtype=np.float64).reshape(-1)
    _validate_commands(cmd_arr)
    if not np.isfinite(initial_state):
        raise ValueError("initial_state must be finite.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)

    b = str(backend).strip().lower()
    if b not in {"auto", "numpy", "jax"}:
        raise ValueError("backend must be one of: auto, numpy, jax.")
    if b == "auto":
        b = "jax" if _HAS_JAX else "numpy"

    if b == "jax":
        return TraceableRuntimeResult(
            state_history=_simulate_jax(cmd_arr, float(initial_state), runtime_spec),
            backend_used="jax",
            compiled=True,
        )

    return TraceableRuntimeResult(
        state_history=_simulate_numpy(cmd_arr, float(initial_state), runtime_spec),
        backend_used="numpy",
        compiled=False,
    )

