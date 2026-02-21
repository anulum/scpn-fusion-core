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

    # Enable float64 for high-precision control analysis
    jax.config.update("jax_enable_x64", True)

    _HAS_JAX = True
except Exception:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


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


@dataclass(frozen=True)
class TraceableRuntimeBatchResult:
    """Result of batched traceable control-loop rollout."""

    state_history: FloatArray
    backend_used: str
    compiled: bool


@dataclass(frozen=True)
class TraceableBackendParityReport:
    """Parity metrics against NumPy reference backend."""

    backend: str
    single_max_abs_err: float
    batch_max_abs_err: float
    single_within_tol: bool
    batch_within_tol: bool


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


def _validate_batch_commands(commands: FloatArray) -> None:
    if commands.ndim != 2 or commands.shape[0] == 0 or commands.shape[1] == 0:
        raise ValueError("commands must have shape (batch, steps) with non-zero sizes.")
    if not np.all(np.isfinite(commands)):
        raise ValueError("commands must contain only finite values.")


def _resolve_backend(backend: str) -> str:
    b = str(backend).strip().lower()
    if b not in {"auto", "numpy", "jax", "torchscript"}:
        raise ValueError("backend must be one of: auto, numpy, jax, torchscript.")
    if b == "auto":
        if _HAS_JAX:
            return "jax"
        if _HAS_TORCH:
            return "torchscript"
        return "numpy"
    return b


def available_traceable_backends() -> list[str]:
    """Return available runtime backends on this machine."""
    out = ["numpy"]
    if _HAS_JAX:
        out.append("jax")
    if _HAS_TORCH:
        out.append("torchscript")
    return out


def _resolve_backend_set(backends: list[str] | tuple[str, ...] | None) -> list[str]:
    available = available_traceable_backends()
    if backends is None:
        return available
    out: list[str] = []
    seen: set[str] = set()
    for raw in backends:
        name = str(raw).strip().lower()
        if name not in {"numpy", "jax", "torchscript"}:
            raise ValueError(
                f"Unsupported backend '{raw}'. Allowed: numpy, jax, torchscript."
            )
        if name not in available:
            raise ValueError(f"Requested backend '{name}' is not available on this host.")
        if name not in seen:
            out.append(name)
            seen.add(name)
    if not out:
        raise ValueError("backends must contain at least one backend when provided.")
    return out


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


if _HAS_TORCH:

    @torch.jit.script
    def _torchscript_rollout(
        cmd: torch.Tensor,
        initial_state: float,
        alpha: float,
        gain: float,
        limit: float,
    ) -> torch.Tensor:
        n = cmd.numel()
        out = torch.empty((n,), dtype=cmd.dtype, device=cmd.device)
        state = torch.tensor(initial_state, dtype=cmd.dtype, device=cmd.device)
        for i in range(n):
            u = torch.clamp(cmd[i], -limit, limit)
            state = state + alpha * ((gain * u) - state)
            out[i] = state
        return out

    @torch.jit.script
    def _torchscript_rollout_batch(
        cmd: torch.Tensor,
        initial_state: torch.Tensor,
        alpha: float,
        gain: float,
        limit: float,
    ) -> torch.Tensor:
        batch = cmd.size(0)
        steps = cmd.size(1)
        out = torch.empty((batch, steps), dtype=cmd.dtype, device=cmd.device)
        state = initial_state.clone()
        for t in range(steps):
            u = torch.clamp(cmd[:, t], -limit, limit)
            state = state + alpha * ((gain * u) - state)
            out[:, t] = state
        return out

else:
    _torchscript_rollout = None
    _torchscript_rollout_batch = None


def _simulate_torchscript(
    commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec
) -> FloatArray:
    if not _HAS_TORCH or _torchscript_rollout is None:
        raise RuntimeError("TorchScript backend requested but torch is not installed.")
    assert torch is not None

    cmd = torch.as_tensor(commands, dtype=torch.float64)
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    hist = _torchscript_rollout(
        cmd,
        float(initial_state),
        alpha,
        float(spec.gain),
        float(spec.command_limit),
    )
    return np.asarray(hist.detach().cpu().numpy(), dtype=np.float64)


def _simulate_numpy_batch(
    commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec
) -> FloatArray:
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    state = np.asarray(initial_state, dtype=np.float64).copy()
    out = np.empty_like(commands, dtype=np.float64)
    for t in range(commands.shape[1]):
        u = np.clip(commands[:, t], -spec.command_limit, spec.command_limit)
        state = state + alpha * ((spec.gain * u) - state)
        out[:, t] = state
    return out


def _simulate_jax_batch(
    commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec
) -> FloatArray:
    if not _HAS_JAX:
        raise RuntimeError("JAX backend requested but JAX is not installed.")
    assert jnp is not None
    assert jax is not None

    cmd = jnp.asarray(commands, dtype=jnp.float64)
    x0 = jnp.asarray(initial_state, dtype=jnp.float64)
    alpha = jnp.asarray(spec.dt_s / (spec.tau_s + spec.dt_s), dtype=jnp.float64)
    gain = jnp.asarray(spec.gain, dtype=jnp.float64)
    limit = jnp.asarray(spec.command_limit, dtype=jnp.float64)

    def _step(state, u_t):
        u_clip = jnp.clip(u_t, -limit, limit)
        next_state = state + alpha * ((gain * u_clip) - state)
        return next_state, next_state

    @jax.jit
    def _rollout_batch(batch_x0, batch_u):
        _, hist_tb = jax.lax.scan(_step, batch_x0, jnp.swapaxes(batch_u, 0, 1))
        return jnp.swapaxes(hist_tb, 0, 1)

    hist = _rollout_batch(x0, cmd)
    return np.asarray(hist, dtype=np.float64)


def _simulate_torchscript_batch(
    commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec
) -> FloatArray:
    if not _HAS_TORCH or _torchscript_rollout_batch is None:
        raise RuntimeError("TorchScript backend requested but torch is not installed.")
    assert torch is not None

    cmd = torch.as_tensor(commands, dtype=torch.float64)
    x0 = torch.as_tensor(initial_state, dtype=torch.float64)
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    hist = _torchscript_rollout_batch(
        cmd,
        x0,
        alpha,
        float(spec.gain),
        float(spec.command_limit),
    )
    return np.asarray(hist.detach().cpu().numpy(), dtype=np.float64)


def run_traceable_control_loop(
    commands: FloatArray,
    *,
    initial_state: float = 0.0,
    spec: TraceableRuntimeSpec | None = None,
    backend: str = "auto",
) -> TraceableRuntimeResult:
    """
    Run a reduced control loop suitable for optional JAX tracing/JIT.

    `backend` can be `auto`, `numpy`, `jax`, or `torchscript`.
    """
    cmd_arr = np.asarray(commands, dtype=np.float64).reshape(-1)
    _validate_commands(cmd_arr)
    if not np.isfinite(initial_state):
        raise ValueError("initial_state must be finite.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)

    b = _resolve_backend(backend)

    if b == "jax":
        return TraceableRuntimeResult(
            state_history=_simulate_jax(cmd_arr, float(initial_state), runtime_spec),
            backend_used="jax",
            compiled=True,
        )

    if b == "torchscript":
        return TraceableRuntimeResult(
            state_history=_simulate_torchscript(
                cmd_arr, float(initial_state), runtime_spec
            ),
            backend_used="torchscript",
            compiled=True,
        )

    return TraceableRuntimeResult(
        state_history=_simulate_numpy(cmd_arr, float(initial_state), runtime_spec),
        backend_used="numpy",
        compiled=False,
    )


def run_traceable_control_batch(
    commands: FloatArray,
    *,
    initial_state: FloatArray | float | None = None,
    spec: TraceableRuntimeSpec | None = None,
    backend: str = "auto",
) -> TraceableRuntimeBatchResult:
    """
    Run batched reduced control loops with optional JAX/TorchScript backends.

    `commands` shape: (batch, steps)
    """
    cmd_arr = np.asarray(commands, dtype=np.float64)
    _validate_batch_commands(cmd_arr)

    batch = int(cmd_arr.shape[0])
    if initial_state is None:
        x0 = np.zeros(batch, dtype=np.float64)
    else:
        arr = np.asarray(initial_state, dtype=np.float64)
        if arr.ndim == 0:
            x0 = np.full(batch, float(arr), dtype=np.float64)
        else:
            x0 = arr.reshape(-1)
        if x0.size != batch:
            raise ValueError("initial_state length must match commands batch dimension.")
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must contain only finite values.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)
    b = _resolve_backend(backend)

    if b == "jax":
        return TraceableRuntimeBatchResult(
            state_history=_simulate_jax_batch(cmd_arr, x0, runtime_spec),
            backend_used="jax",
            compiled=True,
        )
    if b == "torchscript":
        return TraceableRuntimeBatchResult(
            state_history=_simulate_torchscript_batch(cmd_arr, x0, runtime_spec),
            backend_used="torchscript",
            compiled=True,
        )
    return TraceableRuntimeBatchResult(
        state_history=_simulate_numpy_batch(cmd_arr, x0, runtime_spec),
        backend_used="numpy",
        compiled=False,
    )


def validate_traceable_backend_parity(
    *,
    steps: int = 64,
    batch: int = 8,
    seed: int = 42,
    spec: TraceableRuntimeSpec | None = None,
    atol: float = 1e-8,
    backends: list[str] | tuple[str, ...] | None = None,
) -> dict[str, TraceableBackendParityReport]:
    """
    Compare available compiled backends to NumPy for single and batch rollouts.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if batch <= 0:
        raise ValueError("batch must be > 0.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and >= 0.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)

    rng = np.random.default_rng(int(seed))
    single_cmd = np.asarray(rng.normal(0.0, 1.0, size=steps), dtype=np.float64)
    batch_cmd = np.asarray(rng.normal(0.0, 1.0, size=(batch, steps)), dtype=np.float64)
    batch_x0 = np.asarray(rng.normal(0.0, 0.2, size=batch), dtype=np.float64)
    x0 = float(rng.normal(0.0, 0.2))

    ref_single = run_traceable_control_loop(
        single_cmd, initial_state=x0, spec=runtime_spec, backend="numpy"
    ).state_history
    ref_batch = run_traceable_control_batch(
        batch_cmd, initial_state=batch_x0, spec=runtime_spec, backend="numpy"
    ).state_history

    reports: dict[str, TraceableBackendParityReport] = {}
    backend_list = _resolve_backend_set(backends)
    for backend in backend_list:
        out_single = run_traceable_control_loop(
            single_cmd, initial_state=x0, spec=runtime_spec, backend=backend
        ).state_history
        out_batch = run_traceable_control_batch(
            batch_cmd, initial_state=batch_x0, spec=runtime_spec, backend=backend
        ).state_history

        s_err = float(np.max(np.abs(out_single - ref_single)))
        b_err = float(np.max(np.abs(out_batch - ref_batch)))
        reports[backend] = TraceableBackendParityReport(
            backend=backend,
            single_max_abs_err=s_err,
            batch_max_abs_err=b_err,
            single_within_tol=bool(s_err <= atol),
            batch_within_tol=bool(b_err <= atol),
        )
    return reports
