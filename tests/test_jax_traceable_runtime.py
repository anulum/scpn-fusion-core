# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Traceable Runtime Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.control.jax_traceable_runtime as runtime


def test_traceable_runtime_numpy_is_deterministic() -> None:
    commands = np.asarray([0.5, 1.2, -0.7, 0.1, 0.0], dtype=np.float64)
    spec = runtime.TraceableRuntimeSpec(
        dt_s=0.002, tau_s=0.010, gain=2.0, command_limit=1.0
    )
    a = runtime.run_traceable_control_loop(
        commands,
        initial_state=0.25,
        spec=spec,
        backend="numpy",
    )
    b = runtime.run_traceable_control_loop(
        commands,
        initial_state=0.25,
        spec=spec,
        backend="numpy",
    )
    assert a.backend_used == "numpy"
    assert a.compiled is False
    np.testing.assert_allclose(a.state_history, b.state_history, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(a.state_history))


def test_traceable_runtime_auto_falls_back_to_numpy_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_JAX", False)
    commands = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    out = runtime.run_traceable_control_loop(commands, backend="auto")
    assert out.backend_used == "numpy"
    assert out.compiled is False


def test_traceable_runtime_jax_backend_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_JAX", False)
    commands = np.asarray([0.1, 0.2], dtype=np.float64)
    with pytest.raises(RuntimeError, match="JAX backend requested"):
        runtime.run_traceable_control_loop(commands, backend="jax")


def test_traceable_runtime_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        runtime.run_traceable_control_loop(np.asarray([], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        runtime.run_traceable_control_loop(np.asarray([0.0, np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="backend must be one of"):
        runtime.run_traceable_control_loop(
            np.asarray([0.0], dtype=np.float64), backend="torchscript"
        )


def test_traceable_runtime_rejects_invalid_spec() -> None:
    commands = np.asarray([0.1, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="dt_s"):
        runtime.run_traceable_control_loop(
            commands,
            spec=runtime.TraceableRuntimeSpec(dt_s=0.0),
        )
    with pytest.raises(ValueError, match="tau_s"):
        runtime.run_traceable_control_loop(
            commands,
            spec=runtime.TraceableRuntimeSpec(tau_s=-1.0),
        )
    with pytest.raises(ValueError, match="command_limit"):
        runtime.run_traceable_control_loop(
            commands,
            spec=runtime.TraceableRuntimeSpec(command_limit=0.0),
        )
