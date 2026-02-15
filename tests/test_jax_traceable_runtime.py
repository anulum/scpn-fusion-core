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
    monkeypatch.setattr(runtime, "_HAS_TORCH", False)
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


def test_traceable_runtime_torchscript_backend_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_TORCH", False)
    commands = np.asarray([0.1, 0.2], dtype=np.float64)
    with pytest.raises(RuntimeError, match="TorchScript backend requested"):
        runtime.run_traceable_control_loop(commands, backend="torchscript")


@pytest.mark.skipif(not runtime._HAS_TORCH, reason="torch unavailable")
def test_traceable_runtime_torchscript_matches_numpy() -> None:
    commands = np.asarray([0.3, -0.4, 0.5, 1.5], dtype=np.float64)
    spec = runtime.TraceableRuntimeSpec(
        dt_s=0.0025, tau_s=0.010, gain=1.5, command_limit=0.9
    )
    ref = runtime.run_traceable_control_loop(
        commands, initial_state=0.1, spec=spec, backend="numpy"
    )
    out = runtime.run_traceable_control_loop(
        commands, initial_state=0.1, spec=spec, backend="torchscript"
    )
    assert out.backend_used == "torchscript"
    assert out.compiled is True
    np.testing.assert_allclose(out.state_history, ref.state_history, rtol=0.0, atol=1e-8)


def test_traceable_runtime_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        runtime.run_traceable_control_loop(np.asarray([], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        runtime.run_traceable_control_loop(np.asarray([0.0, np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="backend must be one of"):
        runtime.run_traceable_control_loop(
            np.asarray([0.0], dtype=np.float64), backend="flux"
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


def test_traceable_runtime_batch_numpy_deterministic() -> None:
    commands = np.asarray(
        [
            [0.2, 0.1, -0.1, 0.0],
            [0.0, 0.6, -0.4, 0.3],
        ],
        dtype=np.float64,
    )
    spec = runtime.TraceableRuntimeSpec(dt_s=0.002, tau_s=0.01, gain=1.2, command_limit=1.0)
    a = runtime.run_traceable_control_batch(
        commands,
        initial_state=np.asarray([0.1, -0.2], dtype=np.float64),
        spec=spec,
        backend="numpy",
    )
    b = runtime.run_traceable_control_batch(
        commands,
        initial_state=np.asarray([0.1, -0.2], dtype=np.float64),
        spec=spec,
        backend="numpy",
    )
    assert a.backend_used == "numpy"
    assert a.compiled is False
    np.testing.assert_allclose(a.state_history, b.state_history, rtol=0.0, atol=0.0)
    assert a.state_history.shape == commands.shape


def test_traceable_runtime_batch_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        runtime.run_traceable_control_batch(np.asarray([0.1, 0.2], dtype=np.float64))


def test_traceable_runtime_batch_rejects_bad_initial_state_length() -> None:
    commands = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    with pytest.raises(ValueError, match="batch dimension"):
        runtime.run_traceable_control_batch(
            commands,
            initial_state=np.asarray([0.0], dtype=np.float64),
            backend="numpy",
        )


def test_traceable_runtime_batch_auto_falls_back_to_numpy_without_jax_or_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_JAX", False)
    monkeypatch.setattr(runtime, "_HAS_TORCH", False)
    commands = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    out = runtime.run_traceable_control_batch(commands, backend="auto")
    assert out.backend_used == "numpy"
    assert out.compiled is False


def test_traceable_runtime_batch_torchscript_backend_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_TORCH", False)
    commands = np.asarray([[0.1, 0.2]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="TorchScript backend requested"):
        runtime.run_traceable_control_batch(commands, backend="torchscript")


@pytest.mark.skipif(not runtime._HAS_TORCH, reason="torch unavailable")
def test_traceable_runtime_batch_torchscript_matches_numpy() -> None:
    commands = np.asarray([[0.3, -0.4, 0.5], [0.2, 1.1, -1.5]], dtype=np.float64)
    spec = runtime.TraceableRuntimeSpec(dt_s=0.003, tau_s=0.010, gain=1.1, command_limit=0.8)
    ref = runtime.run_traceable_control_batch(commands, spec=spec, backend="numpy")
    out = runtime.run_traceable_control_batch(commands, spec=spec, backend="torchscript")
    assert out.backend_used == "torchscript"
    assert out.compiled is True
    np.testing.assert_allclose(out.state_history, ref.state_history, rtol=0.0, atol=1e-8)


def test_available_traceable_backends_contains_numpy() -> None:
    names = runtime.available_traceable_backends()
    assert "numpy" in names


def test_validate_traceable_backend_parity_invalid_args() -> None:
    with pytest.raises(ValueError, match="steps"):
        runtime.validate_traceable_backend_parity(steps=0)
    with pytest.raises(ValueError, match="batch"):
        runtime.validate_traceable_backend_parity(batch=0)
    with pytest.raises(ValueError, match="atol"):
        runtime.validate_traceable_backend_parity(atol=-1.0)


def test_validate_traceable_backend_parity_reports_numpy() -> None:
    reports = runtime.validate_traceable_backend_parity(
        steps=12,
        batch=3,
        seed=7,
        atol=1e-10,
    )
    assert "numpy" in reports
    np_report = reports["numpy"]
    assert np_report.backend == "numpy"
    assert np_report.single_within_tol is True
    assert np_report.batch_within_tol is True
    assert np_report.single_max_abs_err == pytest.approx(0.0, abs=0.0)
    assert np_report.batch_max_abs_err == pytest.approx(0.0, abs=0.0)


def test_validate_traceable_backend_parity_includes_only_available_backends() -> None:
    reports = runtime.validate_traceable_backend_parity(steps=8, batch=2, seed=11, atol=1e-8)
    expected = set(runtime.available_traceable_backends())
    assert set(reports.keys()) == expected
