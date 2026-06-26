# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RMF Phase-Lock Tests
"""Tests for the RMF phase-lock controller and command-line diagnostic."""

from __future__ import annotations

import json
import importlib
import runpy
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

import scpn_fusion.control.rmf_phase_lock as rmf_phase_lock
from scpn_fusion.control.rmf_phase_lock import (
    RMFPhaseLockConfig,
    RMFPhaseLockController,
    run_rmf_phase_lock_demo,
)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: RMFPhaseLockConfig(f_rmf_nom_hz=float("nan")), "finite"),
        (lambda: RMFPhaseLockConfig(f_rmf_nom_hz=0.0), "positive"),
        (lambda: RMFPhaseLockConfig(min_freq_hz=2.0, max_freq_hz=1.0), "ordered"),
        (lambda: RMFPhaseLockConfig(max_phase_error_rad=0.0), "phase_error"),
        (lambda: RMFPhaseLockConfig(n_neurons=-1), "n_neurons"),
    ],
)
def test_config_rejects_invalid_aot_bounds(
    constructor: Callable[[], RMFPhaseLockConfig], message: str
) -> None:
    """Configuration validation rejects non-finite and unsafe AOT bounds."""
    with pytest.raises(ValueError, match=message):
        constructor()


def test_public_controller_blocks_unsafe_phase_error_without_state_advance() -> None:
    """The Python controller mirrors the Rust AOT phase-error safety gate."""
    cfg = RMFPhaseLockConfig(max_phase_error_rad=0.1)
    ctrl = RMFPhaseLockController(cfg)
    phi_before = ctrl.phi_ant
    omega_before = ctrl.omega_rmf
    t_before = ctrl.t

    out = ctrl.step(np.pi / 2.0)

    assert out == phi_before
    assert ctrl.phi_ant == phi_before
    assert ctrl.omega_rmf == omega_before
    assert ctrl.t == t_before
    assert ctrl.safety_violations == 1


def test_public_controller_clamps_unsafe_frequency_without_nan() -> None:
    """Observed plasma-frequency jumps clamp to the configured AOT envelope."""
    cfg = RMFPhaseLockConfig(
        f_rmf_nom_hz=1.0e6,
        f_sampling_hz=20.0e6,
        k_p=0.0,
        n_neurons=0,
        min_freq_hz=0.8e6,
        max_freq_hz=1.2e6,
        max_phase_error_rad=np.pi,
    )
    ctrl = RMFPhaseLockController(cfg)
    ctrl.step(0.0)
    ctrl.step(float(np.mod(2.0 * np.pi * 2.0e6 * ctrl.dt, 2.0 * np.pi)))

    assert np.isfinite(ctrl.omega_rmf)
    assert np.isclose(ctrl.omega_rmf, 2.0 * np.pi * cfg.max_freq_hz)
    assert ctrl.safety_violations == 1


def test_public_controller_rejects_nonfinite_plasma_phase() -> None:
    """Non-finite plasma phase input fails closed without advancing state."""
    ctrl = RMFPhaseLockController(RMFPhaseLockConfig(n_neurons=0))

    out = ctrl.step(float("nan"))

    assert out == 0.0
    assert ctrl.t == 0.0
    assert ctrl.safety_violations == 1


def test_spiking_detector_branch_advances_public_horizon() -> None:
    """The non-zero neuron branch advances through the public horizon API."""
    cfg = RMFPhaseLockConfig(n_neurons=4, max_phase_error_rad=np.pi)
    ctrl = RMFPhaseLockController(cfg)
    plasma_phis = np.array([0.0, 0.05, 0.10], dtype=np.float64)

    out = ctrl.step_horizon(plasma_phis)

    assert out.shape == plasma_phis.shape
    assert len(ctrl.history["omega"]) == len(plasma_phis)


def test_jax_horizon_falls_back_to_numpy_when_jax_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical JAX wrapper returns NumPy output when JAX cannot import."""
    ctrl = RMFPhaseLockController(RMFPhaseLockConfig(n_neurons=0))

    def _raise_import_error(_name: str) -> object:
        raise ImportError("jax unavailable")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    out = ctrl.step_jax_horizon(np.array([0.0, 0.1], dtype=np.float64))

    assert isinstance(out, np.ndarray)


def test_jax_horizon_uses_imported_array_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The JAX wrapper delegates conversion when the array module is importable."""
    ctrl = RMFPhaseLockController(RMFPhaseLockConfig(n_neurons=0))
    adapter = SimpleNamespace(asarray=lambda value: ("jax-array", np.asarray(value).shape))
    monkeypatch.setattr(importlib, "import_module", lambda _name: adapter)

    out = ctrl.step_jax_horizon(np.array([0.0, 0.1], dtype=np.float64))

    assert out == ("jax-array", (2,))


def test_export_to_fpga_fails_closed() -> None:
    """FPGA export remains blocked until a real RTL generator and timing proof exist."""
    with pytest.raises(NotImplementedError, match="FPGA export is not implemented"):
        RMFPhaseLockController().export_to_fpga("unused")


def test_demo_rejects_invalid_inputs_and_returns_default_summary() -> None:
    """The public diagnostic validates inputs and returns finite summary metrics."""
    with pytest.raises(ValueError, match="horizon"):
        run_rmf_phase_lock_demo(horizon=0)
    with pytest.raises(ValueError, match="plasma_frequency_hz"):
        run_rmf_phase_lock_demo(horizon=1, plasma_frequency_hz=float("nan"))

    summary = run_rmf_phase_lock_demo(horizon=4)

    assert summary["cycles"] == 4
    assert np.isfinite(summary["mean_abs_phase_error"])


def test_main_emits_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The direct CLI entry point emits both text and JSON summaries."""
    assert rmf_phase_lock.main(["--horizon", "4"]) == 0
    text_out = capsys.readouterr().out
    assert "RMF software horizon evaluated: 4 cycles" in text_out

    assert rmf_phase_lock.main(["--horizon", "4", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cycles"] == 4


def test_module_main_executes_with_runpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level ``__main__`` guard delegates to the CLI main function."""
    monkeypatch.setattr(sys, "argv", ["rmf_phase_lock", "--horizon", "1", "--json"])
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as exc,
    ):
        runpy.run_module("scpn_fusion.control.rmf_phase_lock", run_name="__main__")

    assert exc.value.code == 0


def test_module_cli_emits_json_summary() -> None:
    """The module is executable with ``python -m`` and emits machine-readable metrics."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scpn_fusion.control.rmf_phase_lock",
            "--horizon",
            "32",
            "--plasma-frequency-hz",
            "1010000",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["cycles"] == 32
    assert payload["safety_violations"] == 0
    assert np.isfinite(payload["final_phase_rad"])
    assert np.isfinite(payload["final_omega_hz"])
