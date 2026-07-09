# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RMF Phase-Lock Control Simulation Lane
"""
Deterministic high-frequency RMF phase-locking using Stochastic Computing and JAX-Scan.

This module provides software simulation evidence only. FPGA export is
fail-closed until a real RTL generator and timing validation are implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
import importlib
import sys
from dataclasses import dataclass
from typing import Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── RMF Phase-Lock Configuration ─────────────────────────────────────


@dataclass
class RMFPhaseLockConfig:
    """Configuration for the RMF phase-lock loop."""

    f_rmf_nom_hz: float = 1.0e6  # 1 MHz nominal RMF frequency
    f_sampling_hz: float = 10.0e6  # simulation sampling frequency
    k_p: float = 0.5  # Phase detector gain
    k_d: float = 0.01  # Frequency damping
    n_neurons: int = 128  # Population size for stochastic PLL
    bits: int = 16  # reserved fixed-point width for future export
    min_freq_hz: float = 1.0e5  # fail-closed lower AOT frequency bound
    max_freq_hz: float = 5.0e6  # fail-closed upper AOT frequency bound
    max_phase_error_rad: float = float(np.pi / 2.0)  # fail-closed phase-error bound

    def __post_init__(self) -> None:
        """Validate finite positive timing and AOT safety bounds."""
        numeric_fields = (
            self.f_rmf_nom_hz,
            self.f_sampling_hz,
            self.k_p,
            self.k_d,
            self.min_freq_hz,
            self.max_freq_hz,
            self.max_phase_error_rad,
        )
        if not all(np.isfinite(value) for value in numeric_fields):
            raise ValueError("RMF phase-lock configuration values must be finite.")
        if self.f_rmf_nom_hz <= 0.0 or self.f_sampling_hz <= 0.0:
            raise ValueError("RMF nominal and sampling frequencies must be positive.")
        if self.min_freq_hz <= 0.0 or self.max_freq_hz <= self.min_freq_hz:
            raise ValueError("RMF AOT frequency bounds must be positive and ordered.")
        if not (0.0 < self.max_phase_error_rad <= np.pi):
            raise ValueError("RMF max_phase_error_rad must be in (0, pi].")
        if self.n_neurons < 0 or self.bits < 1:
            raise ValueError("RMF n_neurons must be non-negative and bits must be positive.")


class RMFPhaseLockSummary(TypedDict):
    """Machine-readable summary emitted by the RMF phase-lock diagnostic CLI."""

    cycles: int
    final_phase_rad: float
    final_omega_hz: float
    safety_violations: int
    mean_abs_phase_error: float


def _wrapped_phase_delta(delta: float) -> float:
    """Return signed phase delta in [-pi, pi)."""
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


class _SpikingPhaseDetector:
    """Reduced deterministic spiking phase detector matching the Rust lane."""

    def __init__(self, n_neurons: int, dt: float) -> None:
        tau_mem = 0.05e-3
        n = max(1, int(n_neurons))
        self.v_pos = np.zeros(n, dtype=np.float64)
        self.v_neg = np.zeros(n, dtype=np.float64)
        self.v_threshold = 0.2
        self.alpha = dt / tau_mem

    def step(self, error_signal: float) -> float:
        """Advance the detector and return the signed spike-rate proxy."""
        i_scale = 8.0
        i_bias = 0.05
        input_pos = max(error_signal, 0.0) * i_scale + i_bias
        input_neg = max(-error_signal, 0.0) * i_scale + i_bias

        self.v_pos += self.alpha * (-self.v_pos + input_pos)
        spikes_pos = self.v_pos >= self.v_threshold
        self.v_pos[spikes_pos] = 0.0

        self.v_neg += self.alpha * (-self.v_neg + input_neg)
        spikes_neg = self.v_neg >= self.v_threshold
        self.v_neg[spikes_neg] = 0.0

        return float(
            (np.count_nonzero(spikes_pos) - np.count_nonzero(spikes_neg)) / len(self.v_pos)
        )


class RMFPhaseLockController:
    """Software RMF phase-lock controller for JAX horizon simulations."""

    def __init__(self, config: RMFPhaseLockConfig | None = None, seed: int = 42) -> None:
        _ = seed
        self.cfg = RMFPhaseLockConfig() if config is None else config
        self.dt = 1.0 / self.cfg.f_sampling_hz
        self.omega_nom = 2.0 * np.pi * self.cfg.f_rmf_nom_hz

        # State
        self.phi_ant = 0.0
        self.omega_rmf = self.omega_nom
        self.omega_bias = 0.0
        self._last_phi_plasma: float | None = None
        self.t = 0.0
        self.safety_violations = 0
        self._detector = (
            _SpikingPhaseDetector(self.cfg.n_neurons, self.dt) if self.cfg.n_neurons > 0 else None
        )

        self.history: dict[str, list[float]] = {
            "t": [],
            "phi_ant": [],
            "phi_plasma": [],
            "omega": [],
        }

    def step(self, plasma_phi: float) -> float:
        """Advance one deterministic software phase-lock cycle."""
        plasma_phi = float(plasma_phi)
        if not np.isfinite(plasma_phi):
            self.safety_violations += 1
            return self.phi_ant

        raw_error = float(np.sin(self.phi_ant - plasma_phi))
        if abs(raw_error) > self.cfg.max_phase_error_rad:
            self.safety_violations += 1
            return self.phi_ant

        phase_error = self._detector.step(raw_error) if self._detector is not None else raw_error
        observed_bias = (
            _wrapped_phase_delta(plasma_phi - self._last_phi_plasma) / self.dt - self.omega_nom
            if self._last_phi_plasma is not None
            else self.omega_bias
        )
        self.omega_bias = observed_bias - self.cfg.k_p * phase_error * self.dt
        new_omega = self.omega_nom + self.omega_bias
        min_omega = 2.0 * np.pi * self.cfg.min_freq_hz
        max_omega = 2.0 * np.pi * self.cfg.max_freq_hz
        if new_omega < min_omega or new_omega > max_omega:
            self.safety_violations += 1
            self.omega_rmf = float(np.clip(new_omega, min_omega, max_omega))
            self.omega_bias = self.omega_rmf - self.omega_nom
        else:
            self.omega_rmf = new_omega
        self.phi_ant = float((self.phi_ant + self.omega_rmf * self.dt) % (2.0 * np.pi))
        self.t += self.dt
        self._last_phi_plasma = plasma_phi
        self.history["t"].append(self.t)
        self.history["phi_ant"].append(self.phi_ant)
        self.history["phi_plasma"].append(plasma_phi)
        self.history["omega"].append(float(self.omega_rmf))
        return self.phi_ant

    def step_horizon(self, plasma_phis: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a control horizon through the deterministic software PLL."""
        phis = np.asarray(plasma_phis, dtype=np.float64)
        out = np.empty_like(phis, dtype=np.float64)
        for idx, plasma_phi in enumerate(phis):
            out[idx] = self.step(float(plasma_phi))
        return out

    def step_jax_horizon(self, plasma_phis: object) -> object:
        """
        Evaluate a control horizon and return a JAX array when JAX is available.

        The deterministic NumPy path owns the control-law contract. This wrapper
        preserves the historical API without importing JAX at module import time.
        """
        out = self.step_horizon(np.asarray(plasma_phis, dtype=np.float64))
        try:
            jnp = importlib.import_module("jax.numpy")
        except ImportError:
            return out
        return jnp.asarray(out)

    def export_to_fpga(self, out_path: str) -> None:
        """Fail closed until a real RTL generator and timing proof exist."""
        _ = out_path
        raise NotImplementedError(
            "RMF FPGA export is not implemented; step_jax_horizon is software simulation evidence only"
        )


def run_rmf_phase_lock_demo(
    *,
    horizon: int = 10_000,
    plasma_frequency_hz: float | None = None,
    config: RMFPhaseLockConfig | None = None,
) -> RMFPhaseLockSummary:
    """Run the deterministic RMF phase-lock diagnostic horizon.

    Parameters
    ----------
    horizon:
        Number of controller cycles to evaluate.
    plasma_frequency_hz:
        Plasma phase-rotation frequency. Defaults to the configured RMF
        nominal frequency.
    config:
        Optional controller configuration.

    Returns
    -------
    RMFPhaseLockSummary
        Final phase, final frequency, safety-violation count, and mean absolute
        phase error for the evaluated horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    cfg = RMFPhaseLockConfig() if config is None else config
    ctrl = RMFPhaseLockController(cfg)
    plasma_hz = cfg.f_rmf_nom_hz if plasma_frequency_hz is None else float(plasma_frequency_hz)
    if not np.isfinite(plasma_hz) or plasma_hz <= 0.0:
        raise ValueError("plasma_frequency_hz must be finite and positive.")

    steps = np.arange(horizon, dtype=np.float64)
    plasma_traj = np.mod(2.0 * np.pi * plasma_hz * ctrl.dt * steps, 2.0 * np.pi)
    out_phis = ctrl.step_horizon(plasma_traj)
    phase_error = np.abs(np.sin(out_phis - plasma_traj))
    return {
        "cycles": int(len(out_phis)),
        "final_phase_rad": float(out_phis[-1]),
        "final_omega_hz": float(ctrl.omega_rmf / (2.0 * np.pi)),
        "safety_violations": int(ctrl.safety_violations),
        "mean_abs_phase_error": float(np.mean(phase_error)),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line options for the RMF phase-lock diagnostic."""
    parser = argparse.ArgumentParser(description="Run the RMF phase-lock software diagnostic.")
    parser.add_argument("--horizon", type=int, default=10_000, help="controller cycles to evaluate")
    parser.add_argument(
        "--plasma-frequency-hz",
        type=float,
        default=None,
        help="plasma phase-rotation frequency; defaults to nominal RMF frequency",
    )
    parser.add_argument("--rmf-frequency-hz", type=float, default=1.0e6)
    parser.add_argument("--sampling-frequency-hz", type=float, default=10.0e6)
    parser.add_argument("--k-p", type=float, default=0.5)
    parser.add_argument("--k-d", type=float, default=0.01)
    parser.add_argument("--n-neurons", type=int, default=128)
    parser.add_argument("--min-frequency-hz", type=float, default=1.0e5)
    parser.add_argument("--max-frequency-hz", type=float, default=5.0e6)
    parser.add_argument("--max-phase-error-rad", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RMF phase-lock diagnostic command-line interface."""
    args = _parse_args(argv)
    cfg = RMFPhaseLockConfig(
        f_rmf_nom_hz=args.rmf_frequency_hz,
        f_sampling_hz=args.sampling_frequency_hz,
        k_p=args.k_p,
        k_d=args.k_d,
        n_neurons=args.n_neurons,
        min_freq_hz=args.min_frequency_hz,
        max_freq_hz=args.max_frequency_hz,
        max_phase_error_rad=args.max_phase_error_rad,
    )
    summary = run_rmf_phase_lock_demo(
        horizon=args.horizon,
        plasma_frequency_hz=args.plasma_frequency_hz,
        config=cfg,
    )
    if args.json:
        sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            "RMF software horizon evaluated: "
            f"{summary['cycles']} cycles; "
            f"final_omega_hz={summary['final_omega_hz']:.6g}; "
            f"safety_violations={summary['safety_violations']}"
            "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
