# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pure-NumPy SNN Controller (LIF + NEF)
"""Pure-NumPy spiking neural network controller for tokamak plasma control.

Implements LIF neurons with Neural Engineering Framework (NEF) decoding.
Drop-in replacement for the former Nengo-based wrapper; zero external
dependencies beyond NumPy. Compatible with NumPy 1.x and 2.x.

Architecture per channel::

    error → [LP τ_s] → LIF error ensemble → [decode gain·x, LP τ_s]
          → LIF control ensemble → [decode x, LP τ_s] → output

Reference: Eliasmith & Anderson, *Neural Engineering*, MIT Press, 2003.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_nengo_available = True


def nengo_available() -> bool:
    """Always True — no external dependency required."""
    return True


@dataclass
class NengoSNNConfig:
    """Configuration for the SNN controller."""

    n_neurons: int = 200
    n_channels: int = 2
    tau_synapse: float = 0.015  # synaptic time constant [s]
    tau_mem: float = 0.020  # membrane τ_rc [s]
    dt: float = 0.001  # simulation timestep [s]
    max_rate_hz: float = 200.0  # maximum firing rate [Hz]
    intercept_range: tuple[float, float] = (-0.8, 0.8)
    gain: float = 5.0
    seed: int = 42


# ── Internal building blocks ─────────────────────────────────────────


class _Lowpass:
    """First-order exponential lowpass: y += (1-d)(x-y), d = exp(-dt/τ)."""

    __slots__ = ("_decay", "_val")

    def __init__(self, tau: float, dt: float, n: int) -> None:
        self._decay = np.exp(-dt / tau) if tau > 0 else 0.0
        self._val = np.zeros(n)

    def step(self, x: NDArray) -> NDArray:
        self._val = self._decay * self._val + (1.0 - self._decay) * x
        return self._val

    def reset(self) -> None:
        self._val[:] = 0.0


class _LIFPopulation:
    """Vectorized LIF neuron population with NEF gain/bias.

    Membrane dynamics (exact integration per timestep):
        V(t+dt) = J + (V(t) - J) · exp(-Δ/τ_rc)
    Spike when V ≥ 1, then V = 0 for τ_ref seconds.

    Gain α and bias J_bias from per-neuron (max_rate, intercept):
        J_max = 1 / (1 - exp((τ_ref - 1/r_max) / τ_rc))
        α = (J_max - 1) / (1 - intercept)
        J_bias = 1 - α · intercept
    Eliasmith & Anderson 2003, Eq. 4.10–4.12.
    """

    def __init__(
        self,
        n: int,
        tau_rc: float,
        tau_ref: float,
        max_rates: NDArray,
        intercepts: NDArray,
        encoders: NDArray,
        dt: float,
    ) -> None:
        self.n = n
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.dt = dt
        self.encoders = encoders

        J_max = 1.0 / (1.0 - np.exp((tau_ref - 1.0 / max_rates) / tau_rc))
        self.alpha = (J_max - 1.0) / (1.0 - intercepts)
        self.J_bias = 1.0 - self.alpha * intercepts

        self.voltage = np.zeros(n)
        self.ref_time = np.zeros(n)

    def step(self, x: float) -> NDArray:
        """One dt step. Returns spike rates (1/dt on spike, else 0)."""
        J = self.alpha * self.encoders * x + self.J_bias
        delta = np.clip(self.dt - self.ref_time, 0.0, self.dt)
        self.voltage = J + (self.voltage - J) * np.exp(-delta / self.tau_rc)

        spiked = self.voltage >= 1.0
        self.voltage[spiked] = 0.0
        self.ref_time[spiked] = self.tau_ref
        self.ref_time = np.maximum(self.ref_time - self.dt, 0.0)

        return spiked.astype(np.float64) / self.dt

    def steady_rates(self, x_eval: NDArray) -> NDArray:
        """Analytic steady-state firing rates. Shape (n, len(x_eval))."""
        J = self.alpha[:, None] * self.encoders[:, None] * x_eval[None, :] + self.J_bias[:, None]
        rates = np.zeros_like(J)
        ok = J > 1.0
        rates[ok] = 1.0 / (self.tau_ref - self.tau_rc * np.log1p(-1.0 / J[ok]))
        return rates

    def reset(self) -> None:
        self.voltage[:] = 0.0
        self.ref_time[:] = 0.0


def _nef_decoder(pop: _LIFPopulation, fn, n_eval: int = 200, reg: float = 0.1) -> NDArray:
    """Least-squares NEF decoder for target function fn(x).

    Tikhonov regularization matches Nengo's LstsqL2 default.
    Returns shape (n_neurons,).
    """
    x = np.linspace(-1, 1, n_eval)
    A = pop.steady_rates(x)
    Y = fn(x)
    AAt = A @ A.T
    max_a = max(float(np.max(A)), 1e-10)
    gamma = n_eval * reg * max_a**2
    return np.linalg.solve(AAt + gamma * np.eye(pop.n), A @ Y)


class _Channel:
    """One control channel: error → gain·x → control → identity → output."""

    def __init__(self, cfg: NengoSNNConfig, rng: np.random.Generator) -> None:
        n, dt = cfg.n_neurons, cfg.dt

        def _make_pop() -> _LIFPopulation:
            return _LIFPopulation(
                n=n,
                tau_rc=cfg.tau_mem,
                tau_ref=0.002,
                max_rates=rng.uniform(cfg.max_rate_hz * 0.5, cfg.max_rate_hz, n),
                intercepts=rng.uniform(*cfg.intercept_range, n),
                encoders=rng.choice([-1.0, 1.0], n),
                dt=dt,
            )

        self.error_pop = _make_pop()
        self.control_pop = _make_pop()

        gain = cfg.gain
        self.D_gain = _nef_decoder(self.error_pop, lambda x: gain * x)
        self.D_id = _nef_decoder(self.control_pop, lambda x: x)
        self.D_probe = _nef_decoder(self.error_pop, lambda x: x)

        self.syn_in = _Lowpass(cfg.tau_synapse, dt, 1)
        self.syn_mid = _Lowpass(cfg.tau_synapse, dt, 1)
        self.syn_out = _Lowpass(cfg.tau_synapse, dt, 1)

        self.probe_filt = _Lowpass(0.01, dt, 1)
        self.probe_history: list[float] = []

    def step(self, x: float) -> float:
        x_f = float(self.syn_in.step(np.array([x]))[0])
        err_spikes = self.error_pop.step(x_f)
        decoded_gain = float(self.D_gain @ err_spikes)
        mid_f = float(self.syn_mid.step(np.array([decoded_gain]))[0])
        ctl_spikes = self.control_pop.step(mid_f)
        decoded_id = float(self.D_id @ ctl_spikes)
        out = float(self.syn_out.step(np.array([decoded_id]))[0])

        decoded_probe = float(self.D_probe @ err_spikes)
        probe_val = float(self.probe_filt.step(np.array([decoded_probe]))[0])
        self.probe_history.append(probe_val)

        return out

    def reset(self) -> None:
        self.error_pop.reset()
        self.control_pop.reset()
        self.syn_in.reset()
        self.syn_mid.reset()
        self.syn_out.reset()
        self.probe_filt.reset()
        self.probe_history.clear()


# ── Public API ───────────────────────────────────────────────────────


class NengoSNNController:
    """SNN controller for tokamak position control.

    Pure-NumPy LIF + NEF implementation. Per channel: error ensemble
    decodes gain·x into control ensemble, which decodes identity to output.

    Parameters
    ----------
    config : NengoSNNConfig or None
        Controller configuration. Uses defaults if None.
    """

    def __init__(self, config: NengoSNNConfig | None = None) -> None:
        self.cfg = config or NengoSNNConfig()
        self._channels: list[_Channel] = []
        self._built = False
        self._step_count = 0
        self._last_output = np.zeros(self.cfg.n_channels)
        self._output_history: list[NDArray] = []
        self._output_probe_filt = _Lowpass(0.01, self.cfg.dt, self.cfg.n_channels)

        self.build_network()

    def build_network(self) -> None:
        """Build LIF ensembles and compute NEF decoders."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        self._channels = [_Channel(cfg, rng) for _ in range(cfg.n_channels)]
        self._built = True
        self._step_count = 0
        self._output_history.clear()
        logger.info(
            "Built SNN controller: %d neurons/ch x %d ch, dt=%.3fs",
            cfg.n_neurons,
            cfg.n_channels,
            cfg.dt,
        )

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run one control step.

        Parameters
        ----------
        state : array of shape (n_channels,)
            Error vector.

        Returns
        -------
        array of shape (n_channels,)
            Control command.
        """
        if not self._built:
            raise RuntimeError("Network not built.")

        error = np.asarray(state, dtype=float).ravel()[: self.cfg.n_channels]
        output = np.array([ch.step(error[i]) for i, ch in enumerate(self._channels)])
        self._step_count += 1
        self._last_output = output

        probe_out = self._output_probe_filt.step(output)
        self._output_history.append(probe_out.copy())

        return output.copy()

    def reset(self) -> None:
        """Reset all neuron state and filters."""
        for ch in self._channels:
            ch.reset()
        self._step_count = 0
        self._last_output = np.zeros(self.cfg.n_channels)
        self._output_history.clear()
        self._output_probe_filt.reset()

    def get_spike_data(self) -> dict[str, NDArray]:
        """Return probe data from simulation."""
        if not self._output_history:
            return {}
        data: dict[str, NDArray] = {
            "output": np.array(self._output_history),
        }
        for i, ch in enumerate(self._channels):
            if ch.probe_history:
                data[f"error_ch{i}"] = np.array(ch.probe_history)[:, None]
        return data

    def export_weights(self) -> dict[str, NDArray]:
        """Extract decoder, encoder, and gain/bias arrays.

        Returns
        -------
        dict mapping label -> weight array
        """
        if not self._built:
            raise RuntimeError("Network not built.")

        weights: dict[str, NDArray] = {}
        for i, ch in enumerate(self._channels):
            weights[f"ch{i}_error_encoders"] = ch.error_pop.encoders.copy()
            weights[f"ch{i}_error_alpha"] = ch.error_pop.alpha.copy()
            weights[f"ch{i}_error_J_bias"] = ch.error_pop.J_bias.copy()
            weights[f"ch{i}_D_gain"] = ch.D_gain.copy()
            weights[f"ch{i}_control_encoders"] = ch.control_pop.encoders.copy()
            weights[f"ch{i}_control_alpha"] = ch.control_pop.alpha.copy()
            weights[f"ch{i}_control_J_bias"] = ch.control_pop.J_bias.copy()
            weights[f"ch{i}_D_id"] = ch.D_id.copy()
        return weights

    def export_fpga_weights(self, filename: str | Path) -> None:
        """Export weight matrices for FPGA synthesis as .npz."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, NDArray] = {
            "n_neurons": np.array([self.cfg.n_neurons]),
            "n_channels": np.array([self.cfg.n_channels]),
            "dt": np.array([self.cfg.dt]),
            "tau_synapse": np.array([self.cfg.tau_synapse]),
            "tau_mem": np.array([self.cfg.tau_mem]),
            "gain": np.array([self.cfg.gain]),
        }
        payload.update(self.export_weights())
        np.savez(str(path), **payload)
        logger.info("Exported FPGA weights to %s", path)

    def export_loihi(self, filename: str | Path) -> None:
        """Loihi export requires the original nengo + nengo_loihi packages."""
        raise NotImplementedError(
            "Loihi export requires nengo + nengo_loihi. "
            "Use export_fpga_weights() for hardware deployment."
        )

    def benchmark(self, n_steps: int = 1000) -> dict[str, float]:
        """Measure per-step latency.

        Returns
        -------
        dict with mean_us, p50_us, p95_us, p99_us, max_us
        """
        self.reset()
        rng = np.random.default_rng(0)
        times = np.empty(n_steps)
        for i in range(n_steps):
            error = rng.normal(0, 0.1, size=self.cfg.n_channels)
            t0 = time.perf_counter_ns()
            self.step(error)
            times[i] = (time.perf_counter_ns() - t0) / 1e3
        return {
            "mean_us": float(np.mean(times)),
            "p50_us": float(np.percentile(times, 50)),
            "p95_us": float(np.percentile(times, 95)),
            "p99_us": float(np.percentile(times, 99)),
            "max_us": float(np.max(times)),
        }


class NengoSNNControllerStub:
    """Backward-compat stub. No longer needed — controller always works."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise ImportError(
            "NengoSNNControllerStub is deprecated. "
            "Use NengoSNNController directly — no external dependencies required."
        )
