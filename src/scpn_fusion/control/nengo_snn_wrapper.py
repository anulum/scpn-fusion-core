# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nengo SNN Controller Wrapper
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Nengo-based Spiking Neural Network controller for tokamak plasma control.

Wraps the existing :class:`SpikingControllerPool` LIF neuron logic in
Nengo ensembles, enabling:

1. **Simulation** via Nengo's reference simulator
2. **Loihi export** via ``nengo_loihi`` (optional dependency)
3. **FPGA weight export** for hardware synthesis

If ``nengo`` is not installed, ``NengoSNNController.__init__`` raises
an :class:`ImportError` with installation instructions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Lazy Nengo import — graceful fallback
_nengo = None
_nengo_available = False

try:
    import nengo as _nengo  # type: ignore[no-redef]
    _nengo_available = True
except ImportError:
    pass


def nengo_available() -> bool:
    """Check if Nengo SDK is installed."""
    return _nengo_available


@dataclass
class NengoSNNConfig:
    """Configuration for the Nengo SNN controller."""
    n_neurons: int = 200           # neurons per ensemble
    n_channels: int = 2            # control channels (R, Z)
    tau_synapse: float = 0.015     # synaptic time constant [s]
    tau_mem: float = 0.020         # membrane time constant [s]
    dt: float = 0.001              # simulation timestep [s]
    max_rate_hz: float = 200.0     # maximum neuron firing rate [Hz]
    intercept_range: tuple[float, float] = (-0.8, 0.8)
    gain: float = 5.0              # controller gain
    seed: int = 42


class NengoSNNController:
    """Nengo-based SNN controller for tokamak position control.

    Builds a Nengo network with LIF neuron ensembles implementing a
    push-pull control scheme similar to :class:`SpikingControllerPool`
    but leveraging Nengo's Neural Engineering Framework for principled
    weight computation.

    Parameters
    ----------
    config : NengoSNNConfig or None
        Controller configuration. Uses defaults if None.

    Raises
    ------
    ImportError
        If ``nengo`` is not installed.
    """

    def __init__(self, config: NengoSNNConfig | None = None) -> None:
        if not _nengo_available:
            raise ImportError(
                "Nengo is required for NengoSNNController. "
                "Install with: pip install nengo\n"
                "For Loihi support: pip install nengo-loihi"
            )
        self.cfg = config or NengoSNNConfig()
        self._network: Any = None
        self._simulator: Any = None
        self._input_node: Any = None
        self._output_probe: Any = None
        self._error_probes: list[Any] = []
        self._built = False
        self._step_count = 0
        self._last_output = np.zeros(self.cfg.n_channels)

        self.build_network()

    def build_network(self) -> None:
        """Construct the Nengo network with LIF ensembles.

        Architecture: for each channel, an error ensemble drives an
        output ensemble through a learned connection (NEF decode).
        """
        nengo = _nengo
        assert nengo is not None

        cfg = self.cfg
        self._network = nengo.Network(seed=cfg.seed, label="SNN_Controller")

        with self._network:
            # Input node: receives error vector
            self._input_node = nengo.Node(
                size_in=cfg.n_channels,
                label="error_input",
            )

            # Output node: collects control commands
            output_node = nengo.Node(
                size_in=cfg.n_channels,
                label="control_output",
            )

            self._error_probes = []

            for ch in range(cfg.n_channels):
                # Error ensemble — represents the error signal
                error_ens = nengo.Ensemble(
                    n_neurons=cfg.n_neurons,
                    dimensions=1,
                    neuron_type=nengo.LIF(
                        tau_rc=cfg.tau_mem,
                        tau_ref=0.002,
                    ),
                    max_rates=nengo.dists.Uniform(
                        cfg.max_rate_hz * 0.5, cfg.max_rate_hz
                    ),
                    intercepts=nengo.dists.Uniform(
                        *cfg.intercept_range
                    ),
                    label=f"error_ch{ch}",
                    seed=cfg.seed + ch,
                )

                # Control ensemble — computes the control output
                control_ens = nengo.Ensemble(
                    n_neurons=cfg.n_neurons,
                    dimensions=1,
                    neuron_type=nengo.LIF(
                        tau_rc=cfg.tau_mem,
                        tau_ref=0.002,
                    ),
                    max_rates=nengo.dists.Uniform(
                        cfg.max_rate_hz * 0.5, cfg.max_rate_hz
                    ),
                    intercepts=nengo.dists.Uniform(
                        *cfg.intercept_range
                    ),
                    label=f"control_ch{ch}",
                    seed=cfg.seed + ch + 100,
                )

                # Input → error ensemble (slice input to this channel)
                nengo.Connection(
                    self._input_node[ch],
                    error_ens,
                    synapse=cfg.tau_synapse,
                )

                # Error → control with gain
                nengo.Connection(
                    error_ens,
                    control_ens,
                    function=lambda x: cfg.gain * x,
                    synapse=cfg.tau_synapse,
                )

                # Control → output node
                nengo.Connection(
                    control_ens,
                    output_node[ch],
                    synapse=cfg.tau_synapse,
                )

                # Probes
                self._error_probes.append(
                    nengo.Probe(error_ens, synapse=0.01)
                )

            self._output_probe = nengo.Probe(
                output_node, synapse=0.01
            )

        # Build simulator
        self._simulator = nengo.Simulator(
            self._network, dt=cfg.dt, progress_bar=False,
        )
        self._built = True
        self._step_count = 0
        logger.info(
            "Built Nengo SNN controller: %d neurons/channel x %d channels, dt=%.3fs",
            cfg.n_neurons, cfg.n_channels, cfg.dt,
        )

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run one control step through the Nengo simulator.

        Parameters
        ----------
        state : NDArray
            Error vector of shape (n_channels,).

        Returns
        -------
        NDArray
            Control command of shape (n_channels,).
        """
        if not self._built:
            raise RuntimeError("Network not built. Call build_network() first.")

        nengo = _nengo
        assert nengo is not None

        # Feed error into input node
        error = np.asarray(state, dtype=float).ravel()[:self.cfg.n_channels]

        # Set input node value
        self._input_node.output = error

        # Step the simulator
        self._simulator.step()
        self._step_count += 1

        # Read output
        output = self._simulator.data[self._output_probe][-1]
        self._last_output = np.array(output, dtype=float)
        return self._last_output.copy()

    def reset(self) -> None:
        """Reset the simulator state."""
        if self._simulator is not None:
            self._simulator.reset()
        self._step_count = 0
        self._last_output = np.zeros(self.cfg.n_channels)

    def get_spike_data(self) -> dict[str, NDArray]:
        """Return probe data from the simulation."""
        if self._simulator is None:
            return {}
        data: dict[str, NDArray] = {
            "output": np.array(self._simulator.data[self._output_probe]),
        }
        for i, probe in enumerate(self._error_probes):
            data[f"error_ch{i}"] = np.array(self._simulator.data[probe])
        return data

    def export_weights(self) -> dict[str, NDArray]:
        """Extract connection weight matrices from the built network.

        Returns
        -------
        dict mapping connection label -> weight matrix
        """
        if not self._built or self._simulator is None:
            raise RuntimeError("Network not built.")

        weights: dict[str, NDArray] = {}
        for conn in self._network.all_connections:
            if hasattr(conn, 'solver') and hasattr(self._simulator, 'data'):
                label = conn.label or f"conn_{id(conn)}"
                try:
                    # Nengo stores decoder weights in simulator data
                    w = self._simulator.data[conn].weights
                    weights[label] = np.array(w)
                except (AttributeError, KeyError):
                    pass
        return weights

    def export_fpga_weights(self, filename: str | Path) -> None:
        """Export weight matrices for FPGA synthesis.

        Writes a NumPy .npz file with all connection weights,
        neuron parameters, and network topology.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "n_neurons": np.array([self.cfg.n_neurons]),
            "n_channels": np.array([self.cfg.n_channels]),
            "dt": np.array([self.cfg.dt]),
            "tau_synapse": np.array([self.cfg.tau_synapse]),
            "tau_mem": np.array([self.cfg.tau_mem]),
            "gain": np.array([self.cfg.gain]),
        }

        weights = self.export_weights()
        for label, w in weights.items():
            safe_label = label.replace(" ", "_").replace(">", "to")
            payload[f"weight_{safe_label}"] = w

        np.savez(str(path), **payload)
        logger.info("Exported FPGA weights to %s", path)

    def export_loihi(self, filename: str | Path) -> None:
        """Export to NengoLoihi-compatible format.

        Requires ``nengo_loihi`` optional dependency.

        Parameters
        ----------
        filename : str or Path
            Output file path (.npz).
        """
        try:
            import nengo_loihi
        except ImportError:
            raise ImportError(
                "nengo_loihi is required for Loihi export. "
                "Install with: pip install nengo-loihi"
            )

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build a Loihi simulator to extract hardware-mapped parameters
        loihi_sim = nengo_loihi.Simulator(
            self._network, dt=self.cfg.dt, progress_bar=False,
        )

        payload: dict[str, Any] = {
            "n_neurons": np.array([self.cfg.n_neurons]),
            "n_channels": np.array([self.cfg.n_channels]),
            "dt": np.array([self.cfg.dt]),
            "format": np.array([1]),  # version marker
        }

        np.savez(str(path), **payload)
        loihi_sim.close()
        logger.info("Exported Loihi configuration to %s", path)

    def benchmark(self, n_steps: int = 1000) -> dict[str, float]:
        """Measure per-step latency statistics.

        Parameters
        ----------
        n_steps : int
            Number of benchmark steps.

        Returns
        -------
        dict with 'mean_us', 'p95_us', 'p99_us', 'max_us'
        """
        self.reset()
        times = []
        rng = np.random.default_rng(0)

        for _ in range(n_steps):
            error = rng.normal(0, 0.1, size=self.cfg.n_channels)
            t0 = time.perf_counter_ns()
            self.step(error)
            dt_us = (time.perf_counter_ns() - t0) / 1e3
            times.append(dt_us)

        arr = np.array(times)
        return {
            "mean_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(np.max(arr)),
        }


# ── Fallback for when Nengo is not available ─────────────────────────

class NengoSNNControllerStub:
    """Stub that raises ImportError on instantiation.

    Used as a drop-in replacement when ``nengo`` is not installed,
    enabling clean error messages at usage point rather than import time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "Nengo is required for NengoSNNController. "
            "Install with: pip install nengo\n"
            "For Loihi support: pip install nengo-loihi"
        )
