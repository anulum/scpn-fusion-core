# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro Cybernetic Controller
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
    from sc_neurocore.sources.quantum_entropy import QuantumEntropySource

    SC_NEUROCORE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    SC_NEUROCORE_AVAILABLE = False
    StochasticLIFNeuron = None
    QuantumEntropySource = None

# --- CONTROL PARAMETERS ---
SHOT_DURATION = 100
TARGET_R = 6.2
TARGET_Z = 0.0


def _resolve_fusion_kernel() -> Any:
    """Resolve FusionKernel lazily to keep pool-only paths dependency-light."""
    try:
        from scpn_fusion.core._rust_compat import FusionKernel as _FusionKernel

        return _FusionKernel
    except Exception:
        try:
            from scpn_fusion.core.fusion_kernel import FusionKernel as _FusionKernel

            return _FusionKernel
        except Exception as exc:  # pragma: no cover - import-guard path
            raise ImportError(
                "Unable to import FusionKernel. Run with PYTHONPATH=src "
                "or use `python -m scpn_fusion.control.neuro_cybernetic_controller`."
            ) from exc


class SpikingControllerPool:
    """
    Push-pull spiking control population with deterministic fallback.

    Preferred backend is ``sc-neurocore``. If unavailable, a reduced NumPy LIF
    population is used so controller workflows remain executable in CI.
    """

    def __init__(
        self,
        n_neurons: int = 20,
        gain: float = 1.0,
        tau_window: int = 10,
        use_quantum: bool = False,
        *,
        seed: int = 42,
        allow_numpy_fallback: bool = True,
        dt_s: float = 1.0e-3,
        tau_mem_s: float = 15.0e-3,
        noise_std: float = 0.02,
    ) -> None:
        self.n_neurons = max(int(n_neurons), 1)
        self.gain = float(gain)
        self.window_size = max(int(tau_window), 1)
        self.use_quantum = bool(use_quantum)
        self._i_scale = 5.0
        self._i_bias = 0.1
        self.last_rate_pos = 0.0
        self.last_rate_neg = 0.0

        self.history_pos: deque[int] = deque(maxlen=self.window_size)
        self.history_neg: deque[int] = deque(maxlen=self.window_size)
        for _ in range(self.window_size):
            self.history_pos.append(0)
            self.history_neg.append(0)

        if SC_NEUROCORE_AVAILABLE:
            self.backend = "sc_neurocore"
            self.q_source = (
                QuantumEntropySource(n_qubits=4)
                if self.use_quantum and QuantumEntropySource is not None
                else None
            )
            self.pop_pos = [
                StochasticLIFNeuron(seed=i, entropy_source=self.q_source)
                for i in range(self.n_neurons)
            ]
            self.pop_neg = [
                StochasticLIFNeuron(seed=i + 1000, entropy_source=self.q_source)
                for i in range(self.n_neurons)
            ]
            self._v_pos = None
            self._v_neg = None
            self._rng_pos = None
            self._rng_neg = None
            self._alpha = 0.0
            self._noise_std = 0.0
            self._v_threshold = 1.0
            self._v_reset = 0.0
            return

        if not allow_numpy_fallback:
            raise RuntimeError(
                "sc-neurocore is unavailable and allow_numpy_fallback=False."
            )

        self.backend = "numpy_lif"
        self.q_source = None
        self.pop_pos = []
        self.pop_neg = []
        self._rng_pos = np.random.default_rng(int(seed))
        self._rng_neg = np.random.default_rng(int(seed) + 100003)
        self._v_pos = np.zeros(self.n_neurons, dtype=np.float64)
        self._v_neg = np.zeros(self.n_neurons, dtype=np.float64)
        self._alpha = max(float(dt_s), 1.0e-9) / max(float(tau_mem_s), 1.0e-9)
        self._noise_std = max(float(noise_std), 0.0)
        # Reduced threshold keeps fallback lane responsive in low-current control
        # regimes while preserving deterministic push-pull polarity.
        self._v_threshold = 0.35
        self._v_reset = 0.0

    def _step_numpy_population(
        self,
        v: np.ndarray,
        rng: np.random.Generator,
        input_current: float,
    ) -> int:
        noise = rng.normal(0.0, self._noise_std, size=v.shape)
        v += self._alpha * (-v + float(input_current) + noise)
        fired = v >= self._v_threshold
        n_fired = int(np.count_nonzero(fired))
        if n_fired > 0:
            v[fired] = self._v_reset
        return n_fired

    def step(self, error_signal: float) -> float:
        input_pos = max(0.0, float(error_signal)) * self._i_scale
        input_neg = max(0.0, -float(error_signal)) * self._i_scale

        if self.backend == "sc_neurocore":
            spikes_pos = 0
            for neuron in self.pop_pos:
                if neuron.step(self._i_bias + input_pos):
                    spikes_pos += 1

            spikes_neg = 0
            for neuron in self.pop_neg:
                if neuron.step(self._i_bias + input_neg):
                    spikes_neg += 1
        else:
            spikes_pos = self._step_numpy_population(
                self._v_pos, self._rng_pos, self._i_bias + input_pos
            )
            spikes_neg = self._step_numpy_population(
                self._v_neg, self._rng_neg, self._i_bias + input_neg
            )

        self.history_pos.append(spikes_pos)
        self.history_neg.append(spikes_neg)

        self.last_rate_pos = float(
            sum(self.history_pos) / (self.window_size * self.n_neurons)
        )
        self.last_rate_neg = float(
            sum(self.history_neg) / (self.window_size * self.n_neurons)
        )
        return float((self.last_rate_pos - self.last_rate_neg) * self.gain)


class NeuroCyberneticController:
    """
    Replaces PID loops with push-pull spiking populations.
    """

    def __init__(self, config_file: str, seed: int = 42) -> None:
        fusion_kernel_cls = _resolve_fusion_kernel()
        self.kernel = fusion_kernel_cls(config_file)
        self.seed = int(seed)
        self.history: Dict[str, list[float]] = {
            "t": [],
            "Ip": [],
            "R_axis": [],
            "Z_axis": [],
            "Control_R": [],
            "Spike_Rates": [],
        }
        self.brain_R: Optional[SpikingControllerPool] = None
        self.brain_Z: Optional[SpikingControllerPool] = None

    def initialize_brains(self, use_quantum: bool = False) -> None:
        self.brain_R = SpikingControllerPool(
            n_neurons=50,
            gain=10.0,
            tau_window=20,
            use_quantum=use_quantum,
            seed=self.seed + 1,
        )
        self.brain_Z = SpikingControllerPool(
            n_neurons=50,
            gain=20.0,
            tau_window=20,
            use_quantum=use_quantum,
            seed=self.seed + 2,
        )

    def run_shot(self) -> None:
        self.initialize_brains(use_quantum=False)
        self._execute_simulation("Neuro-Cybernetic (Classical SNN)")

    def run_quantum_shot(self) -> None:
        self.initialize_brains(use_quantum=True)
        self._execute_simulation("Quantum-Neuro Hybrid (QNN)")

    def _execute_simulation(self, title: str) -> None:
        assert self.brain_R is not None and self.brain_Z is not None
        print(f"--- {title.upper()} PLASMA INTERFACE ---")
        print("Initializing Stochastic Neural Network (SNN)...")
        print(f"Neurons: {self.brain_R.n_neurons * 4} (Push-Pull Configuration)")

        self.history = {
            "t": [],
            "Ip": [],
            "R_axis": [],
            "Z_axis": [],
            "Control_R": [],
            "Spike_Rates": [],
        }

        self.kernel.solve_equilibrium()

        physics_cfg = self.kernel.cfg.setdefault("physics", {})
        coils = self.kernel.cfg.setdefault("coils", [{} for _ in range(5)])
        while len(coils) < 5:
            coils.append({})
        for coil in coils:
            coil.setdefault("current", 0.0)

        for t in range(SHOT_DURATION):
            target_ip = 5.0 + (10.0 * t / SHOT_DURATION)
            physics_cfg["plasma_current_target"] = target_ip

            idx_max = int(np.argmax(self.kernel.Psi))
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_r = float(self.kernel.R[ir])
            curr_z = float(self.kernel.Z[iz])

            err_r = TARGET_R - curr_r
            err_z = TARGET_Z - curr_z

            ctrl_r = float(self.brain_R.step(err_r))
            ctrl_z = float(self.brain_Z.step(err_z))

            coils[2]["current"] = float(coils[2]["current"]) + ctrl_r
            coils[0]["current"] = float(coils[0]["current"]) - ctrl_z
            coils[4]["current"] = float(coils[4]["current"]) + ctrl_z

            self.kernel.solve_equilibrium()

            self.history["t"].append(float(t))
            self.history["Ip"].append(float(target_ip))
            self.history["R_axis"].append(curr_r)
            self.history["Z_axis"].append(curr_z)
            self.history["Control_R"].append(ctrl_r)
            self.history["Spike_Rates"].append(
                float(self.brain_R.last_rate_pos - self.brain_R.last_rate_neg)
            )

            print(
                f"T={t}: Pos=({curr_r:.2f}, {curr_z:.2f}) | "
                f"Err=({err_r:.3f}, {err_z:.3f}) | "
                f"Brain_Out=({ctrl_r:.3f}, {ctrl_z:.3f})"
            )

        self.visualize(title)

    def visualize(self, title: str) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.set_title(f"{title} Control")
        ax1.plot(self.history["t"], self.history["R_axis"], "b-", label="R (Radial)")
        ax1.plot(self.history["t"], self.history["Z_axis"], "r-", label="Z (Vertical)")
        ax1.axhline(TARGET_R, color="b", linestyle="--", alpha=0.3)
        ax1.axhline(TARGET_Z, color="r", linestyle="--", alpha=0.3)
        ax1.set_ylabel("Position (m)")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Neural Control Activity")
        ax2.plot(self.history["t"], self.history["Control_R"], "k-", label="Radial Command")
        ax2.set_ylabel("Current Delta (A)")
        ax2.set_xlabel("Time Step")
        ax2.legend()

        filename = f"{title.replace(' ', '_')}_Result.png"
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Analysis saved: {filename}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "iter_config.json"
    nc = NeuroCyberneticController(str(cfg))
    if len(sys.argv) > 1 and sys.argv[1] == "quantum":
        nc.run_quantum_shot()
    else:
        nc.run_shot()
