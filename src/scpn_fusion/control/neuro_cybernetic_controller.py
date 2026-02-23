# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neuro Cybernetic Controller
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
from scpn_fusion.scpn.safety_interlocks import SafetyInterlockRuntime

try:
    from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
    from sc_neurocore.sources.quantum_entropy import QuantumEntropySource

    SC_NEUROCORE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    SC_NEUROCORE_AVAILABLE = False
    StochasticLIFNeuron = None
    QuantumEntropySource = None

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
        n_neurons = int(n_neurons)
        if n_neurons < 1:
            raise ValueError("n_neurons must be >= 1.")
        gain = float(gain)
        if not math.isfinite(gain):
            raise ValueError("gain must be finite.")
        tau_window = int(tau_window)
        if tau_window < 1:
            raise ValueError("tau_window must be >= 1.")
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")
        tau_mem_s = float(tau_mem_s)
        if not math.isfinite(tau_mem_s) or tau_mem_s <= 0.0:
            raise ValueError("tau_mem_s must be finite and > 0.")
        noise_std = float(noise_std)
        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std must be finite and >= 0.")

        self.n_neurons = n_neurons
        self.gain = gain
        self.window_size = tau_window
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
        self._alpha = dt_s / tau_mem_s
        self._noise_std = noise_std
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

    def __init__(
        self,
        config_file: str,
        seed: int = 42,
        *,
        shot_duration: int = SHOT_DURATION,
        kernel_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if int(shot_duration) <= 0:
            raise ValueError("shot_duration must be > 0")
        fusion_kernel_cls = kernel_factory if kernel_factory is not None else _resolve_fusion_kernel()
        self.kernel = fusion_kernel_cls(config_file)
        self.seed = int(seed)
        self.shot_duration = int(shot_duration)
        self.history: Dict[str, list[float]] = {}
        self.brain_R: Optional[SpikingControllerPool] = None
        self.brain_Z: Optional[SpikingControllerPool] = None
        self.safety_runtime = SafetyInterlockRuntime()
        self._reset_history()

    def _reset_history(self) -> None:
        self.history = {
            "t": [],
            "Ip": [],
            "R_axis": [],
            "Z_axis": [],
            "Err_R": [],
            "Err_Z": [],
            "Control_R": [],
            "Control_Z": [],
            "Spike_Rates": [],
            "Safety_Position_Allowed": [],
            "Safety_Contract_Violations": [],
        }

    def _check_safety(self, state: Dict[str, float]) -> Dict[str, bool]:
        allowed = self.safety_runtime.update_from_state(state)
        return allowed

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

    def run_shot(
        self,
        *,
        save_plot: bool = True,
        verbose: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.initialize_brains(use_quantum=False)
        return self._execute_simulation(
            "Neuro-Cybernetic (Classical SNN)",
            mode="classical",
            save_plot=save_plot,
            verbose=verbose,
            output_path=output_path,
        )

    def run_quantum_shot(
        self,
        *,
        save_plot: bool = True,
        verbose: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.initialize_brains(use_quantum=True)
        return self._execute_simulation(
            "Quantum-Neuro Hybrid (QNN)",
            mode="quantum",
            save_plot=save_plot,
            verbose=verbose,
            output_path=output_path,
        )

    def _execute_simulation(
        self,
        title: str,
        *,
        mode: str,
        save_plot: bool,
        verbose: bool,
        output_path: Optional[str],
    ) -> Dict[str, Any]:
        assert self.brain_R is not None and self.brain_Z is not None
        if verbose:
            logger.info("--- %s PLASMA INTERFACE ---", title.upper())
            logger.info("Initializing Stochastic Neural Network (SNN)...")
            logger.info("Neurons: %d (Push-Pull Configuration)", self.brain_R.n_neurons * 4)

        self._reset_history()

        self.kernel.solve_equilibrium()

        physics_cfg = self.kernel.cfg.setdefault("physics", {})
        coils = self.kernel.cfg.setdefault("coils", [{} for _ in range(5)])
        while len(coils) < 5:
            coils.append({})
        for coil in coils:
            coil.setdefault("current", 0.0)
        prev_z: Optional[float] = None

        def _mean_profile_or_zero(value: Any) -> float:
            try:
                arr = np.asarray(value, dtype=np.float64)
            except Exception:
                return 0.0
            if arr.size == 0:
                return 0.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.mean(arr))

        for t in range(self.shot_duration):
            target_ip = 5.0 + (10.0 * t / self.shot_duration)
            physics_cfg["plasma_current_target"] = target_ip

            idx_max = int(np.argmax(self.kernel.Psi))
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_r = float(self.kernel.R[ir])
            curr_z = float(self.kernel.Z[iz])

            err_r = TARGET_R - curr_r
            err_z = TARGET_Z - curr_z

            ctrl_r = float(self.brain_R.step(err_r))
            ctrl_z = float(self.brain_Z.step(err_z))

            dz_dt = 0.0 if prev_z is None else float(curr_z - prev_z)
            prev_z = curr_z
            state = {
                "T_e": _mean_profile_or_zero(getattr(self.kernel, "Te", np.zeros(1))),
                "n_e": _mean_profile_or_zero(getattr(self.kernel, "ne", np.zeros(1))),
                "beta_N": float(physics_cfg.get("beta_N", physics_cfg.get("beta_n", 0.0))),
                "I_p": float(target_ip),
                "dZ_dt": float(dz_dt),
            }
            allowed = self._check_safety(state)
            if (
                not allowed.get("heat_ramp", True)
                or not allowed.get("power_ramp", True)
                or not allowed.get("current_ramp", True)
            ):
                ctrl_r = 0.0
            if not allowed.get("position_move", True):
                ctrl_z = 0.0

            coils[2]["current"] = float(coils[2]["current"]) + ctrl_r
            coils[0]["current"] = float(coils[0]["current"]) - ctrl_z
            coils[4]["current"] = float(coils[4]["current"]) + ctrl_z

            self.kernel.solve_equilibrium()

            self.history["t"].append(float(t))
            self.history["Ip"].append(float(target_ip))
            self.history["R_axis"].append(curr_r)
            self.history["Z_axis"].append(curr_z)
            self.history["Err_R"].append(float(err_r))
            self.history["Err_Z"].append(float(err_z))
            self.history["Control_R"].append(ctrl_r)
            self.history["Control_Z"].append(ctrl_z)
            self.history["Spike_Rates"].append(
                float(self.brain_R.last_rate_pos - self.brain_R.last_rate_neg)
            )
            self.history["Safety_Position_Allowed"].append(
                1.0 if allowed.get("position_move", True) else 0.0
            )
            self.history["Safety_Contract_Violations"].append(
                float(len(self.safety_runtime.last_contract_violations))
            )

            if verbose:
                logger.info(
                    "T=%d: Pos=(%.2f, %.2f) | Err=(%.3f, %.3f) | Brain_Out=(%.3f, %.3f)",
                    t, curr_r, curr_z, err_r, err_z, ctrl_r, ctrl_z,
                )

        plot_saved = False
        plot_error: Optional[str] = None
        if save_plot:
            try:
                self.visualize(title, output_path=output_path, verbose=verbose)
                plot_saved = True
            except Exception as exc:
                plot_error = str(exc)
                if verbose:
                    logger.warning("Plot export skipped due to error: %s", exc)

        err_r = np.asarray(self.history["Err_R"], dtype=np.float64)
        err_z = np.asarray(self.history["Err_Z"], dtype=np.float64)
        ctrl_r = np.asarray(self.history["Control_R"], dtype=np.float64)
        ctrl_z = np.asarray(self.history["Control_Z"], dtype=np.float64)
        safety_position_allowed = np.asarray(
            self.history["Safety_Position_Allowed"], dtype=np.float64
        )
        safety_contract_violations = np.asarray(
            self.history["Safety_Contract_Violations"], dtype=np.float64
        )
        summary: Dict[str, Any] = {
            "seed": self.seed,
            "steps": int(self.shot_duration),
            "mode": str(mode),
            "backend_r": self.brain_R.backend,
            "backend_z": self.brain_Z.backend,
            "final_r": float(self.history["R_axis"][-1]),
            "final_z": float(self.history["Z_axis"][-1]),
            "mean_abs_err_r": float(np.mean(np.abs(err_r))),
            "mean_abs_err_z": float(np.mean(np.abs(err_z))),
            "max_abs_control_r": float(np.max(np.abs(ctrl_r))),
            "max_abs_control_z": float(np.max(np.abs(ctrl_z))),
            "mean_spike_imbalance": float(np.mean(self.history["Spike_Rates"])),
            "safety_position_allow_rate": float(
                np.mean(safety_position_allowed)
                if safety_position_allowed.size
                else 1.0
            ),
            "safety_interlock_trips": int(
                np.count_nonzero(safety_position_allowed < 0.5)
            ),
            "safety_contract_violations": int(np.sum(safety_contract_violations)),
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }
        return summary

    def visualize(
        self,
        title: str,
        *,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
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

        filename = (
            output_path
            if output_path is not None
            else f"{title.replace(' ', '_')}_Result.png"
        )
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        if verbose:
            logger.info("Analysis saved: %s", filename)
        return filename


def run_neuro_cybernetic_control(
    *,
    config_file: str,
    shot_duration: int = SHOT_DURATION,
    seed: int = 42,
    quantum: bool = False,
    save_plot: bool = False,
    verbose: bool = False,
    output_path: Optional[str] = None,
    kernel_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Run neuro-cybernetic control in deterministic non-interactive mode."""
    controller = NeuroCyberneticController(
        config_file,
        seed=seed,
        shot_duration=shot_duration,
        kernel_factory=kernel_factory,
    )
    if quantum:
        return controller.run_quantum_shot(
            save_plot=save_plot, verbose=verbose, output_path=output_path
        )
    return controller.run_shot(
        save_plot=save_plot, verbose=verbose, output_path=output_path
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "iter_config.json"
    nc = NeuroCyberneticController(str(cfg))
    if len(sys.argv) > 1 and sys.argv[1] == "quantum":
        nc.run_quantum_shot()
    else:
        nc.run_shot()
