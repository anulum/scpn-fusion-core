# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HIL Demo Runner
"""FPGA-register demo runner extracted from ``hil_harness`` monolith."""

from __future__ import annotations

import time

import numpy as np


class HILDemoRunner:
    """Simulate FPGA register-mapped SNN controller for demo/testing."""

    CLOCK_HZ = 250_000_000
    Q16_SCALE = 65536.0

    def __init__(self, n_neurons: int = 8, n_inputs: int = 4, n_outputs: int = 4):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.registers = np.zeros(512, dtype=np.uint32)
        self.tmr_copies = [np.zeros(n_neurons, dtype=np.float64) for _ in range(3)]
        self.weights: np.ndarray = np.zeros((n_neurons, n_inputs), dtype=np.float64)
        self.output_weights: np.ndarray = np.zeros((n_outputs, n_neurons), dtype=np.float64)
        self.tmr_mismatches = 0
        self.total_steps = 0
        self.latency_cycles: list[int] = []

    @staticmethod
    def float_to_q16_16(x: float) -> int:
        return int(round(x * HILDemoRunner.Q16_SCALE)) & 0xFFFFFFFF

    @staticmethod
    def q16_16_to_float(x: int) -> float:
        if x & 0x80000000:
            x -= 0x100000000
        return x / HILDemoRunner.Q16_SCALE

    def load_weights_from_controller(self, controller: object) -> None:
        """Load weights from a Python SNN controller object."""
        if hasattr(controller, "weights"):
            w = np.asarray(controller.weights, dtype=np.float64)
            self.weights = w[: self.n_neurons, : self.n_inputs]
        if hasattr(controller, "output_weights"):
            ow = np.asarray(controller.output_weights, dtype=np.float64)
            self.output_weights = ow[: self.n_outputs, : self.n_neurons]

    def _lif_step(
        self, state: np.ndarray, inputs: np.ndarray, dt_s: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]:
        tau = 0.02
        threshold = 1.0
        reset = 0.0
        current = self.weights @ inputs
        state = state * (1.0 - dt_s / tau) + current * dt_s
        spikes = (state >= threshold).astype(np.float64)
        state = np.where(spikes > 0, reset, state)
        return state, spikes

    def _tmr_vote(self) -> np.ndarray:
        """Majority vote across TMR copies; median is robust for analog values."""
        stacked = np.stack(self.tmr_copies, axis=0)
        voted = np.median(stacked, axis=0)
        for i in range(3):
            if not np.allclose(self.tmr_copies[i], voted, atol=0.01):
                self.tmr_mismatches += 1
                self.tmr_copies[i] = voted.copy()
                break
        return voted

    def step(self, inputs: np.ndarray) -> np.ndarray:
        """Execute one SNN inference step through simulated register pipeline."""
        t0 = time.perf_counter_ns()
        inp = np.asarray(inputs[: self.n_inputs], dtype=np.float64)

        for i in range(self.n_inputs):
            self.registers[0x60 // 4 + i] = self.float_to_q16_16(float(inp[i]))

        for c in range(3):
            self.tmr_copies[c], _spikes = self._lif_step(self.tmr_copies[c], inp)

        voted = self._tmr_vote()

        for i in range(min(self.n_neurons, 8)):
            self.registers[0x20 // 4 + i] = self.float_to_q16_16(float(voted[i]))

        output = self.output_weights @ voted
        for i in range(self.n_outputs):
            self.registers[0x70 // 4 + i] = self.float_to_q16_16(float(output[i]))

        elapsed_ns = time.perf_counter_ns() - t0
        cycles = max(1, int(elapsed_ns * self.CLOCK_HZ / 1e9))
        self.latency_cycles.append(cycles)
        self.registers[0x200 // 4] = np.uint32(cycles)
        self.total_steps += 1
        return output

    def inject_bitflip(self, neuron_idx: int = 0, bit_idx: int = 15) -> None:
        """Inject a single-bit fault into one TMR copy."""
        state_val = self.tmr_copies[0][neuron_idx]
        raw = np.array([state_val], dtype=np.float64).view(np.uint64)[0]
        flipped = np.uint64(raw ^ (np.uint64(1) << np.uint64(bit_idx)))
        new_val = np.array([flipped], dtype=np.uint64).view(np.float64)[0]
        if np.isfinite(new_val):
            self.tmr_copies[0][neuron_idx] = new_val

    def run_episode(self, n_steps: int = 1000, inject_faults: bool = False) -> dict:
        rng = np.random.default_rng(42)
        for t in range(n_steps):
            inputs = rng.normal(0, 0.1, size=self.n_inputs)
            if inject_faults and t % 100 == 50:
                self.inject_bitflip(neuron_idx=t % self.n_neurons, bit_idx=int(rng.integers(0, 52)))
            self.step(inputs)
        return self.report()

    def report(self) -> dict:
        lat = np.array(self.latency_cycles) if self.latency_cycles else np.array([0])
        return {
            "total_steps": self.total_steps,
            "tmr_mismatches": self.tmr_mismatches,
            "tmr_mismatch_rate": self.tmr_mismatches / max(self.total_steps, 1),
            "latency_mean_cycles": float(np.mean(lat)),
            "latency_p95_cycles": float(np.percentile(lat, 95)),
            "latency_max_cycles": float(np.max(lat)),
            "latency_mean_ns": float(np.mean(lat) / self.CLOCK_HZ * 1e9),
            "n_neurons": self.n_neurons,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }


__all__ = ["HILDemoRunner"]
