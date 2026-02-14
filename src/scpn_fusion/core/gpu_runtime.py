# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GPU Runtime Bridge (GDEP-02)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic CPU vs GPU-sim bridge for multigrid and SNN inference lanes."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class RuntimeBenchmark:
    backend: str
    multigrid_p95_ms_est: float
    snn_p95_ms_est: float
    multigrid_mean_ms_wall: float
    snn_mean_ms_wall: float


class GPURuntimeBridge:
    """Reduced runtime bridge with deterministic benchmark estimates."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(int(seed))
        self.w1 = rng.normal(0.0, 0.15, size=(32, 64))
        self.w2 = rng.normal(0.0, 0.15, size=(64, 8))

    @staticmethod
    def _require_int_at_least(value: int, *, name: str, minimum: int) -> int:
        v = int(value)
        if v < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        return v

    def _gpu_sim_multigrid(self, field: np.ndarray, iterations: int = 4) -> np.ndarray:
        u = field.astype(np.float64, copy=True)
        iterations = self._require_int_at_least(
            iterations, name="iterations", minimum=1
        )
        for _ in range(iterations):
            u = 0.2 * (
                u
                + np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
            )
        return u

    def _cpu_multigrid(self, field: np.ndarray, iterations: int = 4) -> np.ndarray:
        u = field.astype(np.float64, copy=True)
        n0, n1 = u.shape
        iterations = self._require_int_at_least(
            iterations, name="iterations", minimum=1
        )
        for _ in range(iterations):
            next_u = u.copy()
            for i in range(1, n0 - 1):
                for j in range(1, n1 - 1):
                    next_u[i, j] = 0.2 * (
                        u[i, j] + u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]
                    )
            u = next_u
        return u

    def _gpu_sim_snn(self, features: np.ndarray) -> np.ndarray:
        h = np.tanh(features @ self.w1)
        return np.tanh(h @ self.w2)

    def _cpu_snn(self, features: np.ndarray) -> np.ndarray:
        out = np.zeros((features.shape[0], self.w2.shape[1]), dtype=np.float64)
        for b in range(features.shape[0]):
            hidden = np.zeros(self.w1.shape[1], dtype=np.float64)
            for j in range(self.w1.shape[1]):
                acc = 0.0
                for i in range(self.w1.shape[0]):
                    acc += features[b, i] * self.w1[i, j]
                hidden[j] = np.tanh(acc)
            for k in range(self.w2.shape[1]):
                acc = 0.0
                for j in range(self.w2.shape[0]):
                    acc += hidden[j] * self.w2[j, k]
                out[b, k] = np.tanh(acc)
        return out

    @staticmethod
    def _ops_multigrid(grid_size: int, iterations: int) -> float:
        return float(grid_size * grid_size * iterations * 9)

    @staticmethod
    def _ops_snn(batch: int, n_in: int, n_hidden: int, n_out: int) -> float:
        return float(batch * (n_in * n_hidden + n_hidden * n_out))

    def benchmark(self, *, backend: str, trials: int = 64, grid_size: int = 64) -> RuntimeBenchmark:
        if backend not in {"cpu", "gpu_sim"}:
            raise ValueError("backend must be 'cpu' or 'gpu_sim'")

        trials = self._require_int_at_least(trials, name="trials", minimum=8)
        n = self._require_int_at_least(grid_size, name="grid_size", minimum=16)
        field = np.linspace(0.0, 1.0, n * n, dtype=np.float64).reshape(n, n)
        features = np.linspace(-1.0, 1.0, 32 * 8, dtype=np.float64).reshape(8, 32)

        mg_wall = []
        snn_wall = []
        mg_est = []
        snn_est = []

        # Deterministic op-throughput surrogate (ops/ms).
        throughput = 2.0e6 if backend == "cpu" else 2.5e7
        iterations = 4

        for k in range(trials):
            scale = 1.0 + 0.04 * np.sin(2.0 * np.pi * k / max(trials, 1))

            t0 = time.perf_counter()
            if backend == "cpu":
                _ = self._cpu_multigrid(field, iterations=iterations)
            else:
                _ = self._gpu_sim_multigrid(field, iterations=iterations)
            mg_wall.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            if backend == "cpu":
                _ = self._cpu_snn(features)
            else:
                _ = self._gpu_sim_snn(features)
            snn_wall.append((time.perf_counter() - t1) * 1000.0)

            mg_est.append(self._ops_multigrid(n, iterations) * scale / throughput)
            snn_est.append(self._ops_snn(features.shape[0], 32, 64, 8) * scale / throughput)

        return RuntimeBenchmark(
            backend=backend,
            multigrid_p95_ms_est=float(np.percentile(mg_est, 95)),
            snn_p95_ms_est=float(np.percentile(snn_est, 95)),
            multigrid_mean_ms_wall=float(np.mean(mg_wall)),
            snn_mean_ms_wall=float(np.mean(snn_wall)),
        )

    def benchmark_pair(self, *, trials: int = 64, grid_size: int = 64) -> dict[str, float | dict[str, float]]:
        cpu = self.benchmark(backend="cpu", trials=trials, grid_size=grid_size)
        gpu = self.benchmark(backend="gpu_sim", trials=trials, grid_size=grid_size)
        return {
            "cpu": {
                "multigrid_p95_ms_est": cpu.multigrid_p95_ms_est,
                "snn_p95_ms_est": cpu.snn_p95_ms_est,
                "multigrid_mean_ms_wall": cpu.multigrid_mean_ms_wall,
                "snn_mean_ms_wall": cpu.snn_mean_ms_wall,
            },
            "gpu_sim": {
                "multigrid_p95_ms_est": gpu.multigrid_p95_ms_est,
                "snn_p95_ms_est": gpu.snn_p95_ms_est,
                "multigrid_mean_ms_wall": gpu.multigrid_mean_ms_wall,
                "snn_mean_ms_wall": gpu.snn_mean_ms_wall,
            },
            "multigrid_speedup_est": float(
                cpu.multigrid_p95_ms_est / max(gpu.multigrid_p95_ms_est, 1e-9)
            ),
            "snn_speedup_est": float(cpu.snn_p95_ms_est / max(gpu.snn_p95_ms_est, 1e-9)),
        }
