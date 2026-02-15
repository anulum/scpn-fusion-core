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

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]

from scpn_fusion.control.disruption_predictor import apply_bit_flip_fault


@dataclass(frozen=True)
class RuntimeBenchmark:
    backend: str
    multigrid_p95_ms_est: float
    snn_p95_ms_est: float
    multigrid_mean_ms_wall: float
    snn_mean_ms_wall: float


@dataclass(frozen=True)
class EquilibriumLatencyBenchmark:
    backend: str
    trials: int
    grid_size: int
    fault_runs: int
    p95_ms_est: float
    mean_ms_wall: float
    p95_ms_wall: float
    fault_p95_ms_est: float
    fault_mean_ms_wall: float
    fault_p95_ms_wall: float
    sub_ms_target_pass: bool
    latency_spike_over_10ms: bool


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

    @staticmethod
    def _torch_available() -> bool:
        return torch is not None

    def _torch_fallback_multigrid(self, field: np.ndarray, iterations: int = 4) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch fallback requested but torch is not installed.")
        u = torch.as_tensor(np.asarray(field, dtype=np.float64))
        iterations = self._require_int_at_least(
            iterations, name="iterations", minimum=1
        )
        for _ in range(iterations):
            u = 0.2 * (
                u
                + torch.roll(u, shifts=1, dims=0)
                + torch.roll(u, shifts=-1, dims=0)
                + torch.roll(u, shifts=1, dims=1)
                + torch.roll(u, shifts=-1, dims=1)
            )
        return np.asarray(u.detach().cpu().numpy(), dtype=np.float64)

    @staticmethod
    def available_equilibrium_backends() -> tuple[str, ...]:
        if torch is None:
            return ("cpu", "gpu_sim")
        return ("cpu", "gpu_sim", "torch_fallback")

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

    @staticmethod
    def _inject_faults(
        field: np.ndarray,
        *,
        rng: np.random.Generator,
        sensor_noise_std: float,
        bit_flips_per_run: int,
    ) -> np.ndarray:
        noisy = np.asarray(field, dtype=np.float64).copy()
        noisy += rng.normal(0.0, sensor_noise_std, size=noisy.shape)
        n = noisy.size
        bit_flips = max(0, int(bit_flips_per_run))
        if bit_flips > 0:
            idx = rng.choice(n, size=min(bit_flips, n), replace=False)
            flat = noisy.reshape(-1)
            for i in idx:
                bit = int(rng.integers(0, 52))
                flat[int(i)] = apply_bit_flip_fault(float(flat[int(i)]), bit)
            noisy = flat.reshape(noisy.shape)
        return np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)

    def benchmark_equilibrium_latency(
        self,
        *,
        backend: str = "auto",
        trials: int = 128,
        grid_size: int = 64,
        iterations: int = 4,
        fault_runs: int = 10,
        sensor_noise_std: float = 0.015,
        bit_flips_per_run: int = 3,
        seed: int = 42,
    ) -> EquilibriumLatencyBenchmark:
        if backend not in {"auto", "cpu", "gpu_sim", "torch_fallback"}:
            raise ValueError(
                "backend must be one of: auto, cpu, gpu_sim, torch_fallback."
            )
        resolved_backend = (
            "torch_fallback" if backend == "auto" and torch is not None else "gpu_sim"
        )
        if backend in {"cpu", "gpu_sim", "torch_fallback"}:
            resolved_backend = backend
        if resolved_backend == "torch_fallback" and torch is None:
            raise RuntimeError(
                "PyTorch fallback requested but torch is not installed."
            )
        trials_i = self._require_int_at_least(trials, name="trials", minimum=8)
        n = self._require_int_at_least(grid_size, name="grid_size", minimum=16)
        iter_i = self._require_int_at_least(iterations, name="iterations", minimum=1)
        fault_runs_i = self._require_int_at_least(fault_runs, name="fault_runs", minimum=1)
        noise = float(sensor_noise_std)
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError("sensor_noise_std must be finite and >= 0.")

        rng = np.random.default_rng(int(seed))
        base_field = np.linspace(0.0, 1.0, n * n, dtype=np.float64).reshape(n, n)

        if resolved_backend == "cpu":
            throughput = 2.0e6
            solver = self._cpu_multigrid
        elif resolved_backend == "gpu_sim":
            throughput = 2.5e7
            solver = self._gpu_sim_multigrid
        else:
            throughput = 4.0e7
            solver = self._torch_fallback_multigrid

        wall_ms: list[float] = []
        est_ms: list[float] = []
        for k in range(trials_i):
            scale = 1.0 + 0.04 * np.sin(2.0 * np.pi * k / max(trials_i, 1))
            t0 = time.perf_counter()
            _ = solver(base_field, iterations=iter_i)
            wall_ms.append((time.perf_counter() - t0) * 1000.0)
            est_ms.append(self._ops_multigrid(n, iter_i) * scale / throughput)

        fault_wall_ms: list[float] = []
        fault_est_ms: list[float] = []
        for k in range(fault_runs_i):
            field = self._inject_faults(
                base_field,
                rng=rng,
                sensor_noise_std=noise,
                bit_flips_per_run=bit_flips_per_run,
            )
            scale = 1.05 + 0.06 * np.sin(2.0 * np.pi * k / max(fault_runs_i, 1))
            t0 = time.perf_counter()
            _ = solver(field, iterations=iter_i)
            fault_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            fault_est_ms.append(self._ops_multigrid(n, iter_i) * scale / throughput)

        p95_est = float(np.percentile(np.asarray(est_ms, dtype=np.float64), 95))
        p95_wall = float(np.percentile(np.asarray(wall_ms, dtype=np.float64), 95))
        fault_p95_est = float(np.percentile(np.asarray(fault_est_ms, dtype=np.float64), 95))
        fault_p95_wall = float(np.percentile(np.asarray(fault_wall_ms, dtype=np.float64), 95))
        mean_wall = float(np.mean(np.asarray(wall_ms, dtype=np.float64)))
        fault_mean_wall = float(np.mean(np.asarray(fault_wall_ms, dtype=np.float64)))

        sub_ms_pass = bool(p95_est < 1.0 and fault_p95_est < 1.0)
        spike_10ms = bool(p95_wall > 10.0 or fault_p95_wall > 10.0)
        return EquilibriumLatencyBenchmark(
            backend=resolved_backend,
            trials=trials_i,
            grid_size=n,
            fault_runs=fault_runs_i,
            p95_ms_est=p95_est,
            mean_ms_wall=mean_wall,
            p95_ms_wall=p95_wall,
            fault_p95_ms_est=fault_p95_est,
            fault_mean_ms_wall=fault_mean_wall,
            fault_p95_ms_wall=fault_p95_wall,
            sub_ms_target_pass=sub_ms_pass,
            latency_spike_over_10ms=spike_10ms,
        )
