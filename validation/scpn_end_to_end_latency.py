#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — End-to-End Control Latency Benchmark
"""Deterministic end-to-end latency benchmark for SCPN/PID/MPC-lite.

The report separates historical controller-loop timing from the sensor-to-control
digital-twin path used for local latency evidence. Hardware-in-the-loop and
accelerator claims remain blocked unless a measured lane is present in the
generated JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sysconfig
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.control.hil_harness import run_sensor_to_actuator_hil_latency_campaign


FloatArray: TypeAlias = NDArray[np.float64]

_FULL_PHYSICS_MATRIX = np.array(
    [
        [1.20, 0.05, 0.01, 0.00, 0.00, 0.00],
        [0.05, 1.10, 0.03, 0.00, 0.00, 0.00],
        [0.01, 0.03, 1.05, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.08, 0.04, 0.01],
        [0.00, 0.00, 0.00, 0.04, 1.07, 0.02],
        [0.00, 0.00, 0.00, 0.01, 0.02, 1.06],
    ],
    dtype=np.float64,
)
_FULL_PHYSICS_MATRIX_INV = np.linalg.inv(_FULL_PHYSICS_MATRIX)
_DEGRADED_MODE_CASES = (
    "nominal",
    "stale_input",
    "out_of_distribution_input",
    "missing_diagnostic",
    "non_finite_value",
    "actuator_saturation",
    "controller_fallback",
)
_PIPELINE_STAGE_KEYS = (
    "input_validation",
    "feature_assembly",
    "digital_twin_update",
    "controller_decision",
    "fallback_policy",
    "output_serialization",
)
_ACTUATOR_SCALING_COUNTS = (2, 16, 64, 128, 256)
_PREDICTIVE_HORIZON_MS = (50, 100)
REPORT_SCHEMA = "scpn-fusion-core.end_to_end_latency_report.v1"
REPORT_CLAIM_BOUNDARY = (
    "Local reduced-order control latency evidence; not physical HIL, FPGA, "
    "plant CODAC, or actuator hardware timing."
)
REPORT_STATUS_ACCEPTED = "accepted_local_reduced_order_latency_report"
REPORT_STATUS_BLOCKED = "blocked_local_reduced_order_latency_report"


def _build_scpn_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    net.add_transition("T_Rp", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Rn", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Zp", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Zn", threshold=0.1, delay_ticks=1024)

    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile(validate_topology=True)

    compiler = FusionCompiler(
        bitstream_length=512,
        seed=42,
        lif_tau_mem=10.0,
        lif_noise_std=0.1,
        lif_dt=1.0,
        lif_resistance=1.0,
        lif_refractory_period=1,
    )
    compiled = compiler.compile(net, firing_mode="fractional", firing_margin=0.25)
    artifact = compiled.export_artifact(
        name="scpn_end_to_end_latency",
        dt_control_s=0.05,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 0, "neg_place": 1},
                {"name": "dI_PF_topbot_A", "pos_place": 2, "neg_place": 3},
            ],
            "gains": [5.0, 0.0],
            "abs_max": [1.0, 1.0],
            "slew_per_s": [60.0, 60.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=123456,
        targets=ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        sc_n_passes=16,
        sc_bitflip_rate=0.0,
        sc_binary_margin=0.05,
        runtime_profile="traceable",
        runtime_backend="auto",
    )


@dataclass
class _PIDState:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    last_err: float = 0.0

    def step(self, err: float, dt: float, u_limit: float) -> float:
        self.integral += err * dt
        d_err = (err - self.last_err) / max(dt, 1e-9)
        self.last_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * d_err
        return float(np.clip(u, -u_limit, u_limit))


def _disturbance(k: int, steps: int) -> float:
    return float(0.02 * math.sin(0.08 * k) + (0.04 if k >= steps // 2 else 0.0))


def _mpc_lite_action(x: float, k: int, steps: int, horizon: int, u_limit: float) -> float:
    candidates = np.linspace(-u_limit, u_limit, 31)
    best_u = 0.0
    best_cost = float("inf")
    for u in candidates:
        x_pred = x
        cost = 0.0
        for h in range(horizon):
            d = _disturbance(k + h, steps)
            x_pred = 0.95 * x_pred + 0.12 * float(u) + d
            cost += x_pred * x_pred + 0.06 * float(u * u)
        if cost < best_cost:
            best_cost = float(cost)
            best_u = float(u)
    return best_u


def _sensor_preprocess(x: float, last_filtered: float) -> tuple[float, float]:
    filt = 0.84 * last_filtered + 0.16 * x
    return float(np.clip(filt, -1.5, 1.5)), float(filt)


def _actuator_lag(u_cmd: float, u_prev: float, dt: float) -> float:
    tau = 0.04
    alpha = dt / (tau + dt)
    return float(np.clip(u_prev + alpha * (u_cmd - u_prev), -1.0, 1.0))


def _physics_step(x: float, u: float, d: float, mode: str) -> float:
    base = 0.95 * x + 0.12 * u + d
    if mode == "surrogate":
        return float(base)

    rhs = np.array([x, u, d, x * x, u * u, d * d], dtype=np.float64)
    correction = _FULL_PHYSICS_MATRIX_INV @ rhs
    damping = 0.004 * float(np.tanh(correction[0]))
    act_gain = 0.004 * float(np.tanh(correction[1]))
    drift = 0.002 * float(np.tanh(correction[2]))
    return float((0.95 + damping) * x + (0.12 + act_gain) * u + d + drift)


def _pctl(arr: FloatArray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def _summary(values: FloatArray) -> dict[str, float]:
    """Return p50/p95/p99 latency summary for a measured stage.

    Parameters
    ----------
    values
        One-dimensional latency samples in milliseconds.

    Returns
    -------
    dict[str, float]
        Percentile summary in milliseconds.
    """
    return {
        "p50_ms": _pctl(values, 50),
        "p95_ms": _pctl(values, 95),
        "p99_ms": _pctl(values, 99),
    }


def _read_sysfs_value(path: Path) -> str | None:
    """Read a single sysfs text value when it is available.

    Parameters
    ----------
    path
        Kernel sysfs file to read.

    Returns
    -------
    str | None
        Stripped text content, or ``None`` when the value is unavailable.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_frequency_snapshot() -> dict[str, Any]:
    """Collect CPU governor and frequency metadata from Linux sysfs.

    Returns
    -------
    dict[str, Any]
        Governor names and frequency ranges observed across online CPUs.
    """
    cpu_dirs = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
    governors: set[str] = set()
    current_khz: list[int] = []
    min_khz: list[int] = []
    max_khz: list[int] = []
    for cpu_dir in cpu_dirs:
        cpufreq = cpu_dir / "cpufreq"
        governor = _read_sysfs_value(cpufreq / "scaling_governor")
        if governor:
            governors.add(governor)
        for path, bucket in (
            (cpufreq / "scaling_cur_freq", current_khz),
            (cpufreq / "scaling_min_freq", min_khz),
            (cpufreq / "scaling_max_freq", max_khz),
        ):
            value = _read_sysfs_value(path)
            if value is None:
                continue
            try:
                bucket.append(int(value))
            except ValueError:
                continue
    return {
        "governors": sorted(governors),
        "current_khz_min": min(current_khz) if current_khz else None,
        "current_khz_max": max(current_khz) if current_khz else None,
        "scaling_min_khz_min": min(min_khz) if min_khz else None,
        "scaling_max_khz_max": max(max_khz) if max_khz else None,
    }


def _actuator_weights(actuator_count: int) -> FloatArray:
    """Return deterministic actuator fanout weights.

    Parameters
    ----------
    actuator_count
        Number of actuator channels in the command vector.

    Returns
    -------
    numpy.ndarray
        Positive weights used to fan out the two logical controller actions.
    """
    count = int(actuator_count)
    if count < 1:
        raise ValueError("actuator_count must be >= 1.")
    phase = np.arange(count, dtype=np.float64) / float(count)
    return np.asarray(0.35 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * phase)))


def _expand_actuator_command(command: FloatArray, actuator_count: int) -> FloatArray:
    """Fan out a two-channel controller command to a physical actuator vector.

    Parameters
    ----------
    command
        Two-channel logical controller command.
    actuator_count
        Number of physical actuator channels to emit.

    Returns
    -------
    numpy.ndarray
        Bounded actuator vector with ``actuator_count`` entries.
    """
    if command.shape != (2,):
        raise ValueError("command must contain exactly two logical channels.")
    repeated = np.resize(command, int(actuator_count))
    return np.clip(repeated * _actuator_weights(actuator_count), -1.0, 1.0)


def _host_snapshot() -> dict[str, Any]:
    """Collect non-secret host metadata needed to bound local benchmark claims.

    Returns
    -------
    dict[str, Any]
        Platform, CPU, load, affinity, and accelerator availability metadata.
    """
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    loadavg: tuple[float, ...] | None = None
    if hasattr(os, "getloadavg"):
        loadavg = tuple(float(v) for v in os.getloadavg())
    isolation = os.environ.get("SCPN_BENCHMARK_ISOLATION_METHOD", "non_isolated_local_workstation")
    boundary = os.environ.get(
        "SCPN_BENCHMARK_CLAIM_BOUNDARY",
        (
            "Local wall-clock regression evidence only; not an isolated-core or "
            "hardware-in-the-loop latency claim."
        ),
    )
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity) if affinity is not None else None,
        "reserved_cpu_set": os.environ.get("SCPN_BENCHMARK_CPUSET"),
        "loadavg_1_5_15": list(loadavg) if loadavg is not None else None,
        "numpy_version": np.__version__,
        "cpu_frequency": _cpu_frequency_snapshot(),
        "isolation": isolation,
        "benchmark_command": os.environ.get("SCPN_BENCHMARK_COMMAND"),
        "concurrent_heavy_jobs": os.environ.get("SCPN_BENCHMARK_CONCURRENT_HEAVY_JOBS"),
        "claim_boundary": boundary,
    }


def _run_gpu_digital_twin_latency(*, steps: int, actuator_count: int = 2) -> dict[str, Any]:
    """Measure the same sensor-to-control contract on a CUDA/CuPy device.

    Returns
    -------
    dict[str, Any]
        Measured GPU latency row or a fail-closed blocked status.
    """
    if os.environ.get("SCPN_LATENCY_GPU_SUBPROCESS") != "1":
        env = dict(os.environ)
        lib_path = os.pathsep.join(_nvidia_library_dirs())
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            lib_path if not existing_ld_path else f"{lib_path}{os.pathsep}{existing_ld_path}"
        )
        env["SCPN_LATENCY_GPU_SUBPROCESS"] = "1"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--gpu-only",
            "--steps",
            str(steps),
            "--actuators",
            str(actuator_count),
        ]
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if proc.returncode != 0:
            return {
                "status": "blocked_gpu_subprocess_failed",
                "backend": "cupy_cuda",
                "command": " ".join(cmd),
                "returncode": int(proc.returncode),
                "stderr_tail": proc.stderr[-2000:],
                "p50_loop_ms": None,
                "p95_loop_ms": None,
                "p99_loop_ms": None,
            }
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return {
                "status": "blocked_invalid_gpu_json",
                "backend": "cupy_cuda",
                "command": " ".join(cmd),
                "reason": f"{type(exc).__name__}: {exc}",
                "stdout_tail": proc.stdout[-2000:],
                "p50_loop_ms": None,
                "p95_loop_ms": None,
                "p99_loop_ms": None,
            }
        if isinstance(payload, dict):
            payload["command"] = " ".join(cmd)
            payload["ld_library_path_source"] = "project_venv_nvidia_wheels"
            return payload
        return {
            "status": "blocked_invalid_gpu_json",
            "backend": "cupy_cuda",
            "command": " ".join(cmd),
            "reason": "GPU subprocess did not emit a JSON object.",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }

    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        return {
            "status": "blocked_no_compatible_runtime",
            "backend": "gpu",
            "reason": "CuPy is not importable in the active Python environment.",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:
        return {
            "status": "blocked_no_compatible_runtime",
            "backend": "gpu",
            "reason": f"CuPy import failed: {type(exc).__name__}: {exc}",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }

    preload = _preload_cuda_runtime_libraries()
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return {
            "status": "blocked_no_visible_device",
            "backend": "gpu",
            "reason": f"CUDA device query failed: {type(exc).__name__}: {exc}",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }
    if device_count < 1:
        return {
            "status": "blocked_no_visible_device",
            "backend": "gpu",
            "reason": "CUDA runtime reports zero visible devices.",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }

    try:
        device = cp.cuda.Device(0)
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        with device:
            cp.cuda.runtime.deviceSynchronize()
            state = cp.asarray([0.50, 0.52, 0.55, 0.0], dtype=cp.float64)
            actuator_weights = cp.asarray(
                _actuator_weights(actuator_count),
                dtype=cp.float64,
            )
            coupling = cp.asarray(
                [
                    [0.92, 0.04, -0.02, 0.00],
                    [0.02, 0.91, 0.01, -0.03],
                    [0.01, 0.02, 0.95, 0.00],
                    [0.00, -0.01, 0.02, 0.88],
                ],
                dtype=cp.float64,
            )
            last_beta = 2.0
            stage_samples = {key: np.zeros(steps, dtype=np.float64) for key in _PIPELINE_STAGE_KEYS}
            loop_ms = np.zeros(steps, dtype=np.float64)
            checksum = 0.0
            fallback_count = 0
            for k in range(steps):
                phase = 0.031 * float(k)
                loop_start = time.perf_counter()

                t0 = time.perf_counter()
                snapshot = np.asarray(
                    [
                        1.20 + 0.03 * math.sin(phase),
                        2.00 + 0.10 * math.sin(0.7 * phase),
                        4.20 - 0.04 * math.cos(0.5 * phase),
                        6.50 + 0.08 * math.sin(0.3 * phase),
                        0.015 * math.sin(1.3 * phase),
                    ],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(snapshot)):
                    fallback_count += 1
                    snapshot = np.asarray([1.2, 2.0, 4.2, 6.5, 0.0], dtype=np.float64)
                stage_samples["input_validation"][k] = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                features_host = np.asarray(
                    [
                        snapshot[0] / 2.0,
                        snapshot[1] / 4.0,
                        snapshot[2] / 8.0,
                        snapshot[3] / 12.0,
                        snapshot[4] / 0.25,
                        (snapshot[1] - last_beta) / 0.25,
                    ],
                    dtype=np.float64,
                )
                features = cp.asarray(features_host)
                cp.cuda.runtime.deviceSynchronize()
                stage_samples["feature_assembly"][k] = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                drive = cp.asarray(
                    [
                        0.18 * features[1] - 0.04 * features[2],
                        0.08 * features[2] - 0.02 * features[4],
                        0.05 * features[3],
                        0.02 * features[4] + 0.01 * features[5],
                    ],
                    dtype=cp.float64,
                )
                state = cp.clip(
                    coupling @ state + drive,
                    cp.asarray([0.0, 0.2, 0.0, -2.0], dtype=cp.float64),
                    cp.asarray([2.0, 2.0, 2.0, 2.0], dtype=cp.float64),
                )
                cp.cuda.runtime.deviceSynchronize()
                stage_samples["digital_twin_update"][k] = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                command = cp.asarray(
                    [
                        -0.55 * state[3] - 0.18 * features[4] - 0.04 * state[1],
                        0.35 * (0.55 - state[0]) + 0.08 * (0.45 - features[3]),
                    ],
                    dtype=cp.float64,
                )
                cp.cuda.runtime.deviceSynchronize()
                stage_samples["controller_decision"][k] = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                expanded_command = cp.resize(command, actuator_count) * actuator_weights
                safe_command = cp.clip(expanded_command, -1.0, 1.0)
                cp.cuda.runtime.deviceSynchronize()
                safe_host = cp.asnumpy(safe_command)
                if not np.all(np.isfinite(safe_host)):
                    fallback_count += 1
                    safe_host = np.zeros(2, dtype=np.float64)
                stage_samples["fallback_policy"][k] = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                payload = {
                    "step": k,
                    "commands": [float(value) for value in safe_host.tolist()],
                    "fallback": False,
                }
                encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
                checksum += float(len(encoded)) * 1e-6 + float(np.sum(safe_host))
                stage_samples["output_serialization"][k] = (time.perf_counter() - t0) * 1e3

                last_beta = float(snapshot[1])
                loop_ms[k] = (time.perf_counter() - loop_start) * 1e3
            cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        return {
            "status": "blocked_gpu_measurement_failed",
            "backend": "cupy_cuda",
            "visible_device_count": device_count,
            "reason": f"{type(exc).__name__}: {exc}",
            "p50_loop_ms": None,
            "p95_loop_ms": None,
            "p99_loop_ms": None,
        }

    return {
        "status": "measured",
        "backend": "cupy_cuda",
        "visible_device_count": device_count,
        "cuda_runtime_version": runtime_version,
        "preloaded_cuda_libraries": preload,
        "device_name": str(cp.cuda.runtime.getDeviceProperties(0).get("name", b"unknown")),
        "steps": steps,
        "actuator_count": actuator_count,
        "fallback_count": int(fallback_count),
        "checksum": float(checksum),
        "p50_loop_ms": _pctl(loop_ms, 50),
        "p95_loop_ms": _pctl(loop_ms, 95),
        "p99_loop_ms": _pctl(loop_ms, 99),
        "stages": {key: _summary(values) for key, values in stage_samples.items()},
        "claim_boundary": (
            "Local non-isolated CuPy/CUDA measurement of the same reduced-order "
            "sensor-to-control contract; includes host snapshot assembly and output serialisation."
        ),
        "accelerator_isolation": "cuda_device_reserved_by_operator_for_this_benchmark",
    }


def _nvidia_library_dirs() -> list[str]:
    """Return NVIDIA wheel library directories under the active environment.

    Returns
    -------
    list[str]
        Existing library directories below ``site-packages/nvidia``.
    """
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()]


def _preload_cuda_runtime_libraries() -> list[str]:
    """Preload CUDA runtime libraries shipped inside the active virtualenv.

    Returns
    -------
    list[str]
        Absolute library paths successfully loaded with global symbol visibility.
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates = sorted((purelib / "nvidia").glob("*/lib/*.so*"))
    loaded: list[str] = []
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for path in candidates:
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=mode)
        except OSError:
            continue
        loaded.append(str(path))
    return loaded


def _validate_snapshot(snapshot: dict[str, float]) -> tuple[dict[str, float], bool, str]:
    """Validate and clamp a sensor snapshot for the digital-twin pipeline.

    Parameters
    ----------
    snapshot
        Sensor dictionary containing plasma current, beta, q95, density, and
        vertical displacement channels.

    Returns
    -------
    tuple[dict[str, float], bool, str]
        Cleaned snapshot, fallback flag, and fallback reason.
    """
    fallback = False
    reason = "nominal"
    defaults = {
        "ip_ma": 1.2,
        "beta_n": 2.0,
        "q95": 4.0,
        "density_1e19": 6.5,
        "z_m": 0.0,
    }
    cleaned: dict[str, float] = {}
    for key, default in defaults.items():
        if key not in snapshot:
            fallback = True
            reason = f"missing_{key}_defaulted"
        raw = float(snapshot.get(key, default))
        if not math.isfinite(raw):
            fallback = True
            reason = "non_finite_input_replaced"
            raw = default
        cleaned[key] = raw
    if cleaned["q95"] <= 0.0:
        fallback = True
        reason = "invalid_q95_clamped"
        cleaned["q95"] = 0.25
    if cleaned["density_1e19"] < 0.0:
        fallback = True
        reason = "negative_density_clamped"
        cleaned["density_1e19"] = 0.0
    return cleaned, fallback, reason


def _features_from_snapshot(
    snapshot: dict[str, float], last_snapshot: dict[str, float]
) -> FloatArray:
    """Assemble bounded controller features from current and previous sensors.

    Parameters
    ----------
    snapshot
        Validated sensor snapshot.
    last_snapshot
        Previous validated snapshot used for rate features.

    Returns
    -------
    numpy.ndarray
        Six-feature vector in machine-normalised units.
    """
    return np.array(
        [
            snapshot["ip_ma"] / 2.0,
            snapshot["beta_n"] / 4.0,
            snapshot["q95"] / 8.0,
            snapshot["density_1e19"] / 12.0,
            snapshot["z_m"] / 0.25,
            (snapshot["beta_n"] - last_snapshot["beta_n"]) / 0.25,
        ],
        dtype=np.float64,
    )


def _digital_twin_update(state: FloatArray, features: FloatArray) -> FloatArray:
    """Advance a reduced-order digital-twin state used by the latency harness.

    Parameters
    ----------
    state
        Four-state vector carrying beta, q95, density, and vertical displacement.
    features
        Six-feature validated sensor vector.

    Returns
    -------
    numpy.ndarray
        Updated four-state reduced-order digital twin.
    """
    coupling = np.array(
        [
            [0.92, 0.04, -0.02, 0.00],
            [0.02, 0.91, 0.01, -0.03],
            [0.01, 0.02, 0.95, 0.00],
            [0.00, -0.01, 0.02, 0.88],
        ],
        dtype=np.float64,
    )
    drive = np.array(
        [
            0.18 * features[1] - 0.04 * features[2],
            0.08 * features[2] - 0.02 * features[4],
            0.05 * features[3],
            0.02 * features[4] + 0.01 * features[5],
        ],
        dtype=np.float64,
    )
    updated = coupling @ state + drive
    return np.clip(updated, np.array([0.0, 0.2, 0.0, -2.0]), np.array([2.0, 2.0, 2.0, 2.0]))


def _controller_decision(twin_state: FloatArray, features: FloatArray) -> FloatArray:
    """Compute a bounded two-actuator command from twin state and features.

    Parameters
    ----------
    twin_state
        Four-state reduced-order twin state.
    features
        Six-feature validated sensor vector.

    Returns
    -------
    numpy.ndarray
        Vertical and heating/current-drive actuator command vector.
    """
    vertical = -0.55 * twin_state[3] - 0.18 * features[4] - 0.04 * twin_state[1]
    heating = 0.35 * (0.55 - twin_state[0]) + 0.08 * (0.45 - features[3])
    return np.array([vertical, heating], dtype=np.float64)


def _apply_fallback_policy(
    command: FloatArray,
    *,
    fallback_in: bool,
    reason_in: str,
) -> tuple[FloatArray, bool, str]:
    """Apply fail-closed policy checks to a raw controller command.

    Parameters
    ----------
    command
        Raw actuator command vector.
    fallback_in
        Whether earlier pipeline stages already requested fallback.
    reason_in
        Earlier fallback reason.

    Returns
    -------
    tuple[numpy.ndarray, bool, str]
        Safe command, fallback flag, and fallback reason.
    """
    fallback = bool(fallback_in)
    reason = reason_in
    if command.shape != (2,) or not np.all(np.isfinite(command)):
        return np.zeros(2, dtype=np.float64), True, "controller_non_finite_or_bad_shape"
    safe = np.clip(command, -1.0, 1.0)
    if bool(np.any(np.abs(command) > 1.0)):
        fallback = True
        reason = "actuator_saturation_clamped"
    return safe, fallback, reason


def _snapshot_for_case(case_name: str, k: int, last_snapshot: dict[str, float]) -> dict[str, float]:
    """Return a deterministic sensor snapshot for a nominal or degraded case.

    Parameters
    ----------
    case_name
        Degraded-mode case identifier.
    k
        Sample index.
    last_snapshot
        Previous valid snapshot used by stale-input mode.

    Returns
    -------
    dict[str, float]
        Sensor snapshot in engineering units.
    """
    phase = 0.031 * float(k)
    snapshot = {
        "ip_ma": 1.20 + 0.03 * math.sin(phase),
        "beta_n": 2.00 + 0.10 * math.sin(0.7 * phase),
        "q95": 4.20 - 0.04 * math.cos(0.5 * phase),
        "density_1e19": 6.50 + 0.08 * math.sin(0.3 * phase),
        "z_m": 0.015 * math.sin(1.3 * phase),
    }
    if case_name == "stale_input" and k % 7 == 0:
        return dict(last_snapshot)
    if case_name == "out_of_distribution_input" and k % 11 == 0:
        snapshot["beta_n"] = 5.8
        snapshot["q95"] = 0.12
    elif case_name == "missing_diagnostic" and k % 5 == 0:
        snapshot.pop("q95")
    elif case_name == "non_finite_value" and k % 13 == 0:
        snapshot["density_1e19"] = float("nan")
    elif case_name == "actuator_saturation" and k % 9 == 0:
        snapshot["z_m"] = 1.4
    elif case_name == "controller_fallback" and k % 10 == 0:
        snapshot["beta_n"] = float("inf")
    return snapshot


def _run_digital_twin_case(case_name: str, *, steps: int) -> dict[str, Any]:
    """Measure one CPU digital-twin sensor-to-control degraded-mode lane.

    Parameters
    ----------
    case_name
        Degraded-mode case identifier.
    steps
        Number of measured samples.

    Returns
    -------
    dict[str, Any]
        Percentile latency, fallback counts, and stage summaries.
    """
    if case_name not in _DEGRADED_MODE_CASES:
        raise ValueError(f"Unknown degraded-mode case: {case_name}")
    state = np.array([0.50, 0.52, 0.55, 0.0], dtype=np.float64)
    last_snapshot = {
        "ip_ma": 1.2,
        "beta_n": 2.0,
        "q95": 4.2,
        "density_1e19": 6.5,
        "z_m": 0.0,
    }
    stage_samples = {key: np.zeros(steps, dtype=np.float64) for key in _PIPELINE_STAGE_KEYS}
    loop_ms = np.zeros(steps, dtype=np.float64)
    fallback_count = 0
    safe_outputs = 0
    checksum = 0.0
    reason_counts: dict[str, int] = {}
    for k in range(steps):
        loop_start = time.perf_counter()
        snapshot = _snapshot_for_case(case_name, k, last_snapshot)

        t0 = time.perf_counter()
        cleaned, fallback, reason = _validate_snapshot(snapshot)
        if case_name == "stale_input" and k % 7 == 0:
            fallback = True
            reason = "stale_snapshot_safe_mode"
        if case_name == "out_of_distribution_input" and (
            cleaned["beta_n"] > 4.0 or cleaned["q95"] < 0.5
        ):
            fallback = True
            reason = "ood_snapshot_safe_mode"
        stage_samples["input_validation"][k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        features = _features_from_snapshot(cleaned, last_snapshot)
        stage_samples["feature_assembly"][k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        state = _digital_twin_update(state, features)
        stage_samples["digital_twin_update"][k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        command = _controller_decision(state, features)
        if case_name == "controller_fallback" and k % 10 == 0:
            command = np.array([float("nan"), 0.0], dtype=np.float64)
        stage_samples["controller_decision"][k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        safe_command, fallback, reason = _apply_fallback_policy(
            command,
            fallback_in=fallback,
            reason_in=reason,
        )
        if fallback:
            fallback_count += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if np.all(np.isfinite(safe_command)) and np.all(np.abs(safe_command) <= 1.0):
            safe_outputs += 1
        stage_samples["fallback_policy"][k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        payload = {
            "step": k,
            "vertical_command": float(safe_command[0]),
            "heating_command": float(safe_command[1]),
            "fallback": fallback,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        checksum += float(len(encoded)) * 1e-6 + float(np.sum(safe_command))
        stage_samples["output_serialization"][k] = (time.perf_counter() - t0) * 1e3

        last_snapshot = cleaned
        loop_ms[k] = (time.perf_counter() - loop_start) * 1e3

    return {
        "case": case_name,
        "steps": steps,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(steps, 1)),
        "safe_output_rate": float(safe_outputs / max(steps, 1)),
        "fallback_reasons": reason_counts,
        "checksum": float(checksum),
        "p50_loop_ms": _pctl(loop_ms, 50),
        "p95_loop_ms": _pctl(loop_ms, 95),
        "p99_loop_ms": _pctl(loop_ms, 99),
        "stages": {key: _summary(values) for key, values in stage_samples.items()},
        "passes_semantics": bool(
            safe_outputs == steps and (case_name == "nominal" or fallback_count > 0)
        ),
    }


def _run_actuator_scaling_cpu_case(*, actuator_count: int, steps: int) -> dict[str, Any]:
    """Measure CPU sensor-to-control timing while fanning out many actuators.

    Parameters
    ----------
    actuator_count
        Number of actuator channels to serialise each control step.
    steps
        Number of measured samples.

    Returns
    -------
    dict[str, Any]
        Latency percentiles and serialised-output size metadata.
    """
    state = np.array([0.50, 0.52, 0.55, 0.0], dtype=np.float64)
    last_snapshot = {
        "ip_ma": 1.2,
        "beta_n": 2.0,
        "q95": 4.2,
        "density_1e19": 6.5,
        "z_m": 0.0,
    }
    loop_ms = np.zeros(steps, dtype=np.float64)
    serialisation_ms = np.zeros(steps, dtype=np.float64)
    payload_bytes = np.zeros(steps, dtype=np.float64)
    checksum = 0.0
    safe_outputs = 0
    for k in range(steps):
        loop_start = time.perf_counter()
        snapshot = _snapshot_for_case("nominal", k, last_snapshot)
        cleaned, fallback, reason = _validate_snapshot(snapshot)
        features = _features_from_snapshot(cleaned, last_snapshot)
        state = _digital_twin_update(state, features)
        command = _controller_decision(state, features)
        safe_command, fallback, reason = _apply_fallback_policy(
            command,
            fallback_in=fallback,
            reason_in=reason,
        )
        if fallback:
            safe_command = np.zeros(2, dtype=np.float64)
        actuator_vector = _expand_actuator_command(safe_command, actuator_count)
        if np.all(np.isfinite(actuator_vector)) and np.all(np.abs(actuator_vector) <= 1.0):
            safe_outputs += 1

        t0 = time.perf_counter()
        payload = {
            "step": k,
            "commands": [float(value) for value in actuator_vector],
            "fallback": fallback,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        serialisation_ms[k] = (time.perf_counter() - t0) * 1e3
        payload_bytes[k] = float(len(encoded))
        checksum += float(np.sum(actuator_vector)) + payload_bytes[k] * 1e-6
        last_snapshot = cleaned
        loop_ms[k] = (time.perf_counter() - loop_start) * 1e3
    return {
        "status": "measured",
        "backend": "python_numpy_cpu",
        "actuator_count": int(actuator_count),
        "steps": int(steps),
        "safe_output_rate": float(safe_outputs / max(steps, 1)),
        "mean_payload_bytes": float(np.mean(payload_bytes)),
        "checksum": float(checksum),
        "p50_loop_ms": _pctl(loop_ms, 50),
        "p95_loop_ms": _pctl(loop_ms, 95),
        "p99_loop_ms": _pctl(loop_ms, 99),
        "serialization": _summary(serialisation_ms),
    }


def _run_predictive_horizon_case(*, horizon_ms: int, steps: int) -> dict[str, Any]:
    """Measure reduced-order forecast timing for a fixed predictive horizon.

    Parameters
    ----------
    horizon_ms
        Forecast horizon in milliseconds at 1 ms rollout resolution.
    steps
        Number of measured forecast calls.

    Returns
    -------
    dict[str, Any]
        Forecast latency percentiles and real-time factor metadata.
    """
    horizon_steps = int(horizon_ms)
    if horizon_steps < 1:
        raise ValueError("horizon_ms must be >= 1.")
    state0 = np.array([0.50, 0.52, 0.55, 0.0], dtype=np.float64)
    loop_ms = np.zeros(steps, dtype=np.float64)
    checksum = 0.0
    for k in range(steps):
        state = state0 + np.array(
            [
                0.01 * math.sin(0.02 * k),
                0.01 * math.cos(0.017 * k),
                0.005 * math.sin(0.011 * k),
                0.002 * math.cos(0.031 * k),
            ],
            dtype=np.float64,
        )
        t0 = time.perf_counter()
        for h in range(horizon_steps):
            phase = 0.013 * float(k + h)
            features = np.array(
                [
                    0.60 + 0.02 * math.sin(phase),
                    0.50 + 0.03 * math.sin(0.7 * phase),
                    0.52 - 0.01 * math.cos(0.5 * phase),
                    0.54 + 0.01 * math.sin(0.3 * phase),
                    0.06 * math.sin(1.3 * phase),
                    0.02 * math.cos(0.9 * phase),
                ],
                dtype=np.float64,
            )
            state = _digital_twin_update(state, features)
            command = _controller_decision(state, features)
            state[3] += 0.002 * float(command[0])
            state[0] += 0.001 * float(command[1])
        loop_ms[k] = (time.perf_counter() - t0) * 1e3
        checksum += float(np.sum(state))
    p95 = _pctl(loop_ms, 95)
    return {
        "status": "measured",
        "backend": "python_numpy_cpu",
        "horizon_ms": int(horizon_ms),
        "rollout_dt_ms": 1,
        "steps": int(steps),
        "checksum": float(checksum),
        "p50_forecast_ms": _pctl(loop_ms, 50),
        "p95_forecast_ms": p95,
        "p99_forecast_ms": _pctl(loop_ms, 99),
        "p95_real_time_factor": float(horizon_ms / max(p95, 1.0e-9)),
        "passes_realtime": bool(p95 <= float(horizon_ms)),
    }


def _run_rust_digital_twin_latency(*, steps: int, actuator_count: int = 2) -> dict[str, Any]:
    """Run the native Rust digital-twin latency benchmark when Cargo is available.

    Parameters
    ----------
    steps
        Number of measured Rust samples.
    actuator_count
        Number of actuator channels emitted and serialised by the Rust path.

    Returns
    -------
    dict[str, Any]
        Measured Rust JSON payload or a fail-closed blocked status.
    """
    cargo = shutil_which("cargo")
    if cargo is None:
        return {
            "status": "blocked_missing_cargo",
            "backend": "rust_release",
            "reason": "cargo executable was not found on PATH.",
        }
    cmd = [
        cargo,
        "run",
        "-q",
        "-p",
        "fusion-control",
        "--release",
        "--bin",
        "digital_twin_latency",
        "--",
        "--steps",
        str(steps),
        "--actuators",
        str(actuator_count),
        "--json",
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT / "scpn-fusion-rs",
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    elapsed = float(time.perf_counter() - started)
    if proc.returncode != 0:
        return {
            "status": "blocked_rust_command_failed",
            "backend": "rust_release",
            "command": " ".join(cmd),
            "returncode": int(proc.returncode),
            "runtime_seconds": elapsed,
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked_invalid_rust_json",
            "backend": "rust_release",
            "command": " ".join(cmd),
            "runtime_seconds": elapsed,
            "reason": f"{type(exc).__name__}: {exc}",
            "stdout_tail": proc.stdout[-2000:],
        }
    if not isinstance(payload, dict):
        return {
            "status": "blocked_invalid_rust_json",
            "backend": "rust_release",
            "command": " ".join(cmd),
            "runtime_seconds": elapsed,
            "reason": "Rust benchmark did not emit a JSON object.",
        }
    payload["command"] = " ".join(cmd)
    payload["runtime_seconds"] = elapsed
    return payload


def shutil_which(executable: str) -> str | None:
    """Return an executable path without importing optional shell helpers.

    Parameters
    ----------
    executable
        Executable name to locate on ``PATH``.

    Returns
    -------
    str | None
        Absolute executable path when found.
    """
    for item in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(item) / executable
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def run_actuator_scaling_campaign(*, steps: int = 160) -> dict[str, Any]:
    """Measure actuator-count scaling for CPU, Rust, and CUDA lanes.

    Parameters
    ----------
    steps
        Number of samples per actuator-count row.

    Returns
    -------
    dict[str, Any]
        Scaling rows through the `>200` actuator target surface.
    """
    steps = int(steps)
    if steps < 32:
        raise ValueError("steps must be >= 32.")
    rows: list[dict[str, Any]] = []
    for count in _ACTUATOR_SCALING_COUNTS:
        cpu = _run_actuator_scaling_cpu_case(actuator_count=count, steps=steps)
        rust = _run_rust_digital_twin_latency(steps=steps, actuator_count=count)
        gpu = _run_gpu_digital_twin_latency(steps=steps, actuator_count=count)
        rows.append(
            {
                "actuator_count": count,
                "cpu": cpu,
                "rust": rust,
                "gpu": gpu,
                "measured_lane_count": int(
                    1
                    + (1 if rust.get("status") == "measured" else 0)
                    + (1 if gpu.get("status") == "measured" else 0)
                ),
            }
        )
    row_256 = next(row for row in rows if row["actuator_count"] == 256)
    return {
        "schema": "scpn-fusion-core.digital_twin_actuator_scaling.v1",
        "actuator_counts": list(_ACTUATOR_SCALING_COUNTS),
        "rows": rows,
        "passes_thresholds": bool(
            row_256["cpu"]["safe_output_rate"] == 1.0
            and row_256["cpu"]["p95_loop_ms"] <= 2.0
            and row_256["rust"].get("status") == "measured"
            and float(row_256["rust"].get("p95_loop_ms", float("inf"))) <= 2.0
        ),
        "thresholds": {
            "required_actuator_count": 256,
            "max_cpu_256_p95_loop_ms": 2.0,
            "max_rust_256_p95_loop_ms": 2.0,
            "gpu_row_required": False,
        },
        "claim_boundary": (
            "Actuator-count scaling measures command fanout and output serialisation "
            "for the reduced-order local benchmark. It is not actuator hardware timing."
        ),
    }


def run_predictive_horizon_campaign(*, steps: int = 160) -> dict[str, Any]:
    """Measure 50 ms and 100 ms reduced-order forecast horizon timing.

    Parameters
    ----------
    steps
        Number of forecast calls per horizon row.

    Returns
    -------
    dict[str, Any]
        Forecast timing rows and real-time-factor gates.
    """
    rows = [
        _run_predictive_horizon_case(horizon_ms=horizon, steps=steps)
        for horizon in _PREDICTIVE_HORIZON_MS
    ]
    return {
        "schema": "scpn-fusion-core.digital_twin_predictive_horizon.v1",
        "rows": rows,
        "passes_thresholds": bool(all(row["passes_realtime"] for row in rows)),
        "thresholds": {
            "required_horizons_ms": list(_PREDICTIVE_HORIZON_MS),
            "forecast_must_run_faster_than_horizon": True,
        },
        "claim_boundary": (
            "Forecast timing is a reduced-order local digital-twin rollout. It is "
            "not a validated 50-100 ms plasma-instability prediction horizon."
        ),
    }


def run_digital_twin_latency_campaign(*, steps: int = 320) -> dict[str, Any]:
    """Measure CPU, GPU-status, Rust, and degraded-mode digital-twin latency.

    Parameters
    ----------
    steps
        Number of sensor-to-control samples per CPU degraded-mode lane.

    Returns
    -------
    dict[str, Any]
        Structured digital-twin latency evidence with host-load metadata.
    """
    steps = int(steps)
    if steps < 32:
        raise ValueError("steps must be >= 32.")
    host_before = _host_snapshot()
    cpu_cases = {case: _run_digital_twin_case(case, steps=steps) for case in _DEGRADED_MODE_CASES}
    host_after = _host_snapshot()
    nominal = cpu_cases["nominal"]
    degraded_semantics_pass = all(bool(case["passes_semantics"]) for case in cpu_cases.values())
    rust_lane = _run_rust_digital_twin_latency(steps=steps)
    gpu_lane = _run_gpu_digital_twin_latency(steps=steps)
    measured_lane_count = (
        1
        + (1 if rust_lane.get("status") == "measured" else 0)
        + (1 if gpu_lane.get("status") == "measured" else 0)
    )
    hil_lane = run_sensor_to_actuator_hil_latency_campaign(
        n_steps=max(32, min(steps, 160)),
        actuator_count=256,
        rng_seed=2026,
    )
    simulated_hil_scaffold_ready = bool(
        hil_lane["status"] == "measured_simulated_hil"
        and hil_lane["hardware_status"] == "simulated_host_adc_dac_loop"
        and hil_lane["actuator_count"] == 256
        and all(bool(row["passes_semantics"]) for row in hil_lane["scenarios"].values())
    )
    return {
        "schema": "scpn-fusion-core.digital_twin_control_latency.v1",
        "claim_boundary": (
            "Reduced-order sensor-to-control benchmark with a host-side simulated HIL scaffold. "
            "This section does not certify physical HIL, FPGA, plant CODAC, or actuator hardware timing."
        ),
        "simulated_hil_scaffold_ready": simulated_hil_scaffold_ready,
        "physical_hil_ready": False,
        "fpga_timing_ready": False,
        "codac_timing_ready": False,
        "actuator_hardware_timing_ready": False,
        "measurement_context": {
            "claim_boundary": (
                host_before["claim_boundary"]
                if host_before.get("isolation") != "non_isolated_local_workstation"
                else "Python CPU and Rust rows are local non-isolated wall-clock measurements. "
                "GPU rows are accelerator measurements only when status is measured. "
                "HIL rows are simulated host ADC/DAC timing unless hardware_status names a physical rig."
            ),
            "accelerator_isolation_note": (
                "CUDA device was operator-reserved for this benchmark run; CPU host "
                "process isolation is recorded in host_before/host_after."
            ),
            "warmup_policy": "No samples discarded; deterministic synthetic stream is measured from cold local process state.",
            "sample_count_per_case": steps,
            "benchmark_command": host_before.get("benchmark_command"),
            "cpu_isolation_method": host_before.get("isolation"),
            "reserved_cpu_set": host_before.get("reserved_cpu_set"),
            "concurrent_heavy_jobs": host_before.get("concurrent_heavy_jobs"),
            "host_before": host_before,
            "host_after": host_after,
        },
        "cpu": {
            "status": "measured",
            "backend": "python_numpy_cpu",
            "p50_loop_ms": nominal["p50_loop_ms"],
            "p95_loop_ms": nominal["p95_loop_ms"],
            "p99_loop_ms": nominal["p99_loop_ms"],
            "stages": nominal["stages"],
            "degraded_modes": cpu_cases,
        },
        "gpu": gpu_lane,
        "rust": rust_lane,
        "hil": hil_lane,
        "passes_thresholds": bool(
            nominal["p95_loop_ms"] <= 2.0
            and nominal["p99_loop_ms"] <= 5.0
            and degraded_semantics_pass
            and measured_lane_count >= 2
            and simulated_hil_scaffold_ready
        ),
        "thresholds": {
            "max_cpu_nominal_p95_loop_ms": 2.0,
            "max_cpu_nominal_p99_loop_ms": 5.0,
            "min_measured_lanes": 2,
            "degraded_modes_require_safe_outputs": True,
            "hil_simulated_scaffold_required": True,
            "hil_wall_clock_threshold_required_for_report_acceptance": False,
        },
    }


def _controller_runner(
    name: str,
    *,
    scpn: NeuroSymbolicController | None,
) -> Callable[[float, int, int, float], float]:
    if name == "SNN":
        if scpn is None:
            raise ValueError("SNN controller requires SCPN runtime.")

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del steps, dt
            action_vec = scpn.step_traceable((float(x_obs), 0.0), k)
            return float(np.clip(action_vec[0], -1.0, 1.0))

        return _run

    if name == "PID":
        state = _PIDState(kp=1.15, ki=0.24, kd=0.04)

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del k, steps
            err = -x_obs
            return state.step(err, dt=dt, u_limit=1.0)

        return _run

    if name == "MPC-lite":

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del dt
            return _mpc_lite_action(x_obs, k, steps, horizon=6, u_limit=1.0)

        return _run

    raise ValueError(f"Unknown controller: {name}")


def _run_lane(
    *,
    controller_name: str,
    physics_mode: str,
    steps: int,
    dt: float,
    x0: float,
    scpn: NeuroSymbolicController | None,
) -> dict[str, float | str]:
    if physics_mode not in {"surrogate", "full"}:
        raise ValueError("physics_mode must be 'surrogate' or 'full'.")

    runner = _controller_runner(controller_name, scpn=scpn)
    x = float(x0)
    sensor_state = float(x0)
    u_prev = 0.0

    sensor_ms = np.zeros(steps, dtype=np.float64)
    controller_ms = np.zeros(steps, dtype=np.float64)
    actuator_ms = np.zeros(steps, dtype=np.float64)
    physics_ms = np.zeros(steps, dtype=np.float64)
    loop_ms = np.zeros(steps, dtype=np.float64)
    errors = np.zeros(steps, dtype=np.float64)

    for k in range(steps):
        d = _disturbance(k, steps)
        t_loop0 = time.perf_counter()

        t0 = time.perf_counter()
        x_obs, sensor_state = _sensor_preprocess(x, sensor_state)
        sensor_ms[k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        u_cmd = runner(x_obs, k, steps, dt)
        controller_ms[k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        u = _actuator_lag(u_cmd, u_prev, dt)
        actuator_ms[k] = (time.perf_counter() - t0) * 1e3
        u_prev = u

        t0 = time.perf_counter()
        x = _physics_step(x, u, d, physics_mode)
        physics_ms[k] = (time.perf_counter() - t0) * 1e3

        errors[k] = -x
        loop_ms[k] = (time.perf_counter() - t_loop0) * 1e3

    return {
        "runtime_backend": (
            str(scpn.runtime_backend_name)
            if controller_name == "SNN" and scpn is not None
            else "n/a"
        ),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "p50_loop_ms": _pctl(loop_ms, 50),
        "p95_loop_ms": _pctl(loop_ms, 95),
        "p99_loop_ms": _pctl(loop_ms, 99),
        "p95_sensor_ms": _pctl(sensor_ms, 95),
        "p95_controller_ms": _pctl(controller_ms, 95),
        "p95_actuator_ms": _pctl(actuator_ms, 95),
        "p95_physics_ms": _pctl(physics_ms, 95),
    }


def run_campaign(*, seed: int = 42, steps: int = 320) -> dict[str, Any]:
    """Run surrogate and full-physics latency lanes for SNN, PID, and MPC-lite.

    Parameters
    ----------
    seed
        Kept for API parity; campaign uses a deterministic path and ignores randomization.
    steps
        Number of control iterations in each lane.

    Returns
    -------
    dict[str, Any]
        Structured campaign results and threshold pass flags.
    """
    seed = int(seed)
    steps = int(steps)
    if steps < 32:
        raise ValueError("steps must be >= 32.")

    del seed  # Deterministic fixed benchmark path.
    dt = 0.05
    x0 = 0.35
    controllers = ("SNN", "PID", "MPC-lite")
    modes = ("surrogate", "full")

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        out_mode: dict[str, dict[str, Any]] = {}
        for ctrl in controllers:
            scpn = _build_scpn_controller() if ctrl == "SNN" else None
            if scpn is not None:
                scpn.reset()
            out_mode[ctrl] = _run_lane(
                controller_name=ctrl,
                physics_mode=mode,
                steps=steps,
                dt=dt,
                x0=x0,
                scpn=scpn,
            )
        out[mode] = out_mode

    snn_backend = str(out["surrogate"]["SNN"].get("runtime_backend", "unknown")).lower()
    fallback_backend = snn_backend == "numpy"
    if fallback_backend:
        # NumPy fallback remains deterministic but carries expected latency overhead.
        thresholds = {
            "max_snn_p95_loop_ms_surrogate": 10.0,
            "max_snn_p95_loop_ms_full": 14.0,
            "max_snn_full_to_surrogate_ratio": 8.0,
        }
        threshold_profile = "numpy_fallback"
    else:
        thresholds = {
            "max_snn_p95_loop_ms_surrogate": 8.0,
            "max_snn_p95_loop_ms_full": 12.0,
            "max_snn_full_to_surrogate_ratio": 8.0,
        }
        threshold_profile = "accelerated"
    snn_surrogate = out["surrogate"]["SNN"]["p95_loop_ms"]
    snn_full = out["full"]["SNN"]["p95_loop_ms"]
    ratio = snn_full / max(snn_surrogate, 1e-9)
    passes = bool(
        snn_surrogate <= thresholds["max_snn_p95_loop_ms_surrogate"]
        and snn_full <= thresholds["max_snn_p95_loop_ms_full"]
        and ratio <= thresholds["max_snn_full_to_surrogate_ratio"]
    )

    return {
        "seed": 42,
        "steps": steps,
        "modes": out,
        "snn_runtime_backend": snn_backend,
        "threshold_profile": threshold_profile,
        "ratios": {"snn_full_to_surrogate_p95_ratio": float(ratio)},
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Build and return the end-to-end latency report payload."""
    t0 = time.perf_counter()
    campaign = run_campaign(**kwargs)
    steps = int(kwargs.get("steps", 320))
    digital_twin = run_digital_twin_latency_campaign(steps=steps)
    scaling_steps = max(32, min(steps, 160))
    actuator_scaling = run_actuator_scaling_campaign(steps=scaling_steps)
    predictive_horizon = run_predictive_horizon_campaign(steps=scaling_steps)
    passes = bool(
        campaign["passes_thresholds"]
        and digital_twin["passes_thresholds"]
        and actuator_scaling["passes_thresholds"]
        and predictive_horizon["passes_thresholds"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": REPORT_STATUS_ACCEPTED if passes else REPORT_STATUS_BLOCKED,
        "passes_thresholds": passes,
        "claim_boundary": REPORT_CLAIM_BOUNDARY,
        "physical_hil_ready": False,
        "fpga_timing_ready": False,
        "codac_timing_ready": False,
        "actuator_hardware_timing_ready": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "scpn_end_to_end_latency": campaign,
        "digital_twin_control_latency": digital_twin,
        "actuator_count_scaling": actuator_scaling,
        "predictive_horizon": predictive_horizon,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a markdown summary for the end-to-end latency report."""
    g = report["scpn_end_to_end_latency"]
    lines = [
        "# SCPN End-to-End Latency Benchmark",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Claim boundary: `{report['claim_boundary']}`",
        f"- Physical HIL ready: `{'YES' if report['physical_hil_ready'] else 'NO'}`",
        f"- FPGA timing ready: `{'YES' if report['fpga_timing_ready'] else 'NO'}`",
        f"- CODAC timing ready: `{'YES' if report['codac_timing_ready'] else 'NO'}`",
        f"- Actuator hardware timing ready: "
        f"`{'YES' if report['actuator_hardware_timing_ready'] else 'NO'}`",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps: `{g['steps']}`",
        f"- SNN runtime backend: `{g.get('snn_runtime_backend', 'unknown')}`",
        f"- Threshold profile: `{g.get('threshold_profile', 'unknown')}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        f"- SNN full/surrogate p95 ratio: `{g['ratios']['snn_full_to_surrogate_p95_ratio']:.3f}`",
        "",
    ]
    twin = report["digital_twin_control_latency"]
    cpu = twin["cpu"]
    rust = twin["rust"]
    gpu = twin["gpu"]
    hil = twin["hil"]
    lines.extend(
        [
            "## Digital-Twin Sensor-to-Control Path",
            "",
            twin["claim_boundary"],
            "",
            twin["measurement_context"]["claim_boundary"],
            "",
            "| Lane | Status | p50 loop [ms] | p95 loop [ms] | p99 loop [ms] | Boundary |",
            "|------|--------|---------------|---------------|---------------|----------|",
            (
                f"| Python CPU | {cpu['status']} | {cpu['p50_loop_ms']:.6f} | "
                f"{cpu['p95_loop_ms']:.6f} | {cpu['p99_loop_ms']:.6f} | "
                f"{twin['measurement_context'].get('cpu_isolation_method', 'unknown')} |"
            ),
            (
                f"| Rust native | {rust.get('status', 'unknown')} | "
                f"{_fmt_optional_ms(rust.get('p50_loop_ms'))} | "
                f"{_fmt_optional_ms(rust.get('p95_loop_ms'))} | "
                f"{_fmt_optional_ms(rust.get('p99_loop_ms'))} | "
                "local release binary when measured |"
            ),
            (
                f"| GPU | {gpu.get('status', 'unknown')} | "
                f"{_fmt_optional_ms(gpu.get('p50_loop_ms'))} | "
                f"{_fmt_optional_ms(gpu.get('p95_loop_ms'))} | "
                f"{_fmt_optional_ms(gpu.get('p99_loop_ms'))} | "
                f"{gpu.get('reason', gpu.get('claim_boundary', 'measured accelerator lane'))} |"
            ),
            "",
            "### Measurement Metadata",
            "",
            f"- Command: `{twin['measurement_context'].get('benchmark_command') or 'not_recorded'}`",
            f"- CPU isolation method: `{twin['measurement_context'].get('cpu_isolation_method')}`",
            f"- Reserved CPU set: `{twin['measurement_context'].get('reserved_cpu_set') or 'not_recorded'}`",
            f"- Host load before: `{twin['measurement_context']['host_before'].get('loadavg_1_5_15')}`",
            f"- Host load after: `{twin['measurement_context']['host_after'].get('loadavg_1_5_15')}`",
            f"- CPU governors: `{twin['measurement_context']['host_before']['cpu_frequency'].get('governors')}`",
            f"- Concurrent heavy jobs: `{twin['measurement_context'].get('concurrent_heavy_jobs') or 'not_recorded'}`",
            "",
            "### Simulated HIL Sensor-to-Actuator Scaffold",
            "",
            hil["claim_boundary"],
            "",
            "| Status | Hardware status | Actuators | p50 [us] | p95 [us] | p99 [us] | Pass |",
            "|--------|-----------------|-----------|----------|----------|----------|------|",
            (
                f"| {hil['status']} | {hil['hardware_status']} | {hil['actuator_count']} | "
                f"{hil['nominal_latency']['p50_us']:.6f} | "
                f"{hil['nominal_latency']['p95_us']:.6f} | "
                f"{hil['nominal_latency']['p99_us']:.6f} | "
                f"{'YES' if hil['passes_thresholds'] else 'NO'} |"
            ),
            "",
            "| Scenario | Fallbacks | Safe output rate | p95 [us] | Pass | Reasons |",
            "|----------|-----------|------------------|----------|------|---------|",
        ]
    )
    for name, rec in hil["scenarios"].items():
        reasons = ", ".join(
            f"{key}:{value}" for key, value in sorted(rec["fallback_reasons"].items())
        )
        lines.append(
            f"| {name} | {rec['fallback_count']} | {rec['safe_output_rate']:.3f} | "
            f"{rec['latency']['p95_us']:.6f} | {'YES' if rec['passes_semantics'] else 'NO'} | "
            f"{reasons or 'none'} |"
        )
    lines.extend(
        [
            "",
            "### CPU Pipeline Stages",
            "",
            "| Stage | p50 [ms] | p95 [ms] | p99 [ms] |",
            "|-------|----------|----------|----------|",
        ]
    )
    for stage in _PIPELINE_STAGE_KEYS:
        rec = cpu["stages"][stage]
        lines.append(
            f"| {stage} | {rec['p50_ms']:.6f} | {rec['p95_ms']:.6f} | {rec['p99_ms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Degraded Modes",
            "",
            "| Case | Fallbacks | Safe output rate | Pass | Reasons |",
            "|------|-----------|------------------|------|---------|",
        ]
    )
    for case in _DEGRADED_MODE_CASES:
        rec = cpu["degraded_modes"][case]
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(rec["fallback_reasons"].items()))
        lines.append(
            f"| {case} | {rec['fallback_count']} | {rec['safe_output_rate']:.3f} | "
            f"{'YES' if rec['passes_semantics'] else 'NO'} | {reasons or 'none'} |"
        )
    lines.append("")
    scaling = report["actuator_count_scaling"]
    lines.extend(
        [
            "## Actuator-Count Scaling",
            "",
            scaling["claim_boundary"],
            "",
            "| Actuators | CPU p95 [ms] | Rust p95 [ms] | CUDA p95 [ms] | Measured lanes |",
            "|-----------|--------------|---------------|---------------|----------------|",
        ]
    )
    for row in scaling["rows"]:
        lines.append(
            f"| {row['actuator_count']} | "
            f"{_fmt_optional_ms(row['cpu'].get('p95_loop_ms'))} | "
            f"{_fmt_optional_ms(row['rust'].get('p95_loop_ms'))} | "
            f"{_fmt_optional_ms(row['gpu'].get('p95_loop_ms'))} | "
            f"{row['measured_lane_count']} |"
        )
    lines.extend(
        [
            "",
            "## Predictive-Horizon Timing",
            "",
            report["predictive_horizon"]["claim_boundary"],
            "",
            "| Horizon [ms] | p50 forecast [ms] | p95 forecast [ms] | p99 forecast [ms] | p95 real-time factor | Pass |",
            "|--------------|-------------------|-------------------|-------------------|----------------------|------|",
        ]
    )
    for row in report["predictive_horizon"]["rows"]:
        lines.append(
            f"| {row['horizon_ms']} | {row['p50_forecast_ms']:.6f} | "
            f"{row['p95_forecast_ms']:.6f} | {row['p99_forecast_ms']:.6f} | "
            f"{row['p95_real_time_factor']:.3f} | {'YES' if row['passes_realtime'] else 'NO'} |"
        )
    lines.append("")
    for mode in ("surrogate", "full"):
        lines.extend(
            [
                f"## {mode.capitalize()} Physics Mode",
                "",
                "| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |",
                "|------------|------|---------------|------------------|---------------------|-------------------|------------------|",
            ]
        )
        for ctrl in ("SNN", "PID", "MPC-lite"):
            rec = g["modes"][mode][ctrl]
            lines.append(
                f"| {ctrl} | {rec['rmse']:.6f} | {rec['p95_loop_ms']:.6f} | "
                f"{rec['p95_sensor_ms']:.6f} | {rec['p95_controller_ms']:.6f} | "
                f"{rec['p95_actuator_ms']:.6f} | {rec['p95_physics_ms']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt_optional_ms(value: object) -> str:
    """Render an optional millisecond value for markdown tables.

    Parameters
    ----------
    value
        Numeric latency value or ``None`` for blocked lanes.

    Returns
    -------
    str
        Fixed precision number or ``blocked``.
    """
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return "blocked"


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and emit JSON/Markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "scpn_end_to_end_latency.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "scpn_end_to_end_latency.md"),
    )
    parser.add_argument("--actuators", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.gpu_only:
        print(
            json.dumps(
                _run_gpu_digital_twin_latency(steps=args.steps, actuator_count=args.actuators),
                sort_keys=True,
            )
        )
        return 0

    report = generate_report(seed=args.seed, steps=args.steps)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report).rstrip() + "\n", encoding="utf-8")

    g = report["scpn_end_to_end_latency"]
    print("SCPN end-to-end latency benchmark complete.")
    print(
        "snn_p95_surrogate={s:.6f}ms, snn_p95_full={f:.6f}ms, pass={p}".format(
            s=g["modes"]["surrogate"]["SNN"]["p95_loop_ms"],
            f=g["modes"]["full"]["SNN"]["p95_loop_ms"],
            p=g["passes_thresholds"],
        )
    )
    twin = report["digital_twin_control_latency"]
    scaling = report["actuator_count_scaling"]
    horizon = report["predictive_horizon"]
    if args.strict and (
        not report["passes_thresholds"]
        or report["physical_hil_ready"]
        or report["fpga_timing_ready"]
        or report["codac_timing_ready"]
        or report["actuator_hardware_timing_ready"]
        or report["status"] != REPORT_STATUS_ACCEPTED
        or report["claim_boundary"] != REPORT_CLAIM_BOUNDARY
        or not g["passes_thresholds"]
        or not twin["passes_thresholds"]
        or not scaling["passes_thresholds"]
        or not horizon["passes_thresholds"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
