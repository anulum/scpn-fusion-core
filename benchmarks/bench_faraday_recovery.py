#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Faraday Recovery Benchmark
"""Benchmark the classical MIF/FRC Faraday recovery contract."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.faraday_recovery import (
    FaradayRecoveryTrajectoryPoint,
    integrated_recovery_energy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "faraday_recovery_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "faraday_recovery.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "faraday_recovery.rs",
    Path(__file__).resolve(),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(samples: int) -> list[FaradayRecoveryTrajectoryPoint]:
    duration = 1.0e-6
    times = np.linspace(0.0, duration, samples, dtype=np.float64)
    return [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t_s),
            separatrix_radius_m=float(0.15 + 4.0e3 * t_s),
            b_ext_t=20.0,
        )
        for t_s in times
    ]


def _python_case(samples: int) -> dict[str, Any]:
    timings = []
    state = None
    trace = _trajectory(samples)
    for _ in range(7):
        start_ns = time.perf_counter_ns()
        state = integrated_recovery_energy(trace, 6, 0.08)
        timings.append(float(time.perf_counter_ns() - start_ns))
    if state is None:
        raise RuntimeError("benchmark did not run")
    return {
        "language": "python",
        "case": f"python_{samples}_samples",
        "samples": samples,
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "recovered_energy_j": state.recovered_energy_j,
        "max_abs_back_emf_v": state.max_abs_back_emf_v,
        "max_abs_load_current_a": state.max_abs_load_current_a,
        "budget_claim_status": state.budget_claim_status,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "faraday_recovery"
    for estimates_path in sorted(criterion_root.glob("rust_*_samples/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            samples = int(parts[1])
        except (IndexError, ValueError):
            samples = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "samples": samples,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_report(cases: Iterable[int] = (64, 256, 1024)) -> dict[str, Any]:
    rows = [_python_case(samples) for samples in cases]
    rows.extend(_criterion_rows())
    return {
        "schema": "scpn-fusion-core.faraday_recovery_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated regression evidence for the exact classical Faraday recovery "
            "contract over supplied trajectories. This is not FUS-C.6 Hall-MHD trajectory "
            "or Slough compression-work acceptance evidence."
        ),
        "physics_contract": {
            "flux": "Phi = B_ext*pi*R_s^2",
            "emf": "EMF = -N*pi*(R_s^2*dB_ext/dt + 2*B_ext*R_s*dR_s/dt)",
            "load_power": "P = EMF^2/R_load",
            "blocked_acceptance": "self-consistent FUS-C.6 compression-work trajectory absent",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "isolated": False,
        },
        "source_checksums": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in SOURCE_PATHS
        },
        "results": rows,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "rows": len(report["results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
