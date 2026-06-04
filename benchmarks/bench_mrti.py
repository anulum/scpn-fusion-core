#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MRTI Benchmark
"""Benchmark the analytical MRTI growth/spectrum tracker."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.mrti import MRTISpectrumTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "mrti_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "mrti.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "mrti.rs",
    Path(__file__).resolve(),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_case(n_modes: int, steps: int) -> dict[str, Any]:
    samples = []
    for _ in range(5):
        tracker = MRTISpectrumTracker(
            k_max_m_inv=1.0e4,
            n_modes=n_modes,
            initial_perturbation_m=1.0e-9,
            rho_kg_m3=1.0e-3,
            saturation_threshold_m=1.0e-3,
        )
        start_ns = time.perf_counter_ns()
        state = tracker.state()
        for _step in range(steps):
            state = tracker.step(2.0e-7, 6.5e6, B_perp_t=8.0e-4)
        elapsed_ns = time.perf_counter_ns() - start_ns
        samples.append(float(elapsed_ns))
    mean_ns = float(np.mean(np.asarray(samples, dtype=np.float64)))
    return {
        "language": "python",
        "case": f"python_{n_modes}_modes_{steps}_steps",
        "n_modes": n_modes,
        "steps": steps,
        "mean_seconds": mean_ns * 1.0e-9,
        "samples_seconds": [value * 1.0e-9 for value in samples],
        "final_time_s": state.t_s,
        "final_max_amplitude_m": state.max_amplitude_m,
        "saturation_warning": state.saturation_warning,
        "fastest_growing_k_m_inv": state.fastest_growing_k_m_inv,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "mrti"
    for estimates_path in sorted(criterion_root.glob("rust_*_modes_512_steps/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            n_modes = int(parts[1])
        except (IndexError, ValueError):
            n_modes = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "n_modes": n_modes,
                "steps": 512,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_report(
    cases: Iterable[tuple[int, int]] = ((64, 512), (256, 512), (1024, 512)),
) -> dict[str, Any]:
    rows = [_python_case(n_modes, steps) for n_modes, steps in cases]
    rows.extend(_criterion_rows())
    return {
        "schema": "scpn-fusion-core.mrti_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated MRTI analytical growth/spectrum regression evidence. "
            "Rust rows are present only after running the Criterion benchmark."
        ),
        "physics_contract": {
            "name": "MIF/FRC linear MRTI growth spectrum",
            "equation": "gamma^2 = k*a_eff - k^2*B_perp^2/(mu0*rho)",
            "stabilization": "negative radicands clipped to zero for magnetic-tension-stabilized modes",
            "not_claimed": "full Hall-MHD pulsed-compression trajectory coupling or nonlinear saturation physics",
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
