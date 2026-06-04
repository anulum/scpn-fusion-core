#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Tilt-Mode Benchmark
"""Benchmark the conservative FUS-C.5 FRC tilt-mode diagnostic."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.tilt_mode_frc import (
    belova_table1_acceptance_status,
    tilt_mode_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "tilt_mode_frc_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "tilt_mode_frc.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "tilt_mode_frc.rs",
    Path(__file__).resolve(),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equilibrium() -> Any:
    t_i = 10_000.0
    t_e = 5_000.0
    b_ext = 5.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho = np.linspace(0.0, 0.4, 257, dtype=np.float64)
    return solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.2,
            B_ext=b_ext,
            delta=0.02,
        ),
        rho,
    )


def _python_case(iterations: int) -> dict[str, Any]:
    eq = _equilibrium()
    samples = []
    final_report = None
    for _ in range(5):
        checksum = 0.0
        start_ns = time.perf_counter_ns()
        for index in range(iterations):
            elongation = 2.5 + float(index % 16) * 0.125
            final_report = tilt_mode_report(eq, elongation)
            checksum += final_report.growth_rate_s_inv + final_report.s_over_elongation
        elapsed_ns = time.perf_counter_ns() - start_ns
        samples.append(float(elapsed_ns))
    if final_report is None:
        raise RuntimeError("benchmark did not run")
    return {
        "language": "python",
        "case": f"python_{iterations}_reports",
        "iterations": iterations,
        "mean_seconds": float(np.mean(np.asarray(samples, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in samples],
        "final_growth_rate_s_inv": final_report.growth_rate_s_inv,
        "final_s_over_elongation": final_report.s_over_elongation,
        "final_claim_status": final_report.claim_status,
        "checksum": checksum,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "tilt_mode_frc"
    for estimates_path in sorted(criterion_root.glob("rust_*_reports/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            iterations = int(parts[1])
        except (IndexError, ValueError):
            iterations = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "iterations": iterations,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_report(cases: Iterable[int] = (1_000, 10_000, 100_000)) -> dict[str, Any]:
    rows = [_python_case(iterations) for iterations in cases]
    rows.extend(_criterion_rows())
    rows.append({"language": "external_reference", **belova_table1_acceptance_status()})
    return {
        "schema": "scpn-fusion-core.tilt_mode_frc_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated FRC tilt diagnostic regression evidence. "
            "The accepted surface is an MHD Alfvén-time diagnostic with conservative "
            "fail-closed status, not Belova Table I or hybrid eigenvalue parity."
        ),
        "physics_contract": {
            "name": "MIF/FRC n=1 tilt diagnostic",
            "mhd_growth_scaling": "gamma = C*V_A/(E*R_s)",
            "rigid_body_threshold": "s/E thresholds are diagnostic only",
            "not_claimed": "full Belova hybrid eigenvalue solver or Table I same-case parity",
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
