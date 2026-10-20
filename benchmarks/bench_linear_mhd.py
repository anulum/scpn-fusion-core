#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear Reduced-MHD Eigenvalue Solver Benchmark
"""Benchmark the linear reduced-MHD solvers: Python assembly against the compiled eigen-solve.

Each case is timed in two phases, so the share of time spent in interpreted
Python (finite-element assembly) and in compiled libraries (dense QZ from
LAPACK, sparse LU from SuperLU and Arnoldi from ARPACK, reached through SciPy)
is measured rather than assumed.  One row is emitted per (operation, backend);
backends without an implementation are listed as unavailable with the reason.
Runs on a loaded workstation are labelled as local, non-isolated regression
evidence (``BENCHMARK_STANDARD``; benchmark core-isolation policy).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from scpn_fusion.core.linear_mhd_1d import (
    SpectralSearch,
    WessonCylinder,
    assemble,
    clustered_grid,
    unstable_search,
)
from scpn_fusion.core.linear_mhd_2d import ShapedCylinder, assemble_coupled

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "validation" / "reports" / "linear_mhd_benchmark.local.json"
SCHEMA = "scpn_fusion.linear_mhd_benchmark.v1"
ETA = 1e-6


def _stats(samples_ns: list[int]) -> dict[str, float]:
    """Return percentile statistics in milliseconds for a list of nanosecond samples."""
    values = np.asarray(samples_ns, dtype=np.float64) * 1e-6
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_per_s": float(1e3 / values.mean()),
    }


def _time(call: Callable[[], Any], warmup: int, repeats: int) -> tuple[list[int], Any]:
    """Run ``call`` ``warmup`` times untimed, then ``repeats`` times timed; return samples and last result."""
    result = None
    for _ in range(warmup):
        result = call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append(time.perf_counter_ns() - start)
    return samples, result


def _cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str:
    """Return the CPU model name from ``cpuinfo``, or the platform processor string."""
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _commit(root: Path = REPO_ROOT) -> str:
    """Return the commit hash of ``root``, or ``unknown`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _cases() -> list[dict[str, Any]]:
    """Return the cases: Stage 1 production grid (1598 unknowns), a 480-unknown grid, a 2-D case.

    ``paths`` lists the eigen-search paths timed for the case; the iterative path is timed on
    the smaller grid only, because one shift ladder at production size takes tens of minutes on
    a loaded workstation.
    """
    tearing = WessonCylinder(1.2, 1.5, 10.0).geometry(2, 1)
    profile = WessonCylinder(1.02, 1.5, 10.0)
    circular = ShapedCylinder(q=profile.q, major_radius=10.0)
    rs = profile.resonant_radius(2, 1)
    spacing = 0.02 * ETA ** (1.0 / 3.0)
    return [
        {
            "name": "1d_tearing_2_1_n_side_200",
            "assemble": lambda: assemble(
                tearing, clustered_grid(0.0, 1.0, tearing.resonant_x, spacing, 200), ETA
            ),
            "paths": ("dense-qz",),
        },
        {
            "name": "1d_tearing_2_1_n_side_60",
            "assemble": lambda: assemble(
                tearing, clustered_grid(0.0, 1.0, tearing.resonant_x, spacing, 60), ETA
            ),
            "paths": ("dense-qz", "shift-invert"),
        },
        {
            "name": "2d_circular_m2_n_side_60",
            "assemble": lambda: assemble_coupled(
                circular, clustered_grid(0.0, 1.0, rs, spacing, 60), ETA, 1, [2]
            ),
            "paths": ("dense-qz",),
        },
    ]


def _smoke_cases() -> list[dict[str, Any]]:
    """Return one small 1-D case for a quick functional run of the whole report path."""
    tearing = WessonCylinder(1.2, 1.5, 10.0).geometry(2, 1)
    spacing = 0.02 * ETA ** (1.0 / 3.0)
    return [
        {
            "name": "1d_tearing_2_1_n_side_10",
            "assemble": lambda: assemble(
                tearing, clustered_grid(0.0, 1.0, tearing.resonant_x, spacing, 10), ETA
            ),
            "paths": ("dense-qz", "shift-invert"),
        }
    ]


def _write(report: dict[str, Any], output: Path) -> None:
    """Write the report atomically so a partial run still leaves every finished row."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".partial")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def build_report(
    warmup: int,
    repeats: int,
    assembly_repeats: int,
    output: Path,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure every case and backend, writing the report after each row."""
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _commit(),
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "parameters": {
            "warmup": warmup,
            "repeats": repeats,
            "assembly_warmup": 3,
            "assembly_repeats": assembly_repeats,
            "eta": ETA,
        },
        "isolation": {
            "isolated": False,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "load_average_before": list(os.getloadavg()),
            "load_average_after": None,
        },
        "claim_boundary": (
            "Local non-isolated regression evidence on a shared, loaded workstation; not a"
            " production performance claim. Its purpose is the measured split between Python"
            " assembly and the compiled eigen-search."
        ),
        "status": "running",
        "results": [],
    }
    for case in _cases() if cases is None else cases:
        samples, disc = _time(case["assemble"], 3, assembly_repeats)
        size = int(disc.a.shape[0])
        report["results"].append(
            {
                "name": f"{case['name']}/assembly",
                "backend": "python+numpy",
                "unknowns": size,
                "stats": _stats(samples),
                "status": "measured",
            }
        )
        _write(report, output)
        for path in case["paths"]:
            limit = size if path == "dense-qz" else 0

            def search_call(d: Any = disc, lim: int = limit) -> SpectralSearch:
                return unstable_search(d, dense_limit=lim)

            solve_samples, search = _time(search_call, warmup, repeats)
            lam = search.most_unstable
            report["results"].append(
                {
                    "name": f"{case['name']}/search-{path}",
                    "backend": "scipy (LAPACK zggev)"
                    if path == "dense-qz"
                    else "scipy (SuperLU + ARPACK)",
                    "unknowns": size,
                    "stats": _stats(solve_samples),
                    "growth_rate": None if lam is None else lam.real,
                    "complete": search.complete,
                    "status": "measured",
                }
            )
            _write(report, output)
        for backend in ("rust", "mojo", "julia", "go"):
            report["results"].append(
                {
                    "name": f"{case['name']}/search",
                    "backend": backend,
                    "stats": None,
                    "status": f"unavailable: no {backend} implementation of the eigen-search",
                }
            )
    report["isolation"]["load_average_after"] = list(os.getloadavg())
    report["status"] = "complete"
    _write(report, output)
    return report


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the benchmark and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run one small case only")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--assembly-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    cases = _smoke_cases() if args.smoke else None
    report = build_report(args.warmup, args.repeats, args.assembly_repeats, args.output, cases)
    for row in report["results"]:
        stats = row["stats"]
        p50 = "n/a" if stats is None else f"{stats['p50_ms']:.1f} ms"
        print(f"{row['name']:<52} {row['backend']:<26} {p50}")


if __name__ == "__main__":
    main()
