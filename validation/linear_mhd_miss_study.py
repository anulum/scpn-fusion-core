#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — study of the two misses of the linear reduced-MHD acceptance
"""Probe the two criteria that the linear reduced-MHD acceptance misses.

:mod:`validation.linear_mhd_reduced_model_acceptance` passes 7 of 9 fixed criteria. This
study asks what lies behind the two misses, without touching the acceptance or its bands.

1. **Cylinder 2/1 tearing slope.** Over resistivities 1e-6 to 1e-8 the growth rate scales as
   eta^0.554, below the constant-psi exponent 3/5. The constant-psi rate holds when the
   stability index times the resistive-layer width is small, and that product shrinks with
   eta. The study extends the scan to 1e-10 and reports the local slope between neighbouring
   resistivities and the ratio of the growth rate to the asymptotic rate at each one. If the
   explanation holds, the slope approaches 3/5 and the ratio approaches 1. At the smallest
   resistivity the growth rate is also recomputed with twice the elements.
2. **Harris sheet with negative stability index.** At ka = 1.2 the acceptance finds a root of
   7.9e-9 at eta = 1e-6, above its 1e-9 limit. The study repeats the search at three
   resistivities and two element counts and reports the root at each.

Every solve uses the acceptance's grid (smallest spacing ``0.02 eta^(1/3)`` at the resonant
surface) but is a dense QZ solve over all finite eigenvalues: the problems have at most 3200
unknowns, and ``DENSE_LIMIT`` lifts the 2000-unknown limit above which the acceptance's
``growth_rate`` switches to the iterative shift-and-invert search. So a zero root means that
the discrete spectrum has no eigenvalue with real part above 1e-12, at either element count;
the method and completeness of each solve are recorded.

Findings are reported, not gated; the gates concern completeness only: every solve returned
a finite number and was a complete dense search. The record carries the source revision,
source digests, library versions, thread settings and host.

Usage: python validation/linear_mhd_miss_study.py [--output study.json] [--markdown study.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.linear_mhd_1d import (
    Geometry1D,
    WessonCylinder,
    assemble,
    clustered_grid,
    harris_slab,
)
from scpn_fusion.core.linear_mhd_delta_prime import cylinder_delta_prime
from scpn_fusion.core.linear_mhd_spectrum import unstable_search
from scpn_fusion.core.tearing_mode_theory import FKR_MATCHING_COEFFICIENT

REPORT_PATH = ROOT / "validation/reports/linear_mhd_miss_study.json"
MARKDOWN_PATH = ROOT / "validation/reports/linear_mhd_miss_study.md"
SCHEMA = "scpn-fusion.linear-mhd-miss-study.v2"
SLOPE_ETAS = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
ROOT_ETAS = (1e-6, 1e-7, 1e-8)
ELEMENTS = (200, 400)
SPACING_FACTOR = 0.02
SIGMA_MIN = 1e-12
DENSE_LIMIT = 4000
SOURCE_PATHS = (
    "src/scpn_fusion/core/linear_mhd_1d.py",
    "src/scpn_fusion/core/linear_mhd_delta_prime.py",
    "src/scpn_fusion/core/linear_mhd_spectrum.py",
    "src/scpn_fusion/core/tearing_mode_theory.py",
    "validation/linear_mhd_miss_study.py",
)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — study of the two misses of the linear reduced-MHD acceptance -->",
)


@dataclass(frozen=True)
class Solve:
    """One root search: the largest unstable growth rate (0.0 when none) and how it was found."""

    rate: float
    method: str
    complete: bool
    rejected: int


Solver = Callable[[Geometry1D, float, int], Solve]


def solver_solve(geom: Geometry1D, eta: float, elements: int) -> Solve:
    """Search the acceptance's grid with ``elements`` per side by a dense QZ solve."""
    nodes = clustered_grid(
        geom.x_min, geom.x_max, geom.resonant_x, SPACING_FACTOR * eta ** (1.0 / 3.0), elements
    )
    search = unstable_search(
        assemble(geom, nodes, eta), sigma_min=SIGMA_MIN, dense_limit=DENSE_LIMIT
    )
    root = search.most_unstable
    return Solve(
        0.0 if root is None else root.real, search.method, search.complete, search.rejected
    )


def provenance() -> dict[str, object]:
    """Return the source revision, source digests, library versions and host of this run."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    return {
        "source_revision": revision,
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_PATHS
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "cpu_models": sorted(
            {
                line.split(":", 1)[1].strip()
                for line in text.splitlines()
                if line.startswith("model name")
            }
        ),
        "started_at": datetime.now(UTC).isoformat(),
    }


def local_slopes(etas: tuple[float, ...], rates: list[float]) -> list[float]:
    """Return d ln(gamma) / d ln(eta) between each pair of neighbouring resistivities."""
    return [
        math.log(rates[index + 1] / rates[index]) / math.log(etas[index + 1] / etas[index])
        for index in range(len(etas) - 1)
    ]


def slope_scan(solve: Solver = solver_solve) -> dict[str, Any]:
    """Scan the cylinder 2/1 tearing mode to small resistivity against its asymptotic rate."""
    cylinder = WessonCylinder(1.2, 1.5, 10.0)
    geom = cylinder.geometry(2, 1)
    f_prime = abs(float(geom.f_parallel_prime(np.array([geom.resonant_x]))[0]))
    delta_prime = cylinder_delta_prime(cylinder, 2, 1)
    solves = [solve(geom, eta, ELEMENTS[0]) for eta in SLOPE_ETAS]
    rates = [result.rate for result in solves]
    asymptotic = [
        FKR_MATCHING_COEFFICIENT * delta_prime**0.8 * eta**0.6 * f_prime**0.4 for eta in SLOPE_ETAS
    ]
    refined_solve = solve(geom, SLOPE_ETAS[-1], ELEMENTS[1])
    refined = refined_solve.rate
    return {
        "etas": list(SLOPE_ETAS),
        "delta_prime": delta_prime,
        "growth_rates": rates,
        "local_slopes": local_slopes(SLOPE_ETAS, rates),
        "ratio_to_asymptotic": [rate / reference for rate, reference in zip(rates, asymptotic)],
        "smallest_eta_doubled_elements": refined,
        "smallest_eta_doubling_change": abs(refined / rates[-1] - 1.0),
        "solves": [asdict(result) for result in [*solves, refined_solve]],
    }


def root_scan(solve: Solver = solver_solve) -> dict[str, Any]:
    """Repeat the root search of the negative-index Harris sheet over resistivity and elements."""
    stable = harris_slab(1.2, 20.0)
    unstable = harris_slab(0.5, 20.0)
    stable_solves = {
        str(elements): [solve(stable, eta, elements) for eta in ROOT_ETAS] for elements in ELEMENTS
    }
    tearing_solves = [solve(unstable, eta, ELEMENTS[0]) for eta in ROOT_ETAS]
    roots = {key: [result.rate for result in series] for key, series in stable_solves.items()}
    tearing = [result.rate for result in tearing_solves]
    return {
        "etas": list(ROOT_ETAS),
        "roots_by_elements": roots,
        "unstable_ka_0p5_growth": tearing,
        "root_over_tearing": [root / rate for root, rate in zip(roots[str(ELEMENTS[0])], tearing)],
        "solves": [
            asdict(result)
            for result in [
                *(r for series in stable_solves.values() for r in series),
                *tearing_solves,
            ]
        ],
    }


def build_report(solve: Solver = solver_solve) -> dict[str, Any]:
    """Run both probes and return the record with its completeness gates."""
    record_provenance = provenance()
    slopes = slope_scan(solve)
    roots = root_scan(solve)
    solves = [*slopes["solves"], *roots["solves"]]
    checks = {
        "every_solve_finite": all(math.isfinite(result["rate"]) for result in solves),
        "every_solve_complete_dense": all(
            result["complete"] and result["method"] == "dense-qz" for result in solves
        ),
    }
    return {
        "schema": SCHEMA,
        "slope_scan": slopes,
        "root_scan": roots,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "provenance": record_provenance,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render both probes as tables."""
    slopes, roots = report["slope_scan"], report["root_scan"]
    lines = [
        *MARKDOWN_HEADER,
        "# Study of the two misses of the linear reduced-MHD acceptance",
        "",
        f"Status: **{report['status'].upper()}** (completeness only; findings are reported)",
        "",
        f"## Cylinder 2/1 tearing (Δ′ = {slopes['delta_prime']:.4g})",
        "",
        "| η | Growth rate | Ratio to asymptotic | Local slope to next η |",
        "|---|---:|---:|---:|",
    ]
    for index, eta in enumerate(slopes["etas"]):
        slope = slopes["local_slopes"][index] if index < len(slopes["local_slopes"]) else None
        lines.append(
            f"| {eta:.0e} | {slopes['growth_rates'][index]:.4g} "
            f"| {slopes['ratio_to_asymptotic'][index]:.4g} | "
            + ("—" if slope is None else f"{slope:.4f}")
            + " |"
        )
    lines += [
        "",
        f"Doubling the elements at η = {slopes['etas'][-1]:.0e} changes the growth rate by "
        f"{slopes['smallest_eta_doubling_change']:.3g}.",
        "",
        "## Harris sheet, ka = 1.2 (Δ′ < 0)",
        "",
        "Every solve is a dense QZ search over all finite eigenvalues; 0 means no eigenvalue with",
        "real part above 1e-12.",
        "",
        "| η | Root, 200 elements | Root, 400 elements | ka = 0.5 tearing rate |",
        "|---|---:|---:|---:|",
    ]
    for index, eta in enumerate(roots["etas"]):
        lines.append(
            f"| {eta:.0e} | {roots['roots_by_elements']['200'][index]:.3g} "
            f"| {roots['roots_by_elements']['400'][index]:.3g} "
            f"| {roots['unstable_ka_0p5_growth'][index]:.3g} |"
        )
    lines += ["", f"Source revision: `{report['provenance']['source_revision']}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the study and write the JSON and Markdown records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    arguments = parser.parse_args(argv)
    report = build_report()
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arguments.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
