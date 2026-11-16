#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — near-zero spectrum of the linear reduced-MHD pencil
"""Measure how well the near-zero eigenvalues of the Harris pencil are determined.

The linear reduced-MHD acceptance requires the Harris sheet with negative stability index
(ka = 1.2) to have no root above 1e-9, and it finds one of 7.9e-9.
:mod:`validation.linear_mhd_miss_study` found that this root does not converge as the grid
is refined. This diagnostic asks whether the root is determined at all in double precision.
On the acceptance's pencil (200 elements per side, eta = 1e-6) it reports:

- the extreme singular values of ``A``: a vanishing smallest one puts an eigenvalue at 0;
- every finite eigenvalue of modulus below ``NEAR_ZERO``, and for the ``LISTED`` smallest the
  condition number ``|x| |y| / |y^H B x|`` (right and left eigenvectors), the first-order
  error bound ``eps (|A|_1 + |lambda| |B|_1) cond`` and the relative residual;
- the largest real part of the spectrum before and after ``TRIALS`` random relative
  perturbations of size ``PERTURBATION`` to the nonzero entries of ``A``.

An eigenvalue whose real part lies inside its error bound has an undetermined sign. Findings
are reported, not gated. The gates concern completeness: every listed residual is at
rounding level, and every value is finite. The record carries the source revision, source
digests, library versions, thread settings and host.

Usage: python validation/linear_mhd_near_zero_spectrum.py [--output r.json] [--markdown r.md]
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import eig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.linear_mhd_1d import assemble, clustered_grid, harris_slab

ComplexMatrix = NDArray[np.complex128]
REPORT_PATH = ROOT / "validation/reports/linear_mhd_near_zero_spectrum.json"
MARKDOWN_PATH = ROOT / "validation/reports/linear_mhd_near_zero_spectrum.md"
SCHEMA = "scpn-fusion.linear-mhd-near-zero-spectrum.v1"
KA = 1.2
HALF_WIDTH = 20.0
ETA = 1e-6
ELEMENTS = 200
SPACING_FACTOR = 0.02
NEAR_ZERO = 1e-4
LISTED = 12
UNSTABLE_FLOOR = 1e-12
PERTURBATION = 1e-15
TRIALS = 3
SEED = 20261116
RESIDUAL_LIMIT = 1e-12
SOURCE_PATHS = (
    "src/scpn_fusion/core/linear_mhd_1d.py",
    "validation/linear_mhd_near_zero_spectrum.py",
)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — near-zero spectrum of the linear reduced-MHD pencil -->",
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


def harris_pencil(elements: int = ELEMENTS) -> tuple[ComplexMatrix, ComplexMatrix]:
    """Return the dense ``(A, B)`` of the acceptance's ka = 1.2 Harris pencil."""
    geom = harris_slab(KA, HALF_WIDTH)
    nodes = clustered_grid(
        geom.x_min, geom.x_max, geom.resonant_x, SPACING_FACTOR * ETA ** (1.0 / 3.0), elements
    )
    problem = assemble(geom, nodes, ETA)
    return (
        np.asarray(problem.a.toarray(), dtype=np.complex128),
        np.asarray(problem.b.toarray(), dtype=np.complex128),
    )


def near_zero_records(
    a: ComplexMatrix, b: ComplexMatrix, limit: float = NEAR_ZERO, listed: int = LISTED
) -> tuple[int, list[dict[str, float]]]:
    """Return the count of finite eigenvalues below ``limit`` and records of the smallest."""
    values, left, right = eig(a, b, left=True, right=True)
    norm_a, norm_b = float(np.linalg.norm(a, 1)), float(np.linalg.norm(b, 1))
    eps = float(np.finfo(np.float64).eps)
    finite = np.flatnonzero(np.isfinite(values) & (np.abs(values) < limit))
    records = []
    for index in finite[np.argsort(np.abs(values[finite]))][:listed]:
        lam, x, y = complex(values[index]), right[:, index], left[:, index]
        condition = float(np.linalg.norm(x) * np.linalg.norm(y) / abs(y.conj() @ b @ x))
        scale = norm_a + abs(lam) * norm_b
        records.append(
            {
                "real": lam.real,
                "imag": lam.imag,
                "condition": condition,
                "error_bound": eps * scale * condition,
                "relative_residual": float(
                    np.linalg.norm(a @ x - lam * (b @ x)) / (scale * np.linalg.norm(x))
                ),
            }
        )
    return int(finite.size), records


def largest_real_part(a: ComplexMatrix, b: ComplexMatrix) -> tuple[float, int]:
    """Return the largest finite real part and the count above ``UNSTABLE_FLOOR``."""
    values = eig(a, b, right=False)
    real = values[np.isfinite(values)].real
    return float(real.max()), int(np.count_nonzero(real > UNSTABLE_FLOOR))


def perturbation_trials(
    a: ComplexMatrix,
    b: ComplexMatrix,
    trials: int = TRIALS,
    scale: float = PERTURBATION,
    seed: int = SEED,
) -> list[dict[str, float]]:
    """Perturb the nonzero entries of ``A`` by a random relative ``scale`` and re-solve."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(trials):
        noise = rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape)
        largest, unstable = largest_real_part(a * (1.0 + scale * noise), b)
        results.append({"largest_real_part": largest, "unstable_count": unstable})
    return results


def build_report(elements: int = ELEMENTS) -> dict[str, Any]:
    """Run every measurement on the acceptance's pencil and return the record."""
    record_provenance = provenance()
    a, b = harris_pencil(elements)
    singular = np.linalg.svd(a, compute_uv=False)
    count, records = near_zero_records(a, b)
    largest, unstable = largest_real_part(a, b)
    trials = perturbation_trials(a, b)
    positive = [record for record in records if record["real"] > UNSTABLE_FLOOR]
    values = [
        float(singular[0]),
        float(singular[-1]),
        largest,
        *(value for record in records for value in record.values()),
        *(trial["largest_real_part"] for trial in trials),
    ]
    checks = {
        "every_value_finite": all(math.isfinite(value) for value in values),
        "every_listed_residual_at_rounding_level": all(
            record["relative_residual"] <= RESIDUAL_LIMIT for record in records
        ),
    }
    return {
        "schema": SCHEMA,
        "case": {"ka": KA, "half_width": HALF_WIDTH, "eta": ETA, "elements": elements},
        "unknowns": int(a.shape[0]),
        "norm_1": {"a": float(np.linalg.norm(a, 1)), "b": float(np.linalg.norm(b, 1))},
        "singular_values_a": {"largest": float(singular[0]), "smallest": float(singular[-1])},
        "near_zero_count": count,
        "near_zero_limit": NEAR_ZERO,
        "smallest_eigenvalues": records,
        "largest_real_part": largest,
        "unstable_count": unstable,
        "perturbation": {"relative_scale": PERTURBATION, "seed": SEED, "trials": trials},
        "positive_listed_inside_error_bound": all(
            record["real"] < record["error_bound"] for record in positive
        ),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "provenance": record_provenance,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the measurements as text and one table."""
    singular = report["singular_values_a"]
    lines = [
        *MARKDOWN_HEADER,
        "# Near-zero spectrum of the linear reduced-MHD Harris pencil",
        "",
        f"Status: **{report['status'].upper()}** (completeness only; findings are reported)",
        "",
        f"Case: ka = {report['case']['ka']}, η = {report['case']['eta']:.0e}, "
        f"{report['case']['elements']} elements per side, {report['unknowns']} unknowns.",
        "",
        f"Singular values of A: largest {singular['largest']:.3e}, smallest "
        f"{singular['smallest']:.3e}. Finite eigenvalues with |λ| < "
        f"{report['near_zero_limit']:.0e}: {report['near_zero_count']}.",
        "",
        "| Re λ | Im λ | Condition | Error bound | Residual |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in report["smallest_eigenvalues"]:
        lines.append(
            f"| {record['real']:.3e} | {record['imag']:.3e} | {record['condition']:.2e} "
            f"| {record['error_bound']:.2e} | {record['relative_residual']:.1e} |"
        )
    trials = ", ".join(
        f"{trial['largest_real_part']:.3e}" for trial in report["perturbation"]["trials"]
    )
    lines += [
        "",
        f"Largest real part: {report['largest_real_part']:.3e}. After relative perturbations "
        f"of {report['perturbation']['relative_scale']:.0e} to A: {trials}.",
        "",
        f"Source revision: `{report['provenance']['source_revision']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic and write the JSON and Markdown records."""
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
