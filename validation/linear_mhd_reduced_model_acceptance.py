# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fixed acceptance of the 1-D linear resistive reduced-MHD solver.

The criteria were fixed before the solver was written and are kept here
unchanged. A criterion that is missed is reported as a miss; the script never
adjusts a tolerance. Runtime is several minutes; set
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS on a shared machine.

The record carries the source revision, the digests of the solver sources and of
this script, the library versions, the BLAS build, the thread settings and the
host CPU, taken before the run, so that a result can be traced to its code and
host.

Usage: python validation/linear_mhd_reduced_model_acceptance.py [--output result.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from scpn_fusion.core.linear_mhd_1d import (
    WessonCylinder,
    assemble,
    clustered_grid,
    growth_rate,
    harris_slab,
)
from scpn_fusion.core.linear_mhd_delta_prime import cylinder_delta_prime
from scpn_fusion.core.linear_mhd_spectrum import all_eigenvalues, most_unstable
from scpn_fusion.core.tearing_mode_theory import FKR_MATCHING_COEFFICIENT, HarrisSheetTearingModel

ETAS = (1e-6, 1e-7, 1e-8)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/scpn_fusion/core/linear_mhd_1d.py",
    "src/scpn_fusion/core/linear_mhd_spectrum.py",
    "src/scpn_fusion/core/linear_mhd_delta_prime.py",
    "src/scpn_fusion/core/tearing_mode_theory.py",
    "validation/linear_mhd_reduced_model_acceptance.py",
)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _slope(values: list[float]) -> float:
    return float(np.polyfit(np.log(ETAS), np.log(values), 1)[0])


def _cpu_models() -> list[str]:
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    return sorted(
        {
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        }
    )


def provenance() -> dict[str, object]:
    """Return the source revision, source digests, library versions and host of this run."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "source_revision": revision,
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_PATHS
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": {"name": blas["name"], "version": blas["version"]},
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "machine": platform.machine(),
        "cpu_models": _cpu_models(),
        "cpu_count": os.cpu_count(),
        "started_at": datetime.now(UTC).isoformat(),
    }


def run() -> dict[str, object]:
    """Evaluate every acceptance criterion and return the scored record."""
    criteria: dict[str, dict[str, object]] = {}

    slab = harris_slab(0.5, 20.0)
    model = HarrisSheetTearingModel(
        sheet_half_width_m=1.0, wave_number_per_m=0.5, finite_width_coefficient=1.0
    )
    slab_gamma = [growth_rate(slab, eta, n_side=200) for eta in ETAS]
    fkr = [
        model.fkr_growth_rate_per_s(alfven_time_s=1.0, lundquist_number=1.0 / eta) for eta in ETAS
    ]
    slope = _slope(slab_gamma)
    criteria["harris_tearing_slope"] = {
        "value": slope,
        "band": [0.57, 0.63],
        "pass": 0.57 <= slope <= 0.63,
    }
    ratio = slab_gamma[-1] / fkr[-1]
    criteria["harris_tearing_fkr_ratio"] = {
        "value": ratio,
        "band": [0.90, 1.10],
        "pass": abs(ratio - 1.0) <= 0.10,
    }
    stable = growth_rate(harris_slab(1.2, 20.0), 1e-6, n_side=200)
    criteria["harris_negative_delta_prime_stable"] = {
        "value": stable,
        "limit": 1e-9,
        "pass": stable <= 1e-9,
    }

    coarse = growth_rate(slab, 1e-7, n_side=200)
    fine = growth_rate(slab, 1e-7, n_side=400)
    change = abs(fine / coarse - 1.0)
    criteria["harris_element_doubling_change"] = {
        "value": change,
        "limit": 5e-3,
        "pass": change <= 5e-3,
    }

    disc = assemble(slab, clustered_grid(-20.0, 20.0, 0.0, 0.02 * 1e-4 ** (1 / 3), 40), 1e-4)
    dense_max = float(all_eigenvalues(disc).real.max())
    found = most_unstable(disc)
    rel = abs((found.real if found is not None else 0.0) - dense_max) / dense_max
    criteria["sparse_dense_agreement"] = {"value": rel, "limit": 1e-8, "pass": rel <= 1e-8}

    tearing_cyl = WessonCylinder(1.2, 1.5, 10.0)
    tearing = tearing_cyl.geometry(2, 1)
    f_prime = abs(float(tearing.f_parallel_prime(np.array([tearing.resonant_x]))[0]))
    delta_prime = cylinder_delta_prime(tearing_cyl, 2, 1)
    tear_gamma = [growth_rate(tearing, eta, n_side=200) for eta in ETAS]
    tear_ref = FKR_MATCHING_COEFFICIENT * delta_prime**0.8 * ETAS[-1] ** 0.6 * f_prime**0.4
    slope = _slope(tear_gamma)
    criteria["cylinder_tearing_slope"] = {
        "value": slope,
        "band": [0.56, 0.64],
        "pass": 0.56 <= slope <= 0.64,
    }
    ratio = tear_gamma[-1] / tear_ref
    criteria["cylinder_tearing_fkr_ratio"] = {
        "value": ratio,
        "band": [0.85, 1.15],
        "pass": abs(ratio - 1.0) <= 0.15,
    }

    kink = WessonCylinder(0.9, 1.0, 10.0).geometry(1, 1)
    k_prime = abs(float(kink.f_parallel_prime(np.array([kink.resonant_x]))[0]))
    kink_gamma = [growth_rate(kink, eta, n_side=200) for eta in ETAS]
    slope = _slope(kink_gamma)
    criteria["cylinder_resistive_kink_slope"] = {
        "value": slope,
        "band": [1 / 3 - 0.03, 1 / 3 + 0.03],
        "pass": abs(slope - 1 / 3) <= 0.03,
    }
    ratio = kink_gamma[-1] / (ETAS[-1] ** (1 / 3) * k_prime ** (2 / 3))
    criteria["cylinder_resistive_kink_ratio"] = {
        "value": ratio,
        "band": [0.75, 1.25],
        "pass": abs(ratio - 1.0) <= 0.25,
    }

    return {
        "schema": "scpn-fusion.linear-mhd-reduced-model-acceptance.v1",
        "inputs": {
            "etas": list(ETAS),
            "slab_growth": slab_gamma,
            "cylinder_tearing_growth": tear_gamma,
            "cylinder_delta_prime": delta_prime,
            "cylinder_kink_growth": kink_gamma,
        },
        "criteria": criteria,
        "passed": sum(bool(c["pass"]) for c in criteria.values()),
        "total": len(criteria),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the acceptance and print or write the scored record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    origin = provenance()
    started = time.monotonic()
    record = run()
    record["provenance"] = {**origin, "elapsed_s": time.monotonic() - started}
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if record["passed"] == record["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
