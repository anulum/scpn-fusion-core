# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Stage 1 acceptance of the 1-D linear resistive reduced-MHD solver.

The criteria were fixed before the solver was written (reactor group plan
LINEAR_MHD_STABILITY_PLAN_20261019T201226Z, STAGE1_ACCEPTANCE.md) and are
copied here unchanged. A criterion that is missed is reported as a miss; the
script never adjusts a tolerance. Runtime is several minutes; set
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS on a shared machine.

Usage: python validation/linear_mhd_stage1_acceptance.py [--output result.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scpn_fusion.core.linear_mhd_1d import (
    WessonCylinder,
    all_eigenvalues,
    assemble,
    clustered_grid,
    cylinder_delta_prime,
    growth_rate,
    harris_slab,
    most_unstable,
)
from scpn_fusion.core.tearing_mode_theory import FKR_MATCHING_COEFFICIENT, HarrisSheetTearingModel

ETAS = (1e-6, 1e-7, 1e-8)


def _slope(values: list[float]) -> float:
    return float(np.polyfit(np.log(ETAS), np.log(values), 1)[0])


def run() -> dict[str, object]:
    """Evaluate every Stage 1 criterion and return the scored record."""
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
    criteria["S1-A1"] = {"value": slope, "band": [0.57, 0.63], "pass": 0.57 <= slope <= 0.63}
    ratio = slab_gamma[-1] / fkr[-1]
    criteria["S1-A2"] = {"value": ratio, "band": [0.90, 1.10], "pass": abs(ratio - 1.0) <= 0.10}
    stable = growth_rate(harris_slab(1.2, 20.0), 1e-6, n_side=200)
    criteria["S1-A3"] = {"value": stable, "limit": 1e-9, "pass": stable <= 1e-9}

    coarse = growth_rate(slab, 1e-7, n_side=200)
    fine = growth_rate(slab, 1e-7, n_side=400)
    change = abs(fine / coarse - 1.0)
    criteria["S1-B"] = {"value": change, "limit": 5e-3, "pass": change <= 5e-3}

    disc = assemble(slab, clustered_grid(-20.0, 20.0, 0.0, 0.02 * 1e-4 ** (1 / 3), 40), 1e-4)
    dense_max = float(all_eigenvalues(disc).real.max())
    found = most_unstable(disc)
    rel = abs((found.real if found is not None else 0.0) - dense_max) / dense_max
    criteria["S1-C"] = {"value": rel, "limit": 1e-8, "pass": rel <= 1e-8}

    tearing_cyl = WessonCylinder(1.2, 1.5, 10.0)
    tearing = tearing_cyl.geometry(2, 1)
    f_prime = abs(float(tearing.f_parallel_prime(np.array([tearing.resonant_x]))[0]))
    delta_prime = cylinder_delta_prime(tearing_cyl, 2, 1)
    tear_gamma = [growth_rate(tearing, eta, n_side=200) for eta in ETAS]
    tear_ref = FKR_MATCHING_COEFFICIENT * delta_prime**0.8 * ETAS[-1] ** 0.6 * f_prime**0.4
    slope = _slope(tear_gamma)
    criteria["S1-D1"] = {"value": slope, "band": [0.56, 0.64], "pass": 0.56 <= slope <= 0.64}
    ratio = tear_gamma[-1] / tear_ref
    criteria["S1-D2"] = {"value": ratio, "band": [0.85, 1.15], "pass": abs(ratio - 1.0) <= 0.15}

    kink = WessonCylinder(0.9, 1.0, 10.0).geometry(1, 1)
    k_prime = abs(float(kink.f_parallel_prime(np.array([kink.resonant_x]))[0]))
    kink_gamma = [growth_rate(kink, eta, n_side=200) for eta in ETAS]
    slope = _slope(kink_gamma)
    criteria["S1-E1"] = {
        "value": slope,
        "band": [1 / 3 - 0.03, 1 / 3 + 0.03],
        "pass": abs(slope - 1 / 3) <= 0.03,
    }
    ratio = kink_gamma[-1] / (ETAS[-1] ** (1 / 3) * k_prime ** (2 / 3))
    criteria["S1-E2"] = {"value": ratio, "band": [0.75, 1.25], "pass": abs(ratio - 1.0) <= 0.25}

    return {
        "schema": "scpn-fusion.linear-mhd-stage1-acceptance.v1",
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
    record = run()
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if record["passed"] == record["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
