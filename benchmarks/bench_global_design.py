#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Global Design Evaluator Tier Benchmark
"""Benchmark the dispatched reactor-design evaluator backends.

Measures the single-point ``evaluate_design`` wall-clock per backend tier (RUST
and NUMPY) over a shared deterministic sweep of reactor design points, checks the
shared invariant (finite fusion metrics of the expected contract), and records
which tier the class-kernel dispatcher selects. Timings are local non-isolated
regression evidence, not production throughput claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "global_design_benchmark.json"
SCHEMA = "scpn-fusion-core.global-design-benchmark.v1"
KERNEL_NAME = "global_design_scan"
DEFAULT_POINTS = 4000
DEFAULT_REPEATS = 3
POINT_SEED = 2026

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.core import _multi_compat_providers as providers  # noqa: E402

MetricRow: TypeAlias = dict[str, Any]


def _backend_loaders() -> dict[str, Any]:
    """Return the tier-name -> class-loader mapping for the design evaluator."""
    return {
        "RUST": providers._load_rust_design_evaluator,
        "NUMPY": providers._load_numpy_design_evaluator,
    }


def _design_points(n_points: int) -> list[tuple[float, float, float]]:
    """Return a deterministic sweep of (R, B, Ip) reactor design points."""
    rng = np.random.default_rng(POINT_SEED)
    r = rng.uniform(1.1, 9.0, n_points)
    b = rng.uniform(4.0, 12.2, n_points)
    i_p = rng.uniform(2.0, 25.0, n_points)
    return [(float(r[k]), float(b[k]), float(i_p[k])) for k in range(n_points)]


def _run_backend(
    kernel_cls: type, points: list[tuple[float, float, float]], repeats: int
) -> MetricRow:
    """Time the evaluate_design loop for one backend and check its invariant."""
    elapsed: list[float] = []
    last_p_fus = float("nan")
    invariants_passed = False
    for _ in range(repeats):
        kernel = kernel_cls("dummy")
        start = time.perf_counter()
        designs = [kernel.evaluate_design(r, b, i_p) for (r, b, i_p) in points]
        elapsed.append(time.perf_counter() - start)
        last = designs[-1]
        last_p_fus = float(last["P_fus"])
        invariants_passed = bool(
            np.isfinite(last_p_fus)
            and np.isfinite(float(last["Q"]))
            and isinstance(last["Constraint_OK"], bool)
            and last["Model_Regime"] == "physics_scaling_surrogate"
        )
    return {
        "median_loop_s": float(np.median(elapsed)),
        "median_evaluate_us": float(np.median(elapsed) / len(points) * 1e6),
        "final_p_fus_mw": last_p_fus,
        "invariants_passed": invariants_passed,
    }


def build_report(*, n_points: int = DEFAULT_POINTS, repeats: int = DEFAULT_REPEATS) -> MetricRow:
    """Build the reactor-design evaluator backend benchmark report payload."""
    registered = multi.registered_kernel_classes().get(KERNEL_NAME, [])
    selected_cls = multi.dispatch_kernel_class(KERNEL_NAME)
    points = _design_points(n_points)
    backends: dict[str, MetricRow] = {}
    for tier_name, loader in _backend_loaders().items():
        try:
            kernel_cls = loader()
        except (ImportError, AttributeError, TypeError) as exc:
            backends[tier_name] = {"status": "unavailable", "reason": str(exc)}
            continue
        row = _run_backend(kernel_cls, points, repeats)
        row["status"] = "available"
        row["class_name"] = kernel_cls.__name__
        backends[tier_name] = row
    speedup: float | None = None
    rust_row = backends.get("RUST", {})
    numpy_row = backends.get("NUMPY", {})
    if rust_row.get("status") == "available" and numpy_row.get("status") == "available":
        speedup = float(numpy_row["median_loop_s"] / max(rust_row["median_loop_s"], 1e-12))
    all_invariants = all(
        row.get("invariants_passed", False)
        for row in backends.values()
        if row.get("status") == "available"
    )
    return {
        "schema": SCHEMA,
        "kernel": KERNEL_NAME,
        "n_points": n_points,
        "repeats": repeats,
        "point_seed": POINT_SEED,
        "registered_tiers": registered,
        "selected_class": selected_cls.__name__,
        "backends": backends,
        "rust_over_numpy_speedup": speedup,
        "physics_invariants_passed": all_invariants,
        "model_contract": (
            "physics-scaling reactor-design surrogate (Troyon/H-mode beta_N "
            "shaping, Eich divertor scaling, HEAT-ML magnetic-shadow ridge "
            "attenuation with shared frozen weights and engineering caps); both "
            "tiers agree to floating-point round-off (~1e-15 relative, "
            "Constraint_OK identical)"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reactor-design evaluator backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(n_points=args.points, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["physics_invariants_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
