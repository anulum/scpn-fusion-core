#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Surrogate MPC Tier Benchmark
"""Benchmark the dispatched canonical-configuration surrogate-MPC backends.

Measures the gradient-descent ``plan`` wall-clock per backend tier (RUST and
NUMPY) over the linear surrogate ``x_{t+1} = x_t + B u_t``, checks the shared
invariants (finite, action-limit-bounded plan), and records which tier the
class-kernel dispatcher selects. Timings are local non-isolated regression
evidence, not production throughput claims.
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
REPORT = ROOT / "validation" / "reports" / "neural_surrogate_mpc_benchmark.json"
SCHEMA = "scpn-fusion-core.neural-surrogate-mpc-benchmark.v1"
KERNEL_NAME = "neural_surrogate_mpc"
ACTION_LIMIT = 2.0
DEFAULT_STATE_DIM = 4
DEFAULT_COIL_DIM = 3
DEFAULT_PLANS = 2000
DEFAULT_REPEATS = 3
BENCH_SEED = 2026

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.core import _multi_compat_providers as providers  # noqa: E402

MetricRow: TypeAlias = dict[str, Any]


def _backend_loaders() -> dict[str, Any]:
    """Return the tier-name -> class-loader mapping for the surrogate-MPC kernel."""
    return {
        "RUST": providers._load_rust_mpc_controller,
        "NUMPY": providers._load_numpy_mpc_controller,
    }


def _plan_case(state_dim: int, coil_dim: int) -> tuple[Any, Any, Any]:
    """Return a deterministic ``(B, target, state)`` planning case."""
    rng = np.random.default_rng(BENCH_SEED)
    b_matrix = rng.normal(size=(state_dim, coil_dim)) * 0.1
    target = np.zeros(state_dim, dtype=np.float64)
    state = rng.normal(size=state_dim)
    return b_matrix, target, state


def _run_backend(
    controller_cls: type, b_matrix: Any, target: Any, state: Any, plans: int, repeats: int
) -> MetricRow:
    """Time the seeded plan loop for one backend and check its invariants."""
    elapsed: list[float] = []
    final_action_norm = float("nan")
    invariants_passed = False
    for _ in range(repeats):
        controller = controller_cls(b_matrix, target)
        start = time.perf_counter()
        actions = [controller.plan(state) for _ in range(plans)]
        elapsed.append(time.perf_counter() - start)
        stacked = np.asarray(actions, dtype=np.float64)
        final_action_norm = float(np.linalg.norm(stacked[-1]))
        invariants_passed = bool(
            np.all(np.isfinite(stacked)) and np.all(np.abs(stacked) <= ACTION_LIMIT + 1e-9)
        )
    return {
        "median_loop_s": float(np.median(elapsed)),
        "median_plan_us": float(np.median(elapsed) / plans * 1e6),
        "final_action_norm": final_action_norm,
        "invariants_passed": invariants_passed,
    }


def build_report(
    *,
    state_dim: int = DEFAULT_STATE_DIM,
    coil_dim: int = DEFAULT_COIL_DIM,
    plans: int = DEFAULT_PLANS,
    repeats: int = DEFAULT_REPEATS,
) -> MetricRow:
    """Build the surrogate-MPC backend benchmark report payload."""
    registered = multi.registered_kernel_classes().get(KERNEL_NAME, [])
    selected_cls = multi.dispatch_kernel_class(KERNEL_NAME)
    b_matrix, target, state = _plan_case(state_dim, coil_dim)
    backends: dict[str, MetricRow] = {}
    for tier_name, loader in _backend_loaders().items():
        try:
            controller_cls = loader()
        except (ImportError, AttributeError, TypeError) as exc:
            backends[tier_name] = {"status": "unavailable", "reason": str(exc)}
            continue
        row = _run_backend(controller_cls, b_matrix, target, state, plans, repeats)
        row["status"] = "available"
        row["class_name"] = controller_cls.__name__
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
        "state_dim": state_dim,
        "coil_dim": coil_dim,
        "plans": plans,
        "repeats": repeats,
        "seed": BENCH_SEED,
        "registered_tiers": registered,
        "selected_class": selected_cls.__name__,
        "backends": backends,
        "rust_over_numpy_speedup": speedup,
        "physics_invariants_passed": all_invariants,
        "model_contract": (
            "canonical-configuration gradient-descent MPC over the linear "
            "surrogate x_{t+1} = x_t + B u_t (horizon 10, learning rate 0.5, "
            "20 iterations, action limit 2.0, regularisation 0.1); both tiers "
            "run the identical deterministic planner and agree to round-off"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the surrogate-MPC backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dim", type=int, default=DEFAULT_STATE_DIM)
    parser.add_argument("--coil-dim", type=int, default=DEFAULT_COIL_DIM)
    parser.add_argument("--plans", type=int, default=DEFAULT_PLANS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(
        state_dim=args.state_dim, coil_dim=args.coil_dim, plans=args.plans, repeats=args.repeats
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["physics_invariants_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
