#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO Turbulence Tier Benchmark
"""Benchmark the dispatched FNO turbulence surrogate backends.

Measures the spectral FNO forward (``predict``) wall-clock per backend tier
(RUST and NUMPY) over a shared deterministic weight archive, checks the shared
invariant (finite prediction of the expected shape), and records which tier
the class-kernel dispatcher selects. Timings are local non-isolated regression
evidence, not production throughput claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "fno_turbulence_benchmark.json"
SCHEMA = "scpn-fusion-core.fno-turbulence-benchmark.v1"
KERNEL_NAME = "fno_turbulence"
DEFAULT_GRID = 64
DEFAULT_PREDICTS = 200
DEFAULT_REPEATS = 3
WEIGHT_SEED = 2026
FIELD_SEED = 7

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.core import _multi_compat_providers as providers  # noqa: E402
from scpn_fusion.core.fno_training import MultiLayerFNO  # noqa: E402

MetricRow: TypeAlias = dict[str, Any]


def _backend_loaders() -> dict[str, Any]:
    """Return the tier-name -> class-loader mapping for the FNO kernel."""
    return {
        "RUST": providers._load_rust_fno,
        "NUMPY": providers._load_numpy_fno,
    }


def _write_weights(directory: Path) -> Path:
    """Write a deterministic FNO weight archive and return its path."""
    model = MultiLayerFNO(seed=WEIGHT_SEED)
    path = directory / "fno_weights.npz"
    model.save_weights(path)
    return path


def _run_backend(
    kernel_cls: type, weights: Path, field: Any, predicts: int, repeats: int
) -> MetricRow:
    """Time the predict loop for one backend and check its invariant."""
    elapsed: list[float] = []
    final_energy = float("nan")
    invariants_passed = False
    for _ in range(repeats):
        kernel = kernel_cls(weights)
        start = time.perf_counter()
        predictions = [kernel.predict(field) for _ in range(predicts)]
        elapsed.append(time.perf_counter() - start)
        last = np.asarray(predictions[-1], dtype=np.float64)
        final_energy = float(np.mean(last**2))
        invariants_passed = bool(np.all(np.isfinite(last)) and last.shape == field.shape)
    return {
        "median_loop_s": float(np.median(elapsed)),
        "median_predict_us": float(np.median(elapsed) / predicts * 1e6),
        "final_prediction_energy": final_energy,
        "invariants_passed": invariants_passed,
    }


def build_report(
    *, grid: int = DEFAULT_GRID, predicts: int = DEFAULT_PREDICTS, repeats: int = DEFAULT_REPEATS
) -> MetricRow:
    """Build the FNO turbulence backend benchmark report payload."""
    registered = multi.registered_kernel_classes().get(KERNEL_NAME, [])
    with tempfile.TemporaryDirectory() as tmp:
        weights = _write_weights(Path(tmp))
        selected_cls = multi.dispatch_kernel_class(KERNEL_NAME)
        field = np.random.default_rng(FIELD_SEED).standard_normal((grid, grid)) * 0.5
        backends: dict[str, MetricRow] = {}
        for tier_name, loader in _backend_loaders().items():
            try:
                kernel_cls = loader()
            except (ImportError, AttributeError, TypeError) as exc:
                backends[tier_name] = {"status": "unavailable", "reason": str(exc)}
                continue
            row = _run_backend(kernel_cls, weights, field, predicts, repeats)
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
        "grid": grid,
        "predicts": predicts,
        "repeats": repeats,
        "weight_seed": WEIGHT_SEED,
        "field_seed": FIELD_SEED,
        "registered_tiers": registered,
        "selected_class": selected_cls.__name__,
        "backends": backends,
        "rust_over_numpy_speedup": speedup,
        "physics_invariants_passed": all_invariants,
        "model_contract": (
            "spectral FNO forward (lift -> Fourier spectral convolution + "
            "pointwise skip + GELU per layer -> project) over a shared weight "
            "archive; both tiers run the identical forward and agree to "
            "floating-point round-off"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FNO turbulence backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--predicts", type=int, default=DEFAULT_PREDICTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(grid=args.grid, predicts=args.predicts, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["physics_invariants_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
