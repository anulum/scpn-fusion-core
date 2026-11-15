# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — benchmark of the stochastic AND + popcount matrix-vector product
"""Time ``CompiledNet.dense_forward`` against the former per-stream popcount path.

Until 2026-11 ``dense_forward`` tested ``hasattr(np, "bit_count")``, which no NumPy release
provides, so it always ran a Python loop over stream pairs. It now ANDs every weight stream with
its input stream and counts all words at once with ``np.bitwise_count``. This benchmark rebuilds
the former loop as a reference, checks that both produce identical outputs on every case, and
records the wall time of each on the executing host. Timings are orientation only; the report
names the host.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.neurocore_compat import (
    RNG,
    generate_bernoulli_bitstream,
    pack_bitstream,
    vec_and,
)
from scpn_fusion.scpn.compiler import CompiledNet

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation" / "reports" / "stochastic_popcount_benchmark.json"
SCHEMA = "scpn-fusion.stochastic-popcount-benchmark.v1"
CASES = ((8, 8, 1024), (32, 32, 1024), (64, 16, 4096))
REPEATS = 5
SEED = 20261115


def _net(n_out: int, length: int) -> CompiledNet:
    return CompiledNet(
        n_places=1,
        n_transitions=n_out,
        place_names=["p0"],
        transition_names=[f"t{i}" for i in range(n_out)],
        W_in=np.zeros((n_out, 1), dtype=np.float64),
        W_out=np.zeros((1, n_out), dtype=np.float64),
        neurons=[],
        thresholds=np.full(n_out, 0.5, dtype=np.float64),
        firing_mode="binary",
        bitstream_length=length,
    )


def per_stream_reference(
    net: CompiledNet, w_packed: NDArray[np.uint64], input_probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reproduce the former loop: one AND and one byte-unpacking popcount per stream pair."""
    n_out, n_in, _n_words = w_packed.shape
    length = net.bitstream_length
    streams = [
        pack_bitstream(
            generate_bernoulli_bitstream(
                float(np.clip(input_probs[j], 0.0, 1.0)),
                length,
                rng=RNG(net.seed + 1_000_000 + j),
            )
        )
        for j in range(n_in)
    ]
    output = np.zeros(n_out, dtype=np.float64)
    for i in range(n_out):
        total = 0
        for j in range(n_in):
            words = vec_and(w_packed[i, j, :], streams[j])
            total += int(np.unpackbits(words.view(np.uint8), bitorder="little").sum())
        output[i] = total / length
    return output


def _median_seconds(function: Callable[[], NDArray[np.float64]], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def run(cases: tuple[tuple[int, int, int], ...] = CASES, repeats: int = REPEATS) -> dict[str, Any]:
    """Run every case and return the report; ``identical`` is False if any output differs."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n_out, n_in, length in cases:
        net = _net(n_out, length)
        w_packed = rng.integers(0, 2**64, size=(n_out, n_in, length // 64), dtype=np.uint64)
        input_probs = rng.uniform(0.0, 1.0, size=n_in)
        vectorised = net.dense_forward(w_packed, input_probs)
        reference = per_stream_reference(net, w_packed, input_probs)
        vectorised_s = _median_seconds(lambda: net.dense_forward(w_packed, input_probs), repeats)
        reference_s = _median_seconds(
            lambda: per_stream_reference(net, w_packed, input_probs), repeats
        )
        rows.append(
            {
                "bitstream_length": length,
                "identical": bool(np.array_equal(vectorised, reference)),
                "n_in": n_in,
                "n_out": n_out,
                "per_stream_median_s": reference_s,
                "speedup": reference_s / vectorised_s,
                "vectorised_median_s": vectorised_s,
            }
        )
    return {
        "cases": rows,
        "host": {
            "logical_cpus": os.cpu_count() or 0,
            "machine": platform.machine(),
            "numpy": np.__version__,
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
        },
        "identical": all(row["identical"] for row in rows),
        "repeats": repeats,
        "schema": SCHEMA,
        "seed": SEED,
        "timing_scope": "orientation only; wall time on the named host",
    }


def main() -> int:
    """Write the report; exit 1 if the two paths ever disagree."""
    report = run(CASES, REPEATS)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
