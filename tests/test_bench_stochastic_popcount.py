# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests of the stochastic popcount benchmark
"""The popcount benchmark checks parity on every case and reports a well-formed record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import bench_stochastic_popcount as bench


def test_run_reports_identical_outputs_and_positive_timings() -> None:
    report = bench.run(cases=((3, 4, 128), (2, 2, 64)), repeats=1)
    assert report["schema"] == bench.SCHEMA
    assert report["identical"] is True
    assert len(report["cases"]) == 2
    for row in report["cases"]:
        assert row["identical"] is True
        assert row["vectorised_median_s"] > 0.0
        assert row["per_stream_median_s"] > 0.0
        assert row["speedup"] == pytest.approx(
            row["per_stream_median_s"] / row["vectorised_median_s"]
        )
    assert report["host"]["numpy"] == np.__version__


def test_reference_reproduces_the_former_loop_on_a_known_word() -> None:
    net = bench._net(1, 64)
    w_packed = np.array([[[0xFFFFFFFFFFFFFFFF]]], dtype=np.uint64)
    probs = np.array([1.0])
    # With every weight bit set and p = 1 every AND bit is set: the estimate is exactly 1.
    np.testing.assert_array_equal(bench.per_stream_reference(net, w_packed, probs), [1.0])
    np.testing.assert_array_equal(net.dense_forward(w_packed, probs), [1.0])


def test_main_writes_the_report_and_fails_on_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "reports" / "stochastic_popcount_benchmark.json"
    monkeypatch.setattr(bench, "REPORT", target)
    monkeypatch.setattr(bench, "CASES", ((2, 2, 64),))
    monkeypatch.setattr(bench, "REPEATS", 1)
    assert bench.main() == 0
    assert json.loads(target.read_text(encoding="utf-8"))["identical"] is True

    def disagreeing(net: object, w_packed: object, input_probs: object) -> np.ndarray:
        return np.full(2, -1.0)

    monkeypatch.setattr(bench, "per_stream_reference", disagreeing)
    assert bench.main() == 1
