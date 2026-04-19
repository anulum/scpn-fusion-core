# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runtime Parity/Perf Guard Tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "runtime_parity_perf_guard.py"
SPEC = importlib.util.spec_from_file_location("runtime_parity_perf_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def test_evaluate_passes_for_valid_reports() -> None:
    summary = guard.evaluate(
        parity={
            "strict_ok": True,
            "reports": [{"backend": "numpy", "single_within_tol": True, "batch_within_tol": True}],
        },
        latency={
            "scpn_end_to_end_latency": {
                "passes_thresholds": True,
                "modes": {
                    "surrogate": {"SNN": {"p95_loop_ms": 1.2}},
                    "full": {"SNN": {"p95_loop_ms": 2.4}},
                },
                "ratios": {"snn_full_to_surrogate_p95_ratio": 2.0},
            }
        },
        thresholds={
            "parity": {"min_report_count": 1},
            "latency": {
                "max_snn_p95_surrogate_ms": 6.0,
                "max_snn_p95_full_ms": 10.0,
                "max_full_to_surrogate_ratio": 6.5,
            },
        },
    )
    assert summary["overall_pass"] is True


def test_evaluate_fails_when_parity_not_strict_ok() -> None:
    summary = guard.evaluate(
        parity={"strict_ok": False, "reports": [{"backend": "numpy"}]},
        latency={
            "scpn_end_to_end_latency": {
                "passes_thresholds": True,
                "modes": {
                    "surrogate": {"SNN": {"p95_loop_ms": 1.2}},
                    "full": {"SNN": {"p95_loop_ms": 2.4}},
                },
                "ratios": {"snn_full_to_surrogate_p95_ratio": 2.0},
            }
        },
        thresholds={"parity": {"min_report_count": 1}, "latency": {}},
    )
    assert summary["parity"]["passes"] is False
    assert summary["overall_pass"] is False


def test_main_writes_summary_and_nonzero_on_latency_regression(tmp_path: Path) -> None:
    parity_path = tmp_path / "parity.json"
    latency_path = tmp_path / "latency.json"
    thresholds_path = tmp_path / "thresholds.json"
    summary_path = tmp_path / "summary.json"

    parity_path.write_text(
        json.dumps({"strict_ok": True, "reports": [{"backend": "numpy"}]}),
        encoding="utf-8",
    )
    latency_path.write_text(
        json.dumps(
            {
                "scpn_end_to_end_latency": {
                    "passes_thresholds": False,
                    "modes": {
                        "surrogate": {"SNN": {"p95_loop_ms": 9.0}},
                        "full": {"SNN": {"p95_loop_ms": 20.0}},
                    },
                    "ratios": {"snn_full_to_surrogate_p95_ratio": 10.0},
                }
            }
        ),
        encoding="utf-8",
    )
    thresholds_path.write_text(
        json.dumps(
            {
                "parity": {"min_report_count": 1},
                "latency": {
                    "max_snn_p95_surrogate_ms": 6.0,
                    "max_snn_p95_full_ms": 10.0,
                    "max_full_to_surrogate_ratio": 6.5,
                },
            }
        ),
        encoding="utf-8",
    )

    rc = guard.main(
        [
            "--parity",
            str(parity_path),
            "--latency",
            str(latency_path),
            "--thresholds",
            str(thresholds_path),
            "--summary-json",
            str(summary_path),
        ]
    )
    assert rc == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is False
