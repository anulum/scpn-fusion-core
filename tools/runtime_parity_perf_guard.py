#!/usr/bin/env python
"""Runtime parity + latency performance regression guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARITY = REPO_ROOT / "artifacts" / "traceable_runtime_parity.json"
DEFAULT_LATENCY = REPO_ROOT / "artifacts" / "scpn_end_to_end_latency_ci.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "runtime_parity_perf_thresholds.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "runtime_parity_perf_guard_summary.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def evaluate(
    *,
    parity: dict[str, Any],
    latency: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    parity_cfg = dict(thresholds.get("parity", {}))
    latency_cfg = dict(thresholds.get("latency", {}))

    reports = parity.get("reports", [])
    if not isinstance(reports, list):
        reports = []
    strict_ok = bool(parity.get("strict_ok", False))
    min_reports = int(parity_cfg.get("min_report_count", 1))

    campaign = dict(latency.get("scpn_end_to_end_latency", {}))
    modes = dict(campaign.get("modes", {}))
    surrogate_snn = float(modes.get("surrogate", {}).get("SNN", {}).get("p95_loop_ms", float("inf")))
    full_snn = float(modes.get("full", {}).get("SNN", {}).get("p95_loop_ms", float("inf")))
    ratio = float(campaign.get("ratios", {}).get("snn_full_to_surrogate_p95_ratio", float("inf")))
    latency_passes = bool(campaign.get("passes_thresholds", False))

    parity_pass = bool(strict_ok and len(reports) >= min_reports)
    perf_pass = bool(
        latency_passes
        and surrogate_snn <= float(latency_cfg.get("max_snn_p95_surrogate_ms", 6.0))
        and full_snn <= float(latency_cfg.get("max_snn_p95_full_ms", 10.0))
        and ratio <= float(latency_cfg.get("max_full_to_surrogate_ratio", 6.5))
    )

    return {
        "parity": {
            "strict_ok": strict_ok,
            "report_count": len(reports),
            "min_report_count": min_reports,
            "passes": parity_pass,
        },
        "latency": {
            "passes_thresholds": latency_passes,
            "snn_p95_surrogate_ms": surrogate_snn,
            "snn_p95_full_ms": full_snn,
            "full_to_surrogate_ratio": ratio,
            "limits": {
                "max_snn_p95_surrogate_ms": float(latency_cfg.get("max_snn_p95_surrogate_ms", 6.0)),
                "max_snn_p95_full_ms": float(latency_cfg.get("max_snn_p95_full_ms", 10.0)),
                "max_full_to_surrogate_ratio": float(latency_cfg.get("max_full_to_surrogate_ratio", 6.5)),
            },
            "passes": perf_pass,
        },
        "overall_pass": bool(parity_pass and perf_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", default=str(DEFAULT_PARITY))
    parser.add_argument("--latency", default=str(DEFAULT_LATENCY))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    parity_path = _resolve(args.parity)
    latency_path = _resolve(args.latency)
    thresholds_path = _resolve(args.thresholds)
    summary_path = _resolve(args.summary_json)

    summary = evaluate(
        parity=_load_json(parity_path),
        latency=_load_json(latency_path),
        thresholds=_load_json(thresholds_path),
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Runtime guard summary: "
        f"parity_pass={summary['parity']['passes']} "
        f"latency_pass={summary['latency']['passes']} "
        f"snn_p95_surrogate_ms={summary['latency']['snn_p95_surrogate_ms']:.4f}"
    )
    if not bool(summary["overall_pass"]):
        print("Runtime parity/perf guard failed.")
        return 1
    print("Runtime parity/perf guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
