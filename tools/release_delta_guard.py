#!/usr/bin/env python
"""Guard release deltas against pinned underdeveloped/claims baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "docs" / "release_delta_baseline.json"
DEFAULT_UNDERDEV_SUMMARY = REPO_ROOT / "docs" / "UNDERDEVELOPED_SCOPE_SUMMARY.json"
DEFAULT_CLAIMS_MANIFEST = REPO_ROOT / "validation" / "claims_manifest.json"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "artifacts" / "release_delta_guard_summary.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _coerce_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be >= 0.")
    return value


def _parse_underdeveloped_summary(path: Path) -> dict[str, int]:
    payload = _load_json(path, label="underdeveloped summary")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("underdeveloped summary must contain snapshots list.")
    by_scope: dict[str, dict[str, Any]] = {}
    for item in snapshots:
        if isinstance(item, dict) and isinstance(item.get("scope"), str):
            by_scope[item["scope"]] = item

    source = by_scope.get("source")
    docs_claims = by_scope.get("docs_claims")
    if source is None or docs_claims is None:
        raise ValueError("underdeveloped summary must include source and docs_claims snapshots.")

    return {
        "source_total": _coerce_int(source.get("total_entries"), label="source_total"),
        "source_p0p1": _coerce_int(source.get("p0_p1_entries"), label="source_p0p1"),
        "docs_claims_total": _coerce_int(
            docs_claims.get("total_entries"),
            label="docs_claims_total",
        ),
        "docs_claims_p0p1": _coerce_int(
            docs_claims.get("p0_p1_entries"),
            label="docs_claims_p0p1",
        ),
    }


def _parse_claims_count(path: Path) -> int:
    payload = _load_json(path, label="claims manifest")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims manifest must contain claims list.")
    return len(claims)


def evaluate(
    *,
    baseline: dict[str, Any],
    current: dict[str, int],
    claims_tracked: int,
    require_positive_delta: bool,
) -> dict[str, Any]:
    required_keys = (
        "source_total",
        "source_p0p1",
        "docs_claims_total",
        "docs_claims_p0p1",
        "claims_tracked",
    )
    baseline_metrics: dict[str, int] = {}
    for key in required_keys:
        baseline_metrics[key] = _coerce_int(baseline.get(key), label=f"baseline[{key}]")

    current_metrics = dict(current)
    current_metrics["claims_tracked"] = int(claims_tracked)

    deltas: dict[str, int] = {
        key: current_metrics[key] - baseline_metrics[key] for key in required_keys
    }

    checks = {
        "source_p0p1_non_regression": current_metrics["source_p0p1"]
        <= baseline_metrics["source_p0p1"],
        "docs_claims_p0p1_non_regression": (
            current_metrics["docs_claims_p0p1"] <= baseline_metrics["docs_claims_p0p1"]
        ),
        "claims_tracked_non_regression": (
            current_metrics["claims_tracked"] >= baseline_metrics["claims_tracked"]
        ),
    }
    positive_improvements = {
        "source_p0p1_reduction": current_metrics["source_p0p1"] < baseline_metrics["source_p0p1"],
        "docs_claims_p0p1_reduction": (
            current_metrics["docs_claims_p0p1"] < baseline_metrics["docs_claims_p0p1"]
        ),
        "claims_tracked_growth": (
            current_metrics["claims_tracked"] > baseline_metrics["claims_tracked"]
        ),
    }

    overall_pass = all(checks.values())
    if require_positive_delta:
        overall_pass = overall_pass and any(positive_improvements.values())

    return {
        "baseline": baseline_metrics,
        "current": current_metrics,
        "deltas": deltas,
        "checks": checks,
        "positive_improvements": positive_improvements,
        "require_positive_delta": bool(require_positive_delta),
        "overall_pass": bool(overall_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Pinned release-delta baseline JSON.",
    )
    parser.add_argument(
        "--underdeveloped-summary",
        default=str(DEFAULT_UNDERDEV_SUMMARY),
        help="Generated underdeveloped scope summary JSON.",
    )
    parser.add_argument(
        "--claims-manifest",
        default=str(DEFAULT_CLAIMS_MANIFEST),
        help="Claims manifest JSON used to generate claims/evidence map.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--require-positive-delta",
        action="store_true",
        help="Require at least one positive improvement vs baseline.",
    )
    args = parser.parse_args(argv)

    baseline_path = _resolve(args.baseline)
    underdev_path = _resolve(args.underdeveloped_summary)
    claims_manifest = _resolve(args.claims_manifest)
    summary_path = _resolve(args.summary_json)

    baseline = _load_json(baseline_path, label="release delta baseline")
    current = _parse_underdeveloped_summary(underdev_path)
    claims_tracked = _parse_claims_count(claims_manifest)
    summary = evaluate(
        baseline=baseline,
        current=current,
        claims_tracked=claims_tracked,
        require_positive_delta=bool(args.require_positive_delta),
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Release delta guard "
        f"{'PASS' if summary['overall_pass'] else 'FAIL'} "
        f"(positive_required={summary['require_positive_delta']})"
    )
    for key, value in summary["deltas"].items():
        sign = "+" if value >= 0 else ""
        print(f"- delta[{key}]={sign}{value}")
    print(f"- {_display_path(summary_path)}")

    return 0 if bool(summary["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
