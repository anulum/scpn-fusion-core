#!/usr/bin/env python
"""Fallback-budget regression guard for benchmark provenance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "fallback_budget_thresholds.json"
DEFAULT_TORAX = REPO_ROOT / "artifacts" / "torax_benchmark.json"
DEFAULT_SPARC = REPO_ROOT / "artifacts" / "sparc_geqdsk_rmse_benchmark.json"
DEFAULT_FREEGS = REPO_ROOT / "artifacts" / "freegs_benchmark.json"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "fallback_budget_summary.json"


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


def _case_backends(cases: list[dict[str, Any]], key: str) -> list[str]:
    return [str(case.get(key, "unknown")) for case in cases]


def _fallback_rate_for_cases(
    cases: list[dict[str, Any]],
    *,
    backend_key: str,
    preferred_backend: str,
) -> float:
    if not cases:
        return 1.0
    fallback_count = 0
    for case in cases:
        backend = str(case.get(backend_key, "unknown"))
        if backend != preferred_backend:
            fallback_count += 1
    return float(fallback_count / len(cases))


def evaluate(
    *,
    torax: dict[str, Any],
    sparc: dict[str, Any],
    freegs: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    torax_cfg = dict(thresholds.get("torax", {}))
    sparc_cfg = dict(thresholds.get("sparc", {}))
    freegs_cfg = dict(thresholds.get("freegs", {}))

    torax_cases = [dict(c) for c in torax.get("cases", []) if isinstance(c, dict)]
    sparc_cases = [dict(c) for c in sparc.get("cases", []) if isinstance(c, dict)]
    freegs_cases = [dict(c) for c in freegs.get("cases", []) if isinstance(c, dict)]
    if not torax_cases:
        raise ValueError("torax benchmark artifact has zero cases")
    if not sparc_cases:
        raise ValueError("sparc benchmark artifact has zero cases")
    if not freegs_cases:
        raise ValueError("freegs benchmark artifact has zero cases")

    torax_preferred = str(torax_cfg.get("preferred_backend", "neural_transport"))
    sparc_preferred = str(sparc_cfg.get("preferred_backend", "neural_equilibrium"))
    torax_allowed = {str(v) for v in torax_cfg.get("allowed_backends", [torax_preferred])}
    sparc_allowed = {str(v) for v in sparc_cfg.get("allowed_backends", [sparc_preferred])}
    freegs_allowed_modes = {str(v) for v in freegs_cfg.get("allowed_modes", ["solovev_manufactured_source", "freegs"])}

    torax_rate = _fallback_rate_for_cases(
        torax_cases,
        backend_key="transport_backend",
        preferred_backend=torax_preferred,
    )
    sparc_rate = _fallback_rate_for_cases(
        sparc_cases,
        backend_key="surrogate_backend",
        preferred_backend=sparc_preferred,
    )
    torax_backends = sorted(set(_case_backends(torax_cases, "transport_backend")))
    sparc_backends = sorted(set(_case_backends(sparc_cases, "surrogate_backend")))
    freegs_modes = sorted(set(_case_backends(freegs_cases, "mode")))

    torax_pass = (
        torax_rate <= float(torax_cfg.get("max_fallback_rate", 0.0))
        and set(torax_backends).issubset(torax_allowed)
        and (all(bool(c.get("passes", False)) for c in torax_cases) if bool(torax_cfg.get("require_all_cases_pass", True)) else True)
    )
    sparc_pass = (
        sparc_rate <= float(sparc_cfg.get("max_fallback_rate", 1.0))
        and set(sparc_backends).issubset(sparc_allowed)
        and (all(bool(c.get("passes", False)) for c in sparc_cases) if bool(sparc_cfg.get("require_all_cases_pass", True)) else True)
    )
    freegs_pass = (
        set(freegs_modes).issubset(freegs_allowed_modes)
        and (all(bool(c.get("passes", False)) for c in freegs_cases) if bool(freegs_cfg.get("require_all_cases_pass", True)) else True)
    )

    return {
        "torax": {
            "preferred_backend": torax_preferred,
            "observed_backends": torax_backends,
            "fallback_rate": torax_rate,
            "max_fallback_rate": float(torax_cfg.get("max_fallback_rate", 0.0)),
            "passes": bool(torax_pass),
        },
        "sparc": {
            "preferred_backend": sparc_preferred,
            "observed_backends": sparc_backends,
            "fallback_rate": sparc_rate,
            "max_fallback_rate": float(sparc_cfg.get("max_fallback_rate", 1.0)),
            "passes": bool(sparc_pass),
        },
        "freegs": {
            "observed_modes": freegs_modes,
            "allowed_modes": sorted(freegs_allowed_modes),
            "passes": bool(freegs_pass),
        },
        "overall_pass": bool(torax_pass and sparc_pass and freegs_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torax", default=str(DEFAULT_TORAX))
    parser.add_argument("--sparc", default=str(DEFAULT_SPARC))
    parser.add_argument("--freegs", default=str(DEFAULT_FREEGS))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args(argv)

    torax_path = _resolve(args.torax)
    sparc_path = _resolve(args.sparc)
    freegs_path = _resolve(args.freegs)
    thresholds_path = _resolve(args.thresholds)
    summary_path = _resolve(args.summary_json)

    summary = evaluate(
        torax=_load_json(torax_path),
        sparc=_load_json(sparc_path),
        freegs=_load_json(freegs_path),
        thresholds=_load_json(thresholds_path),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Fallback budget summary: "
        f"torax={summary['torax']['fallback_rate']:.3f} "
        f"sparc={summary['sparc']['fallback_rate']:.3f} "
        f"freegs_modes={summary['freegs']['observed_modes']}"
    )
    if not bool(summary["overall_pass"]):
        print("Fallback budget guard failed.")
        return 1
    print("Fallback budget guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
