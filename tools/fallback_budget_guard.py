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
DEFAULT_TELEMETRY = REPO_ROOT / "artifacts" / "fallback_telemetry.json"


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


def _mean_flag(cases: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(case.get(key)) for case in cases if key in case]
    if not values:
        return None
    return float(sum(1 for v in values if v) / len(values))


def _rate_for_backend(
    cases: list[dict[str, Any]],
    *,
    backend_key: str,
    preferred_backend: str,
) -> float:
    if not cases:
        return 0.0
    preferred_hits = sum(
        1 for case in cases if str(case.get(backend_key, "unknown")) == preferred_backend
    )
    return float(preferred_hits / len(cases))


def evaluate(
    *,
    torax: dict[str, Any],
    sparc: dict[str, Any],
    freegs: dict[str, Any],
    thresholds: dict[str, Any],
    runtime_telemetry: dict[str, Any] | None = None,
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
    torax_strict_requested = bool(torax.get("require_neural_transport", False))
    sparc_strict_requested = bool(sparc.get("require_neural_backend", False))
    freegs_strict_requested = bool(freegs.get("require_freegs_backend", False))

    torax_allowed = {str(v) for v in torax_cfg.get("allowed_backends", [torax_preferred])}
    sparc_allowed = {str(v) for v in sparc_cfg.get("allowed_backends", [sparc_preferred])}
    freegs_allowed_modes = {
        str(v) for v in freegs_cfg.get("allowed_modes", ["solovev_manufactured_source", "freegs"])
    }
    require_freegs_mode_when_available = bool(
        freegs_cfg.get("require_freegs_mode_when_available", False)
    )
    min_freegs_cases_when_available = int(freegs_cfg.get("min_freegs_cases_when_available", 1))
    freegs_available = bool(freegs.get("freegs_available", False))
    force_solovev = bool(freegs.get("force_solovev", False))

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
    torax_preferred_rate = _rate_for_backend(
        torax_cases,
        backend_key="transport_backend",
        preferred_backend=torax_preferred,
    )
    sparc_preferred_rate = _rate_for_backend(
        sparc_cases,
        backend_key="surrogate_backend",
        preferred_backend=sparc_preferred,
    )
    torax_backend_requirement_rate = _mean_flag(torax_cases, "backend_requirement_satisfied")
    sparc_backend_requirement_rate = _mean_flag(sparc_cases, "backend_requirement_satisfied")
    freegs_modes = sorted(set(_case_backends(freegs_cases, "mode")))
    freegs_reference_backends = sorted(set(_case_backends(freegs_cases, "reference_backend")))
    freegs_case_count = sum(1 for case in freegs_cases if str(case.get("mode", "")) == "freegs")
    torax_min_cases = int(torax_cfg.get("min_case_count", 0))
    sparc_min_cases = int(sparc_cfg.get("min_case_count", 0))
    freegs_min_cases = int(freegs_cfg.get("min_case_count", 0))
    torax_min_preferred_rate = torax_cfg.get("min_preferred_backend_rate")
    sparc_min_preferred_rate = sparc_cfg.get("min_preferred_backend_rate")
    torax_require_backend_requirement = bool(
        torax_cfg.get("require_backend_requirement_satisfied", False)
    )
    sparc_require_backend_requirement = bool(
        sparc_cfg.get("require_backend_requirement_satisfied", False)
    )
    freegs_required_reference_backend = str(
        freegs_cfg.get("require_reference_backend_when_available", "")
    ).strip()
    torax_max_fallback_rate = float(torax_cfg.get("max_fallback_rate", 0.0))
    sparc_max_fallback_rate = float(sparc_cfg.get("max_fallback_rate", 1.0))

    if torax_strict_requested:
        torax_allowed = {torax_preferred}
        torax_min_preferred_rate = 1.0
        torax_require_backend_requirement = True
        torax_max_fallback_rate = 0.0
    if sparc_strict_requested:
        sparc_allowed = {sparc_preferred}
        sparc_min_preferred_rate = 1.0
        sparc_require_backend_requirement = True
        sparc_max_fallback_rate = 0.0
    if freegs_strict_requested:
        require_freegs_mode_when_available = True
        min_freegs_cases_when_available = max(min_freegs_cases_when_available, 1)

    torax_pass = (
        len(torax_cases) >= torax_min_cases
        and torax_rate <= torax_max_fallback_rate
        and set(torax_backends).issubset(torax_allowed)
        and (
            True
            if torax_min_preferred_rate is None
            else torax_preferred_rate >= float(torax_min_preferred_rate)
        )
        and (
            True
            if (not torax_require_backend_requirement or torax_backend_requirement_rate is None)
            else torax_backend_requirement_rate >= 1.0
        )
        and (
            all(bool(c.get("passes", False)) for c in torax_cases)
            if bool(torax_cfg.get("require_all_cases_pass", True))
            else True
        )
    )
    sparc_pass = (
        len(sparc_cases) >= sparc_min_cases
        and sparc_rate <= sparc_max_fallback_rate
        and set(sparc_backends).issubset(sparc_allowed)
        and (
            True
            if sparc_min_preferred_rate is None
            else sparc_preferred_rate >= float(sparc_min_preferred_rate)
        )
        and (
            True
            if (not sparc_require_backend_requirement or sparc_backend_requirement_rate is None)
            else sparc_backend_requirement_rate >= 1.0
        )
        and (
            all(bool(c.get("passes", False)) for c in sparc_cases)
            if bool(sparc_cfg.get("require_all_cases_pass", True))
            else True
        )
    )
    freegs_pass = (
        len(freegs_cases) >= freegs_min_cases
        and set(freegs_modes).issubset(freegs_allowed_modes)
        and (
            all(bool(c.get("passes", False)) for c in freegs_cases)
            if bool(freegs_cfg.get("require_all_cases_pass", True))
            else True
        )
        and (
            True
            if not (freegs_required_reference_backend and freegs_available and not force_solovev)
            else (freegs_reference_backends == [freegs_required_reference_backend])
        )
        and (
            True
            if (not require_freegs_mode_when_available)
            else (
                (not freegs_available)
                or force_solovev
                or (freegs_case_count >= min_freegs_cases_when_available)
            )
        )
    )

    runtime_cfg = dict(thresholds.get("runtime", {}))
    runtime_payload = dict(runtime_telemetry or {})
    runtime_total = int(runtime_payload.get("total_count", 0) or 0)
    raw_domain_counts = runtime_payload.get("domain_counts", {})
    if not isinstance(raw_domain_counts, dict):
        raw_domain_counts = {}
    runtime_domain_counts = {
        str(k): int(v)
        for k, v in raw_domain_counts.items()
        if isinstance(k, str) and isinstance(v, (int, float))
    }
    runtime_max_total_raw = runtime_cfg.get("max_total_events")
    runtime_max_total = None if runtime_max_total_raw is None else int(runtime_max_total_raw)
    raw_runtime_domain_limits = runtime_cfg.get("max_domain_events", {})
    if not isinstance(raw_runtime_domain_limits, dict):
        raw_runtime_domain_limits = {}
    runtime_domain_limits = {
        str(k): int(v)
        for k, v in raw_runtime_domain_limits.items()
        if isinstance(k, str) and isinstance(v, (int, float))
    }
    runtime_total_ok = True if runtime_max_total is None else runtime_total <= runtime_max_total
    runtime_domain_ok = all(
        int(runtime_domain_counts.get(domain, 0)) <= int(limit)
        for domain, limit in runtime_domain_limits.items()
    )
    runtime_pass = runtime_total_ok and runtime_domain_ok

    return {
        "torax": {
            "preferred_backend": torax_preferred,
            "observed_backends": torax_backends,
            "fallback_rate": torax_rate,
            "preferred_backend_rate": torax_preferred_rate,
            "max_fallback_rate": torax_max_fallback_rate,
            "min_preferred_backend_rate": (
                None if torax_min_preferred_rate is None else float(torax_min_preferred_rate)
            ),
            "min_case_count": torax_min_cases,
            "case_count": len(torax_cases),
            "backend_requirement_satisfied_rate": torax_backend_requirement_rate,
            "require_backend_requirement_satisfied": torax_require_backend_requirement,
            "strict_requested": torax_strict_requested,
            "passes": bool(torax_pass),
        },
        "sparc": {
            "preferred_backend": sparc_preferred,
            "observed_backends": sparc_backends,
            "fallback_rate": sparc_rate,
            "preferred_backend_rate": sparc_preferred_rate,
            "max_fallback_rate": sparc_max_fallback_rate,
            "min_preferred_backend_rate": (
                None if sparc_min_preferred_rate is None else float(sparc_min_preferred_rate)
            ),
            "min_case_count": sparc_min_cases,
            "case_count": len(sparc_cases),
            "backend_requirement_satisfied_rate": sparc_backend_requirement_rate,
            "require_backend_requirement_satisfied": sparc_require_backend_requirement,
            "strict_requested": sparc_strict_requested,
            "passes": bool(sparc_pass),
        },
        "freegs": {
            "freegs_available": freegs_available,
            "force_solovev": force_solovev,
            "observed_modes": freegs_modes,
            "observed_reference_backends": freegs_reference_backends,
            "allowed_modes": sorted(freegs_allowed_modes),
            "freegs_case_count": freegs_case_count,
            "case_count": len(freegs_cases),
            "min_case_count": freegs_min_cases,
            "require_reference_backend_when_available": freegs_required_reference_backend,
            "require_freegs_mode_when_available": require_freegs_mode_when_available,
            "min_freegs_cases_when_available": min_freegs_cases_when_available,
            "strict_requested": freegs_strict_requested,
            "passes": bool(freegs_pass),
        },
        "runtime": {
            "max_total_events": runtime_max_total,
            "total_events": runtime_total,
            "total_passes": runtime_total_ok,
            "max_domain_events": runtime_domain_limits,
            "domain_counts": runtime_domain_counts,
            "domain_passes": runtime_domain_ok,
            "passes": runtime_pass,
        },
        "overall_pass": bool(torax_pass and sparc_pass and freegs_pass and runtime_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torax", default=str(DEFAULT_TORAX))
    parser.add_argument("--sparc", default=str(DEFAULT_SPARC))
    parser.add_argument("--freegs", default=str(DEFAULT_FREEGS))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--telemetry",
        default=str(DEFAULT_TELEMETRY),
        help="Optional runtime fallback telemetry snapshot JSON.",
    )
    args = parser.parse_args(argv)

    torax_path = _resolve(args.torax)
    sparc_path = _resolve(args.sparc)
    freegs_path = _resolve(args.freegs)
    thresholds_path = _resolve(args.thresholds)
    summary_path = _resolve(args.summary_json)
    telemetry_path = _resolve(args.telemetry)
    runtime_telemetry = _load_json(telemetry_path) if telemetry_path.exists() else None

    summary = evaluate(
        torax=_load_json(torax_path),
        sparc=_load_json(sparc_path),
        freegs=_load_json(freegs_path),
        thresholds=_load_json(thresholds_path),
        runtime_telemetry=runtime_telemetry,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Fallback budget summary: "
        f"torax={summary['torax']['fallback_rate']:.3f} "
        f"sparc={summary['sparc']['fallback_rate']:.3f} "
        f"freegs_modes={summary['freegs']['observed_modes']} "
        f"runtime_events={summary['runtime']['total_events']}"
    )
    if not bool(summary["overall_pass"]):
        print("Fallback budget guard failed.")
        return 1
    print("Fallback budget guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
