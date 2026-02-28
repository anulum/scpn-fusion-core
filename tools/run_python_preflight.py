from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECK_TIMEOUT_SECONDS = 1800.0


def _normalize_check_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("check_timeout_seconds must be finite and > 0.")
    return timeout


def _build_release_checks(
    *,
    skip_version_metadata: bool,
    skip_claims_audit: bool,
    skip_claims_map: bool,
    skip_release_checklist: bool,
    skip_shot_manifest: bool,
    skip_shot_splits: bool,
    skip_disruption_calibration: bool,
    skip_disruption_replay_pipeline: bool,
    skip_eped_domain_contract: bool,
    skip_transport_uncertainty: bool,
    skip_torax_strict_backend: bool,
    skip_sparc_strict_backend: bool,
    skip_multi_ion_conservation: bool,
    skip_end_to_end_latency: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if not skip_version_metadata:
        # Hardening: Run metadata sync first to ensure consistency
        checks.append(
            (
                "Metadata auto-synchronization",
                [
                    sys.executable,
                    "tools/sync_metadata.py",
                ],
            )
        )
        checks.append(
            (
                "Version metadata consistency",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_version_metadata.py",
                    "-q",
                ],
            )
        )
    if not skip_claims_audit:
        checks.append(
            (
                "Claims evidence audit",
                [
                    sys.executable,
                    "tools/claims_audit.py",
                ],
            )
        )
    if not skip_claims_map:
        checks.append(
            (
                "Claims evidence map drift check",
                [
                    sys.executable,
                    "tools/generate_claims_evidence_map.py",
                    "--check",
                ],
            )
        )
    if not skip_release_checklist:
        checks.append(
            (
                "Release acceptance checklist gate",
                [
                    sys.executable,
                    "tools/check_release_acceptance.py",
                ],
            )
        )
    if not skip_shot_manifest:
        checks.append(
            (
                "Disruption shot provenance manifest check",
                [
                    sys.executable,
                    "tools/generate_disruption_shot_manifest.py",
                    "--check",
                ],
            )
        )
    if not skip_shot_splits:
        checks.append(
            (
                "Disruption shot split leakage check",
                [
                    sys.executable,
                    "tools/check_disruption_shot_splits.py",
                ],
            )
        )
    if not skip_disruption_calibration:
        checks.append(
            (
                "Disruption risk calibration holdout check",
                [
                    sys.executable,
                    "tools/generate_disruption_risk_calibration.py",
                    "--check",
                ],
            )
        )
    if not skip_disruption_replay_pipeline:
        checks.append(
            (
                "Disruption replay pipeline contract benchmark",
                [
                    sys.executable,
                    "validation/benchmark_disruption_replay_pipeline.py",
                    "--strict",
                ],
            )
        )
    if not skip_eped_domain_contract:
        checks.append(
            (
                "EPED domain contract benchmark",
                [
                    sys.executable,
                    "validation/benchmark_eped_domain_contract.py",
                    "--strict",
                ],
            )
        )
    if not skip_transport_uncertainty:
        checks.append(
            (
                "Transport uncertainty envelope benchmark",
                [
                    sys.executable,
                    "validation/benchmark_transport_uncertainty_envelope.py",
                    "--strict",
                ],
            )
        )
    if not skip_torax_strict_backend:
        checks.append(
            (
                "TORAX strict-backend benchmark",
                [
                    sys.executable,
                    "validation/benchmark_vs_torax.py",
                    "--strict-backend",
                ],
            )
        )
    if not skip_sparc_strict_backend:
        checks.append(
            (
                "SPARC GEQDSK strict-backend benchmark",
                [
                    sys.executable,
                    "validation/benchmark_sparc_geqdsk_rmse.py",
                    "--strict-backend",
                ],
            )
        )
    if not skip_multi_ion_conservation:
        checks.append(
            (
                "Multi-ion transport conservation benchmark",
                [
                    sys.executable,
                    "validation/benchmark_multi_ion_transport_conservation.py",
                    "--strict",
                ],
            )
        )
    if not skip_end_to_end_latency:
        checks.append(
            (
                "SCPN end-to-end latency benchmark",
                [
                    sys.executable,
                    "validation/scpn_end_to_end_latency.py",
                    "--strict",
                ],
            )
        )
    if not skip_notebook_quality:
        checks.append(
            (
                "Golden notebook quality gate",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_neuro_symbolic_control_demo_notebook.py",
                    "-q",
                ],
            )
        )
    if not skip_threshold_smoke:
        checks.append(
            (
                "Task 5/6 threshold smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_task5_disruption_mitigation_integration.py::test_task5_campaign_passes_thresholds_smoke",
                    "tests/test_task6_heating_neutronics_realism.py::test_task6_campaign_passes_thresholds_smoke",
                    "-q",
                ],
            )
        )
    if not skip_mypy:
        checks.append(("mypy strict", [sys.executable, "tools/run_mypy_strict.py"]))
    return checks


def _build_research_checks(*, skip_research_suite: bool) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if not skip_research_suite:
        checks.append(
            (
                "Experimental-only pytest suite",
                [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"],
            )
        )
    return checks


def _build_checks(
    *,
    gate: str,
    skip_version_metadata: bool,
    skip_claims_audit: bool,
    skip_claims_map: bool,
    skip_release_checklist: bool,
    skip_shot_manifest: bool,
    skip_shot_splits: bool,
    skip_disruption_calibration: bool,
    skip_disruption_replay_pipeline: bool,
    skip_eped_domain_contract: bool,
    skip_transport_uncertainty: bool,
    skip_torax_strict_backend: bool,
    skip_sparc_strict_backend: bool,
    skip_multi_ion_conservation: bool,
    skip_end_to_end_latency: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
    skip_research_suite: bool,
) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if gate in {"release", "all"}:
        checks.extend(
            _build_release_checks(
                skip_version_metadata=skip_version_metadata,
                skip_claims_audit=skip_claims_audit,
                skip_claims_map=skip_claims_map,
                skip_release_checklist=skip_release_checklist,
                skip_shot_manifest=skip_shot_manifest,
                skip_shot_splits=skip_shot_splits,
                skip_disruption_calibration=skip_disruption_calibration,
                skip_disruption_replay_pipeline=skip_disruption_replay_pipeline,
                skip_eped_domain_contract=skip_eped_domain_contract,
                skip_transport_uncertainty=skip_transport_uncertainty,
                skip_torax_strict_backend=skip_torax_strict_backend,
                skip_sparc_strict_backend=skip_sparc_strict_backend,
                skip_multi_ion_conservation=skip_multi_ion_conservation,
                skip_end_to_end_latency=skip_end_to_end_latency,
                skip_notebook_quality=skip_notebook_quality,
                skip_threshold_smoke=skip_threshold_smoke,
                skip_mypy=skip_mypy,
            )
        )
    if gate in {"research", "all"}:
        checks.extend(
            _build_research_checks(skip_research_suite=skip_research_suite)
        )
    return checks


def _run_check(name: str, cmd: list[str], *, timeout_seconds: float) -> int:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[preflight] {name}: {rendered}")
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            (
                f"[preflight] TIMEOUT at '{name}' "
                f"after {timeout_seconds:.1f}s."
            ),
            file=sys.stderr,
        )
        return 124
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run local/CI Python preflight checks with gate profiles "
            "(release, research, or both)."
        )
    )
    parser.add_argument(
        "--gate",
        choices=("release", "research", "all"),
        default="release",
        help=(
            "Gate profile to run. "
            "'release' excludes experimental-only lanes, "
            "'research' runs experimental-only lanes, "
            "'all' runs both."
        ),
    )
    parser.add_argument(
        "--skip-version-metadata",
        action="store_true",
        help="Skip tests/test_version_metadata.py",
    )
    parser.add_argument(
        "--skip-notebook-quality",
        action="store_true",
        help="Skip tests/test_neuro_symbolic_control_demo_notebook.py",
    )
    parser.add_argument(
        "--skip-claims-audit",
        action="store_true",
        help="Skip tools/claims_audit.py",
    )
    parser.add_argument(
        "--skip-claims-map",
        action="store_true",
        help="Skip tools/generate_claims_evidence_map.py --check",
    )
    parser.add_argument(
        "--skip-release-checklist",
        action="store_true",
        help="Skip tools/check_release_acceptance.py",
    )
    parser.add_argument(
        "--skip-shot-manifest",
        action="store_true",
        help="Skip tools/generate_disruption_shot_manifest.py --check",
    )
    parser.add_argument(
        "--skip-shot-splits",
        action="store_true",
        help="Skip tools/check_disruption_shot_splits.py",
    )
    parser.add_argument(
        "--skip-disruption-calibration",
        action="store_true",
        help="Skip tools/generate_disruption_risk_calibration.py --check",
    )
    parser.add_argument(
        "--skip-disruption-replay-pipeline",
        action="store_true",
        help="Skip validation/benchmark_disruption_replay_pipeline.py --strict",
    )
    parser.add_argument(
        "--skip-eped-domain-contract",
        action="store_true",
        help="Skip validation/benchmark_eped_domain_contract.py --strict",
    )
    parser.add_argument(
        "--skip-transport-uncertainty",
        action="store_true",
        help="Skip validation/benchmark_transport_uncertainty_envelope.py --strict",
    )
    parser.add_argument(
        "--skip-torax-strict-backend",
        action="store_true",
        help="Skip validation/benchmark_vs_torax.py --strict-backend",
    )
    parser.add_argument(
        "--skip-sparc-strict-backend",
        action="store_true",
        help="Skip validation/benchmark_sparc_geqdsk_rmse.py --strict-backend",
    )
    parser.add_argument(
        "--skip-multi-ion-conservation",
        action="store_true",
        help="Skip validation/benchmark_multi_ion_transport_conservation.py --strict",
    )
    parser.add_argument(
        "--skip-end-to-end-latency",
        action="store_true",
        help="Skip validation/scpn_end_to_end_latency.py --strict",
    )
    parser.add_argument(
        "--skip-threshold-smoke",
        action="store_true",
        help=(
            "Skip Task 5/6 threshold smoke tests "
            "(tests/test_task5_disruption_mitigation_integration.py and "
            "tests/test_task6_heating_neutronics_realism.py)."
        ),
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip tools/run_mypy_strict.py",
    )
    parser.add_argument(
        "--skip-research-suite",
        action="store_true",
        help="Skip pytest experimental-only lane (tests/ -m experimental).",
    )
    parser.add_argument(
        "--check-timeout-seconds",
        type=float,
        default=DEFAULT_CHECK_TIMEOUT_SECONDS,
        help="Per-check subprocess timeout in seconds.",
    )
    args = parser.parse_args(argv)
    try:
        check_timeout_seconds = _normalize_check_timeout_seconds(
            args.check_timeout_seconds
        )
    except ValueError as exc:
        parser.error(str(exc))

    checks = _build_checks(
        gate=args.gate,
        skip_version_metadata=args.skip_version_metadata,
        skip_claims_audit=args.skip_claims_audit,
        skip_claims_map=args.skip_claims_map,
        skip_release_checklist=args.skip_release_checklist,
        skip_shot_manifest=args.skip_shot_manifest,
        skip_shot_splits=args.skip_shot_splits,
        skip_disruption_calibration=args.skip_disruption_calibration,
        skip_disruption_replay_pipeline=args.skip_disruption_replay_pipeline,
        skip_eped_domain_contract=args.skip_eped_domain_contract,
        skip_transport_uncertainty=args.skip_transport_uncertainty,
        skip_torax_strict_backend=args.skip_torax_strict_backend,
        skip_sparc_strict_backend=args.skip_sparc_strict_backend,
        skip_multi_ion_conservation=args.skip_multi_ion_conservation,
        skip_end_to_end_latency=args.skip_end_to_end_latency,
        skip_notebook_quality=args.skip_notebook_quality,
        skip_threshold_smoke=args.skip_threshold_smoke,
        skip_mypy=args.skip_mypy,
        skip_research_suite=args.skip_research_suite,
    )
    if not checks:
        print("[preflight] No checks selected.")
        return 0

    for name, cmd in checks:
        rc = _run_check(name, cmd, timeout_seconds=check_timeout_seconds)
        if rc != 0:
            print(
                f"[preflight] FAILED at '{name}' with exit code {rc}.",
                file=sys.stderr,
            )
            return rc

    print("[preflight] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
