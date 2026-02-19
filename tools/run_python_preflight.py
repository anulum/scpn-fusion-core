from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_checks(
    *,
    skip_version_metadata: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if not skip_version_metadata:
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


def _run_check(name: str, cmd: list[str]) -> int:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[preflight] {name}: {rendered}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fast local/CI Python preflight checks "
            "(version metadata, Golden notebook gate, threshold smokes, mypy strict)."
        )
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
    args = parser.parse_args(argv)

    checks = _build_checks(
        skip_version_metadata=args.skip_version_metadata,
        skip_notebook_quality=args.skip_notebook_quality,
        skip_threshold_smoke=args.skip_threshold_smoke,
        skip_mypy=args.skip_mypy,
    )
    if not checks:
        print("[preflight] No checks selected.")
        return 0

    for name, cmd in checks:
        rc = _run_check(name, cmd)
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
