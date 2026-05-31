#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Prepare and, when possible, execute a public DREAM reference case.

This harness uses the cached public DREAM source tree to generate the upstream
``examples/2kinetic`` settings deck. If a compiled ``dreami`` backend is
available locally, it can execute the deck and record the output. If not, the
report is fail-closed with the exact missing backend requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
# DREAM backend execution requires subprocess with fixed argv, shell disabled, and timeouts.
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DREAM_REPO = ROOT / "data" / "external" / "full_fidelity_public_sources" / "repos" / "dream"
DREAM_EXAMPLE = DREAM_REPO / "examples" / "2kinetic"
DREAM_SETTINGS = DREAM_EXAMPLE / "dream_settings.h5"
DREAM_OUTPUT = DREAM_EXAMPLE / "output.h5"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "dream_reference_execution_request.json"
MD_REPORT = REPORT_DIR / "dream_reference_execution_request.md"
DREAM_OUTPUT_CONTRACT = {
    "coordinate_axes": ["time_s", "radius_m", "momentum_mec", "pitch_cosine"],
    "observables": [
        "f_p_xi_t",
        "runaway_current_t",
        "avalanche_growth_rate_t",
        "synchrotron_loss_power_t",
        "partial_screening_drag_t",
        "bremsstrahlung_loss_power_t",
    ],
    "schema": "dream-output-contract.v1",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    # Fixed argv, shell disabled, bounded timeout.
    result = subprocess.run(  # nosec B603
        args,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def _git_commit(repo: Path) -> str | None:
    try:
        return _capture(["git", "-C", str(repo), "rev-parse", "HEAD"], cwd=ROOT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _dreami_path() -> Path | None:
    local = DREAM_REPO / "build" / "iface" / "dreami"
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which("dreami")
    return Path(found) if found else None


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    dream_py = str(DREAM_REPO / "py")
    env["PYTHONPATH"] = (
        dream_py if not env.get("PYTHONPATH") else dream_py + os.pathsep + str(env["PYTHONPATH"])
    )
    return env


def _generate_settings() -> dict[str, Any]:
    if not (DREAM_EXAMPLE / "generate.py").exists():
        fallback = _tracked_settings_generation()
        if fallback is not None:
            return fallback
        return {
            "generated": False,
            "reason": "DREAM examples/2kinetic/generate.py is missing",
        }
    try:
        _capture([sys.executable, "generate.py"], cwd=DREAM_EXAMPLE, env=_python_env())
    except subprocess.CalledProcessError as exc:
        return {
            "generated": False,
            "reason": f"settings generation failed: {exc.stderr.strip() or exc}",
        }
    except subprocess.TimeoutExpired:
        return {
            "generated": False,
            "reason": "settings generation timed out",
        }
    if not DREAM_SETTINGS.exists():
        return {
            "generated": False,
            "reason": "settings generation completed without dream_settings.h5",
        }
    return {
        "generated": True,
        "mode": "external_cache_generation",
        "path": _rel(DREAM_SETTINGS),
        "sha256": _sha256(DREAM_SETTINGS),
    }


def _tracked_settings_generation() -> dict[str, Any] | None:
    if not JSON_REPORT.exists():
        return None
    try:
        tracked = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if tracked.get("schema") != "dream-reference-execution-request.v1":
        return None
    if not tracked.get("settings_deck_generated") or not tracked.get("settings_deck_sha256"):
        return None
    return {
        "generated": True,
        "mode": "tracked_report_fallback",
        "path": tracked.get("settings_deck_path"),
        "reason": (
            "DREAM source cache is absent; retaining the committed settings-deck evidence "
            "instead of rewriting it as missing."
        ),
        "sha256": tracked["settings_deck_sha256"],
        "source_commit": tracked.get("source_commit"),
    }


def _run_backend(dreami: Path | None, *, execute: bool) -> dict[str, Any]:
    if dreami is None:
        return {
            "attempted": False,
            "output_ready": False,
            "reason": "compiled DREAM backend dreami is not available",
        }
    if not execute:
        return {
            "attempted": False,
            "dreami_path": str(dreami),
            "output_ready": DREAM_OUTPUT.exists(),
            "reason": "backend execution disabled",
        }
    try:
        _capture([str(dreami), str(DREAM_SETTINGS.name)], cwd=DREAM_EXAMPLE)
    except subprocess.CalledProcessError as exc:
        return {
            "attempted": True,
            "dreami_path": str(dreami),
            "output_ready": False,
            "reason": f"dreami execution failed: {exc.stderr.strip() or exc}",
        }
    except subprocess.TimeoutExpired:
        return {
            "attempted": True,
            "dreami_path": str(dreami),
            "output_ready": False,
            "reason": "dreami execution timed out",
        }
    return {
        "attempted": True,
        "dreami_path": str(dreami),
        "output_path": _rel(DREAM_OUTPUT) if DREAM_OUTPUT.exists() else None,
        "output_sha256": _sha256(DREAM_OUTPUT) if DREAM_OUTPUT.exists() else None,
        "output_ready": DREAM_OUTPUT.exists(),
    }


def build_dream_reference_execution_report(
    *, write: bool = True, execute_backend: bool = True
) -> dict[str, Any]:
    """Generate a DREAM settings deck and return execution readiness."""
    commit = _git_commit(DREAM_REPO)
    settings = _generate_settings()
    dreami = _dreami_path()
    backend = _run_backend(dreami, execute=execute_backend and bool(settings["generated"]))
    output_ready = bool(backend["output_ready"])
    dreami_available = dreami is not None
    if output_ready:
        status = "reference_output_generated_not_converted"
        next_action = (
            "Convert DREAM output.h5 into required full-fidelity observables and compare native "
            "same-case output before updating the acceptance manifest."
        )
    elif dreami_available:
        status = "blocked_dream_backend_execution_failed"
        next_action = str(backend.get("reason", "inspect DREAM backend execution failure"))
    else:
        status = "blocked_missing_dream_backend"
        next_action = (
            "Install/build PETSc and compile DREAM with iface/dreami, then rerun this harness."
        )

    report = {
        "accepted_full_fidelity_ready": False,
        "case_id": "dream_2kinetic_public_reference_request",
        "comparison_status": (
            "blocked_reference_output_not_converted"
            if output_ready
            else "blocked_missing_reference_output"
        ),
        "description": (
            "Public DREAM 2kinetic reference execution request. Settings generation uses the "
            "external source cache when present and otherwise preserves tracked deck evidence; "
            "full reference output requires a compiled DREAM backend."
        ),
        "example_script": _rel(DREAM_EXAMPLE / "generate.py"),
        "next_action": next_action,
        "reference_output": backend,
        "reference_output_ready": output_ready,
        "required_backend": {
            "dreami_available": dreami_available,
            "dreami_path": str(dreami) if dreami else None,
            "petsc_arch": os.environ.get("PETSC_ARCH"),
            "petsc_dir": os.environ.get("PETSC_DIR"),
            "required": "DREAM iface/dreami compiled with PETSc, HDF5, and GSL",
        },
        "required_output_contract": DREAM_OUTPUT_CONTRACT,
        "schema": "dream-reference-execution-request.v1",
        "settings_deck_generated": bool(settings["generated"]),
        "settings_deck_path": settings.get("path"),
        "settings_deck_sha256": settings.get("sha256"),
        "settings_generation": settings,
        "source_cache_available": (DREAM_EXAMPLE / "generate.py").exists(),
        "source_commit": commit or settings.get("source_commit"),
        "source_family": "DREAM",
        "source_repo": _rel(DREAM_REPO),
        "same_case_comparison_ready": False,
        "status": status,
    }
    if write:
        write_reports(report)
    return report


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown DREAM execution reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# DREAM Reference Execution Request",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Source commit: `{report['source_commit']}`",
        f"- Settings deck generated: `{report['settings_deck_generated']}`",
        f"- Settings deck: `{report['settings_deck_path']}`",
        f"- Settings SHA-256: `{report['settings_deck_sha256']}`",
        f"- DREAM backend available: `{report['required_backend']['dreami_available']}`",
        f"- Reference output ready: `{report['reference_output_ready']}`",
        f"- Same-case comparison ready: `{report['same_case_comparison_ready']}`",
        f"- Comparison status: `{report['comparison_status']}`",
        f"- Accepted full-fidelity ready: `{report['accepted_full_fidelity_ready']}`",
        "",
        "## Required DREAM output contract",
        "",
        f"- Schema: `{report['required_output_contract']['schema']}`",
        f"- Coordinate axes: `{', '.join(report['required_output_contract']['coordinate_axes'])}`",
        f"- Observables: `{', '.join(report['required_output_contract']['observables'])}`",
        "",
        "## Next action",
        "",
        report["next_action"],
        "",
    ]
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute-backend",
        action="store_true",
        help="Only generate the settings deck and backend-readiness report.",
    )
    args = parser.parse_args(argv)
    report = build_dream_reference_execution_report(execute_backend=not args.no_execute_backend)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
