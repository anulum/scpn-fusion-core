#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate a deterministic retrain-request payload for external FNO services."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.9/3.10 CI lanes
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_REPORT = REPO_ROOT / "validation" / "reports" / "full_validation_pipeline.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "fno_external_retrain_request.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _safe_git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _load_project_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        version = (
            data.get("project", {}).get("version")
            if isinstance(data.get("project", {}), dict)
            else None
        )
        if isinstance(version, str) and version.strip():
            return version.strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise ValueError("Unable to determine project version from pyproject.toml")


def _load_validation_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def _extract_retrain_context(report: dict[str, Any], *, report_found: bool) -> dict[str, Any]:
    controller_metrics = report.get("controller_metrics", {})
    rewrite_required = bool(report.get("rewrite_required", False))
    config = report.get("config", {})
    return {
        "validation_report_found": report_found,
        "rewrite_required": rewrite_required,
        "controller_metrics": (controller_metrics if isinstance(controller_metrics, dict) else {}),
        "campaign_config": config if isinstance(config, dict) else {},
    }


def build_request_payload(report: dict[str, Any], *, report_found: bool) -> dict[str, Any]:
    version = _load_project_version()
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "scpn-fusion",
            "version": version,
            "git_sha": _safe_git_sha(),
        },
        "request": {
            "model_family": "FNO turbulence surrogate",
            "training_route": "external-service",
            "required_training_sources": [
                "GENE flux-tube runs",
                "CGYRO flux-tube runs",
            ],
            "required_outputs": {
                "weights_npz": "weights/fno_turbulence_retrained_from_empirical.npz",
                "manifest_json": "artifacts/external_fno_retrain_manifest.json",
            },
            "quality_gates": {
                "max_eval_relative_l2_mean": 0.25,
                "must_report_error_bars": True,
                "must_include_data_license_provenance": True,
            },
        },
        "local_validation_context": _extract_retrain_context(report, report_found=report_found),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", default=str(DEFAULT_VALIDATION_REPORT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    report_path = _resolve(args.validation_report)
    out_path = _resolve(args.output_json)

    report_found = report_path.exists()
    if report_found:
        report = _load_validation_report(report_path)
    else:
        report = {}
    payload = build_request_payload(report, report_found=report_found)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "External retrain request exported: "
        f"output={out_path.as_posix()} "
        f"rewrite_required={payload['local_validation_context']['rewrite_required']} "
        f"report_found={payload['local_validation_context']['validation_report_found']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
