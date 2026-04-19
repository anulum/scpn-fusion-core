#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Import externally retrained FNO weights with checksum/provenance validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "external_fno_retrain_manifest.json"
DEFAULT_WEIGHTS = REPO_ROOT / "artifacts" / "external_fno_retrain_weights.npz"
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "fno_turbulence_retrained_from_empirical.npz"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "external_fno_retrain_import_summary.json"
REQUIRED_DATASET_TOKENS = ("gene", "cgyro")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object payload")
    return payload


def _validate_manifest(manifest: dict[str, Any], weights_sha: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        errors.append("manifest.schema_version is missing")

    service = manifest.get("service")
    if not isinstance(service, str) or not service.strip():
        errors.append("manifest.service is missing")

    declared_sha = manifest.get("weights_sha256")
    if not isinstance(declared_sha, str) or not declared_sha.strip():
        errors.append("manifest.weights_sha256 is missing")
    elif declared_sha.strip().lower() != weights_sha.lower():
        errors.append("manifest.weights_sha256 does not match provided weights file")

    datasets = manifest.get("trained_datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append("manifest.trained_datasets is missing or empty")
    else:
        lowered = " ".join(str(item).lower() for item in datasets)
        if not any(token in lowered for token in REQUIRED_DATASET_TOKENS):
            errors.append("manifest.trained_datasets must include GENE and/or CGYRO provenance")

    license_info = manifest.get("data_license")
    if not isinstance(license_info, str) or not license_info.strip():
        errors.append("manifest.data_license is missing")

    return (len(errors) == 0), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when manifest validation fails (default behavior).",
    )
    args = parser.parse_args(argv)

    manifest_path = _resolve(args.manifest)
    weights_path = _resolve(args.weights)
    output_path = _resolve(args.output)
    summary_path = _resolve(args.summary_json)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path.as_posix(),
        "weights_path": weights_path.as_posix(),
        "output_path": output_path.as_posix(),
        "overall_pass": False,
        "errors": [],
    }

    try:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        manifest = _load_json(manifest_path)
        weights_sha = _sha256(weights_path)
        ok, errors = _validate_manifest(manifest, weights_sha)
        summary["weights_sha256"] = weights_sha
        summary["errors"] = errors
        summary["manifest_service"] = manifest.get("service")
        summary["manifest_schema_version"] = manifest.get("schema_version")
        summary["manifest_trained_datasets"] = manifest.get("trained_datasets")

        if not ok:
            summary["overall_pass"] = False
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_path, output_path)
            summary["overall_pass"] = True
            summary["imported_bytes"] = int(output_path.stat().st_size)
    except Exception as exc:  # pragma: no cover - exercised in integration usage
        summary["errors"] = [str(exc)]
        summary["overall_pass"] = False

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"External FNO import: pass={summary['overall_pass']} output={output_path.as_posix()}")
    if not bool(summary["overall_pass"]):
        print("External FNO import failed; see summary JSON for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
