"""Regression tests for external FNO retrain workflow tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_fno_external_retrain_request",
    ROOT / "tools" / "export_fno_external_retrain_request.py",
)
assert EXPORT_SPEC and EXPORT_SPEC.loader
export_fno_external_retrain_request = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(export_fno_external_retrain_request)

IMPORT_SPEC = importlib.util.spec_from_file_location(
    "import_external_fno_weights",
    ROOT / "tools" / "import_external_fno_weights.py",
)
assert IMPORT_SPEC and IMPORT_SPEC.loader
import_external_fno_weights = importlib.util.module_from_spec(IMPORT_SPEC)
IMPORT_SPEC.loader.exec_module(import_external_fno_weights)


def test_export_payload_allows_missing_report_context() -> None:
    payload = export_fno_external_retrain_request.build_request_payload(
        {},
        report_found=False,
    )
    assert payload["request"]["training_route"] == "external-service"
    assert payload["local_validation_context"]["validation_report_found"] is False


def test_import_manifest_validation_accepts_valid_sha(tmp_path: Path) -> None:
    weights = tmp_path / "external_weights.npz"
    weights.write_bytes(b"synthetic-weights")
    sha = import_external_fno_weights._sha256(weights)
    ok, errors = import_external_fno_weights._validate_manifest(
        {
            "schema_version": "1.0",
            "service": "external-fno-train-v1",
            "weights_sha256": sha,
            "trained_datasets": ["GENE flux-tube", "CGYRO validation subset"],
            "data_license": "Licensed for SCPN validation use",
        },
        sha,
    )
    assert ok is True
    assert errors == []
