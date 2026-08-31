# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FAIR-MAST magnetic diagnostic qualification verification
"""Reproduce magnetic diagnostic qualification from the complete real archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import requests

from scpn_fusion.io import (
    MastCompleteMagneticArchiveEnvelope,
    MastMagneticDiagnosticQualification,
    acquire_mast_complete_magnetic_archive,
    build_mast_complete_magnetic_archive_envelope,
    build_mast_magnetic_diagnostic_qualification,
    decode_mast_complete_magnetic_archive_envelope,
    decode_mast_magnetic_diagnostic_qualification,
    verify_mast_complete_magnetic_archive_source,
)

_DEFAULT_PROVENANCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_PROVENANCE.json"
)
_DEFAULT_ENVELOPE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json"
)
_DEFAULT_QUALIFICATION = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
)


def verify_magnetic_diagnostic_qualification(
    provenance_path: Path,
    expected_envelope_path: Path,
    expected_qualification_path: Path,
    *,
    archive_root: Path | None,
    mapping_path: Path | None,
    attempts: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Reproduce the complete archive and every qualification fact."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    expected_archive = decode_mast_complete_magnetic_archive_envelope(
        expected_envelope_path.read_bytes()
    )
    expected_qualification = decode_mast_magnetic_diagnostic_qualification(
        expected_qualification_path.read_bytes()
    )
    with ExitStack() as stack:
        if archive_root is None:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="scpn-mast-")))
            actual_archive = acquire_mast_complete_magnetic_archive(
                provenance_path,
                directory,
                attempts=attempts,
                timeout_seconds=timeout_seconds,
            )
            materialised_root = directory / f"{actual_archive.payload['shot_id']}.zarr"
        else:
            materialised_root = archive_root
            actual_archive = build_mast_complete_magnetic_archive_envelope(
                provenance_path, materialised_root
            )
        if actual_archive.to_bytes() != expected_archive.to_bytes():
            raise RuntimeError("authentic source does not reproduce the tracked archive envelope")
        verify_mast_complete_magnetic_archive_source(actual_archive, materialised_root)

        if mapping_path is None:
            mapping_directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="scpn-mast-mapping-"))
            )
            materialised_mapping = mapping_directory / "mast.yml"
            _download_expected_mapping(
                expected_qualification,
                materialised_mapping,
                attempts=attempts,
                timeout_seconds=timeout_seconds,
            )
        else:
            materialised_mapping = mapping_path
        actual_qualification = build_mast_magnetic_diagnostic_qualification(
            actual_archive,
            materialised_root,
            materialised_mapping,
        )
        if actual_qualification.to_bytes() != expected_qualification.to_bytes():
            raise RuntimeError(
                "authentic archive and mapping do not reproduce tracked qualification"
            )
        return _verification_report(actual_archive, actual_qualification)


def _download_expected_mapping(
    expected: MastMagneticDiagnosticQualification,
    destination: Path,
    *,
    attempts: int,
    timeout_seconds: float,
) -> None:
    mapping = cast(dict[str, object], expected.payload["ingestion_mapping"])
    url = cast(str, mapping["mapping_url"])
    expected_sha256 = cast(str, mapping["mapping_sha256"])
    failure: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            content = response.content
            if hashlib.sha256(content).hexdigest() != expected_sha256:
                raise RuntimeError("downloaded ingestion mapping SHA-256 differs")
            destination.write_bytes(content)
            return
        except (requests.RequestException, RuntimeError) as exc:
            failure = exc
            if attempt < attempts:
                time.sleep(min(float(attempt), 5.0))
    raise RuntimeError("pinned ingestion mapping download failed") from failure


def _verification_report(
    archive: MastCompleteMagneticArchiveEnvelope,
    qualification: MastMagneticDiagnosticQualification,
) -> dict[str, object]:
    payload = qualification.payload
    completeness = cast(dict[str, object], payload["completeness"])
    summary = cast(dict[str, object], payload["qualification_summary"])
    return {
        "archive_array_count": completeness["archive_array_count"],
        "archive_envelope_sha256": archive.sha256,
        "channel_record_count": completeness["channel_record_count"],
        "clock_count": completeness["clock_count"],
        "measurement_count": completeness["measurement_count"],
        "qualification_sha256": qualification.sha256,
        "qualification_summary": summary,
        "shot_id": payload["shot_id"],
        "status": "pass",
    }


def main() -> int:
    """Parse arguments and run or regenerate the qualification proof."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, default=_DEFAULT_PROVENANCE)
    parser.add_argument("--expected-envelope", type=Path, default=_DEFAULT_ENVELOPE)
    parser.add_argument("--expected-qualification", type=Path, default=_DEFAULT_QUALIFICATION)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--mapping-path", type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--regenerate-expected", action="store_true")
    arguments = parser.parse_args()
    if arguments.regenerate_expected:
        if arguments.archive_root is None or arguments.mapping_path is None:
            parser.error("--regenerate-expected requires --archive-root and --mapping-path")
        archive = build_mast_complete_magnetic_archive_envelope(
            arguments.provenance, arguments.archive_root
        )
        expected_archive = decode_mast_complete_magnetic_archive_envelope(
            arguments.expected_envelope.read_bytes()
        )
        if archive.to_bytes() != expected_archive.to_bytes():
            raise RuntimeError("local archive does not reproduce the tracked archive envelope")
        qualification = build_mast_magnetic_diagnostic_qualification(
            archive, arguments.archive_root, arguments.mapping_path
        )
        arguments.expected_qualification.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.expected_qualification.with_suffix(".json.tmp")
        temporary.write_bytes(qualification.to_bytes())
        temporary.replace(arguments.expected_qualification)
        print(
            json.dumps(
                {
                    "output": str(arguments.expected_qualification),
                    "qualification_sha256": qualification.sha256,
                    "status": "generated",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = verify_magnetic_diagnostic_qualification(
        arguments.provenance,
        arguments.expected_envelope,
        arguments.expected_qualification,
        archive_root=arguments.archive_root,
        mapping_path=arguments.mapping_path,
        attempts=arguments.attempts,
        timeout_seconds=arguments.timeout_seconds,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.report.with_suffix(arguments.report.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(arguments.report)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
