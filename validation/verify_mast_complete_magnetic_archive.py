# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — complete FAIR-MAST magnetic archive verification
"""Run the real complete-group FAIR-MAST acquisition and fidelity proof."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

from scpn_fusion.io import (
    MastCompleteMagneticArchiveEnvelope,
    acquire_mast_complete_magnetic_archive,
    build_mast_complete_magnetic_archive_envelope,
    decode_mast_complete_magnetic_archive_envelope,
    verify_mast_complete_magnetic_archive_source,
)

_DEFAULT_PROVENANCE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_PROVENANCE.json"
)
_DEFAULT_ENVELOPE = Path(
    "validation/reference_data/mast/disruption_shots/MAGNETIC_ARCHIVE_ENVELOPE.json"
)


def verify_complete_magnetic_archive(
    provenance_path: Path,
    expected_envelope_path: Path,
    *,
    archive_root: Path | None,
    attempts: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Verify the tracked contract against every authentic source object and array."""
    expected = decode_mast_complete_magnetic_archive_envelope(expected_envelope_path.read_bytes())
    if archive_root is None:
        with tempfile.TemporaryDirectory(prefix="scpn-mast-complete-") as directory:
            actual = acquire_mast_complete_magnetic_archive(
                provenance_path,
                Path(directory),
                attempts=attempts,
                timeout_seconds=timeout_seconds,
            )
            materialised_root = Path(directory) / f"{actual.payload['shot_id']}.zarr"
            return _verification_report(expected, actual, materialised_root)
    actual = build_mast_complete_magnetic_archive_envelope(provenance_path, archive_root)
    return _verification_report(expected, actual, archive_root)


def _verification_report(
    expected: MastCompleteMagneticArchiveEnvelope,
    actual: MastCompleteMagneticArchiveEnvelope,
    archive_root: Path,
) -> dict[str, object]:
    if actual.to_bytes() != expected.to_bytes():
        raise RuntimeError(
            "authentic source does not reproduce the tracked complete magnetic envelope"
        )
    verify_mast_complete_magnetic_archive_source(actual, archive_root)
    payload = actual.payload
    completeness = cast(dict[str, object], payload["completeness"])
    provenance = cast(dict[str, object], payload["provenance"])
    clocks = cast(list[dict[str, object]], payload["clocks"])
    return {
        "array_count": completeness["array_count"],
        "clock_count": completeness["clock_count"],
        "clocks": [clock["name"] for clock in clocks],
        "envelope_sha256": actual.sha256,
        "object_count": provenance["object_count"],
        "shot_id": payload["shot_id"],
        "source_ingestion_revision": payload["source_ingestion_revision"],
        "source_ingestion_tree_state": payload["source_ingestion_tree_state"],
        "status": "pass",
        "total_bytes": provenance["total_bytes"],
    }


def main() -> int:
    """Parse CLI arguments and run the complete-group verifier."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, default=_DEFAULT_PROVENANCE)
    parser.add_argument("--expected-envelope", type=Path, default=_DEFAULT_ENVELOPE)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--regenerate-expected",
        action="store_true",
        help="Regenerate the canonical expected envelope from --archive-root.",
    )
    arguments = parser.parse_args()
    if arguments.regenerate_expected:
        if arguments.archive_root is None:
            parser.error("--regenerate-expected requires --archive-root")
        generated = build_mast_complete_magnetic_archive_envelope(
            arguments.provenance, arguments.archive_root
        )
        arguments.expected_envelope.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.expected_envelope.with_suffix(".json.tmp")
        temporary.write_bytes(generated.to_bytes())
        temporary.replace(arguments.expected_envelope)
        print(
            json.dumps(
                {
                    "envelope_sha256": generated.sha256,
                    "output": str(arguments.expected_envelope),
                    "status": "generated",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = verify_complete_magnetic_archive(
        arguments.provenance,
        arguments.expected_envelope,
        archive_root=arguments.archive_root,
        attempts=arguments.attempts,
        timeout_seconds=arguments.timeout_seconds,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
