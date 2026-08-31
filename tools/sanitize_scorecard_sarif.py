# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — OpenSSF Scorecard SARIF Sanitizer
"""Remove upstream placeholder locations from repository-level Scorecard SARIF."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PLACEHOLDER_URI = "no file associated with this alert"


class ScorecardSarifError(ValueError):
    """Raised when Scorecard SARIF cannot be safely sanitized."""


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScorecardSarifError(f"{context} must be an object")
    return value


def _array(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScorecardSarifError(f"{context} must be an array")
    return value


def sanitize_scorecard_sarif(payload: Any) -> int:
    """Remove only the known non-URI placeholder and return the removal count."""
    root = _object(payload, context="SARIF root")
    if root.get("version") != "2.1.0":
        raise ScorecardSarifError("SARIF version must be '2.1.0'")
    runs = _array(root.get("runs"), context="SARIF runs")

    removed = 0
    for run_index, run_value in enumerate(runs):
        run = _object(run_value, context=f"runs[{run_index}]")
        results = _array(run.get("results", []), context=f"runs[{run_index}].results")
        for result_index, result_value in enumerate(results):
            result = _object(
                result_value,
                context=f"runs[{run_index}].results[{result_index}]",
            )
            locations_value = result.get("locations")
            if locations_value is None:
                continue
            locations = _array(
                locations_value,
                context=f"runs[{run_index}].results[{result_index}].locations",
            )
            retained: list[Any] = []
            for location_index, location_value in enumerate(locations):
                location = _object(
                    location_value,
                    context=(
                        f"runs[{run_index}].results[{result_index}].locations[{location_index}]"
                    ),
                )
                physical = location.get("physicalLocation")
                if not isinstance(physical, dict):
                    retained.append(location_value)
                    continue
                artifact = physical.get("artifactLocation")
                if not isinstance(artifact, dict) or artifact.get("uri") != PLACEHOLDER_URI:
                    retained.append(location_value)
                    continue
                removed += 1

            if retained:
                result["locations"] = retained
            else:
                result.pop("locations", None)

    return removed


def sanitize_file(path: Path) -> int:
    """Sanitize one UTF-8 SARIF file and atomically replace it on success."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScorecardSarifError(f"{path}: cannot read valid UTF-8 JSON: {exc}") from exc

    removed = sanitize_scorecard_sarif(payload)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary_path = temporary.name
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise ScorecardSarifError(f"{path}: cannot atomically replace SARIF: {exc}") from exc
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    """Sanitize the requested Scorecard SARIF file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Scorecard SARIF file")
    args = parser.parse_args(argv)
    removed = sanitize_file(args.path)
    print(f"sanitized {args.path}: removed {removed} placeholder location(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
