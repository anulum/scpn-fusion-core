#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Diagnostic
"""Execute the fixed-physical CVGC2 forcing and inverse-response diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from validation import diagnose_ida_coil_vacuum_grid_convergence as grid_diagnostic
from validation import ida_coil_vacuum_fixed_physical_contract as contract
from validation.ida_coil_vacuum_fixed_physical_response import (
    build_fixed_physical_convergence,
    build_fixed_physical_grid,
)

ROOT = grid_diagnostic.ROOT
REPORT_PATH = ROOT / contract.REPORT_PATH
MARKDOWN_PATH = ROOT / contract.MARKDOWN_PATH


def _source_artifacts() -> dict[str, dict[str, Any]]:
    """Bind every executed CVGC2 source file and the clean source commit."""
    artifacts: dict[str, dict[str, Any]] = {
        name: {
            "path": path,
            "sha256": grid_diagnostic._file_sha256(ROOT / path),
        }
        for name, path in sorted(contract.SOURCE_PATHS.items())
    }
    artifacts["repository"] = grid_diagnostic._repository_artifact()
    return artifacts


def _verify_cvgc1_arrays(
    execution: grid_diagnostic.GridLadderExecution,
    upstream: dict[str, Any],
) -> None:
    """Require CVGC2 to reuse every upstream total forcing and response array."""
    if execution.anchor != upstream["anchor"]:
        raise ValueError("CVGC2 exact 129 anchor drifted from CVGC1")
    if execution.coil_manifest != upstream["coil_manifest"]:
        raise ValueError("CVGC2 coil manifest drifted from CVGC1")
    upstream_grids = {int(row["resolution"]): row for row in upstream["grids"]}
    for result in execution.results:
        expected = upstream_grids[result.resolution]
        if (
            result.report["forcing_partition"]["total"]["field_sha256"]
            != expected["forcing_partition"]["total"]["field_sha256"]
        ):
            raise ValueError(f"CVGC2 {result.resolution} total forcing drifted from CVGC1")
        if (
            result.report["response_partition"]["total"]["field_sha256"]
            != expected["response_partition"]["total"]["field_sha256"]
        ):
            raise ValueError(f"CVGC2 {result.resolution} total response drifted from CVGC1")
        radius = float(result.report["masks"]["fixed_physical_radius_m"])
        if radius != contract.FIXED_PHYSICAL_RADIUS_M:
            raise ValueError("CVGC2 fixed physical radius drifted from the frozen value")


def run_diagnostic(*, generated_at: str) -> dict[str, Any]:
    """Execute the exact CVGC1 ladder and fixed-physical inverse partitions."""
    upstream = contract.load_upstream_report(ROOT)
    execution = grid_diagnostic.execute_grid_ladder()
    _verify_cvgc1_arrays(execution, upstream)
    fixed_results = [build_fixed_physical_grid(row) for row in execution.results]
    source_artifacts = _source_artifacts()
    return contract.build_report(
        generated_at=generated_at,
        environment=execution.environment,
        execution_binding=contract.build_execution_binding(
            anchor=execution.anchor,
            coil_manifest=execution.coil_manifest,
            source_artifacts=source_artifacts,
        ),
        source_artifacts=source_artifacts,
        upstream=contract.upstream_binding(ROOT, upstream),
        grids=[row.report for row in fixed_results],
        convergence=build_fixed_physical_convergence(fixed_results),
    )


def write_report(
    report: dict[str, Any],
    *,
    output: Path = REPORT_PATH,
    markdown_output: Path = MARKDOWN_PATH,
) -> None:
    """Write validated JSON and Markdown evidence to explicit repository paths."""
    contract.validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(contract.render_markdown(report), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one written evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run, write, and optionally revalidate the CVGC2 report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = run_diagnostic(generated_at=args.generated_at)
    write_report(report, output=args.output, markdown_output=args.markdown_output)
    if args.check:
        loaded = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("written CVGC2 report must remain a JSON object")
        contract.validate_report(loaded)
        if loaded["payload_sha256"] != report["payload_sha256"]:
            raise ValueError("written CVGC2 payload drifted after serialisation")
        if not args.markdown_output.read_text(encoding="utf-8").endswith("\n"):
            raise ValueError("written CVGC2 Markdown must end with a newline")
    sys.stdout.write(
        json.dumps(
            {
                "json_sha256": _file_sha256(args.output),
                "markdown_sha256": _file_sha256(args.markdown_output),
                "payload_sha256": report["payload_sha256"],
                "routing": report["routing"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
