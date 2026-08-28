#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Constrained TGLF Model-Selection CLI
"""Run the frozen constrained TGLF family study against a fresh holdout."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.io.tglf_constrained_model_selection import (
    run_tglf_constrained_model_selection,
    write_tglf_constrained_selection_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--selection-lock",
        type=Path,
        action="append",
        required=True,
        help="Pre-computation lock file; repeat for independently frozen locks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=31)
    return parser


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the fail-closed constrained model-selection boundary."""
    args = _parser().parse_args(argv)
    dataset_argument = cast(Path, args.dataset_root)
    dataset_root = dataset_argument.resolve()
    output = cast(Path, args.output).resolve()
    summary: dict[str, object]
    try:
        if dataset_argument.is_symlink():
            raise ValueError("dataset root must not be a symlink")
        if _is_within(output, dataset_root):
            raise ValueError("output must be outside the immutable source corpus")
        report = run_tglf_constrained_model_selection(
            dataset_root,
            selection_lock_paths=tuple(cast(list[Path], args.selection_lock)),
            latency_repeats=cast(int, args.latency_repeats),
        )
        written = write_tglf_constrained_selection_report(report, output)
        selection = cast(dict[str, object], report["selection"])
        summary = {
            "status": report["status"],
            "output": str(written),
            "calibration_leader": selection["calibration_leader"],
            "calibration_leader_eligible": selection["calibration_leader_eligible"],
            "test_gate_passed": selection["test_gate_passed"],
        }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        summary = {"status": "failed", "failures": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
