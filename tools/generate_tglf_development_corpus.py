#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Official TGLF Development Corpus CLI
"""Plan, generate, resume or verify a deterministic official TGLF corpus."""

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

from scpn_fusion.io.tglf_development_corpus import (
    TGLF_DEVELOPMENT_SEED,
    build_tglf_development_plan,
    generate_tglf_development_corpus,
    verify_tglf_development_corpus,
)
from scpn_fusion.io.tglf_development_plan import TGLF_EXPANDED_SELECTION_SEED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    plan = subparsers.add_parser("plan", help="Write or print the frozen plan without solving")
    plan.add_argument("--seed", type=int)
    plan.add_argument(
        "--profile", choices=("development", "expanded", "fixture"), default="development"
    )
    plan.add_argument("--output", type=Path)

    generate = subparsers.add_parser("generate", help="Generate a new recoverable corpus")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--seed", type=int)
    generate.add_argument(
        "--profile", choices=("development", "expanded", "fixture"), default="development"
    )
    generate.add_argument("--command", default="tglf")
    generate.add_argument("--timeout-s", type=float, default=120.0)
    generate.add_argument("--max-retries", type=int, default=2)
    generate.add_argument("--max-runs", type=int)
    generate.add_argument("--resume", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify a completed corpus")
    verify.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the official development-corpus command line boundary.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments without the executable name.

    Returns
    -------
    int
        Zero for a written plan or passing completed corpus, two for a durable
        partial checkpoint, and one for invalid or failed work.
    """
    args = _parser().parse_args(argv)
    try:
        if args.operation == "plan":
            seed = args.seed
            if seed is None:
                seed = (
                    TGLF_EXPANDED_SELECTION_SEED
                    if args.profile == "expanded"
                    else TGLF_DEVELOPMENT_SEED
                )
            result = build_tglf_development_plan(seed=seed, profile=args.profile)
            if args.output is not None:
                output = cast(Path, args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        elif args.operation == "generate":
            seed = args.seed
            if seed is None:
                seed = (
                    TGLF_EXPANDED_SELECTION_SEED
                    if args.profile == "expanded"
                    else TGLF_DEVELOPMENT_SEED
                )
            result = generate_tglf_development_corpus(
                cast(Path, args.output_dir),
                seed=cast(int, seed),
                profile=cast(str, args.profile),
                command=cast(str, args.command),
                timeout_s=cast(float, args.timeout_s),
                max_retries=cast(int, args.max_retries),
                resume=cast(bool, args.resume),
                max_runs=cast(int | None, args.max_runs),
            )
        else:
            result = verify_tglf_development_corpus(cast(Path, args.dataset_root))
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        result = {"status": "failed", "failures": [str(exc)]}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    if result.get("status") == "partial":
        return 2
    return 0 if result.get("status") not in {"failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
