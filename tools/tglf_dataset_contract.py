#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GACODE TGLF Dataset Contract CLI
"""Build or verify a versioned official-GACODE TGLF dataset manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_dataset_contract import (
    MAX_TGLF_RECORDS_BYTES,
    build_tglf_dataset_manifest,
    verify_tglf_dataset,
    write_tglf_dataset_manifest,
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = checked_json_load(path, max_bytes=MAX_TGLF_RECORDS_BYTES)
    if not isinstance(payload, list) or not payload:
        raise ValueError("records JSON must contain a non-empty array")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"records[{index}] must be an object")
        records.append(cast(dict[str, Any], item))
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and immediately verify manifest.json")
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--records", default="dataset.json")
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--gacode-revision", required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--purpose", choices=("pilot", "development"), default="pilot")

    verify = subparsers.add_parser("verify", help="Verify an existing manifest.json")
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset-contract command line interface.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments without the executable name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero only when the requested build or verification passes.
    """
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            root = cast(Path, args.dataset_root)
            records_name = cast(str, args.records)
            records = _load_records(root / records_name)
            manifest = build_tglf_dataset_manifest(
                root,
                records,
                dataset_id=cast(str, args.dataset_id),
                gacode_revision=cast(str, args.gacode_revision),
                seed=cast(int, args.seed),
                purpose=cast(str, args.purpose),
                records_file=records_name,
            )
            manifest_path = write_tglf_dataset_manifest(root, manifest)
            result = verify_tglf_dataset(root, manifest_path=manifest_path)
        else:
            result = verify_tglf_dataset(
                cast(Path, args.dataset_root), manifest_path=cast(Path | None, args.manifest)
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "dataset_id": None, "failures": [str(exc)]}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
