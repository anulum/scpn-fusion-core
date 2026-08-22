# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — checksum and RNG-provenance verifier for the ITER-like baseline
"""Verify the published ITER-like synthetic dataset against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_plasma_current_feature(
    *,
    samples: int,
    workers: int,
    base_seed: int,
    coil_count: int,
    base_current: float,
    coil_multiplier_range: tuple[float, float],
    current_multiplier_range: tuple[float, float],
    divisor: float,
) -> NDArray[np.float64]:
    """Rebuild the historical per-worker RNG stream for feature zero."""
    values: list[float] = []
    for worker in range(workers):
        count = samples // workers + (1 if worker < samples % workers else 0)
        rng = np.random.default_rng(base_seed + worker)
        for _ in range(count):
            rng.uniform(*coil_multiplier_range, size=coil_count)
            values.append(base_current * rng.uniform(*current_multiplier_range) / divisor)
    return np.asarray(values, dtype=np.float64)


def _array_contract(array: NDArray[np.generic], spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if list(array.shape) != spec["shape"]:
        failures.append(f"shape {list(array.shape)} != {spec['shape']}")
    if str(array.dtype) != spec["dtype"]:
        failures.append(f"dtype {array.dtype} != {spec['dtype']}")
    return failures


def verify_dataset(*, data_dir: Path, manifest_path: Path, full_field_scan: bool) -> dict[str, Any]:
    """Return a machine-readable verification result."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = manifest["generation"]
    x_spec = manifest["arrays"]["X"]
    y_spec = manifest["arrays"]["Y"]
    x_path = data_dir / x_spec["release_asset"]
    y_path = data_dir / y_spec["release_asset"]
    failures: list[str] = []
    for path, spec in ((x_path, x_spec), (y_path, y_spec)):
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        if path.stat().st_size != spec["size_bytes"]:
            failures.append(f"size mismatch for {path}")
        if sha256_file(path) != spec["sha256"]:
            failures.append(f"SHA-256 mismatch for {path}")
    if failures:
        return {"status": "failed", "failures": failures}

    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    y = np.load(y_path, mmap_mode="r", allow_pickle=False)
    failures.extend(_array_contract(x, x_spec))
    failures.extend(_array_contract(y, y_spec))
    if not np.all(np.isfinite(x)):
        failures.append("X contains non-finite values")
    expected_ip = expected_plasma_current_feature(
        samples=generation["requested_samples"],
        workers=generation["workers"],
        base_seed=generation["worker_seeds"][0],
        coil_count=generation["coil_count"],
        base_current=generation["plasma_current_base_config_value"],
        coil_multiplier_range=(
            float(generation["coil_current_multiplier_range"][0]),
            float(generation["coil_current_multiplier_range"][1]),
        ),
        current_multiplier_range=(
            float(generation["plasma_current_multiplier_range"][0]),
            float(generation["plasma_current_multiplier_range"][1]),
        ),
        divisor=generation["plasma_current_feature_divisor"],
    )
    if not np.array_equal(np.asarray(x[:, 0]), expected_ip):
        failures.append("feature zero does not match the historical worker RNG stream")
    if full_field_scan:
        for start in range(0, len(y), 128):
            if not np.all(np.isfinite(y[start : start + 128])):
                failures.append(f"Y contains non-finite values at or after row {start}")
                break
    return {
        "status": "passed" if not failures else "failed",
        "dataset_id": manifest["dataset_id"],
        "x_sha256": x_spec["sha256"],
        "y_sha256": y_spec["sha256"],
        "rng_rows_verified": int(len(expected_ip)),
        "full_field_scan": full_field_scan,
        "failures": failures,
    }


def main() -> None:
    """Verify files and print a JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with release arrays")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-field-scan", action="store_true")
    args = parser.parse_args()
    result = verify_dataset(
        data_dir=args.data,
        manifest_path=args.manifest,
        full_field_scan=args.full_field_scan,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
