# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — isolated RustBCA seed-batch execution
"""Execute caller-identified RustBCA binaries and retain per-seed diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from validation.rustbca_request import validate_rustbca_request
from validation.rustbca_build_receipt import verify_rustbca_build_receipt


def _digest(raw: bytes) -> str:
    """Hash the exact input or native-extension bytes used for custody checks."""
    return hashlib.sha256(raw).hexdigest()


def _diagnostic_counts(
    sputtering: float, number: float, energy: float, samples: int
) -> tuple[int, int]:
    """Validate native or retained diagnostics before deriving discrete event counts."""
    if any(type(value) not in (int, float) for value in (sputtering, number, energy)):
        raise ValueError("RustBCA diagnostics must be numeric scalars")
    if not all(math.isfinite(v) for v in (sputtering, number, energy)):
        raise ValueError("RustBCA returned nonfinite diagnostics")
    if sputtering < 0 or not 0 <= number <= 1 or not 0 <= energy <= 1:
        raise ValueError("RustBCA diagnostics exceed the declared coefficient contract")
    counts = (sputtering * samples, number * samples)
    if any(not math.isfinite(count) or abs(count - round(count)) > 1e-7 for count in counts):
        raise ValueError("RustBCA aggregate count is not integral")
    return round(counts[0]), round(counts[1])


def _worker(batch: Path, output: Path, expected_binary_sha256: str) -> None:
    """Verify the native binary, execute one seeded batch and retain validated diagnostics."""
    raw = batch.read_bytes()
    case = json.loads(raw)
    spec = importlib.util.find_spec("libRustBCA")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Install the optional RustBCA runtime")
    binary = Path(spec.origin)
    if _digest(binary.read_bytes()) != expected_binary_sha256:
        raise ValueError("RustBCA binary digest mismatch")
    backend = importlib.import_module("libRustBCA")
    ion = {key: value for key, value in case["ion"].items() if key not in {"label", "provenance"}}
    target = {
        key: value for key, value in case["target"].items() if key not in {"label", "provenance"}
    }
    args = (ion, target, case["energy_eV"], case["angle_deg"], case["samples"])
    sputtering = float(backend.sputtering_yield(*args))
    number, energy = map(float, backend.reflection_coefficient(*args))
    counts = _diagnostic_counts(sputtering, number, energy, case["samples"])
    if batch.read_bytes() != raw or _digest(binary.read_bytes()) != expected_binary_sha256:
        raise ValueError("RustBCA binary or request changed during execution")
    result = {
        "request_sha256": _digest(raw),
        "binary_sha256": expected_binary_sha256,
        "seed": int(os.environ["LIBRUSTBCA_SEED"]),
        "threads": int(os.environ["RAYON_NUM_THREADS"]),
        "sputtering_yield": sputtering,
        "number_reflection": number,
        "energy_reflection": energy,
        "sputtered_count": counts[0],
        "reflected_count": counts[1],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def run_rustbca_reference(
    request: object,
    output_dir: Path,
    *,
    python: Path,
    expected_binary_sha256: str,
    timeout_seconds: float = 240.0,
    build_receipt: Path | None = None,
    expected_build_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Run every energy/angle/seed point with a bounded, identified real backend.

    Parameters
    ----------
    request : object
        Mapping accepted by validate_rustbca_request. Caller attribution is kept.
    output_dir : Path
        New directory for the normalized request, per-seed inputs, logs and outputs.
    python : Path
        Interpreter in the separately installed optional RustBCA environment.
    expected_binary_sha256 : str
        Previously recorded SHA256 of the runtime extension. Each child verifies it.
    timeout_seconds : float
        Total execution budget in (0, 3600] seconds, including every batch.

    build_receipt : Path or None
        Optional retained native build observation, verified before and after the run.
    expected_build_receipt_sha256 : str or None
        Required caller-pinned digest when build_receipt is supplied.

    Returns
    -------
    dict[str, Any]
        Complete seed batches and empirical means/standard errors with local file
        hashes. No source, material, physical or actuation approval is inferred.

    Raises
    ------
    ValueError
        For invalid requests, digest syntax, execution limits or changed custody.
    FileExistsError
        If the output directory already exists.
    subprocess.SubprocessError
        For child failure or exhausted execution budget; previous files are retained.

    Notes
    -----
    Binary identity is caller-supplied, not source/build attestation. The two upstream
    observables use different recoil settings and are not one cascade energy ledger.
    Seed-batch standard errors describe sampling dispersion only; zero observed
    events or zero dispersion do not establish zero uncertainty or a confidence bound.
    """
    normalized = validate_rustbca_request(request)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_binary_sha256):
        raise ValueError("expected_binary_sha256 must be a lowercase SHA256 digest")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 3600:
        raise ValueError("timeout_seconds must be finite and in (0, 3600]")
    if (build_receipt is None) != (expected_build_receipt_sha256 is None):
        raise ValueError("build_receipt and its expected digest must be supplied together")
    build_custody = None
    if build_receipt is not None and expected_build_receipt_sha256 is not None:
        build_custody = verify_rustbca_build_receipt(
            build_receipt,
            expected_receipt_sha256=expected_build_receipt_sha256,
            expected_binary_sha256=expected_binary_sha256,
        )
    python = python.absolute()
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen: dict[str, bytes] = {}
    if build_receipt is not None:
        receipt_raw = build_receipt.read_bytes()
        if _digest(receipt_raw) != expected_build_receipt_sha256:
            raise ValueError("RustBCA build receipt changed before retention")
        (output_dir / "build-receipt.json").write_bytes(receipt_raw)
        frozen["build-receipt.json"] = receipt_raw
    raw = (json.dumps(normalized, indent=2, allow_nan=False) + "\n").encode()
    (output_dir / "request.json").write_bytes(raw)
    frozen["request.json"] = raw
    deadline = time.monotonic() + timeout_seconds
    root = Path(__file__).resolve().parent.parent
    rows = []
    ordinal = 0
    for energy in normalized["energies_eV"]:
        for angle in normalized["angles_deg"]:
            batches = []
            for seed in normalized["seeds"]:
                case = {
                    "ion": normalized["ion"],
                    "target": normalized["target"],
                    "energy_eV": energy,
                    "angle_deg": angle,
                    "samples": normalized["samples_per_seed"],
                }
                name = f"batch-{ordinal:04d}"
                ordinal += 1
                input_file = output_dir / (name + "-input.json")
                output_file = output_dir / (name + "-output.json")
                batch_raw = (json.dumps(case, sort_keys=True, allow_nan=False) + "\n").encode()
                input_file.write_bytes(batch_raw)
                frozen[input_file.name] = batch_raw
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("RustBCA sweep", timeout_seconds)
                with (output_dir / (name + ".log")).open("w") as log:
                    subprocess.run(
                        [
                            str(python),
                            "-m",
                            "validation.rustbca_reference",
                            "--worker",
                            str(input_file),
                            str(output_file),
                            expected_binary_sha256,
                        ],
                        cwd=output_dir,
                        env=os.environ
                        | {
                            "PYTHONPATH": str(root),
                            "LIBRUSTBCA_SEED": str(seed),
                            "RAYON_NUM_THREADS": str(normalized["threads"]),
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=remaining,
                    )
                output_raw = output_file.read_bytes()
                frozen[output_file.name] = output_raw
                frozen[name + ".log"] = (output_dir / (name + ".log")).read_bytes()
                batch = json.loads(output_raw)
                if (
                    batch["request_sha256"] != _digest(batch_raw)
                    or batch["binary_sha256"] != expected_binary_sha256
                    or batch["seed"] != seed
                    or batch["threads"] != normalized["threads"]
                ):
                    raise ValueError("RustBCA child receipt does not match requested custody")
                counts = _diagnostic_counts(
                    batch["sputtering_yield"],
                    batch["number_reflection"],
                    batch["energy_reflection"],
                    normalized["samples_per_seed"],
                )
                if any(
                    type(batch[key]) is not int or batch[key] != expected
                    for key, expected in zip(("sputtered_count", "reflected_count"), counts)
                ):
                    raise ValueError("RustBCA child counts do not match diagnostics")
                batches.append(batch)
            summary = {}
            for key in ("sputtering_yield", "number_reflection", "energy_reflection"):
                values = [batch[key] for batch in batches]
                dispersion = statistics.stdev(values)
                summary[key] = {
                    "mean": statistics.mean(values),
                    "batch_standard_error": dispersion / math.sqrt(len(values))
                    if dispersion > 0
                    else None,
                    "uncertainty_status": "empirical_seed_batch_dispersion"
                    if dispersion > 0
                    else "unresolved_zero_observed_dispersion",
                    "confidence_interval": None,
                }
            rows.append(
                {"energy_eV": energy, "angle_deg": angle, "batches": batches, "summary": summary}
            )
    if any((output_dir / name).read_bytes() != raw for name, raw in frozen.items()):
        raise ValueError("RustBCA retained artifact changed during execution")
    if build_receipt is not None and expected_build_receipt_sha256 is not None:
        verify_rustbca_build_receipt(
            build_receipt,
            expected_receipt_sha256=expected_build_receipt_sha256,
            expected_binary_sha256=expected_binary_sha256,
        )
    report = {
        "build_custody": build_custody,
        "schema": "scpn-fusion.rustbca-diagnostics.v1",
        "request": normalized,
        "request_sha256": _digest(frozen["request.json"]),
        "binary_sha256": expected_binary_sha256,
        "interpreter": str(python),
        "timeout_seconds": timeout_seconds,
        "rows": rows,
        "files": {name: _digest(raw) for name, raw in frozen.items()},
        "provenance_verified": False,
        "material_evidence_verified": False,
        "actionable": False,
        "federated": False,
        "evidence_claimed": False,
        "scope": "caller-identified binary diagnostics; source/build and physical admission unestablished",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    """Execute an internal worker or a caller-identified RustBCA request from JSON."""
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--build-receipt-sha256")
    args = parser.parse_args()
    run_rustbca_reference(
        json.loads(args.request.read_bytes()),
        args.output,
        python=args.python,
        expected_binary_sha256=args.binary_sha256,
        timeout_seconds=args.timeout_seconds,
        build_receipt=args.build_receipt,
        expected_build_receipt_sha256=args.build_receipt_sha256,
    )


if __name__ == "__main__":
    main()
