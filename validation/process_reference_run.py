# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source-bound PROCESS reference execution
"""Run the pinned real PROCESS evaluation case and preserve diagnostics and custody."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from validation.process_power_report import read_process_power_report

MANIFEST = Path(__file__).parent / "reference_data/process_source.json"


def verify_process_source() -> dict[str, Any]:
    """Verify installed package source/data against the immutable reference manifest.

    Returns
    -------
    dict[str, Any]
        Source commit, manifest digest, installed version and generated-version hash.

    Raises
    ------
    ModuleNotFoundError
        When the optional upstream package is unavailable.
    ValueError
        When installed source/data differ, are absent or include unexpected files.

    Notes
    -----
    Python/Numba caches are excluded. The unused build-generated _version.py is
    recorded separately; all 220 upstream package source/data files are checked.
    This does not certify transitive dependencies or physical model validity.
    """
    spec = importlib.util.find_spec("process")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Install the pinned PROCESS optional runtime")
    package = Path(spec.origin).parent
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    actual = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p != package / "_version.py"
    }
    if actual != manifest["package_files"]:
        raise ValueError("Installed PROCESS source/data do not match the pinned manifest")
    generated = package / "_version.py"
    return {
        "commit": manifest["commit"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "installed_version": importlib.metadata.version("process"),
        "generated_version_sha256": hashlib.sha256(generated.read_bytes()).hexdigest()
        if generated.exists()
        else None,
    }


def run_process_reference(
    case: Path, output_dir: Path, *, equality_tolerance: float, timeout_seconds: float = 240.0
) -> dict[str, Any]:
    """Execute the pinned case in a separate process, retaining even rejected results.

    Parameters
    ----------
    case : Path
        Unmodified upstream large_tokamak_eval_IN.DAT from the pinned snapshot.
    output_dir : Path
        New directory exclusively created for inputs, log, outputs and JSON receipt.
    equality_tolerance : float
        Caller-declared positive finite equality diagnostic tolerance.
    timeout_seconds : float
        Finite positive child execution limit in seconds; defaults to 240.

    Returns
    -------
    dict[str, Any]
        Power diagnostics with local execution/source/file custody. Scientific and
        control authority remain false regardless of convergence or recorded checks.

    Raises
    ------
    ValueError
        For input/source mismatch or malformed output diagnostics.
    FileExistsError
        If the requested output directory exists; nothing is overwritten.
    subprocess.SubprocessError
        If execution fails or exceeds the timeout. Existing log/input stay available.

    Notes
    -----
    Run this API in the isolated PROCESS environment. The child uses its exact
    interpreter with two OpenBLAS/Numba threads and a noninteractive plotting backend.
    Verification is local observed custody, not a signed third-party attestation.
    """
    if not math.isfinite(equality_tolerance) or equality_tolerance <= 0:
        raise ValueError("equality_tolerance must be finite and positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    source = verify_process_source()
    manifest = json.loads(MANIFEST.read_bytes())
    case_bytes = case.read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != manifest["case_sha256"]:
        raise ValueError("PROCESS reference input differs from the pinned case")
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_case = output_dir / "IN.DAT"
    frozen_case.write_bytes(case_bytes)
    code = "from process.main import SingleRun; import sys; SingleRun(sys.argv[1], filepath_out=sys.argv[1]).run()"
    env = os.environ | {"OPENBLAS_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    with (output_dir / "execution.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c", code, str(frozen_case)],
            cwd=output_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_seconds,
        )
    if verify_process_source() != source or frozen_case.read_bytes() != case_bytes:
        raise ValueError("PROCESS source or case changed during execution")
    mfile = output_dir / "MFILE.DAT"
    report = read_process_power_report(
        mfile,
        expected_sha256=hashlib.sha256(mfile.read_bytes()).hexdigest(),
        expected_constraints=tuple(manifest["constraints"]),
        equality_tolerance=equality_tolerance,
    )
    report["execution"] = {
        "source": source,
        "input_sha256": manifest["case_sha256"],
        "timeout_seconds": timeout_seconds,
        "python": sys.version,
        "interpreter": sys.executable,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output_dir.iterdir()
            if p.is_file()
        },
        "source_verified_locally": True,
        "third_party_attested": False,
    }
    report["provenance_verified"] = True
    report["scope"] = "local source-bound execution diagnostics; physical admission not established"
    (output_dir / "power-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Run one pinned case and write a source-bound diagnostic report via the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--equality-tolerance", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()
    run_process_reference(
        args.case,
        args.output_dir,
        equality_tolerance=args.equality_tolerance,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
