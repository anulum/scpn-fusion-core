#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — official GACODE TGLF runtime evidence
"""Exercise the PATH-resolved official GACODE TGLF runtime and record evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core._tglf_interface_runtime import (
    _parse_gacode_tglf_spectrum,
    _resolve_tglf_command,
)
from scpn_fusion.core.tglf_interface import TGLFInputDeck, run_tglf_binary
from validation.evidence_output import add_evidence_output_arguments, resolve_evidence_outputs

EXPECTED_GACODE_COMMIT = "b49339750a4aa4cf2b089fa9ff3afe098005f0f8"
CANONICAL_JSON = ROOT / "validation" / "reports" / "tglf_gacode_runtime.json"
CANONICAL_MD = ROOT / "validation" / "reports" / "tglf_gacode_runtime.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unreported"


def _cyclone_like_deck() -> TGLFInputDeck:
    return TGLFInputDeck(
        rho=0.5,
        q=1.4,
        s_hat=0.78,
        R_LTi=6.9,
        R_LTe=6.9,
        R_Lne=2.2,
        R_Lni=2.2,
        beta_e=0.0,
        Z_eff=1.0,
        xnue=0.01,
        T_e_keV=2.0,
        T_i_keV=2.0,
        R_major=2.78,
        a_minor=1.0,
        B_toroidal=2.0,
        kappa=1.0,
        delta=0.0,
    )


def run_benchmark(*, command: str, work_dir: Path, timeout_s: float) -> dict[str, Any]:
    """Run one public-interface case plus the official nine-case regression suite."""
    resolved = _resolve_tglf_command(command)
    work_dir.mkdir(parents=True, exist_ok=True)
    case_dir = work_dir / "cyclone_like"
    regression_dir = work_dir / "upstream_regression"
    regression_dir.mkdir(parents=True, exist_ok=True)

    deck = _cyclone_like_deck()
    started = time.perf_counter()
    output = run_tglf_binary(
        deck,
        tglf_command=command,
        timeout_s=timeout_s,
        work_dir=case_dir,
        max_retries=0,
    )
    case_seconds = time.perf_counter() - started
    gamma, omega_r, k_y = _parse_gacode_tglf_spectrum(case_dir)

    regression_started = time.perf_counter()
    regression = subprocess.run(
        [resolved, "-r"],
        cwd=regression_dir,
        capture_output=True,
        text=True,
        timeout=max(timeout_s, 120.0),
    )
    regression_seconds = time.perf_counter() - regression_started
    pass_cases = re.findall(r"^(tglf\d+): PASS", regression.stdout, flags=re.MULTILINE)
    version = (case_dir / "out.tglf.version").read_text(encoding="utf-8").splitlines()[0]
    expected_short = EXPECTED_GACODE_COMMIT[:8]

    output_files = [
        "input.tglf",
        "out.tglf.gbflux",
        "out.tglf.eigenvalue_spectrum",
        "out.tglf.ky_spectrum",
        "out.tglf.version",
    ]
    hashes = {name: _sha256(case_dir / name) for name in output_files}
    serialized_output = asdict(output)
    scalar_values = [
        value for value in serialized_output.values() if isinstance(value, (int, float))
    ]
    for species_flux in serialized_output["species_fluxes"]:
        scalar_values.extend(
            value for value in species_flux.values() if isinstance(value, (int, float))
        )
    gates = {
        "expected_gacode_revision": version.startswith(expected_short),
        "finite_public_output": bool(np.all(np.isfinite(np.asarray(scalar_values)))),
        "nonempty_consistent_spectrum": bool(
            len(k_y) > 0 and len(k_y) == len(gamma) == len(omega_r)
        ),
        "official_regression_9_of_9": regression.returncode == 0
        and pass_cases == [f"tglf{i:02d}" for i in range(1, 10)],
        "signed_fluxes_preserved": output.particle_e < 0.0 and output.particle_i < 0.0,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "official GACODE runtime activation and parser fidelity",
        "gacode": {
            "repository": "https://github.com/gafusion/gacode",
            "expected_commit": EXPECTED_GACODE_COMMIT,
            "reported_version": version,
            "command_name": command,
            "path_resolution_required": True,
        },
        "machine": {
            "cpu": _cpu_model(),
            "architecture": platform.machine(),
            "system": platform.system(),
        },
        "case": {
            "name": "Cyclone-like electrostatic two-species activation case",
            "input": asdict(deck),
            "output": serialized_output,
            "spectrum_points": int(len(k_y)),
            "dominant_ky": float(k_y[int(np.argmax(gamma))]),
            "elapsed_seconds_orientation_only": case_seconds,
            "sha256": hashes,
        },
        "official_regression": {
            "passed_cases": pass_cases,
            "pass_count": len(pass_cases),
            "elapsed_seconds_orientation_only": regression_seconds,
        },
        "gates": gates,
        "limitations": [
            "Elapsed times are orientation-only measurements from the reported workstation.",
            "This activation benchmark does not establish surrogate accuracy or uncertainty calibration.",
            "This activation benchmark does not establish superiority over another transport solver.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    case = report["case"]
    output = case["output"]
    regression = report["official_regression"]
    gates = report["gates"]
    rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items()
    )
    return f"""# GACODE TGLF runtime evidence

Status: **{report["status"]}**

This report records a PATH-resolved official GACODE TGLF activation run. It is
runtime and parser evidence, not a cross-solver accuracy claim.

| Gate | Result |
|---|---:|
{rows}

- GACODE revision: `{report["gacode"]["reported_version"]}`
- Official regression: {regression["pass_count"]}/9 cases passed
- Activation spectrum: {case["spectrum_points"]} ky points
- Ion/electron heat flux: {output["q_i"]:.8g} / {output["q_e"]:.8g} gyro-Bohm
- Electron/ion particle flux: {output["particle_e"]:.8g} / {output["particle_i"]:.8g} gyro-Bohm
- Activation runtime: {case["elapsed_seconds_orientation_only"]:.6f} s (orientation only)
- Regression runtime: {regression["elapsed_seconds_orientation_only"]:.6f} s (orientation only)
- Machine CPU: {report["machine"]["cpu"]}

## Limits

{chr(10).join(f"- {item}" for item in report["limitations"])}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="tglf")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "artifacts" / "tmp_task" / "tglf_gacode",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    add_evidence_output_arguments(parser)
    args = parser.parse_args()
    outputs = resolve_evidence_outputs(
        root=ROOT,
        canonical_json=CANONICAL_JSON,
        canonical_markdown=CANONICAL_MD,
        requested_json=args.output_json,
        requested_markdown=args.output_md,
        commit_evidence=args.commit_evidence,
    )
    report = run_benchmark(command=args.command, work_dir=args.work_dir, timeout_s=args.timeout_s)
    outputs.json.parent.mkdir(parents=True, exist_ok=True)
    outputs.markdown.parent.mkdir(parents=True, exist_ok=True)
    outputs.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(outputs.json),
                "markdown": str(outputs.markdown),
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
