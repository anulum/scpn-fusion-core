#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-DREAM Fluid Runaway Comparison Gate
"""Compare our reduced runaway lane against a really-executed DREAM reference.

The reference artifact (``validation/reference_data/dream/``) was produced by
actually running the open-source DREAM code (Chalmers,
doi:10.1016/j.cpc.2021.108098) through ``validation/dream_reference_runner.py``
in its dedicated pixi environment (``external/dream-env``; DREAM is a C++
code needing GSL/HDF5/PETSc, built in ``external/DREAM``). The artifact
carries full provenance (DREAM git SHA, settings SHA-256, timestamp), and
this lane runs everywhere against the committed artifact, so its checks are
deterministic.

Claim boundary: this gate asserts reference integrity, finite rate output,
and that the cross-code rate ratios are RECORDED — it does not claim physics
equivalence. The first real-reference comparison exposed two genuine findings
in ``scpn_fusion.core.runaway_electrons`` (documented in
``rate_model_findings``): the avalanche exponential rate is ~32x DREAM's
Rosenbluth-Putvinski fluid rate at the same state — consistent with the
missing RP geometric factor ``sqrt(pi/(3(Z+5)))/lnLambda`` (= 1/35.9 at
Z=1, lnLambda=15) — and the Dreicer rate is ~5.4x LOW versus DREAM's
Connor-Hastie implementation (fixed lnLambda=15 versus DREAM's computed
Coulomb logarithm, and a differing prefactor variant). Equivalence
thresholds may only be introduced after that correctness row is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE = ROOT / "validation" / "reference_data" / "dream" / "dream_fluid_runaway_reference.json"
REPORT = ROOT / "validation" / "reports" / "dream_fluid_parity.json"
SCHEMA = "scpn-fusion-core.dream-fluid-parity.v1"

sys.path.insert(0, str(SRC))


def _load_reference() -> dict[str, Any]:
    """Load and integrity-check the committed DREAM reference artifact."""
    payload = cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))
    provenance = payload["provenance"]
    required = ("code", "code_url", "dream_git_sha", "settings_sha256", "paper_doi")
    missing = [key for key in required if not provenance.get(key)]
    if missing:
        raise ValueError(f"DREAM reference provenance incomplete: missing {missing}")
    if provenance["code"] != "DREAM":
        raise ValueError("reference artifact is not a DREAM export")
    return payload


def _series_checksum(series: dict[str, Any]) -> str:
    """Return a stable checksum of the reference series payload."""
    canonical = json.dumps(series, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_report() -> dict[str, Any]:
    """Build the DREAM fluid-parity report payload."""
    runaway = import_module("scpn_fusion.core.runaway_electrons")

    reference = _load_reference()
    scenario = reference["scenario"]
    series = reference["series"]

    params = runaway.RunawayParams(
        ne_20=float(scenario["n_e_m3"]) / 1e20,
        Te_keV=float(scenario["T_e_eV"]) / 1e3,
        E_par=float(scenario["E_field_V_m"]),
        Z_eff=float(scenario["Z_eff"]),
        B0=float(scenario["B0_T"]),
        R0=1.65,  # not used by the closed-form rates; recorded for the contract
        a=float(scenario["minor_radius_m"]),
    )

    ours_dreicer = float(runaway.dreicer_generation_rate(params))
    # avalanche_growth_rate is linear in n_RE; evaluating at n_RE=1 yields the
    # exponential rate [1/s] directly comparable with DREAM's GammaAva.
    ours_avalanche_exp = float(runaway.avalanche_growth_rate(params, 1.0))

    dream_dreicer = float(np.median(series["other_fluid"]["gammaDreicer"]))
    dream_avalanche_exp = float(np.median(series["other_fluid"]["GammaAva"]))

    dreicer_ratio = ours_dreicer / dream_dreicer if dream_dreicer > 0.0 else float("inf")
    avalanche_ratio = (
        ours_avalanche_exp / dream_avalanche_exp if dream_avalanche_exp > 0.0 else float("inf")
    )

    n_re = np.asarray(series["n_re_m3"], dtype=np.float64)
    time_s = np.asarray(series["time_s"], dtype=np.float64)

    checks = {
        "reference_provenance_complete": True,
        "reference_series_finite": bool(np.all(np.isfinite(n_re)) and np.all(np.isfinite(time_s))),
        "reference_n_re_monotonic_growth": bool(np.all(np.diff(n_re) >= 0.0)),
        "our_rates_finite_and_positive": bool(
            np.isfinite(ours_dreicer)
            and ours_dreicer > 0.0
            and np.isfinite(ours_avalanche_exp)
            and ours_avalanche_exp > 0.0
        ),
        "cross_code_ratios_recorded": bool(
            np.isfinite(dreicer_ratio) and np.isfinite(avalanche_ratio)
        ),
    }

    z_eff = float(scenario["Z_eff"])
    rp_geometric_factor = float(15.0 * np.sqrt(3.0 * (z_eff + 5.0) / np.pi))

    return {
        "schema": SCHEMA,
        "reference": {
            "artifact": str(REFERENCE.relative_to(ROOT)),
            "dream_git_sha": reference["provenance"]["dream_git_sha"],
            "settings_sha256": reference["provenance"]["settings_sha256"],
            "series_checksum": _series_checksum(series),
        },
        "scenario": scenario,
        "rates": {
            "ours_dreicer_m3_s": ours_dreicer,
            "dream_gamma_dreicer_m3_s": dream_dreicer,
            "dreicer_ratio_ours_over_dream": dreicer_ratio,
            "ours_avalanche_exponential_s": ours_avalanche_exp,
            "dream_gamma_ava_s": dream_avalanche_exp,
            "avalanche_ratio_ours_over_dream": avalanche_ratio,
        },
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "rate_model_findings": {
            "avalanche_overestimate": (
                "our avalanche_growth_rate cites Rosenbluth-Putvinski Eq. 66 "
                "but omits the geometric factor sqrt(pi/(3(Z+5)))/lnLambda; "
                f"lnLambda*sqrt(3(Z+5)/pi) = {rp_geometric_factor:.1f} at this "
                f"scenario versus the measured ratio {avalanche_ratio:.1f} - "
                "the missing factor explains the discrepancy to within the "
                "difference between our fixed lnLambda=15 and DREAM's computed "
                "Coulomb logarithm and effective critical momentum"
            ),
            "dreicer_underestimate": (
                f"our Connor-Hastie rate is {1.0 / dreicer_ratio:.1f}x LOW "
                "versus DREAM's CONNOR_HASTIE implementation at the same "
                "state; suspects: fixed lnLambda=15 versus computed, and the "
                "prefactor/exponent variant of the fit"
            ),
            "disposition": (
                "correctness row tracked in TODO BACKLOG 3; equivalence "
                "thresholds are deliberately NOT asserted until the rate "
                "models are reconciled against the open-access DREAM paper "
                "(arXiv:2103.16457) with source-verified formulas"
            ),
        },
        "claim_boundary": (
            "integrity + finiteness + recorded cross-code ratios only; "
            "physics equivalence NOT claimed"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the DREAM fluid-parity gate and write the tracked report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["rates"], indent=2, sort_keys=True))
    print(f"all_checks_passed: {report['all_checks_passed']}")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
