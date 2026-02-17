#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — CI RMSE Regression Gate
# Parses rmse_dashboard_ci.json and fails if key metrics regress.
# ──────────────────────────────────────────────────────────────────────
"""CI gate: fail the build if physics RMSE metrics exceed thresholds.

Thresholds are set as **regression guards** — slightly above current
best values so that future changes cannot silently degrade physics
fidelity.  They are NOT publication-quality targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────
# Set ~30% above current best so they catch regressions, not noise.
THRESHOLDS: dict[str, float] = {
    # tau_E absolute RMSE (s) across 20 ITPA H-mode points
    # Current best: ~0.129 s
    "confinement_itpa_tau_rmse_s": 0.20,
    # Magnetic axis RMSE (m) across SPARC GEQDSKs
    # Current best: ~1.60 m (dominated by synthetic lmode files;
    # real EFIT files achieve <0.01 m)
    "sparc_axis_rmse_m": 2.50,
    # beta_N absolute RMSE across ITER/SPARC design points
    # Current best: ~1.26 (beta_N estimation from 0-D model is coarse)
    "beta_iter_sparc_beta_n_rmse": 2.00,
}


def main() -> int:
    artifact = Path("artifacts/rmse_dashboard_ci.json")
    if not artifact.exists():
        print(f"ERROR: {artifact} not found — run rmse_dashboard.py first.")
        return 1

    data = json.loads(artifact.read_text(encoding="utf-8"))
    failures: list[str] = []

    # ── confinement_itpa ──────────────────────────────────────────────
    itpa = data.get("confinement_itpa", {})
    if itpa:
        tau_rmse = itpa.get("tau_rmse_s", 0.0)
        thresh = THRESHOLDS["confinement_itpa_tau_rmse_s"]
        if tau_rmse > thresh:
            failures.append(
                f"confinement_itpa: tau_rmse {tau_rmse:.4f} s > {thresh:.4f} s"
            )
        else:
            print(f"PASS  confinement_itpa: tau_rmse {tau_rmse:.4f} s <= {thresh:.4f} s")

    # ── sparc_axis ────────────────────────────────────────────────────
    sparc = data.get("sparc_axis", {})
    if sparc:
        axis_rmse = sparc.get("axis_rmse_m", 0.0)
        thresh = THRESHOLDS["sparc_axis_rmse_m"]
        if axis_rmse > thresh:
            failures.append(
                f"sparc_axis: RMSE {axis_rmse:.4f} m > {thresh:.4f} m"
            )
        else:
            print(f"PASS  sparc_axis: RMSE {axis_rmse:.4f} m <= {thresh:.4f} m")

    # ── beta_iter_sparc ───────────────────────────────────────────────
    beta = data.get("beta_iter_sparc", {})
    if beta:
        beta_rmse = beta.get("beta_n_rmse", 0.0)
        thresh = THRESHOLDS["beta_iter_sparc_beta_n_rmse"]
        if beta_rmse > thresh:
            failures.append(
                f"beta_iter_sparc: beta_N RMSE {beta_rmse:.4f} > {thresh:.4f}"
            )
        else:
            print(f"PASS  beta_iter_sparc: beta_N RMSE {beta_rmse:.4f} <= {thresh:.4f}")

    if failures:
        print("\nFAILED RMSE regression gate:")
        for f in failures:
            print(f"  FAIL  {f}")
        return 1

    print("\nAll RMSE regression gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
