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
from typing import Any

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
    # Current best: ~0.042 (DynamicBurnModel + profile peaking factor 1.446)
    "beta_iter_sparc_beta_n_rmse": 0.10,
    # Disruption false-positive rate — hard fail (promoted from soft warn in v3.1)
    "disruption_fpr": 0.15,
    # TBR realistic range [1.0, 1.4] — corrected with port-coverage + streaming
    "tbr_min": 1.00,
    "tbr_max": 1.40,
    # Q peak — 0-D model artifact ceiling
    "q_max": 15.0,
}
_MAX_ARTIFACT_JSON_BYTES = 4 * 1024 * 1024
_MAX_TOP_LEVEL_KEYS = 256


def _load_json_object(path: Path) -> dict[str, Any]:
    size = int(path.stat().st_size)
    if size > _MAX_ARTIFACT_JSON_BYTES:
        raise ValueError(
            f"{path} exceeds max JSON size "
            f"({_MAX_ARTIFACT_JSON_BYTES} bytes)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level JSON object.")
    if len(data) > _MAX_TOP_LEVEL_KEYS:
        raise ValueError(
            f"{path} has {len(data)} top-level keys, exceeding max "
            f"{_MAX_TOP_LEVEL_KEYS}."
        )
    return data


def main() -> int:
    artifact = Path("artifacts/rmse_dashboard_ci.json")
    if not artifact.exists():
        print(f"ERROR: {artifact} not found — run rmse_dashboard.py first.")
        return 1

    try:
        data = _load_json_object(artifact)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
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

    # ── disruption FPR (hard gate since v3.1) ──────────────────────────
    real_shot_artifact = Path("artifacts/real_shot_validation.json")
    if real_shot_artifact.exists():
        try:
            rs_data = _load_json_object(real_shot_artifact)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        dis = rs_data.get("disruption", {})
        fpr = dis.get("false_positive_rate", 0.0)
        fpr_thresh = THRESHOLDS["disruption_fpr"]
        if fpr > fpr_thresh:
            failures.append(
                f"disruption FPR: {fpr:.2f} > {fpr_thresh:.2f} "
                f"(hard gate since v3.1 — FPR must be <= 15%)"
            )
        else:
            print(f"PASS  disruption FPR: {fpr:.2f} <= {fpr_thresh:.2f}")

        # ── TBR gate (corrected range [1.0, 1.4]) ──────────────────
        tbr = rs_data.get("blanket", {}).get("tbr_corrected", None)
        if tbr is not None:
            tbr_lo = THRESHOLDS["tbr_min"]
            tbr_hi = THRESHOLDS["tbr_max"]
            if tbr < tbr_lo:
                failures.append(
                    f"TBR: {tbr:.3f} < {tbr_lo:.2f} "
                    f"(below tritium self-sufficiency)"
                )
            elif tbr > tbr_hi:
                failures.append(
                    f"TBR: {tbr:.3f} > {tbr_hi:.2f} "
                    f"(unrealistically high — missing correction factors?)"
                )
            else:
                print(f"PASS  TBR: {tbr:.3f} in [{tbr_lo:.2f}, {tbr_hi:.2f}]")

        # ── Q peak gate (0-D artifact ceiling) ──────────────────────
        q_peak = rs_data.get("burn", {}).get("Q_peak", None)
        if q_peak is not None:
            q_thresh = THRESHOLDS["q_max"]
            if q_peak > q_thresh:
                failures.append(
                    f"Q_peak: {q_peak:.1f} > {q_thresh:.1f} "
                    f"(0-D model artifact — check temperature/density caps)"
                )
            else:
                print(f"PASS  Q_peak: {q_peak:.1f} <= {q_thresh:.1f}")

    if failures:
        print("\nFAILED RMSE regression gate:")
        for f in failures:
            print(f"  FAIL  {f}")
        return 1

    print("\nAll RMSE regression gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
