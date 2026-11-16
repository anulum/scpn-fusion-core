#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — per-regime coverage of the QLKNN-10D split-conformal certificate
"""Measure how the QLKNN-10D split-conformal certificate covers each transport regime.

The tracked certificate (``validation/surrogate_conformal_certificates.json``) gives 90 %
marginal intervals per output channel. This report splits the untouched holdout split by
two regime definitions, both fixed here before any result was seen:

* **input labels**, the split's ``regimes`` array. It is an input-gradient heuristic, not a
  stability label: R/L_Ti > 4 is ``itg_label``, R/L_Te > 5 with R/L_Ti <= 4 is
  ``tem_label``, and every other row is ``neither_label``. These labels are known at
  prediction time.
* **target regimes**, per channel: ``stable`` where the target flux is zero (QLKNN-10D
  clips negative fluxes to zero, so this class includes clipped rows); ``near_threshold``
  where it is positive and at most the :data:`NEAR_THRESHOLD_QUANTILE` quantile of the
  calibration split's positive targets; ``unstable`` above that.

For every group the report gives the coverage of the certificate's interval with a 95 %
Wilson interval, and, where the target is positive, the median interval width relative to
the target. A group below the target coverage is a finding, not a failure. The report
gates only its own completeness: the certificate, weights and splits match their recorded
digests, the recomputed marginal coverage equals the certificate's, and every row falls
into exactly one group of each definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import generate_surrogate_conformal_certificates as certificates

FloatArray = NDArray[np.float64]
REPORT_PATH = ROOT / "validation/reports/qlknn_conformal_regime_coverage.json"
MARKDOWN_PATH = ROOT / "validation/reports/qlknn_conformal_regime_coverage.md"
CERTIFICATE_PATH = ROOT / "validation/surrogate_conformal_certificates.json"
SCHEMA = "scpn-fusion-core.qlknn-conformal-regime-coverage.v1"
NEAR_THRESHOLD_QUANTILE = 0.10
WILSON_Z = 1.959963984540054
INPUT_LABELS = {0: "neither_label", 1: "itg_label", 2: "tem_label"}
TARGET_REGIMES = ("stable", "near_threshold", "unstable")
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — per-regime coverage of the QLKNN-10D conformal certificate -->",
)


def wilson_interval(covered: int, total: int) -> tuple[float, float]:
    """Return the 95 % Wilson score interval of a binomial proportion."""
    if total <= 0 or not 0 <= covered <= total:
        raise ValueError("need 0 <= covered <= total and total > 0")
    p = covered / total
    denominator = 1.0 + WILSON_Z**2 / total
    centre = (p + WILSON_Z**2 / (2 * total)) / denominator
    half = WILSON_Z * math.sqrt(p * (1 - p) / total + WILSON_Z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def group_coverage(
    covered: NDArray[np.bool_], members: NDArray[np.bool_], relative_width: FloatArray
) -> dict[str, Any]:
    """Coverage of one group, its Wilson interval and its median relative width."""
    total = int(members.sum())
    if total == 0:
        return {
            "n": 0,
            "covered": 0,
            "coverage": None,
            "wilson95": None,
            "median_relative_width": None,
        }
    hits = int(covered[members].sum())
    widths = relative_width[members]
    widths = widths[np.isfinite(widths)]
    return {
        "n": total,
        "covered": hits,
        "coverage": hits / total,
        "wilson95": list(wilson_interval(hits, total)),
        "median_relative_width": float(np.median(widths)) if widths.size else None,
    }


def target_regimes(targets: FloatArray, threshold: float) -> dict[str, NDArray[np.bool_]]:
    """Split one channel's targets into stable, near-threshold and unstable rows."""
    return {
        "stable": targets == 0.0,
        "near_threshold": (targets > 0.0) & (targets <= threshold),
        "unstable": targets > threshold,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_labels(path: Path, rows: int) -> NDArray[np.int64]:
    """Return a split's input-gradient labels, refusing a missing row or an unknown label."""
    with np.load(path, allow_pickle=False) as payload:
        labels = np.asarray(payload["regimes"], dtype=np.int64)
    if labels.shape != (rows,) or not np.isin(labels, list(INPUT_LABELS)).all():
        raise ValueError(f"{path} regimes must hold one label in {sorted(INPUT_LABELS)} per row")
    return labels


def build_report(certificate_path: Path = CERTIFICATE_PATH) -> dict[str, Any]:
    """Recompute the certificate's holdout scores and measure coverage per regime."""
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    entry = certificate["certificates"][0]
    weights = ROOT / entry["model"]["weights"]
    calibration = ROOT / entry["calibration"]["split"]
    holdout = ROOT / entry["holdout"]["split"]
    digests = {
        "weights": _sha256(weights) == entry["model"]["weights_file_sha256"],
        "calibration": _sha256(calibration) == entry["calibration"]["split_sha256"],
        "holdout": _sha256(holdout) == entry["holdout"]["split_sha256"],
    }
    if not all(digests.values()):
        raise ValueError(f"inputs differ from the certificate's digests: {digests}")
    scores = certificates.residual_scores(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout
    )
    calibration_targets = scores.calibration_targets
    labels = input_labels(holdout, scores.holdout_scores.shape[0])
    channels: dict[str, Any] = {}
    reproduced = True
    partitions = True
    for index, channel in enumerate(entry["channels"]):
        radius = float(channel["radius"])
        covered = scores.holdout_scores[:, index] <= radius
        marginal = float(np.mean(covered))
        reproduced &= marginal == channel["holdout_empirical_coverage"]
        target = scores.holdout_targets[:, index]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_width = np.where(target > 0.0, 2.0 * radius / target, np.nan)
        positive = calibration_targets[:, index][calibration_targets[:, index] > 0.0]
        threshold = float(np.quantile(positive, NEAR_THRESHOLD_QUANTILE))
        by_label = {name: labels == code for code, name in INPUT_LABELS.items()}
        by_target = target_regimes(target, threshold)
        for groups in (by_label, by_target):
            partitions &= bool((np.sum(list(groups.values()), axis=0) == 1).all())
        channels[channel["name"]] = {
            "radius": radius,
            "marginal_coverage": marginal,
            "near_threshold_upper_value": threshold,
            "input_labels": {
                name: group_coverage(covered, members, relative_width)
                for name, members in by_label.items()
            },
            "target_regimes": {
                name: group_coverage(covered, members, relative_width)
                for name, members in by_target.items()
            },
        }
    checks = {
        "marginal_coverage_reproduces_certificate": bool(reproduced),
        "every_row_in_exactly_one_group": bool(partitions),
    }
    return {
        "schema": SCHEMA,
        "certificate": {
            "path": (
                certificate_path.relative_to(ROOT).as_posix()
                if certificate_path.is_relative_to(ROOT)
                else str(certificate_path)
            ),
            "sha256": _sha256(certificate_path),
            "target_marginal_coverage": certificate["method"]["target_marginal_coverage"],
        },
        "definitions": {
            "input_labels": "split 'regimes': R/L_Ti > 4 itg; R/L_Te > 5 and R/L_Ti <= 4 tem; else neither",
            "target_regimes": "per channel: 0 stable; (0, q] near_threshold; > q unstable",
            "near_threshold_quantile_of_positive_calibration_targets": NEAR_THRESHOLD_QUANTILE,
            "coverage": "holdout canonical absolute residual <= certificate radius",
            "relative_width": "2 * radius / target, positive targets only",
        },
        "holdout_rows": int(scores.holdout_scores.shape[0]),
        "output_unit": scores.output_unit,
        "channels": channels,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }


def _fmt(group: dict[str, Any]) -> str:
    if not group["n"]:
        return "— (0 rows)"
    low, high = group["wilson95"]
    return f"{group['coverage']:.3f} [{low:.3f}, {high:.3f}] (n={group['n']})"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the per-channel, per-group coverage table."""
    target = report["certificate"]["target_marginal_coverage"]
    lines = [
        *MARKDOWN_HEADER,
        "# Per-regime coverage of the QLKNN-10D split-conformal certificate",
        "",
        f"Status: **{report['status'].upper()}** (completeness gates). Target marginal "
        f"coverage {target:.2f}; a group below it is a finding, not a gate.",
        "",
        f"Holdout rows: {report['holdout_rows']}. Coverage with 95 % Wilson interval.",
        "",
        "| Channel | Group | Coverage | Median 2r/y |",
        "|---|---|---|---:|",
    ]
    for name, channel in report["channels"].items():
        lines.append(f"| {name} | marginal | {channel['marginal_coverage']:.3f} | — |")
        for kind in ("input_labels", "target_regimes"):
            for group_name, group in channel[kind].items():
                width = group["median_relative_width"]
                lines.append(
                    f"| {name} | {group_name} | {_fmt(group)} "
                    f"| {'—' if width is None else f'{width:.3g}'} |"
                )
    lines += ["", f"Source revision: `{report['source_revision']}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 when a completeness gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=CERTIFICATE_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = build_report(args.certificate.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
