#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — split-conformal variants for the QLKNN-10D surrogate, compared per regime
"""Compare split-conformal variants for the QLKNN-10D surrogate, regime by regime.

Each variant is calibrated on the certificate's calibration split at its level ``1 - alpha``
with the finite-sample rank, and evaluated on the untouched holdout split. The variants and
their scores are fixed here before any result was seen:

* ``marginal_absolute``: score ``|y - y_hat|``, one radius per channel. This is the tracked
  certificate, and the report checks that it reproduces the certificate exactly.
* ``mondrian_input_label``: the same score, with one radius per channel and per input label
  of the split (an input-gradient heuristic, known at prediction time). Coverage is then
  guaranteed within each label group, under exchangeability within the group.
* ``normalised_gyro_bohm``: score ``|y - y_hat| / chi_gB(Te)``, where ``chi_gB`` is the
  gyro-Bohm diffusivity that converts the database's normalised targets to m^2/s. The
  interval is ``y_hat +- q chi_gB``.
* ``normalised_prediction``: score ``|y - y_hat| / (|y_hat| + r0)``, with ``r0`` the
  channel's certificate radius. The interval is ``y_hat +- q (|y_hat| + r0)``.

Two further variants were added after the first comparison (940937b9, report 753c658a),
so they were chosen after seeing its results and are reported as post hoc:

* ``mondrian_predicted_flux``: the absolute score, with one radius per predicted-flux class
  (``y_hat <= r0`` or ``y_hat > r0``), a group known at prediction time.
* ``mondrian_predicted_flux_normalised_prediction``: the ``|y_hat| + r0`` score with the same
  two groups.

Per channel the report also gives two diagnostics of the surrogate itself on the unstable
rows: the share predicted at most ``r0``, and the median ratio of prediction to target.

Every variant is scored on the same groups as the per-regime coverage report (target
regimes, which are known only for evaluation, and input labels), with Wilson intervals and
the median relative width ``2h/y`` on positive targets. The report gates only its
completeness: the inputs match the certificate's digests and the marginal variant reproduces
the certificate's radii and coverage exactly. A Mondrian group without enough calibration
rows for its finite-sample rank stops the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from validation import qlknn_conformal_regime_coverage as regimes

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
REPORT_PATH = ROOT / "validation/reports/qlknn_conformal_variants.json"
MARKDOWN_PATH = ROOT / "validation/reports/qlknn_conformal_variants.md"
SCHEMA = "scpn-fusion-core.qlknn-conformal-variants.v1"
VARIANTS = (
    "marginal_absolute",
    "mondrian_input_label",
    "normalised_gyro_bohm",
    "normalised_prediction",
)
POST_HOC_VARIANTS = (
    "mondrian_predicted_flux",
    "mondrian_predicted_flux_normalised_prediction",
)
POST_HOC_NOTE = (
    "added after the first comparison (940937b9, report 753c658a); chosen after seeing its results"
)
PREDICTED_CLASSES = {0: "predicted_at_most_r0", 1: "predicted_above_r0"}
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — split-conformal variants for the QLKNN-10D surrogate -->",
)


def rank_quantile(scores: FloatArray, alpha: float) -> float:
    """Return the finite-sample corrected split-conformal quantile of ``scores``."""
    rank = certificates.finite_sample_rank(int(scores.size), alpha)
    return float(np.partition(scores, rank - 1)[rank - 1])


def group_quantiles(
    scores: FloatArray, groups: IntArray, needed: IntArray, alpha: float
) -> dict[int, float]:
    """Return one finite-sample quantile per group label (Mondrian calibration).

    Raises
    ------
    ValueError
        If a group in ``needed`` has no calibration rows, or too few for the rank.
    """
    missing = sorted(set(np.unique(needed).tolist()) - set(np.unique(groups).tolist()))
    if missing:
        raise ValueError(f"no calibration rows for groups {missing}")
    return {int(g): rank_quantile(scores[groups == g], alpha) for g in np.unique(groups)}


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features(path: Path) -> FloatArray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["X"], dtype=np.float64)


def evaluate(
    holdout_scores: FloatArray,
    quantiles: FloatArray,
    scale: FloatArray,
    targets: FloatArray,
    labels: IntArray,
    threshold: float,
) -> dict[str, Any]:
    """Score one variant of one channel on the holdout split.

    ``quantiles`` is the per-row calibrated quantile (one value for a marginal variant,
    the row's group value for a Mondrian one) and ``scale`` the per-row score scale, so a
    row is covered when ``holdout_scores <= quantiles`` and its half-width is
    ``quantiles * scale``.
    """
    covered = holdout_scores <= quantiles
    half_width = quantiles * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(targets > 0.0, 2.0 * half_width / targets, np.nan)
    every = np.ones(targets.size, dtype=bool)
    return {
        "marginal": regimes.group_coverage(covered, every, relative),
        "median_half_width": float(np.median(half_width)),
        "target_regimes": {
            name: regimes.group_coverage(covered, members, relative)
            for name, members in regimes.target_regimes(targets, threshold).items()
        },
        "input_labels": {
            name: regimes.group_coverage(covered, labels == code, relative)
            for code, name in regimes.INPUT_LABELS.items()
        },
    }


def build_report(certificate_path: Path = regimes.CERTIFICATE_PATH) -> dict[str, Any]:
    """Calibrate every variant and score it per regime on the holdout split."""
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    entry = certificate["certificates"][0]
    alpha = float(certificate["method"]["alpha"])
    weights = ROOT / entry["model"]["weights"]
    calibration = ROOT / entry["calibration"]["split"]
    holdout = ROOT / entry["holdout"]["split"]
    expected = (
        (weights, entry["model"]["weights_file_sha256"]),
        (calibration, entry["calibration"]["split_sha256"]),
        (holdout, entry["holdout"]["split_sha256"]),
    )
    if any(_sha256(path) != digest for path, digest in expected):
        raise ValueError("inputs differ from the certificate's digests")
    data = certificates.residual_scores(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout
    )
    cal_labels = regimes.input_labels(calibration, data.calibration_scores.shape[0])
    test_labels = regimes.input_labels(holdout, data.holdout_scores.shape[0])
    cal_gb = certificates.gyro_bohm_scale(_features(calibration))
    test_gb = certificates.gyro_bohm_scale(_features(holdout))
    reproduced = True
    channels: dict[str, Any] = {}
    for index, channel in enumerate(entry["channels"]):
        r0 = float(channel["radius"])
        cal_abs = data.calibration_scores[:, index]
        test_abs = data.holdout_scores[:, index]
        targets = data.holdout_targets[:, index]
        positive = data.calibration_targets[:, index]
        threshold = float(np.quantile(positive[positive > 0.0], regimes.NEAR_THRESHOLD_QUANTILE))
        cal_pred_scale = np.abs(data.calibration_predictions[:, index]) + r0
        test_pred_scale = np.abs(data.holdout_predictions[:, index]) + r0
        ones = np.ones_like(test_abs)
        cal_class = (data.calibration_predictions[:, index] > r0).astype(np.int64)
        test_class = (data.holdout_predictions[:, index] > r0).astype(np.int64)
        by_class = group_quantiles(cal_abs, cal_class, test_class, alpha)
        by_class_norm = group_quantiles(cal_abs / cal_pred_scale, cal_class, test_class, alpha)
        marginal_q = rank_quantile(cal_abs, alpha)
        mondrian = group_quantiles(cal_abs, cal_labels, test_labels, alpha)
        gb_q = rank_quantile(cal_abs / cal_gb, alpha)
        pred_q = rank_quantile(cal_abs / cal_pred_scale, alpha)
        settings: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {
            "marginal_absolute": (test_abs, np.full_like(test_abs, marginal_q), ones),
            "mondrian_input_label": (
                test_abs,
                np.array([mondrian[int(g)] for g in test_labels], dtype=np.float64),
                ones,
            ),
            "normalised_gyro_bohm": (test_abs / test_gb, np.full_like(test_abs, gb_q), test_gb),
            "normalised_prediction": (
                test_abs / test_pred_scale,
                np.full_like(test_abs, pred_q),
                test_pred_scale,
            ),
            "mondrian_predicted_flux": (
                test_abs,
                np.array([by_class[int(g)] for g in test_class], dtype=np.float64),
                ones,
            ),
            "mondrian_predicted_flux_normalised_prediction": (
                test_abs / test_pred_scale,
                np.array([by_class_norm[int(g)] for g in test_class], dtype=np.float64),
                test_pred_scale,
            ),
        }
        results = {
            name: evaluate(scores, q, scale, targets, test_labels, threshold)
            for name, (scores, q, scale) in settings.items()
        }
        baseline = results["marginal_absolute"]["marginal"]["coverage"]
        reproduced &= marginal_q == r0 and baseline == channel["holdout_empirical_coverage"]
        unstable = targets > threshold
        predicted = data.holdout_predictions[:, index][unstable]
        channels[channel["name"]] = {
            "near_threshold_upper_value": threshold,
            "unstable_diagnostics": {
                "rows": int(unstable.sum()),
                "share_predicted_at_most_r0": float(np.mean(predicted <= r0)),
                "median_prediction_over_target": float(np.median(predicted / targets[unstable])),
            },
            "quantiles": {
                "marginal_absolute": marginal_q,
                "mondrian_input_label": {regimes.INPUT_LABELS[g]: q for g, q in mondrian.items()},
                "normalised_gyro_bohm": gb_q,
                "normalised_prediction": pred_q,
                "mondrian_predicted_flux": {PREDICTED_CLASSES[g]: q for g, q in by_class.items()},
                "mondrian_predicted_flux_normalised_prediction": {
                    PREDICTED_CLASSES[g]: q for g, q in by_class_norm.items()
                },
            },
            "variants": results,
        }
    checks = {"marginal_variant_reproduces_certificate": bool(reproduced)}
    return {
        "schema": SCHEMA,
        "alpha": alpha,
        "certificate_sha256": _sha256(certificate_path),
        "variants": list(VARIANTS),
        "post_hoc_variants": list(POST_HOC_VARIANTS),
        "post_hoc_note": POST_HOC_NOTE,
        "holdout_rows": int(data.holdout_scores.shape[0]),
        "output_unit": data.output_unit,
        "channels": channels,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "source_revision": _git_revision(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render coverage per target regime and the median relative width, per variant."""
    lines = [
        *MARKDOWN_HEADER,
        "# Split-conformal variants for the QLKNN-10D surrogate, per regime",
        "",
        f"Status: **{report['status'].upper()}** (completeness gates). Level "
        f"{1 - report['alpha']:.2f}; holdout rows {report['holdout_rows']}.",
        "",
        "Cells: coverage, then the median relative width 2h/y in brackets (positive targets).",
        f"Variants marked (post hoc) were {POST_HOC_NOTE}.",
        "",
        "| Channel | Variant | Marginal | Zero flux | Near threshold | Unstable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, channel in report["channels"].items():
        for variant, result in channel["variants"].items():
            cells = []
            for regime in regimes.TARGET_REGIMES:
                group = result["target_regimes"][regime]
                width = group["median_relative_width"]
                cell = "—" if group["coverage"] is None else f"{group['coverage']:.3f}"
                cells.append(cell + ("" if width is None else f" ({width:.3g})"))
            label = f"{variant} (post hoc)" if variant in POST_HOC_VARIANTS else variant
            lines.append(
                f"| {name} | {label} | {result['marginal']['coverage']:.3f} | "
                + " | ".join(cells)
                + " |"
            )
    lines += ["", "Surrogate on the unstable rows:", ""]
    for name, channel in report["channels"].items():
        diagnostics = channel["unstable_diagnostics"]
        lines.append(
            f"- {name}: {diagnostics['share_predicted_at_most_r0']:.3f} of "
            f"{diagnostics['rows']} rows predicted at most r0; median prediction/target "
            f"{diagnostics['median_prediction_over_target']:.3f}."
        )
    lines += ["", f"Source revision: `{report['source_revision']}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 when a completeness gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=regimes.CERTIFICATE_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = build_report(args.certificate.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
