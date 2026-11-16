#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — class-conditional conformal stability sets for the QLKNN-10D surrogate
"""Bound the QLKNN-10D surrogate's false-stable rate with class-conditional conformal sets.

The variant comparison found that the surrogate predicts at most the certificate radius on
about two thirds of the truly unstable holdout rows, so no flux interval centred on its
prediction covers them. This experiment was designed after that finding; its own settings
below were fixed before it was run.

Per output channel, with the stability label of a row being ``stable`` (zero target flux)
or ``unstable`` (positive target flux):

1. **Stability set.** Class-conditional (Mondrian by the true label) split conformal on
   the calibration split, with the prediction ``y_hat`` as the score: ``unstable`` enters a
   row's set when ``-y_hat`` is at most the ``1 - alpha`` quantile of ``-y_hat`` over the
   calibration's unstable rows, and ``stable`` enters when ``y_hat`` is at most that
   quantile of ``y_hat`` over its stable rows. Under exchangeability within each class, a
   row's true label is in its set with probability at least ``1 - alpha``.
2. **Flux interval.** ``unstable`` in the set gives ``[max(0, y_hat - q_u), y_hat + q_u]``,
   with ``q_u`` the ``1 - alpha`` quantile of ``|y - y_hat|`` over the calibration's
   unstable rows; ``stable`` alone gives ``[0, 0]``; both give ``[0, y_hat + q_u]``; an
   empty set abstains and covers nothing. A truly unstable row is then covered with
   probability at least ``1 - 2 alpha`` (a union bound over the two steps).

The report gives, on the untouched holdout split and per true class, the set coverage, the
set-size shares (empty, one label, both), and the flux coverage with Wilson intervals, plus
the median interval width. A robustness section repeats the experiment on
:data:`RESPLIT_SEEDS` random halvings of the union of both splits and summarises the per-class
coverages, so a single split's deviation can be judged. It has no pass or fail status: it
stops, instead of reporting,
when the inputs differ from the certificate's digests or a class has too few calibration
rows for its rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from validation import qlknn_conformal_variants as variants

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
REPORT_PATH = ROOT / "validation/reports/qlknn_conformal_stability_sets.json"
MARKDOWN_PATH = ROOT / "validation/reports/qlknn_conformal_stability_sets.md"
SCHEMA = "scpn-fusion-core.qlknn-conformal-stability-sets.v1"
RESPLIT_SEEDS = tuple(range(20))
DESIGN_NOTE = (
    "designed after the variant comparison (9bd8a7a7, report b2f13559) found the false-stable "
    "bias; the settings of this experiment were fixed before it was run"
)
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — conformal stability sets for the QLKNN-10D surrogate -->",
)


def stability_sets(
    calibration_predictions: FloatArray,
    calibration_unstable: BoolArray,
    holdout_predictions: FloatArray,
    alpha: float,
) -> tuple[BoolArray, BoolArray, dict[str, float]]:
    """Return, per holdout row, whether ``stable`` and ``unstable`` are in its set.

    Raises
    ------
    ValueError
        If either class has too few calibration rows for the finite-sample rank.
    """
    q_unstable = variants.rank_quantile(-calibration_predictions[calibration_unstable], alpha)
    q_stable = variants.rank_quantile(calibration_predictions[~calibration_unstable], alpha)
    has_unstable = -holdout_predictions <= q_unstable
    has_stable = holdout_predictions <= q_stable
    return has_stable, has_unstable, {"stable": q_stable, "unstable": q_unstable}


def flux_intervals(
    predictions: FloatArray, has_stable: BoolArray, has_unstable: BoolArray, q_flux: float
) -> tuple[FloatArray, FloatArray]:
    """Return the lower and upper flux bounds implied by each row's stability set.

    An empty set abstains: its interval is empty (lower above upper) and covers nothing.
    """
    lower = np.where(has_unstable, np.maximum(0.0, predictions - q_flux), 0.0)
    upper = np.where(has_unstable, predictions + q_flux, 0.0)
    lower = np.where(has_stable & has_unstable, 0.0, lower)
    empty = ~(has_stable | has_unstable)
    return np.where(empty, np.inf, lower), np.where(empty, -np.inf, upper)


def score_channel(
    calibration_predictions: FloatArray,
    calibration_targets: FloatArray,
    calibration_scores: FloatArray,
    holdout_predictions: FloatArray,
    holdout_targets: FloatArray,
    alpha: float,
) -> dict[str, Any]:
    """Calibrate one channel's sets and intervals and return the holdout arrays they imply."""
    cal_unstable = calibration_targets > 0.0
    has_stable, has_unstable, set_quantiles = stability_sets(
        calibration_predictions, cal_unstable, holdout_predictions, alpha
    )
    q_flux = variants.rank_quantile(calibration_scores[cal_unstable], alpha)
    lower, upper = flux_intervals(holdout_predictions, has_stable, has_unstable, q_flux)
    return {
        "has_stable": has_stable,
        "has_unstable": has_unstable,
        "set_quantiles": set_quantiles,
        "unstable_flux_quantile": q_flux,
        "covered": (lower <= holdout_targets) & (holdout_targets <= upper),
        "width": np.where(np.isfinite(upper - lower), upper - lower, np.nan),
    }


def resplit_summary(
    predictions: FloatArray, targets: FloatArray, scores: FloatArray, alpha: float
) -> dict[str, dict[str, float]]:
    """Repeat one channel on random halvings of the pooled rows; summarise each coverage."""
    values: dict[str, list[float]] = {"unstable_set": [], "stable_set": [], "unstable_flux": []}
    for seed in RESPLIT_SEEDS:
        order = np.random.default_rng(seed).permutation(targets.size)
        cal, test = order[: targets.size // 2], order[targets.size // 2 :]
        result = score_channel(
            predictions[cal], targets[cal], scores[cal], predictions[test], targets[test], alpha
        )
        unstable = targets[test] > 0.0
        values["unstable_set"].append(float(np.mean(result["has_unstable"][unstable])))
        values["stable_set"].append(float(np.mean(result["has_stable"][~unstable])))
        values["unstable_flux"].append(float(np.mean(result["covered"][unstable])))
    return {
        name: {
            "mean": float(np.mean(series)),
            "std": float(np.std(series, ddof=1)),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
        }
        for name, series in values.items()
    }


def _class_summary(
    members: BoolArray,
    in_set: BoolArray,
    has_stable: BoolArray,
    has_unstable: BoolArray,
    covered: BoolArray,
    width: FloatArray,
) -> dict[str, Any]:
    total = int(members.sum())
    widths = width[members]
    widths = widths[np.isfinite(widths)]
    size = has_stable.astype(np.int64) + has_unstable.astype(np.int64)
    hits, flux = int(in_set[members].sum()), int(covered[members].sum())
    return {
        "n": total,
        "set_coverage": hits / total,
        "set_coverage_wilson95": list(regimes.wilson_interval(hits, total)),
        "set_size_share": {str(k): float(np.mean(size[members] == k)) for k in (0, 1, 2)},
        "flux_coverage": flux / total,
        "flux_coverage_wilson95": list(regimes.wilson_interval(flux, total)),
        "median_interval_width": float(np.median(widths)) if widths.size else None,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(certificate_path: Path = regimes.CERTIFICATE_PATH) -> dict[str, Any]:
    """Calibrate the stability sets and flux intervals and score them on the holdout split."""
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
    channels: dict[str, Any] = {}
    for index, channel in enumerate(entry["channels"]):
        targets = data.holdout_targets[:, index]
        result = score_channel(
            data.calibration_predictions[:, index],
            data.calibration_targets[:, index],
            data.calibration_scores[:, index],
            data.holdout_predictions[:, index],
            targets,
            alpha,
        )
        has_stable, has_unstable = result["has_stable"], result["has_unstable"]
        covered, width = result["covered"], result["width"]
        test_unstable = targets > 0.0
        pooled = (
            np.concatenate(
                [data.calibration_predictions[:, index], data.holdout_predictions[:, index]]
            ),
            np.concatenate([data.calibration_targets[:, index], targets]),
            np.concatenate([data.calibration_scores[:, index], data.holdout_scores[:, index]]),
        )
        channels[channel["name"]] = {
            "set_quantiles": result["set_quantiles"],
            "unstable_flux_quantile": result["unstable_flux_quantile"],
            "resplits": resplit_summary(*pooled, alpha),
            "unstable": _class_summary(
                test_unstable, has_unstable, has_stable, has_unstable, covered, width
            ),
            "stable": _class_summary(
                ~test_unstable, has_stable, has_stable, has_unstable, covered, width
            ),
            "abstention_share": float(np.mean(~(has_stable | has_unstable))),
        }
    return {
        "schema": SCHEMA,
        "alpha": alpha,
        "design_note": DESIGN_NOTE,
        "guarantees": {
            "set": "true label in the set with probability >= 1 - alpha, per class",
            "unstable_flux": "unstable rows covered with probability >= 1 - 2 alpha",
        },
        "certificate_sha256": _sha256(certificate_path),
        "holdout_rows": int(data.holdout_scores.shape[0]),
        "output_unit": data.output_unit,
        "channels": channels,
        "source_revision": variants.git_revision(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render per-class set and flux coverage per channel."""
    alpha = report["alpha"]
    lines = [
        *MARKDOWN_HEADER,
        "# Class-conditional conformal stability sets for the QLKNN-10D surrogate",
        "",
        f"Set level {1 - alpha:.2f} per class; "
        f"unstable flux bound {1 - 2 * alpha:.2f}. This experiment was {report['design_note']}.",
        "",
        "| Channel | Class | n | Set coverage | Both labels | Flux coverage | Median width |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, channel in report["channels"].items():
        for label in ("unstable", "stable"):
            row = channel[label]
            width = row["median_interval_width"]
            lines.append(
                f"| {name} | {label} | {row['n']} | {row['set_coverage']:.3f} "
                f"| {row['set_size_share']['2']:.3f} | {row['flux_coverage']:.3f} "
                f"| {'—' if width is None else f'{width:.3g}'} |"
            )
    lines += ["", "Widths in m^2/s, on the rows of that true class. Abstention share:", ""]
    for name, channel in report["channels"].items():
        lines.append(f"- {name}: {channel['abstention_share']:.3f}.")
    lines += [
        "",
        f"Random halvings of the pooled rows ({len(RESPLIT_SEEDS)} seeds): mean [min, max].",
        "",
        "| Channel | Unstable set | Stable set | Unstable flux |",
        "|---|---:|---:|---:|",
    ]
    for name, channel in report["channels"].items():
        cells = [
            f"{s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}]"
            for s in (
                channel["resplits"][k] for k in ("unstable_set", "stable_set", "unstable_flux")
            )
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"Source revision: `{report['source_revision']}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=regimes.CERTIFICATE_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = build_report(args.certificate.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
