# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Model-Selection Metrics
"""Per-channel, stratified, closure, eligibility and timing metrics."""

from __future__ import annotations

import time
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.tglf_surrogate_candidates import TGLFRegressionCandidate
from scpn_fusion.io._tglf_model_selection_data import (
    COMPOSITIONS,
    FLUX_FIELDS,
    SPECIES_SLOTS,
    STRATA,
    BoolArray,
    FloatArray,
    TGLFModelStudyData,
)

CHANNEL_NRMSE_MAX: Final = 1.0
ABS_NORMALISED_BIAS_MAX: Final = 0.5
SIGN_AGREEMENT_MIN: Final = 0.75
THRESHOLD_NRMSE_MAX: Final = 1.25
CLOSURE_P95_MAX: Final = 0.20
NEAR_ZERO_FRACTION: Final = 0.05


def indices(labels: tuple[str, ...], value: str) -> NDArray[np.int64]:
    """Return stable integer indices for one categorical label."""
    return np.asarray(
        [index for index, label in enumerate(labels) if label == value],
        dtype=np.int64,
    )


def percentile(values: list[float], quantile: float) -> float:
    """Return a float64 percentile for a non-empty value sequence."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def channel_metrics(
    truth: FloatArray,
    prediction: FloatArray,
    active: BoolArray,
    scales: FloatArray,
    target_names: tuple[str, ...],
) -> dict[str, dict[str, float | int | None]]:
    """Compute signed scale-aware metrics independently for each target."""
    result: dict[str, dict[str, float | int | None]] = {}
    for target_index, target_name in enumerate(target_names):
        mask = active[:, target_index]
        if not np.any(mask):
            result[target_name] = {
                "active_rows": 0,
                "rmse": None,
                "mae": None,
                "normalised_rmse": None,
                "normalised_bias": None,
                "relative_mae": None,
                "sign_agreement": None,
            }
            continue
        expected = truth[mask, target_index]
        observed = prediction[mask, target_index]
        error = observed - expected
        scale = float(scales[target_index])
        rmse = float(np.sqrt(np.mean(error * error)))
        mae = float(np.mean(np.abs(error)))
        nonzero = np.abs(expected) >= NEAR_ZERO_FRACTION * scale
        sign_agreement = (
            float(np.mean(np.sign(observed[nonzero]) == np.sign(expected[nonzero])))
            if np.any(nonzero)
            else None
        )
        result[target_name] = {
            "active_rows": int(expected.size),
            "rmse": rmse,
            "mae": mae,
            "normalised_rmse": rmse / scale,
            "normalised_bias": float(np.mean(error)) / scale,
            "relative_mae": float(
                np.mean(np.abs(error) / np.maximum(np.abs(expected), NEAR_ZERO_FRACTION * scale))
            ),
            "sign_agreement": sign_agreement,
        }
    return result


def channel_summary(
    metrics: dict[str, dict[str, float | int | None]],
) -> dict[str, float]:
    """Summarise active-channel normalised RMSE without hiding the worst channel."""
    values = [
        float(item["normalised_rmse"])
        for item in metrics.values()
        if item["normalised_rmse"] is not None
    ]
    if not values:
        raise ValueError("channel summary has no active target")
    return {
        "median_normalised_rmse": float(np.median(values)),
        "p95_normalised_rmse": percentile(values, 95.0),
        "worst_normalised_rmse": max(values),
    }


def closure_ratio(components: FloatArray) -> FloatArray:
    """Return absolute summed flow divided by summed component magnitude."""
    numerator = np.abs(np.sum(components, axis=1))
    denominator = np.sum(np.abs(components), axis=1)
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 1.0e-30)
    return cast(FloatArray, result)


def closure_metrics(
    data: TGLFModelStudyData,
    row_indices: NDArray[np.int64],
    truth: FloatArray,
    prediction: FloatArray,
) -> dict[str, dict[str, float]]:
    """Measure charge-weighted particle and exchange closure distributions."""
    particle_indices = np.asarray(
        [1 + slot * len(FLUX_FIELDS) for slot in range(SPECIES_SLOTS)],
        dtype=np.int64,
    )
    exchange_indices = particle_indices + 3
    charges = data.charges[row_indices]

    def summarize(values: FloatArray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95.0)),
            "maximum": float(np.max(values)),
        }

    return {
        "charge_weighted_particle_truth": summarize(
            closure_ratio(truth[:, particle_indices] * charges)
        ),
        "charge_weighted_particle_prediction": summarize(
            closure_ratio(prediction[:, particle_indices] * charges)
        ),
        "exchange_truth": summarize(closure_ratio(truth[:, exchange_indices])),
        "exchange_prediction": summarize(closure_ratio(prediction[:, exchange_indices])),
    }


def evaluate_rows(
    data: TGLFModelStudyData,
    row_indices: NDArray[np.int64],
    prediction: FloatArray,
    target_scales: FloatArray,
) -> dict[str, Any]:
    """Evaluate one immutable split with channel and categorical detail."""
    truth = data.targets[row_indices]
    active = data.active_targets[row_indices]
    if prediction.shape != truth.shape:
        raise ValueError(f"prediction shape {prediction.shape} != truth shape {truth.shape}")
    finite_rows = np.all(np.isfinite(prediction), axis=1)
    failed_rows = int(np.count_nonzero(~finite_rows))
    if failed_rows:
        return {
            "rows": int(row_indices.size),
            "failed_rows": failed_rows,
            "failure_rate": failed_rows / float(row_indices.size),
            "channels": {},
            "summary": None,
            "by_stratum": {},
            "by_composition": {},
            "closure": None,
        }
    channels = channel_metrics(truth, prediction, active, target_scales, data.target_names)
    by_stratum: dict[str, Any] = {}
    for stratum in STRATA:
        local = np.asarray(
            [
                offset
                for offset, index in enumerate(row_indices)
                if data.strata[int(index)] == stratum
            ],
            dtype=np.int64,
        )
        if local.size:
            metrics = channel_metrics(
                truth[local], prediction[local], active[local], target_scales, data.target_names
            )
            by_stratum[stratum] = {
                "rows": int(local.size),
                "channels": metrics,
                "summary": channel_summary(metrics),
            }
    by_composition: dict[str, Any] = {}
    for composition in COMPOSITIONS:
        local = np.asarray(
            [
                offset
                for offset, index in enumerate(row_indices)
                if data.compositions[int(index)] == composition
            ],
            dtype=np.int64,
        )
        if local.size:
            metrics = channel_metrics(
                truth[local], prediction[local], active[local], target_scales, data.target_names
            )
            by_composition[composition] = {
                "rows": int(local.size),
                "channels": metrics,
                "summary": channel_summary(metrics),
            }
    return {
        "rows": int(row_indices.size),
        "failed_rows": 0,
        "failure_rate": 0.0,
        "channels": channels,
        "summary": channel_summary(channels),
        "by_stratum": by_stratum,
        "by_composition": by_composition,
        "closure": closure_metrics(data, row_indices, truth, prediction),
    }


def eligibility(evaluation: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply every predeclared calibration or test admission threshold."""
    reasons: list[str] = []
    if evaluation.get("failed_rows") != 0:
        return False, ["non-finite prediction rows"]
    channels = cast(dict[str, dict[str, float | int | None]], evaluation["channels"])
    for name, metrics in channels.items():
        nrmse = metrics["normalised_rmse"]
        bias = metrics["normalised_bias"]
        sign = metrics["sign_agreement"]
        if nrmse is not None and float(nrmse) > CHANNEL_NRMSE_MAX:
            reasons.append(f"{name} normalised RMSE exceeds {CHANNEL_NRMSE_MAX}")
        if bias is not None and abs(float(bias)) > ABS_NORMALISED_BIAS_MAX:
            reasons.append(f"{name} absolute normalised bias exceeds {ABS_NORMALISED_BIAS_MAX}")
        if sign is not None and float(sign) < SIGN_AGREEMENT_MIN:
            reasons.append(f"{name} sign agreement is below {SIGN_AGREEMENT_MIN}")
    threshold = cast(dict[str, Any], evaluation["by_stratum"]).get("threshold")
    if not isinstance(threshold, dict):
        reasons.append("threshold stratum is absent")
    else:
        threshold_channels = cast(dict[str, dict[str, float | int | None]], threshold["channels"])
        for name, metrics in threshold_channels.items():
            nrmse = metrics["normalised_rmse"]
            if nrmse is not None and float(nrmse) > THRESHOLD_NRMSE_MAX:
                reasons.append(f"threshold {name} normalised RMSE exceeds {THRESHOLD_NRMSE_MAX}")
    closure = cast(dict[str, dict[str, float]], evaluation["closure"])
    if closure["charge_weighted_particle_prediction"]["p95"] > CLOSURE_P95_MAX:
        reasons.append(f"particle closure p95 exceeds {CLOSURE_P95_MAX}")
    if closure["exchange_prediction"]["p95"] > CLOSURE_P95_MAX:
        reasons.append(f"exchange closure p95 exceeds {CLOSURE_P95_MAX}")
    return not reasons, reasons


def candidate_rank(report: dict[str, Any]) -> tuple[float, float, float, float, int, float]:
    """Return the frozen lexicographic calibration-only selection key."""
    summary = cast(dict[str, float], cast(dict[str, Any], report["calibration"])["summary"])
    closure = cast(
        dict[str, dict[str, float]],
        cast(dict[str, Any], report["calibration"])["closure"],
    )
    latency = cast(dict[str, float], report["latency_orientation"])
    return (
        summary["median_normalised_rmse"],
        summary["p95_normalised_rmse"],
        summary["worst_normalised_rmse"],
        closure["charge_weighted_particle_prediction"]["p95"]
        + closure["exchange_prediction"]["p95"],
        cast(int, report["state_bytes"]),
        latency["median_microseconds_per_row"],
    )


def latency_orientation(
    candidate: TGLFRegressionCandidate,
    features: FloatArray,
    *,
    repeats: int,
) -> dict[str, float | int]:
    """Measure loaded-host prediction latency without admitting a performance claim."""
    if repeats < 3:
        raise ValueError("latency repeats must be at least three")
    for _ in range(3):
        candidate.predict(features)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        candidate.predict(features)
        samples.append((time.perf_counter_ns() - start) / (1000.0 * features.shape[0]))
    return {
        "batch_rows": int(features.shape[0]),
        "repeats": repeats,
        "median_microseconds_per_row": float(np.median(samples)),
        "p95_microseconds_per_row": percentile(samples, 95.0),
        "minimum_microseconds_per_row": min(samples),
        "maximum_microseconds_per_row": max(samples),
    }


__all__ = ["candidate_rank", "eligibility", "evaluate_rows", "indices", "latency_orientation"]
