# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Public sliding-window disruption-risk scan tests.

Exercises alarm and non-alarm signals, exact window indexing, public-facade
wiring, and fail-closed input validation without duplicating detector logic.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_fusion.control as control
from scpn_fusion.control.disruption_risk_runtime import (
    DisruptionRiskWindowScan,
    scan_disruption_risk_windows,
)

WINDOW_SIZE = 100
THRESHOLD = 0.50
FloatArray = NDArray[np.float64]


def _make_disruptive_signal(rng: np.random.Generator, length: int = 500) -> FloatArray:
    """High-amplitude ramp + noise. Amplitude sufficient to overcome logit bias."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.asarray(50.0 * t**2 + rng.normal(0.0, 5.0, length), dtype=np.float64)


def _make_safe_signal(rng: np.random.Generator, length: int = 500) -> FloatArray:
    """Low-amplitude stationary noise near zero."""
    return np.asarray(0.01 * rng.normal(0.0, 0.01, length), dtype=np.float64)


def test_public_window_scan_classifies_seeded_signals_exactly() -> None:
    """The public scan separates seeded disruptive and safe signals exactly."""
    rng = np.random.default_rng(42)
    shots = [
        (_make_disruptive_signal(rng), 1),
        (_make_disruptive_signal(rng), 1),
        (_make_safe_signal(rng), 0),
        (_make_safe_signal(rng), 0),
    ]

    labels = np.asarray([label for _, label in shots], dtype=np.int64)
    scans = [
        scan_disruption_risk_windows(signal, window_size=WINDOW_SIZE, threshold=THRESHOLD)
        for signal, _ in shots
    ]
    predictions = np.asarray([int(scan.detected) for scan in scans], dtype=np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    recall = tp / n_pos
    fpr = fp / n_neg

    assert recall == 1.0
    assert fpr == 0.0
    assert all(scan.first_alarm_index is not None for scan in scans[:2])
    assert all(scan.first_alarm_index is None for scan in scans[2:])


def test_short_signal_is_scored_once() -> None:
    """A signal shorter than the requested window retains all samples once."""
    scan = scan_disruption_risk_windows([0.1, 0.2, 0.3], window_size=100, threshold=1.0)

    assert isinstance(scan, DisruptionRiskWindowScan)
    assert scan.window_size == 3
    assert scan.stride == 1
    assert scan.threshold == 1.0
    np.testing.assert_array_equal(scan.window_end_indices, np.asarray([3], dtype=np.int64))
    assert scan.risk_scores.shape == (1,)
    assert not scan.detected


def test_unaligned_stride_includes_the_final_sample() -> None:
    """The final window is added when the stride misses its start index."""
    scan = scan_disruption_risk_windows(
        np.linspace(0.0, 1.0, 10),
        window_size=4,
        stride=4,
        threshold=1.0,
    )

    assert scan.stride == 4
    np.testing.assert_array_equal(
        scan.window_end_indices,
        np.asarray([4, 8, 10], dtype=np.int64),
    )
    assert scan.risk_scores.shape == (3,)


def test_control_facade_exports_window_scan_contract() -> None:
    """The stable control facade exposes both the scan function and result type."""
    assert control.scan_disruption_risk_windows is scan_disruption_risk_windows
    assert control.DisruptionRiskWindowScan is DisruptionRiskWindowScan


def test_empty_or_non_finite_signal_is_rejected() -> None:
    """Empty and non-finite signal inputs fail closed before scoring."""
    with pytest.raises(ValueError, match="at least one sample"):
        scan_disruption_risk_windows([])
    with pytest.raises(ValueError, match="signal must be finite"):
        scan_disruption_risk_windows([0.0, float("nan")])


@pytest.mark.parametrize("window_size", [True, 0])
def test_invalid_window_size_is_rejected(window_size: object) -> None:
    """Boolean and non-positive window sizes fail the integer contract."""
    with pytest.raises(ValueError, match="window_size must be an integer >= 1"):
        scan_disruption_risk_windows([0.0], window_size=cast(int, window_size))


@pytest.mark.parametrize("stride", [True, 0])
def test_invalid_stride_is_rejected(stride: object) -> None:
    """Boolean and non-positive strides fail the integer contract."""
    with pytest.raises(ValueError, match="stride must be an integer >= 1"):
        scan_disruption_risk_windows([0.0], stride=cast(int, stride))


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    """Non-finite and out-of-range alarm thresholds fail closed."""
    with pytest.raises(ValueError, match=r"threshold must be finite and in \[0, 1\]"):
        scan_disruption_risk_windows([0.0], threshold=threshold)
