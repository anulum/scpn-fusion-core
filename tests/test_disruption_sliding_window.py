# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Synthetic disruption sliding-window detection test (C-M1).

Creates synthetic shot signals with known labels and runs a sliding-window
risk detector using predict_disruption_risk. Asserts recall and FPR.
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.disruption_risk_runtime import predict_disruption_risk

WINDOW_SIZE = 100
THRESHOLD = 0.50


def _make_disruptive_signal(rng, length=500):
    """High-amplitude ramp + noise. Amplitude sufficient to overcome logit bias."""
    t = np.linspace(0, 1, length)
    return 50.0 * t**2 + rng.normal(0, 5.0, length)


def _make_safe_signal(rng, length=500):
    """Low-amplitude stationary noise near zero."""
    return 0.01 * rng.normal(0, 0.01, length)


def _sliding_window_detect(signal, window_size=WINDOW_SIZE, threshold=THRESHOLD):
    """Return True if any window's risk exceeds threshold."""
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n <= window_size:
        return predict_disruption_risk(sig) > threshold
    for start in range(0, n - window_size + 1, window_size // 2):
        window = sig[start : start + window_size]
        if predict_disruption_risk(window) > threshold:
            return True
    return False


def test_sliding_window_recall_and_fpr():
    """2 disruptive + 2 safe shots with known labels; check recall and FPR."""
    rng = np.random.default_rng(42)
    shots = [
        (_make_disruptive_signal(rng), 1),
        (_make_disruptive_signal(rng), 1),
        (_make_safe_signal(rng), 0),
        (_make_safe_signal(rng), 0),
    ]

    labels = np.array([s[1] for s in shots])
    predictions = np.array([int(_sliding_window_detect(s[0])) for s in shots])

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    recall = tp / n_pos
    fpr = fp / n_neg

    assert recall >= 0.5, f"Recall too low: {recall}"
    assert fpr <= 0.5, f"FPR too high: {fpr}"
