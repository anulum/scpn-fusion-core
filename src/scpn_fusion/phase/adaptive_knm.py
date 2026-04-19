# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-Time Adaptive Knm Engine
r"""
Online-adaptive coupling matrix K(t) driven by tokamak diagnostics.

Each control tick, diagnostic signals (β_N, disruption risk, Mirnov RMS,
per-layer order parameters) modulate the baseline Knm through independent
channels:

  1. Beta channel:     K *= (1 + β_scale · β_N)       clamped to β_max_boost
  2. Risk channel:     K[MHD pairs] += risk_gain · risk
  3. Coherence PI:     K[m,m] += PI(R_target − R_m)    per-layer integral control
  4. Rate limiter:     |ΔK_ij| ≤ max_delta per tick
  5. Symmetry/pos:     K = ½(K+Kᵀ), K ≥ 0
  6. Guard veto:       if guard_approved=False → revert to last known-good K

No existing codebase has Kuramoto phase dynamics with online-adaptive
coupling driven by tokamak diagnostics.  This is the novelty claim for
the Frontiers submission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.phase.knm import KnmSpec


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Per-tick plasma diagnostic bundle."""

    R_layer: NDArray[np.float64]
    V_layer: NDArray[np.float64]
    lambda_exp: float
    beta_n: float
    q95: float
    disruption_risk: float
    mirnov_rms: float
    guard_approved: bool


@dataclass(frozen=True)
class AdaptiveKnmConfig:
    """Tuning knobs for each adaptation channel."""

    beta_scale: float = 0.3
    beta_max_boost: float = 0.5
    risk_pairs: tuple[tuple[int, int], ...] = ((2, 5), (3, 5), (2, 4))
    risk_gain: float = 0.4
    coherence_Kp: float = 0.15
    coherence_Ki: float = 0.02
    coherence_R_target: float = 0.6
    coherence_max_boost: float = 0.3
    max_delta_per_tick: float = 0.02
    revert_on_guard_refusal: bool = True


class AdaptiveKnmEngine:
    """Diagnostic-driven online adaptation of the Knm coupling matrix.

    Holds a baseline K from a KnmSpec and produces K_adapted each tick.
    """

    def __init__(
        self,
        baseline_spec: KnmSpec,
        config: AdaptiveKnmConfig | None = None,
    ) -> None:
        self._baseline = np.asarray(baseline_spec.K, dtype=np.float64).copy()
        self._L = self._baseline.shape[0]
        self._cfg = config or AdaptiveKnmConfig()
        self._K_current = self._baseline.copy()
        self._K_last_good = self._baseline.copy()
        self._integral = np.zeros(self._L, dtype=np.float64)

    def update(self, snap: DiagnosticSnapshot) -> NDArray[np.float64]:
        """Apply all adaptation channels and return K_adapted."""
        cfg = self._cfg

        # Guard veto: revert before computing if previous tick was refused
        if not snap.guard_approved and cfg.revert_on_guard_refusal:
            self._K_current[:] = self._K_last_good
            self._integral[:] = 0.0
            return self._K_current.copy()

        K_new = self._baseline.copy()

        # 1. Beta channel: scale entire matrix
        beta_boost = min(cfg.beta_scale * snap.beta_n, cfg.beta_max_boost)
        K_new *= 1.0 + beta_boost

        # 2. Risk channel: amplify MHD-relevant pairs
        for i, j in cfg.risk_pairs:
            if i < self._L and j < self._L:
                delta = cfg.risk_gain * snap.disruption_risk
                K_new[i, j] += delta
                K_new[j, i] += delta

        # 3. Coherence PI: per-layer diagonal boost
        R = np.asarray(snap.R_layer, dtype=np.float64)
        error = cfg.coherence_R_target - R[: self._L]
        self._integral += cfg.coherence_Ki * error
        np.clip(self._integral, 0.0, cfg.coherence_max_boost, out=self._integral)
        for m in range(self._L):
            boost = cfg.coherence_Kp * max(error[m], 0.0) + self._integral[m]
            K_new[m, m] += min(boost, cfg.coherence_max_boost)

        # 4. Invariants: symmetry + non-negativity
        K_new = 0.5 * (K_new + K_new.T)
        np.maximum(K_new, 0.0, out=K_new)

        # 5. Rate limit: per-element clamp
        dK = K_new - self._K_current
        np.clip(dK, -cfg.max_delta_per_tick, cfg.max_delta_per_tick, out=dK)
        K_new = self._K_current + dK

        # Re-enforce invariants after rate limiting
        K_new = 0.5 * (K_new + K_new.T)
        np.maximum(K_new, 0.0, out=K_new)

        # 6. Commit — K_last_good tracks the most recent approved K
        self._K_current[:] = K_new
        self._K_last_good[:] = K_new
        return self._K_current.copy()

    def reset(self) -> None:
        """Revert to baseline and clear integral state."""
        self._K_current[:] = self._baseline
        self._K_last_good[:] = self._baseline
        self._integral[:] = 0.0

    @property
    def K_current(self) -> NDArray[np.float64]:
        return self._K_current.copy()

    @property
    def adaptation_summary(self) -> dict:
        """Snapshot of engine state for dashboard export."""
        diff = self._K_current - self._baseline
        return {
            "L": self._L,
            "K_mean": float(self._K_current.mean()),
            "K_max": float(self._K_current.max()),
            "delta_frobenius": float(np.linalg.norm(diff)),
            "delta_max_element": float(np.abs(diff).max()),
            "integral_sum": float(self._integral.sum()),
        }
