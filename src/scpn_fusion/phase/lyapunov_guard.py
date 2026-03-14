# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Lyapunov Stability Guard
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Real-time Lyapunov stability guardrail for phase-sync control loops.

Monitors V(t) = (1/N) Σ (1 − cos(θ_i − Ψ)) over a sliding window.
When the Lyapunov exponent λ > 0 for K consecutive windows, the guard
flags instability and can trigger a halt or parameter clamp.

Interface mirrors DIRECTOR_AI's CoherenceScorer pattern:
    guard.check(theta, psi) → LyapunovVerdict
    verdict.approved / verdict.score / verdict.lambda_exp

Integration with DIRECTOR_AI (optional):
    from director_ai.core import CoherenceScorer
    scorer.review() can call guard.check() to incorporate
    Lyapunov stability into the dual-entropy coherence score.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.phase.kuramoto import lyapunov_exponent, lyapunov_v

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LyapunovVerdict:
    """Result of a stability check — mirrors director_ai.CoherenceScore."""

    v: float
    lambda_exp: float
    approved: bool
    consecutive_violations: int

    @property
    def score(self) -> float:
        """Stability score ∈ [0, 1].  1 = fully stable (λ ≪ 0)."""
        return float(np.clip(1.0 / (1.0 + np.exp(10.0 * self.lambda_exp)), 0.0, 1.0))


class LyapunovGuard:
    """Sliding-window Lyapunov stability monitor.

    Parameters
    ----------
    window : int
        Number of V(t) samples in the sliding window.
    dt : float
        Timestep between samples (for λ computation).
    lambda_threshold : float
        λ above this value counts as a violation.  Default 0 (any growth).
    max_violations : int
        Consecutive violations before guard refuses.
    """

    def __init__(
        self,
        window: int = 50,
        dt: float = 1e-3,
        lambda_threshold: float = 0.0,
        max_violations: int = 3,
    ):
        self._window = window
        self._dt = dt
        self._lambda_threshold = lambda_threshold
        self._max_violations = max_violations
        self._v_buffer: deque[float] = deque(maxlen=window)
        self._consecutive = 0

    def reset(self) -> None:
        self._v_buffer.clear()
        self._consecutive = 0

    def check(self, theta: FloatArray, psi: float) -> LyapunovVerdict:
        """Feed one sample and return stability verdict."""
        v = lyapunov_v(theta, psi)
        self._v_buffer.append(v)

        if len(self._v_buffer) < 2:
            return LyapunovVerdict(v=v, lambda_exp=0.0, approved=True, consecutive_violations=0)

        lam = lyapunov_exponent(list(self._v_buffer), self._dt)

        if lam > self._lambda_threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        approved = self._consecutive < self._max_violations
        if not approved:
            logger.warning(
                "Lyapunov guard REFUSED: λ=%.4f > %.4f for %d consecutive windows",
                lam,
                self._lambda_threshold,
                self._consecutive,
            )

        return LyapunovVerdict(
            v=v,
            lambda_exp=lam,
            approved=approved,
            consecutive_violations=self._consecutive,
        )

    def check_trajectory(self, v_hist: list[float]) -> LyapunovVerdict:
        """Batch check: compute λ from a full V(t) trajectory."""
        if len(v_hist) < 2:
            return LyapunovVerdict(
                v=v_hist[-1] if v_hist else 0.0,
                lambda_exp=0.0,
                approved=True,
                consecutive_violations=0,
            )
        lam = lyapunov_exponent(v_hist, self._dt)
        approved = lam <= self._lambda_threshold
        return LyapunovVerdict(
            v=v_hist[-1],
            lambda_exp=lam,
            approved=approved,
            consecutive_violations=0 if approved else 1,
        )

    def to_director_ai_dict(self, verdict: LyapunovVerdict) -> dict:
        """Export verdict in DIRECTOR_AI AuditLogger format."""
        return {
            "query": "lyapunov_stability_check",
            "response": f"V={verdict.v:.6f}, λ={verdict.lambda_exp:.6f}",
            "approved": verdict.approved,
            "score": verdict.score,
            "h_logical": 0.0,
            "h_factual": float(max(0.0, verdict.lambda_exp)),
            "halt_reason": (
                ""
                if verdict.approved
                else f"λ={verdict.lambda_exp:.4f} > threshold for {verdict.consecutive_violations} windows"
            ),
        }
