# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GK Spot-Check Scheduler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Scheduler for GK spot-check validation of surrogate transport models.

Decides *when* and *where* to invoke the expensive GK solver based on
configurable strategies: periodic, adaptive (OOD-triggered), or
critical-region policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_ood_detector import OODResult


@dataclass
class SchedulerConfig:
    """Scheduler parameters."""

    strategy: str = "adaptive"  # "periodic" / "adaptive" / "critical_region"
    period: int = 5  # validate every N transport steps (periodic mode)
    budget: int = 5  # max GK calls per transport step
    anchor_rho: tuple[float, ...] = (0.3, 0.5, 0.8)  # always-validated surfaces
    pedestal_rho: float = 0.85  # critical-region inner edge
    axis_rho: float = 0.15  # critical-region axis edge
    chi_change_threshold: float = 0.5  # adaptive: flag if |delta chi|/chi > this


@dataclass
class SpotCheckRequest:
    """Which flux surfaces to validate and why."""

    rho_indices: list[int]
    reasons: dict[int, str]  # index → reason string
    step_number: int


class GKScheduler:
    """Spot-check scheduler for hybrid surrogate+GK transport."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._step = 0
        self._prev_chi_i: NDArray[np.float64] | None = None

    def step(
        self,
        rho: NDArray[np.float64],
        chi_i: NDArray[np.float64],
        ood_results: list[OODResult] | None = None,
    ) -> SpotCheckRequest | None:
        """Decide whether to run GK spot-checks this step.

        Parameters
        ----------
        rho : array
            Radial grid.
        chi_i : array
            Current ion diffusivity profile (from surrogate).
        ood_results : list of OODResult or None
            Per-surface OOD detector results (adaptive mode).

        Returns
        -------
        SpotCheckRequest or None
            None if no validation needed this step.
        """
        self._step += 1
        indices: dict[int, str] = {}

        if self.config.strategy == "periodic":
            if self._step % self.config.period != 0:
                self._prev_chi_i = chi_i.copy()
                return None
            # Add anchor surfaces
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho - rho_val)))
                indices[idx] = "periodic_anchor"

        elif self.config.strategy == "adaptive":
            # OOD-triggered surfaces
            if ood_results is not None:
                for i, result in enumerate(ood_results):
                    if result.is_ood and len(indices) < self.config.budget:
                        indices[i] = f"ood_{result.method}"

            # Large chi change
            if self._prev_chi_i is not None:
                safe_prev = np.maximum(np.abs(self._prev_chi_i), 1e-10)
                rel_change = np.abs(chi_i - self._prev_chi_i) / safe_prev
                big_change = np.where(rel_change > self.config.chi_change_threshold)[0]
                for idx in big_change:
                    if len(indices) < self.config.budget:
                        indices[int(idx)] = "chi_change"

            # Always add anchors if budget allows
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho - rho_val)))
                if idx not in indices and len(indices) < self.config.budget:
                    indices[idx] = "anchor"

            if not indices:
                self._prev_chi_i = chi_i.copy()
                return None

        elif self.config.strategy == "critical_region":
            for i, r in enumerate(rho):
                if (r > self.config.pedestal_rho or r < self.config.axis_rho) and len(
                    indices
                ) < self.config.budget:
                    indices[i] = "critical_region"
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho - rho_val)))
                if idx not in indices and len(indices) < self.config.budget:
                    indices[idx] = "anchor"

        self._prev_chi_i = chi_i.copy()

        if not indices:
            return None

        # Enforce budget
        selected = dict(list(indices.items())[: self.config.budget])
        return SpotCheckRequest(
            rho_indices=list(selected.keys()),
            reasons=selected,
            step_number=self._step,
        )

    def reset(self) -> None:
        self._step = 0
        self._prev_chi_i = None
