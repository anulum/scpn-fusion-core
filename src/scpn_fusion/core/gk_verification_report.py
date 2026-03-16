# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GK Verification Report
"""
Per-session verification report for hybrid surrogate+GK transport.

Tracks spot-check statistics, correction factors, OOD trigger rates,
and error distributions across a simulation session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scpn_fusion.core.gk_corrector import CorrectionRecord


@dataclass
class VerificationReport:
    """Accumulated verification statistics for one simulation session."""

    total_steps: int = 0
    steps_verified: int = 0
    total_spot_checks: int = 0
    ood_triggers: int = 0
    records: list[CorrectionRecord] = field(default_factory=list)
    correction_factors: list[float] = field(default_factory=list)

    @property
    def verification_fraction(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.steps_verified / self.total_steps

    @property
    def max_rel_error(self) -> float:
        if not self.records:
            return 0.0
        return max(abs(r.rel_error_chi_i) for r in self.records)

    @property
    def mean_rel_error(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([abs(r.rel_error_chi_i) for r in self.records]))

    def add_step(self, verified: bool, n_spot_checks: int = 0, n_ood: int = 0) -> None:
        self.total_steps += 1
        if verified:
            self.steps_verified += 1
        self.total_spot_checks += n_spot_checks
        self.ood_triggers += n_ood

    def add_records(self, records: list[CorrectionRecord]) -> None:
        self.records.extend(records)

    def add_correction_factor(self, factor: float) -> None:
        self.correction_factors.append(factor)

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "steps_verified": self.steps_verified,
            "verification_fraction": round(self.verification_fraction, 4),
            "total_spot_checks": self.total_spot_checks,
            "ood_triggers": self.ood_triggers,
            "max_rel_error_chi_i": round(self.max_rel_error, 4),
            "mean_rel_error_chi_i": round(self.mean_rel_error, 4),
            "n_correction_records": len(self.records),
            "mean_correction_factor": (
                round(float(np.mean(self.correction_factors)), 4)
                if self.correction_factors
                else 0.0
            ),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text
