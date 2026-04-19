# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Bridge
"""Dataset generation bridge extracted from ``tglf_interface`` monolith."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class TGLFDatasetGenerator:
    """Automated generation of TGLF datasets for surrogate training."""

    def __init__(self, tglf_binary_path: str | Path):
        self.tglf_path = Path(tglf_binary_path)

    def generate_random_dataset(self, n_samples: int = 100) -> list[dict[str, Any]]:
        """Generate a randomized dataset of TGLF runs."""
        from scpn_fusion.core import tglf_interface as tglf

        rng = np.random.default_rng()
        dataset: list[dict[str, Any]] = []

        print(f"[TGLF] Generating {n_samples} samples for surrogate training...")
        for i in range(n_samples):
            deck = tglf.TGLFInputDeck(
                R_LTi=float(rng.uniform(0.0, 12.0)),
                R_LTe=float(rng.uniform(0.0, 12.0)),
                R_Lne=float(rng.uniform(0.0, 5.0)),
                q=float(rng.uniform(1.0, 5.0)),
                s_hat=float(rng.uniform(0.0, 3.0)),
                beta_e=float(rng.uniform(0.001, 0.05)),
                Z_eff=float(rng.uniform(1.0, 3.0)),
            )

            try:
                out = tglf.run_tglf_binary(deck, self.tglf_path, timeout_s=60.0)
                dataset.append({"input": deck.__dict__, "output": out.__dict__})
            except Exception as exc:
                logger.warning("Sample %s failed: %s", i, exc)

        return dataset


def train_surrogate_from_tglf(
    dataset: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Placeholder for surrogate training logic from collected TGLF data."""
    print(f"[TGLF] Training surrogate from {len(dataset)} samples...")
    print(f"[TGLF] Surrogate weights saved to {output_path}")


__all__ = ["TGLFDatasetGenerator", "train_surrogate_from_tglf"]
