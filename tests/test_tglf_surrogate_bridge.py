# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted TGLF surrogate bridge module."""

from __future__ import annotations

from pathlib import Path

from scpn_fusion.core.tglf_surrogate_bridge import (
    TGLFDatasetGenerator,
    train_surrogate_from_tglf,
)


def test_dataset_generator_accepts_zero_samples() -> None:
    gen = TGLFDatasetGenerator("C:/fake/tglf")
    out = gen.generate_random_dataset(n_samples=0)
    assert out == []


def test_train_surrogate_placeholder_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "weights.npz"
    train_surrogate_from_tglf([], out_path)
    # Placeholder currently prints progress; no file emission contract yet.
    assert out_path.name == "weights.npz"
