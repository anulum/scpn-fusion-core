# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO QLKNN Spatial Generator Tests
"""Production-contract tests for the FNO/QLKNN spatial-data generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import neural_transport
from tools import generate_fno_qlknn_spatial as generator


class _NeuralModel:
    is_neural = True
    _weights = {"layer": "weights"}

    def __init__(self, _weights_path: object) -> None:
        pass


ROOT = Path(__file__).resolve().parents[1]
TRAINED_WEIGHTS = ROOT / "weights" / "neural_transport_qlknn.npz"


def test_generate_refuses_without_a_trained_oracle_and_writes_nothing(tmp_path: Path) -> None:
    """Without trained QLKNN weights the analytic fallback is not accepted as the oracle."""
    output_dir = tmp_path / "dataset"

    with pytest.raises(RuntimeError, match="requires trained QLKNN neural weights"):
        generator.generate(tmp_path / "missing.npz", output_dir, n_equilibria=2, grid_size=4)

    assert not output_dir.exists()


def test_generate_writes_bounded_spatial_splits_from_the_trained_oracle(tmp_path: Path) -> None:
    """The trained oracle yields finite flux fields on bounded normalised-psi equilibria."""
    output_dir = tmp_path / "dataset"

    generator.generate(TRAINED_WEIGHTS, output_dir, n_equilibria=4, grid_size=5, seed=11)

    splits = {}
    for split in ("train", "val"):
        with np.load(output_dir / f"{split}.npz", allow_pickle=False) as archive:
            splits[split] = (np.asarray(archive["X"]), np.asarray(archive["Y"]))
    assert sum(x.shape[0] for x, _ in splits.values()) == 4
    for psi, chi in splits.values():
        assert psi.shape[1:] == chi.shape[1:] == (5, 5)
        assert np.isfinite(psi).all() and np.isfinite(chi).all()
        assert float(psi.min()) >= 0.0 and float(psi.max()) <= 1.2
        assert float(chi.min()) >= 0.0
    assert (output_dir / "metadata.json").is_file()


@pytest.mark.parametrize(
    ("n_equilibria", "grid_size", "message"),
    [
        (0, 8, "n_equilibria must be >= 2"),
        (1, 8, "n_equilibria must be >= 2"),
        (2, 1, "grid_size must be >= 2"),
    ],
)
def test_generate_rejects_invalid_dataset_dimensions_before_writes(
    tmp_path: Path, n_equilibria: int, grid_size: int, message: str
) -> None:
    """Reject dimensions that cannot produce two spatial train/validation splits."""
    output_dir = tmp_path / "dataset"
    with pytest.raises(ValueError, match=message):
        generator.generate(
            Path("missing.npz"),
            output_dir,
            n_equilibria=n_equilibria,
            grid_size=grid_size,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("invalid_kind", ["shape", "nonfinite"])
def test_generate_rejects_invalid_oracle_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_kind: str
) -> None:
    """Fail before artifact writes on malformed or non-finite oracle batches."""
    monkeypatch.setattr(
        generator,
        "_require_neural_transport_oracle",
        lambda _weights: _NeuralModel(_weights),
    )

    def _invalid_forward(x_batch: Any, _weights: Any) -> np.ndarray[Any, Any]:
        if invalid_kind == "shape":
            return np.zeros((len(x_batch), 2), dtype=np.float64)
        result = np.zeros((len(x_batch), 3), dtype=np.float64)
        result[0, 0] = np.nan
        return result

    monkeypatch.setattr(neural_transport, "_mlp_forward", _invalid_forward)
    output_dir = tmp_path / "dataset"
    expected = "returned shape" if invalid_kind == "shape" else "non-finite"
    with pytest.raises(RuntimeError, match=expected):
        generator.generate(Path("weights.npz"), output_dir, n_equilibria=2, grid_size=2)
    assert not output_dir.exists()


def test_generate_real_qlknn_dataset_and_progress_boundary(tmp_path: Path) -> None:
    """Generate real QLKNN-derived train/validation NPZs through the 50-row progress path."""
    output_dir = tmp_path / "dataset"
    weights = generator.REPO_ROOT / "weights" / "neural_transport_qlknn.npz"
    generator.generate(weights, output_dir, n_equilibria=50, grid_size=2, seed=17)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["neural_oracle"] is True
    assert metadata["n_equilibria"] == 50
    assert metadata["n_train"] == 42
    assert metadata["n_val"] == 8
    assert metadata["grid_size"] == 2
    assert metadata["seed"] == 17
    for name, expected_rows in (("train.npz", 42), ("val.npz", 8)):
        with np.load(output_dir / name, allow_pickle=False) as payload:
            assert payload.files == ["X", "Y"]
            assert payload["X"].shape == payload["Y"].shape == (expected_rows, 2, 2)
            assert payload["X"].dtype == payload["Y"].dtype == np.float32
            assert np.isfinite(payload["X"]).all()
            assert np.isfinite(payload["Y"]).all()
            assert float(payload["X"].min()) >= 0.0
            assert float(payload["X"].max()) <= 1.0
            assert float(payload["Y"].min()) >= 0.0
            assert float(payload["Y"].max()) <= 1.0


def test_main_parses_cli_and_rejects_invalid_sample_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route CLI arguments into the production dimension guard before writes."""
    output_dir = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_fno_qlknn_spatial.py",
            "--weights",
            "missing.npz",
            "--output-dir",
            str(output_dir),
            "--n-equilibria",
            "0",
            "--grid-size",
            "8",
            "--seed",
            "9",
        ],
    )
    with pytest.raises(ValueError, match="n_equilibria must be >= 2"):
        generator.main()
    assert not output_dir.exists()
