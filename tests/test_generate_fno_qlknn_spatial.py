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


class _NonNeuralModel:
    is_neural = False
    _weights = None

    def __init__(self, _weights_path: object) -> None:
        pass


class _NeuralModel:
    is_neural = True
    _weights = {"layer": "weights"}

    def __init__(self, _weights_path: object) -> None:
        pass


class _NeuralModelWithoutWeights:
    is_neural = True
    _weights = None

    def __init__(self, _weights_path: object) -> None:
        pass


def test_require_neural_transport_oracle_rejects_non_neural_model() -> None:
    with pytest.raises(RuntimeError, match="requires trained QLKNN neural weights"):
        generator._require_neural_transport_oracle(  # noqa: SLF001
            "missing.npz", model_cls=_NonNeuralModel
        )


def test_require_neural_transport_oracle_rejects_missing_weights() -> None:
    """Reject a nominally neural model that did not load any weight payload."""
    with pytest.raises(RuntimeError, match="requires trained QLKNN neural weights"):
        generator._require_neural_transport_oracle(  # noqa: SLF001
            "missing.npz", model_cls=_NeuralModelWithoutWeights
        )


def test_require_neural_transport_oracle_accepts_loaded_neural_model() -> None:
    model = generator._require_neural_transport_oracle(  # noqa: SLF001
        "weights.npz", model_cls=_NeuralModel
    )
    assert model.is_neural is True
    assert model._weights is not None


def test_equilibrium_and_profiles_preserve_spatial_contracts() -> None:
    """Produce finite bounded equilibrium fields and physical profile floors."""
    rng = np.random.default_rng(11)
    psi, rho, rr, zz = generator._make_tokamak_equilibrium(  # noqa: SLF001
        5, 4, 6.2, 2.0, 1.8, 0.3, 5.3, 15.0, rng
    )
    profiles = generator._profiles_from_rho(rho, 15.0, 12.0, 8.0, 1.0, 4.0)  # noqa: SLF001

    assert psi.shape == rho.shape == rr.shape == zz.shape == (5, 4)
    assert np.isfinite(psi).all()
    assert float(psi.min()) >= 0.0
    assert float(psi.max()) <= 1.2
    assert float(rho.min()) >= 0.0
    assert float(rho.max()) <= 1.5
    assert set(profiles) == {
        "Te",
        "Ti",
        "ne",
        "q",
        "s_hat",
        "beta_e",
        "grad_Te",
        "grad_Ti",
        "grad_ne",
    }
    assert all(value.shape == (5, 4) for value in profiles.values())
    assert all(np.isfinite(value).all() for value in profiles.values())
    assert float(profiles["Te"].min()) >= 0.1
    assert float(profiles["Ti"].min()) >= 0.1
    assert float(profiles["ne"].min()) >= 0.1


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
