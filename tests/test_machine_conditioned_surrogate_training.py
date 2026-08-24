# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — machine-conditioned successor training tests
"""Contracts for authenticated loading and recovery-safe train-only PCA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from scpn_fusion.io.machine_conditioned_surrogate_training import (
    deterministic_split,
    fit_streaming_randomized_pca,
    load_machine_conditioned_training_data,
)
from scpn_fusion.io.machine_conditioned_surrogate_cli import run_training_cli
from tools.train_machine_conditioned_equilibrium_surrogate import (
    _load_optimizer_checkpoint,
    _save_optimizer_checkpoint,
    adam_step,
    init_mlp_params,
)

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "validation/reference/iter_machine_conditioned_v2_n3_seed20260822_33x33"


def test_successor_cli_adapter_is_directly_linked() -> None:
    assert callable(run_training_cli)


def _low_rank_fields(*, validation_offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(17)
    factors = rng.normal(size=(8, 3))
    basis = rng.normal(size=(3, 5, 4))
    fields = np.einsum("nk,kij->nij", factors, basis)
    fields[7] += validation_offset
    return np.asarray(fields, dtype=np.float64)


def test_authenticated_loader_exposes_v2_training_contract() -> None:
    data = load_machine_conditioned_training_data(REFERENCE, full_field_scan=True)
    assert data.inputs.shape == (3, 17)
    assert data.fields.shape == (3, 33, 33)
    assert data.grid_shape == (33, 33)
    assert len(data.feature_names) == 17
    assert data.manifest_sha256 == data.manifest_sha256.lower()


def test_deterministic_split_is_disjoint_complete_and_stable() -> None:
    first_train, first_validation = deterministic_split(50, validation_fraction=0.2, seed=42)
    second_train, second_validation = deterministic_split(50, validation_fraction=0.2, seed=42)
    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert np.intersect1d(first_train, first_validation).size == 0
    assert np.array_equal(np.sort(np.concatenate((first_train, first_validation))), np.arange(50))


def test_streaming_pca_is_train_only_and_resume_exact(tmp_path: Path) -> None:
    train_indices = np.arange(7, dtype=np.int64)
    identity = {"dataset": "low-rank", "source_sha256": "a" * 64}
    first_fields = _low_rank_fields(validation_offset=0.0)
    shifted_validation = _low_rank_fields(validation_offset=1.0e9)

    first, first_latent = fit_streaming_randomized_pca(
        first_fields,
        train_indices,
        n_components=3,
        oversampling=1,
        power_iterations=1,
        seed=11,
        chunk_rows=2,
        checkpoint_dir=tmp_path / "first",
        identity=identity,
    )
    shifted, shifted_latent = fit_streaming_randomized_pca(
        shifted_validation,
        train_indices,
        n_components=3,
        oversampling=1,
        power_iterations=1,
        seed=11,
        chunk_rows=2,
        checkpoint_dir=tmp_path / "shifted",
        identity=identity,
    )
    resumed, resumed_latent = fit_streaming_randomized_pca(
        first_fields,
        train_indices,
        n_components=3,
        oversampling=1,
        power_iterations=1,
        seed=11,
        chunk_rows=2,
        checkpoint_dir=tmp_path / "first",
        identity=identity,
        resume=True,
    )

    assert np.array_equal(first.mean, shifted.mean)
    assert np.allclose(first.components, shifted.components, rtol=0.0, atol=1.0e-12)
    assert np.allclose(first_latent, shifted_latent, rtol=0.0, atol=1.0e-12)
    assert np.array_equal(first.mean, resumed.mean)
    assert np.array_equal(first.components, resumed.components)
    assert np.array_equal(first_latent, resumed_latent)
    reconstructed = first.inverse_transform(first_latent)
    expected = first_fields[train_indices].reshape(len(train_indices), -1)
    assert np.allclose(reconstructed, expected, rtol=0.0, atol=1.0e-10)
    assert float(np.sum(first.explained_variance_ratio)) == pytest.approx(1.0)


def test_streaming_pca_rejects_tampered_recovery_stage(tmp_path: Path) -> None:
    fields = _low_rank_fields()
    train_indices = np.arange(7, dtype=np.int64)
    checkpoint_dir = tmp_path / "pca"
    identity = {"dataset": "low-rank", "source_sha256": "b" * 64}
    fit_streaming_randomized_pca(
        fields,
        train_indices,
        n_components=3,
        oversampling=1,
        power_iterations=0,
        seed=7,
        chunk_rows=2,
        checkpoint_dir=checkpoint_dir,
        identity=identity,
    )
    recovery = json.loads((checkpoint_dir / "pca_recovery.json").read_text())
    stage_path = checkpoint_dir / recovery["stage_file"]
    with stage_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fit_streaming_randomized_pca(
            fields,
            train_indices,
            n_components=3,
            oversampling=1,
            power_iterations=0,
            seed=7,
            chunk_rows=2,
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            resume=True,
        )


def test_successor_adam_step_is_seeded_and_deterministic() -> None:
    first = init_mlp_params(random.PRNGKey(42), input_dim=17, hidden_sizes=(8,), output_dim=3)
    second = init_mlp_params(random.PRNGKey(42), input_dim=17, hidden_sizes=(8,), output_dim=3)
    zeros = [{name: jnp.zeros_like(value) for name, value in layer.items()} for layer in first]
    features = jnp.arange(68, dtype=jnp.float64).reshape(4, 17) / 68.0
    targets = jnp.arange(12, dtype=jnp.float64).reshape(4, 3) / 12.0
    first_result = adam_step(first, zeros, zeros, features, targets, 1.0e-4, 0.5, 1)
    second_result = adam_step(second, zeros, zeros, features, targets, 1.0e-4, 0.5, 1)
    for first_tree, second_tree in zip(first_result[:3], second_result[:3], strict=True):
        for first_layer, second_layer in zip(first_tree, second_tree, strict=True):
            for name in ("W", "b"):
                assert np.array_equal(first_layer[name], second_layer[name])
    assert np.array_equal(first_result[3], second_result[3])


def test_optimizer_recovery_is_exact_and_rejects_tamper(tmp_path: Path) -> None:
    params = init_mlp_params(random.PRNGKey(9), input_dim=17, hidden_sizes=(8,), output_dim=3)
    moments = [{name: jnp.zeros_like(value) for name, value in layer.items()} for layer in params]
    identity = {"seed": np.asarray([9], dtype=np.int64)}
    _save_optimizer_checkpoint(
        tmp_path,
        identity=identity,
        params=params,
        first_moment=moments,
        second_moment=moments,
        best_params=params,
        completed_epochs=7,
        final_training_loss=0.5,
        best_validation_loss=0.75,
        best_epoch=6,
        evaluations_without_improvement=1,
        evaluation_epochs=[1, 6],
        training_losses=[1.0, 0.5],
        validation_losses=[1.25, 0.75],
    )
    loaded = _load_optimizer_checkpoint(tmp_path, identity=identity)
    assert loaded[4:] == (7, 0.5, 0.75, 6, 1, [1, 6], [1.0, 0.5], [1.25, 0.75])

    recovery = json.loads((tmp_path / "optimizer_recovery.json").read_text())
    with (tmp_path / recovery["stage_file"]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_optimizer_checkpoint(tmp_path, identity=identity)
