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
    MachineConditionedTrainingData,
    deterministic_four_way_split,
    deterministic_split,
    fit_streaming_randomized_pca,
    load_machine_conditioned_training_data,
)
from tools import train_machine_conditioned_equilibrium_surrogate as trainer
from scpn_fusion.io.machine_conditioned_surrogate_cli import run_training_cli
from tools.train_machine_conditioned_equilibrium_surrogate import (
    _load_optimizer_checkpoint,
    _save_optimizer_checkpoint,
    adam_step,
    hybrid_latent_loss,
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


def test_four_way_split_is_disjoint_complete_stable_and_role_sized() -> None:
    first = deterministic_four_way_split(
        50_000,
        validation_fraction=0.10,
        calibration_fraction=0.05,
        test_fraction=0.05,
        seed=42,
    )
    second = deterministic_four_way_split(
        50_000,
        validation_fraction=0.10,
        calibration_fraction=0.05,
        test_fraction=0.05,
        seed=42,
    )
    assert len(first.training) == 40_000
    assert len(first.validation) == 5_000
    assert len(first.calibration) == 2_500
    assert len(first.test) == 2_500
    for first_role, second_role in zip(first.__dict__.values(), second.__dict__.values()):
        assert np.array_equal(first_role, second_role)
    combined = np.concatenate((first.training, first.validation, first.calibration, first.test))
    assert np.array_equal(np.sort(combined), np.arange(50_000))
    assert len(np.unique(combined)) == len(combined)


@pytest.mark.parametrize(
    ("validation", "calibration", "test"),
    [
        (0.0, 0.05, 0.05),
        (0.10, -0.05, 0.05),
        (0.10, 0.05, 1.0),
        (0.40, 0.30, 0.30),
    ],
)
def test_four_way_split_rejects_invalid_fractions(
    validation: float, calibration: float, test: float
) -> None:
    with pytest.raises(ValueError):
        deterministic_four_way_split(
            100,
            validation_fraction=validation,
            calibration_fraction=calibration,
            test_fraction=test,
            seed=42,
        )


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


def test_hybrid_loss_reduces_to_standardized_mse_at_zero_weight() -> None:
    predictions = jnp.asarray([[1.0, 2.0], [3.0, 5.0]])
    targets = jnp.asarray([[0.0, 1.0], [1.0, 1.0]])
    loss = hybrid_latent_loss(
        predictions,
        targets,
        component_weights=jnp.asarray([100.0, 0.01]),
        sample_weights=jnp.asarray([7.0, 0.2]),
        field_loss_weight=0.0,
    )
    assert float(loss) == pytest.approx(float(jnp.mean(jnp.square(predictions - targets))))


def test_field_aligned_loss_prioritizes_physical_component_and_relative_sample() -> None:
    targets = jnp.zeros((2, 2), dtype=jnp.float64)
    component_weights = jnp.asarray([10.0, 1.0], dtype=jnp.float64)
    sample_weights = jnp.asarray([4.0, 1.0], dtype=jnp.float64)
    leading_small_field_error = hybrid_latent_loss(
        jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
        targets,
        component_weights,
        sample_weights,
        1.0,
    )
    tail_large_field_error = hybrid_latent_loss(
        jnp.asarray([[0.0, 0.0], [0.0, 1.0]]),
        targets,
        component_weights,
        sample_weights,
        1.0,
    )
    assert float(leading_small_field_error) == pytest.approx(10.0)
    assert float(tail_large_field_error) == pytest.approx(0.25)


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


def test_field_aware_training_emits_four_way_evidence_and_runtime_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(81)
    inputs = rng.normal(size=(12, 17))
    factors = inputs[:, :3]
    basis = rng.normal(size=(3, 5, 4))
    fields = np.einsum("nk,kij->nij", factors, basis)
    data = MachineConditionedTrainingData(
        root=tmp_path / "dataset",
        manifest={"dataset_id": "unit-four-way"},
        manifest_sha256="a" * 64,
        inputs=np.asarray(inputs, dtype=np.float64),
        fields=np.asarray(fields, dtype=np.float64),
        feature_names=tuple(f"feature_{index}" for index in range(17)),
        grid_shape=(5, 4),
        inputs_sha256="b" * 64,
        fields_sha256="c" * 64,
    )
    monkeypatch.setattr(trainer, "load_machine_conditioned_training_data", lambda *a, **k: data)

    report = trainer.run_training(
        dataset_dir=data.root,
        output_path=tmp_path / "candidate.npz",
        report_path=tmp_path / "report.json",
        checkpoint_dir=tmp_path / "checkpoints",
        epochs=2,
        seed=7,
        validation_fraction=1.0 / 6.0,
        calibration_fraction=1.0 / 6.0,
        test_fraction=1.0 / 6.0,
        field_loss_weight=0.9,
        n_components=3,
        pca_oversampling=1,
        pca_power_iterations=0,
        pca_chunk_rows=2,
        hidden_sizes=(8,),
        evaluation_every=1,
        checkpoint_every=1,
        early_stopping_patience=3,
    )

    assert report["status"] == "completed_local_candidate_not_promoted"
    assert report["split"]["training_samples"] == 6
    assert report["split"]["validation_samples"] == 2
    assert report["split"]["calibration_samples"] == 2
    assert report["split"]["test_samples"] == 2
    assert report["training"]["field_loss_weight"] == pytest.approx(0.9)
    assert report["conformal_relative_l2"]["calibration_samples"] == 2
    assert 0.0 <= report["conformal_relative_l2"]["test_empirical_coverage"] <= 1.0
    assert report["artifact"]["runtime_load_predict_finite"] is True
    assert (tmp_path / "candidate.npz").is_file()
    assert (tmp_path / "report.json").is_file()
