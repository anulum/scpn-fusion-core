# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER Surrogate Tool Tests
"""Real-surface tests for the recoverable ITER surrogate training tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core import iter_surrogate_artifact_status as public_iter_surrogate_artifact_status
from scpn_fusion.core.neural_equilibrium import (
    ITER_SURROGATE_VALIDATION_REPORT,
    NeuralEquilibriumAccelerator,
    iter_surrogate_artifact_status,
)
from tools.train_iter_surrogate import (
    deterministic_split,
    default_iter_dataset_paths,
    inspect_iter_dataset,
    load_iter_dataset,
    run_training,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ITER_WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_equilibrium_iter_v1.npz"


def test_default_iter_dataset_paths_are_directory_relative(tmp_path: Path) -> None:
    """Default array paths remain relative to the caller's data directory."""
    x_path, y_path = default_iter_dataset_paths(tmp_path)

    assert x_path == tmp_path / "iter_X.npy"
    assert y_path == tmp_path / "iter_Y.npy"


def test_load_iter_dataset_from_directory(tmp_path: Path) -> None:
    """Directory datasets load through the public array contract."""
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    y = np.arange(8, dtype=np.float64).reshape(2, 4)
    np.save(tmp_path / "iter_X.npy", x)
    np.save(tmp_path / "iter_Y.npy", y)

    loaded_x, loaded_y = load_iter_dataset(tmp_path)

    np.testing.assert_allclose(loaded_x, x)
    np.testing.assert_allclose(loaded_y, y)


def test_load_iter_dataset_from_npz(tmp_path: Path) -> None:
    """Legacy NPZ datasets remain readable without pickle."""
    x = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.arange(6, dtype=np.float64).reshape(2, 3)
    dataset_path = tmp_path / "iter_dataset.npz"
    np.savez(dataset_path, X=x, Y=y)

    loaded_x, loaded_y = load_iter_dataset(dataset_path)

    np.testing.assert_allclose(loaded_x, x)
    np.testing.assert_allclose(loaded_y, y)


def test_iter_dataset_report_distinguishes_development_from_full_fidelity() -> None:
    """Dataset reports preserve the configured sample-count claim boundary."""
    x = np.ones((10, 12), dtype=np.float64)
    y = np.ones((10, 16), dtype=np.float64)

    dev_report = inspect_iter_dataset(x, y, min_full_fidelity_samples=50)
    ready_report = inspect_iter_dataset(x, y, min_full_fidelity_samples=10)

    assert dev_report["status"] == "development_dataset_below_full_fidelity_sample_count"
    assert ready_report["status"] == "full_fidelity_iter_dataset_ready"


def test_iter_dataset_report_blocks_invalid_shapes_and_nonfinite_values() -> None:
    """Malformed or non-finite datasets fail the evidence gate."""
    x = np.ones((10, 11), dtype=np.float64)
    y = np.ones((10, 16), dtype=np.float64)
    assert inspect_iter_dataset(x, y)["status"] == "blocked_invalid_feature_shape"

    x = np.ones((10, 12), dtype=np.float64)
    y = np.ones((9, 16), dtype=np.float64)
    assert inspect_iter_dataset(x, y)["status"] == "blocked_invalid_field_shape"

    y = np.ones((10, 16), dtype=np.float64)
    y[0, 0] = np.nan
    assert inspect_iter_dataset(x, y)["status"] == "blocked_nonfinite_values"


def test_deterministic_split_is_disjoint_complete_and_reproducible() -> None:
    """Held-out membership is stable and never overlaps training data."""
    train_a, validation_a = deterministic_split(20, validation_fraction=0.2, seed=42)
    train_b, validation_b = deterministic_split(20, validation_fraction=0.2, seed=42)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert len(train_a) == 16
    assert len(validation_a) == 4
    assert set(train_a).isdisjoint(validation_a)
    assert sorted(np.concatenate((train_a, validation_a)).tolist()) == list(range(20))


def test_recoverable_training_writes_candidate_report_and_exact_resume(tmp_path: Path) -> None:
    """The real training surface checkpoints both stages and resumes at an epoch boundary."""
    rng = np.random.default_rng(7)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "iter_X.npy", rng.normal(size=(20, 12)))
    np.save(data_dir / "iter_Y.npy", rng.normal(size=(20, 16)))
    out_path = tmp_path / "candidate.npz"
    report_path = tmp_path / "report.json"
    checkpoint_dir = tmp_path / "checkpoints"

    first = run_training(
        data_path=data_dir,
        out_path=out_path,
        report_path=report_path,
        checkpoint_dir=checkpoint_dir,
        epochs=2,
        seed=42,
        checkpoint_every=1,
        min_full_fidelity_samples=50,
    )

    assert first["status"] == "completed_candidate_below_full_fidelity_claim_threshold"
    assert first["facility_validated"] is False
    assert first["split"]["training_samples"] == 16
    assert first["split"]["validation_samples"] == 4
    assert out_path.exists()
    assert (checkpoint_dir / "pca_checkpoint.npz").exists()
    optimizer_path = checkpoint_dir / "optimizer_checkpoint.npz"
    with np.load(optimizer_path, allow_pickle=False) as checkpoint:
        assert int(checkpoint["completed_epochs"][0]) == 2

    resumed = run_training(
        data_path=data_dir,
        out_path=out_path,
        report_path=report_path,
        checkpoint_dir=checkpoint_dir,
        epochs=3,
        seed=42,
        checkpoint_every=1,
        min_full_fidelity_samples=50,
        resume=True,
    )

    assert resumed["training"]["epochs"] == 3
    assert np.isfinite(resumed["held_out_validation"]["mean_relative_l2"])
    with np.load(optimizer_path, allow_pickle=False) as checkpoint:
        assert int(checkpoint["completed_epochs"][0]) == 3

    uninterrupted_path = tmp_path / "uninterrupted.npz"
    run_training(
        data_path=data_dir,
        out_path=uninterrupted_path,
        report_path=tmp_path / "uninterrupted_report.json",
        checkpoint_dir=tmp_path / "uninterrupted_checkpoints",
        epochs=3,
        seed=42,
        checkpoint_every=1,
        min_full_fidelity_samples=50,
    )
    with (
        np.load(out_path, allow_pickle=False) as resumed_weights,
        np.load(uninterrupted_path, allow_pickle=False) as uninterrupted_weights,
    ):
        assert resumed_weights.files == uninterrupted_weights.files
        for name in resumed_weights.files:
            np.testing.assert_array_equal(resumed_weights[name], uninterrupted_weights[name])


def test_resume_rejects_dataset_changed_after_checkpoint(tmp_path: Path) -> None:
    """Recovery fails closed if either raw dataset digest changes."""
    rng = np.random.default_rng(11)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    x_path = data_dir / "iter_X.npy"
    np.save(x_path, rng.normal(size=(10, 12)))
    np.save(data_dir / "iter_Y.npy", rng.normal(size=(10, 16)))
    out_path = tmp_path / "candidate.npz"
    report_path = tmp_path / "report.json"
    checkpoint_dir = tmp_path / "checkpoints"
    run_training(
        data_path=data_dir,
        out_path=out_path,
        report_path=report_path,
        checkpoint_dir=checkpoint_dir,
        epochs=1,
        seed=42,
        checkpoint_every=1,
    )
    changed = np.load(x_path, allow_pickle=False)
    changed[0, 0] += 1.0
    np.save(x_path, changed)

    with pytest.raises(ValueError, match="Checkpoint identity mismatch for x_sha256"):
        run_training(
            data_path=data_dir,
            out_path=out_path,
            report_path=report_path,
            checkpoint_dir=checkpoint_dir,
            epochs=1,
            seed=42,
            checkpoint_every=1,
            resume=True,
        )


def test_iter_surrogate_weights_load_and_predict_finite_field() -> None:
    """The committed standard artifact remains runtime-loadable."""
    accel = NeuralEquilibriumAccelerator()
    accel.load_weights(ITER_WEIGHTS_PATH)

    assert accel.cfg.n_input_features == 12
    assert accel.cfg.grid_shape == (128, 128)
    assert accel.cfg.n_components == 20
    assert accel._input_mean is not None
    assert accel._input_mean.shape == (12,)

    psi = accel.predict(np.asarray(accel._input_mean, dtype=np.float64))

    assert psi.shape == (128, 128)
    assert np.all(np.isfinite(psi))


def test_iter_surrogate_artifact_status_preserves_standard_vs_high_fidelity_boundary() -> None:
    """Public status does not promote the standard artifact to high fidelity."""
    status = iter_surrogate_artifact_status()
    public_status = public_iter_surrogate_artifact_status()

    assert status == public_status
    assert status["status"] == "standard_iter_surrogate_artifact_present_and_runtime_loadable"
    assert status["artifact"] == "weights/neural_equilibrium_iter_v1.npz"
    assert status["artifact_exists"] is True
    assert status["artifact_size_bytes"] == 3_124_396
    assert status["input_features"] == 12
    assert status["grid_shape"] == (128, 128)
    assert status["pca_components"] == 20
    assert status["high_fidelity_gpu_retraining_complete"] is False
    assert status["required_high_fidelity_report"] == (
        "validation/reports/iter_surrogate_training_report.json"
    )
    assert "does not claim high-fidelity GPU retraining" in str(status["claim_boundary"])
    assert ITER_SURROGATE_VALIDATION_REPORT.exists()
