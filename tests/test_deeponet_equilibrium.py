# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Equilibrium Tests
"""Runtime and artifact contracts for the equilibrium branch-trunk operator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core import DeepONetEquilibriumAccelerator as PublicDeepONetRuntime
from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator
from scpn_fusion.io.machine_conditioned_surrogate_training import MachineConditionedTrainingData
from tools import train_machine_conditioned_deeponet as trainer


def _write_artifact(path: Path, *, schema: str = "scpn-fusion.equilibrium-deeponet.v1") -> None:
    np.savez(
        path,
        artifact_schema=np.asarray([schema]),
        branch_n_layers=np.asarray([2], dtype=np.int64),
        branch_0_W=np.ones((3, 4), dtype=np.float64) * 0.1,
        branch_0_b=np.zeros(4, dtype=np.float64),
        branch_1_W=np.ones((4, 2), dtype=np.float64) * 0.2,
        branch_1_b=np.zeros(2, dtype=np.float64),
        trunk_n_layers=np.asarray([2], dtype=np.int64),
        trunk_0_W=np.ones((2, 4), dtype=np.float64) * 0.1,
        trunk_0_b=np.zeros(4, dtype=np.float64),
        trunk_1_W=np.ones((4, 2), dtype=np.float64) * 0.2,
        trunk_1_b=np.zeros(2, dtype=np.float64),
        input_mean=np.zeros(3, dtype=np.float64),
        input_std=np.ones(3, dtype=np.float64),
        coordinates_rz_m=np.asarray([[3.0, -1.0], [4.0, -1.0], [3.0, 1.0], [4.0, 1.0]]),
        coordinate_mean=np.asarray([3.5, 0.0]),
        coordinate_std=np.asarray([0.5, 1.0]),
        field_mean=np.arange(4, dtype=np.float64),
        field_scale=np.asarray([2.0]),
        basis_width=np.asarray([2], dtype=np.int64),
        grid_nh=np.asarray([2], dtype=np.int64),
        grid_nw=np.asarray([2], dtype=np.int64),
        feature_names=np.asarray(["a", "b", "c"]),
        dataset_manifest_sha256=np.asarray(["a" * 64]),
    )


def test_deeponet_runtime_loads_and_predicts_single_and_batch(tmp_path: Path) -> None:
    artifact = tmp_path / "deeponet.npz"
    _write_artifact(artifact)
    runtime = DeepONetEquilibriumAccelerator()
    runtime.load_weights(artifact)
    single = runtime.predict(np.asarray([1.0, 2.0, 3.0]))
    batch = runtime.predict_batch(np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert single.shape == (2, 2)
    assert batch.shape == (2, 2, 2)
    assert np.array_equal(single, batch[0])
    assert np.all(np.isfinite(batch))
    assert runtime.machine_manifest_sha256 == "a" * 64
    assert PublicDeepONetRuntime is DeepONetEquilibriumAccelerator


def test_deeponet_runtime_rejects_wrong_schema_and_shape(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong.npz"
    _write_artifact(artifact, schema="wrong")
    runtime = DeepONetEquilibriumAccelerator()
    with pytest.raises(ValueError, match="unsupported"):
        runtime.load_weights(artifact)
    with pytest.raises(RuntimeError, match="not been loaded"):
        runtime.predict(np.zeros(3))


def test_deeponet_runtime_rejects_nonfinite_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "deeponet.npz"
    _write_artifact(artifact)
    runtime = DeepONetEquilibriumAccelerator()
    runtime.load_weights(artifact)
    with pytest.raises(ValueError, match="finite"):
        runtime.predict(np.asarray([1.0, np.nan, 3.0]))


def test_deeponet_training_is_recoverable_and_emits_untouched_test_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(91)
    root = tmp_path / "dataset"
    root.mkdir()
    np.save(root / "grid_r_m.npy", np.linspace(3.0, 4.0, 4))
    np.save(root / "grid_z_m.npy", np.linspace(-1.0, 1.0, 3))
    inputs = rng.normal(size=(12, 17))
    coordinates = np.stack(np.meshgrid(np.linspace(3.0, 4.0, 4), np.linspace(-1.0, 1.0, 3)))
    spatial = np.stack((coordinates[0], coordinates[1], coordinates[0] * coordinates[1]))
    fields = np.einsum("nk,kij->nij", inputs[:, :3], spatial)
    data = MachineConditionedTrainingData(
        root=root,
        manifest={
            "dataset_id": "unit-deeponet",
            "arrays": {
                "grid_r_m": {"file": "grid_r_m.npy"},
                "grid_z_m": {"file": "grid_z_m.npy"},
            },
            "machine": {"name": "unit-fixed-machine"},
        },
        manifest_sha256="d" * 64,
        inputs=np.asarray(inputs, dtype=np.float64),
        fields=np.asarray(fields, dtype=np.float64),
        feature_names=tuple(f"feature_{index}" for index in range(17)),
        grid_shape=(3, 4),
        inputs_sha256="e" * 64,
        fields_sha256="f" * 64,
    )
    monkeypatch.setattr(trainer, "load_machine_conditioned_training_data", lambda *a, **k: data)
    arguments = {
        "dataset_dir": root,
        "output_path": tmp_path / "candidate.npz",
        "report_path": tmp_path / "report.json",
        "checkpoint_dir": tmp_path / "checkpoints",
        "seed": 7,
        "validation_fraction": 1.0 / 6.0,
        "calibration_fraction": 1.0 / 6.0,
        "test_fraction": 1.0 / 6.0,
        "branch_hidden": (8,),
        "trunk_hidden": (8,),
        "basis_width": 4,
        "shot_batch_size": 4,
        "coordinate_batch_size": 6,
        "validation_probe_shots": 2,
        "validation_probe_coordinates": 6,
        "statistics_chunk_rows": 2,
        "evaluation_every": 1,
        "checkpoint_every": 1,
        "early_stopping_patience": 4,
    }
    first = trainer.run_training(**arguments, steps=2)
    resumed = trainer.run_training(**arguments, steps=3, resume=True)
    fresh_arguments = {
        **arguments,
        "output_path": tmp_path / "fresh_candidate.npz",
        "report_path": tmp_path / "fresh_report.json",
        "checkpoint_dir": tmp_path / "fresh_checkpoints",
    }
    fresh = trainer.run_training(**fresh_arguments, steps=3)

    assert first["training"]["completed_steps"] == 2
    assert resumed["training"]["completed_steps"] == 3
    assert resumed["split"]["samples"] == {
        "training": 6,
        "validation": 2,
        "calibration": 2,
        "test": 2,
    }
    assert resumed["claims"]["cross_machine_validated"] is False
    assert resumed["artifact"]["runtime_load_predict_finite"] is True
    assert resumed["artifact"]["runtime_training_path_parity_max_abs"] < 1.0e-4
    assert 0.0 <= resumed["conformal_relative_l2"]["test_empirical_coverage"] <= 1.0
    assert resumed["training"]["training_losses"] == fresh["training"]["training_losses"]
    assert resumed["training"]["validation_losses"] == fresh["training"]["validation_losses"]
    with (
        np.load(arguments["output_path"], allow_pickle=False) as resumed_weights,
        np.load(fresh_arguments["output_path"], allow_pickle=False) as fresh_weights,
    ):
        for name in resumed_weights.files:
            assert np.array_equal(resumed_weights[name], fresh_weights[name])
