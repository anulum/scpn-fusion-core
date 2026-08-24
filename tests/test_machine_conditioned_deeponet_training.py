# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned DeepONet Training Tests
"""End-to-end split, recovery, evidence, and failure contracts."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core import DeepONetEquilibriumAccelerator
from scpn_fusion.io.deeponet_training_data import (
    field_metrics,
    load_coordinates,
    runtime_backend_parity,
)
from scpn_fusion.io.machine_conditioned_equilibrium_dataset import array_contract
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    deterministic_four_way_split,
    load_machine_conditioned_training_data,
)
from tools import train_machine_conditioned_deeponet as trainer

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "validation/reference/iter_machine_conditioned_v2_n3_seed20260822_33x33"


def _refresh_array_contracts(root: Path, manifest: dict[str, Any], *names: str) -> None:
    for name in names:
        spec = manifest["arrays"][name]
        path = root / spec["file"]
        array = np.load(path, allow_pickle=False)
        manifest["arrays"][name] = array_contract(path, array, role=spec["role"])


def _training_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    shutil.copytree(REFERENCE, root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sample_arrays = ("inputs", "psi_total", "psi_vacuum", "diagnostics")
    for name in sample_arrays:
        path = root / manifest["arrays"][name]["file"]
        expanded = np.concatenate([np.load(path, allow_pickle=False)] * 4, axis=0)
        if name == "diagnostics":
            expanded[:, 0] = np.arange(len(expanded), dtype=np.float64)
        np.save(path, np.asarray(expanded, dtype=np.float64))
    manifest["dataset_id"] = "iter-like-fixed-support-v2-n12-test-contract"
    manifest["generation"]["accepted_samples"] = 12
    manifest["generation"]["requested_samples"] = 12
    _refresh_array_contracts(root, manifest, *sample_arrays)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _shift_post_selection_fields(root: Path, indices: NDArray[np.int64]) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_path = root / manifest["arrays"]["psi_total"]["file"]
    total = np.load(total_path, allow_pickle=False)
    total[indices] += 1.0e6
    np.save(total_path, total)
    vacuum = np.load(root / manifest["arrays"]["psi_vacuum"]["file"], allow_pickle=False)
    diagnostics_path = root / manifest["arrays"]["diagnostics"]["file"]
    diagnostics = np.load(diagnostics_path, allow_pickle=False)
    delta_column = manifest["diagnostic_names"].index("plasma_delta_max_abs_wb")
    diagnostics[:, delta_column] = np.max(np.abs(total - vacuum), axis=(1, 2))
    np.save(diagnostics_path, diagnostics)
    _refresh_array_contracts(root, manifest, "psi_total", "diagnostics")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _arguments(tmp_path: Path, dataset_dir: Path) -> dict[str, Any]:
    return {
        "dataset_dir": dataset_dir,
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


def test_training_resume_matches_uninterrupted_runtime_artifact(tmp_path: Path) -> None:
    dataset_dir = _training_fixture(tmp_path)
    arguments = _arguments(tmp_path, dataset_dir)
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
    assert resumed["artifact"]["runtime_backend"] in {"numpy", "rust"}
    assert resumed["artifact"]["runtime_training_path_parity_max_abs"] < 1.0e-4
    backend_parity = resumed["artifact"]["rust_numpy_untouched_test_parity"]
    assert backend_parity["sample_count"] == 2
    if resumed["artifact"]["runtime_backend"] == "rust":
        assert backend_parity["evaluated"] is True
        assert backend_parity["within_tolerance"] is True
        assert backend_parity["max_tolerance_ratio"] <= 1.0
        assert isinstance(backend_parity["max_ulp_difference"], int)
    else:
        assert backend_parity["evaluated"] is False
        assert backend_parity["within_tolerance"] is None
    assert 0.0 <= resumed["conformal_relative_l2"]["test_empirical_coverage"] <= 1.0
    assert resumed["training"]["training_losses"] == fresh["training"]["training_losses"]
    assert resumed["training"]["validation_losses"] == fresh["training"]["validation_losses"]

    data = load_machine_conditioned_training_data(dataset_dir, full_field_scan=True)
    split = deterministic_four_way_split(
        len(data.inputs),
        validation_fraction=1.0 / 6.0,
        calibration_fraction=1.0 / 6.0,
        test_fraction=1.0 / 6.0,
        seed=7,
    )
    numpy_runtime = DeepONetEquilibriumAccelerator(prefer_rust=False)
    numpy_runtime.load_weights(arguments["output_path"])
    unavailable = runtime_backend_parity(
        numpy_runtime,
        numpy_runtime,
        data,
        split.test,
        chunk_rows=1,
    )
    assert unavailable["evaluated"] is False
    with pytest.raises(ValueError, match="at least one sample"):
        field_metrics(
            numpy_runtime,
            data,
            np.asarray([], dtype=np.int64),
            chunk_rows=1,
        )
    with pytest.raises(ValueError, match="chunk size must be positive"):
        field_metrics(numpy_runtime, data, split.test, chunk_rows=0)
    with pytest.raises(ValueError, match="at least one sample"):
        runtime_backend_parity(
            numpy_runtime,
            numpy_runtime,
            data,
            np.asarray([], dtype=np.int64),
            chunk_rows=1,
        )
    with pytest.raises(ValueError, match="chunk size must be positive"):
        runtime_backend_parity(
            numpy_runtime,
            numpy_runtime,
            data,
            split.test,
            chunk_rows=0,
        )
    with pytest.raises(ValueError, match="tolerances must be positive"):
        runtime_backend_parity(
            numpy_runtime,
            numpy_runtime,
            data,
            split.test,
            chunk_rows=1,
            relative_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="reference must use the NumPy backend"):
        runtime_backend_parity(
            numpy_runtime,
            DeepONetEquilibriumAccelerator(),
            data,
            split.test,
            chunk_rows=1,
        )
    with (
        np.load(arguments["output_path"], allow_pickle=False) as resumed_weights,
        np.load(fresh_arguments["output_path"], allow_pickle=False) as fresh_weights,
    ):
        source_names = set(resumed_weights["source_sha256_names"].tolist())
        assert {
            "src/scpn_fusion/core/_multi_compat.py",
            "src/scpn_fusion/io/safe_loaders.py",
            "scpn-fusion-rs/Cargo.lock",
            "scpn-fusion-rs/crates/fusion-ml/Cargo.toml",
            "scpn-fusion-rs/crates/fusion-ml/src/deeponet_equilibrium.rs",
            "scpn-fusion-rs/crates/fusion-python/src/bindings/ml.rs",
        } <= source_names
        for name in resumed_weights.files:
            assert np.array_equal(resumed_weights[name], fresh_weights[name])


def test_training_coordinates_reject_grid_shape_drift(tmp_path: Path) -> None:
    dataset_dir = _training_fixture(tmp_path)
    data = load_machine_conditioned_training_data(dataset_dir, full_field_scan=True)
    np.save(dataset_dir / "grid_r_m.npy", np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="coordinate arrays do not match"):
        load_coordinates(data)


def test_training_rejects_tampered_statistics_recovery(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    with (arguments["checkpoint_dir"] / "statistics.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="statistics recovery SHA-256 mismatch"):
        trainer.run_training(**arguments, steps=2, resume=True)


def test_training_rejects_invalid_statistics_recovery_metadata(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    statistics_recovery = arguments["checkpoint_dir"] / "statistics_recovery.json"
    recovery = json.loads(statistics_recovery.read_text(encoding="utf-8"))
    recovery["file"] = "wrong.npz"
    statistics_recovery.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(ValueError, match="statistics recovery metadata is invalid"):
        trainer.run_training(**arguments, steps=2, resume=True)


def test_training_rejects_symlinked_statistics_recovery(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    statistics = arguments["checkpoint_dir"] / "statistics.npz"
    target = arguments["checkpoint_dir"] / "statistics_target.npz"
    statistics.rename(target)
    statistics.symlink_to(target.name)
    with pytest.raises(ValueError, match="statistics recovery SHA-256 mismatch"):
        trainer.run_training(**arguments, steps=2, resume=True)


def test_training_rejects_tampered_optimizer_recovery(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    recovery = json.loads(
        (arguments["checkpoint_dir"] / "optimizer_recovery.json").read_text(encoding="utf-8")
    )
    with (arguments["checkpoint_dir"] / recovery["stage_file"]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="optimizer recovery SHA-256 mismatch"):
        trainer.run_training(**arguments, steps=2, resume=True)


def test_training_rejects_optimizer_recovery_step_mismatch(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    optimizer_recovery = arguments["checkpoint_dir"] / "optimizer_recovery.json"
    recovery = json.loads(optimizer_recovery.read_text(encoding="utf-8"))
    recovery["completed_steps"] = 2
    optimizer_recovery.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer recovery step mismatch"):
        trainer.run_training(**arguments, steps=2, resume=True)


def test_training_rejects_changed_resume_identity(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=1)
    with pytest.raises(ValueError, match="recovery identity mismatch"):
        trainer.run_training(**arguments, steps=2, learning_rate=2.0e-4, resume=True)


def test_training_rejects_recovery_beyond_requested_steps(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    trainer.run_training(**arguments, steps=2)
    with pytest.raises(ValueError, match="exceeds the requested step target"):
        trainer.run_training(**arguments, steps=1, resume=True)


def test_training_stops_after_validation_ceases_improving(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    arguments["early_stopping_patience"] = 1
    report = trainer.run_training(
        **arguments,
        steps=3,
        learning_rate=1.0e-20,
    )
    assert report["training"]["stopped_early"] is True
    assert report["training"]["completed_steps"] == 2


def test_training_obeys_sparse_evaluation_and_checkpoint_cadence(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    arguments["evaluation_every"] = 3
    arguments["checkpoint_every"] = 3
    report = trainer.run_training(**arguments, steps=4)
    assert report["training"]["evaluation_steps"] == [1, 3, 4]
    assert report["recovery"]["optimizer_completed_steps"] == 4


def test_calibration_and_test_values_cannot_change_selected_weights(tmp_path: Path) -> None:
    first = _training_fixture(tmp_path / "first")
    shifted = _training_fixture(tmp_path / "shifted")
    split = deterministic_four_way_split(
        12,
        validation_fraction=1.0 / 6.0,
        calibration_fraction=1.0 / 6.0,
        test_fraction=1.0 / 6.0,
        seed=7,
    )
    _shift_post_selection_fields(shifted, np.concatenate((split.calibration, split.test)))
    first_arguments = _arguments(tmp_path / "first_run", first)
    shifted_arguments = _arguments(tmp_path / "shifted_run", shifted)
    first_report = trainer.run_training(**first_arguments, steps=2)
    shifted_report = trainer.run_training(**shifted_arguments, steps=2)

    assert (
        first_report["training"]["validation_losses"]
        == shifted_report["training"]["validation_losses"]
    )
    assert first_report["untouched_final_test"] != shifted_report["untouched_final_test"]
    with (
        np.load(first_arguments["output_path"], allow_pickle=False) as first_weights,
        np.load(shifted_arguments["output_path"], allow_pickle=False) as shifted_weights,
    ):
        network_keys = [
            name for name in first_weights.files if name.startswith(("branch_", "trunk_"))
        ]
        for name in network_keys:
            assert np.array_equal(first_weights[name], shifted_weights[name])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"steps": 0}, "integer hyperparameters"),
        ({"learning_rate": float("nan")}, "continuous hyperparameters must be finite"),
        ({"weight_decay": float("inf")}, "continuous hyperparameters must be finite"),
        ({"weight_decay": -1.0}, "learning rate or weight decay"),
        ({"gradient_clip": 0.0}, "gradient clip"),
    ],
)
def test_training_rejects_invalid_hyperparameters(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    arguments = _arguments(tmp_path, _training_fixture(tmp_path))
    arguments["steps"] = 1
    arguments.update(override)
    with pytest.raises(ValueError, match=message):
        trainer.run_training(**arguments)


def test_deeponet_cli_trains_a_runtime_loadable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dataset_dir = _training_fixture(tmp_path)
    artifact = tmp_path / "cli_candidate.npz"
    report = tmp_path / "cli_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/train_machine_conditioned_deeponet.py",
            "--dataset-dir",
            str(dataset_dir),
            "--out",
            str(artifact),
            "--report",
            str(report),
            "--checkpoint-dir",
            str(tmp_path / "cli_checkpoints"),
            "--steps",
            "1",
            "--validation-fraction",
            str(1.0 / 6.0),
            "--calibration-fraction",
            str(1.0 / 6.0),
            "--test-fraction",
            str(1.0 / 6.0),
            "--basis-width",
            "4",
            "--shot-batch-size",
            "4",
            "--coordinate-batch-size",
            "16",
            "--validation-probe-shots",
            "2",
            "--validation-probe-coordinates",
            "16",
            "--statistics-chunk-rows",
            "2",
            "--evaluation-every",
            "1",
            "--checkpoint-every",
            "1",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert "completed_local_candidate_not_promoted" in result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    runtime = DeepONetEquilibriumAccelerator()
    runtime.load_weights(artifact)
    assert payload["artifact"]["runtime_load_predict_finite"] is True
    assert runtime.predict(np.load(dataset_dir / "inputs.npy", allow_pickle=False)[0]).shape == (
        33,
        33,
    )

    direct_artifact = tmp_path / "direct_cli_candidate.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_machine_conditioned_deeponet.py",
            "--dataset-dir",
            str(dataset_dir),
            "--out",
            str(direct_artifact),
            "--report",
            str(tmp_path / "direct_cli_report.json"),
            "--checkpoint-dir",
            str(tmp_path / "direct_cli_checkpoints"),
            "--steps",
            "1",
            "--validation-fraction",
            str(1.0 / 6.0),
            "--calibration-fraction",
            str(1.0 / 6.0),
            "--test-fraction",
            str(1.0 / 6.0),
            "--basis-width",
            "4",
            "--shot-batch-size",
            "4",
            "--coordinate-batch-size",
            "16",
            "--validation-probe-shots",
            "2",
            "--validation-probe-coordinates",
            "16",
            "--statistics-chunk-rows",
            "2",
            "--evaluation-every",
            "1",
            "--checkpoint-every",
            "1",
        ],
    )
    with caplog.at_level(
        logging.INFO,
        logger="scpn_fusion.io.machine_conditioned_deeponet_cli",
    ):
        trainer.main()
    assert caplog.messages[-1] == "completed_local_candidate_not_promoted"
    DeepONetEquilibriumAccelerator().load_weights(direct_artifact)
