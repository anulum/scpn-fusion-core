# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — machine-conditioned equilibrium dataset tests
"""Contract tests for the measured ITER-like v2 reference cohort."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.io.machine_conditioned_equilibrium_dataset import (
    sha256_file,
    verify_machine_conditioned_dataset,
)
from scpn_fusion.io.recoverable_npy_dataset import RecoverableNpyDataset
from tools.generate_iter_machine_conditioned_dataset import (
    feature_ranges,
    latin_hypercube,
    load_generation_spec,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "validation" / "iter_machine_conditioned_v2_spec.json"
REFERENCE = REPO / "validation" / "reference" / "iter_machine_conditioned_v2_n3_seed20260822_33x33"


def test_measured_reference_passes_full_verification() -> None:
    result = verify_machine_conditioned_dataset(REFERENCE, full_field_scan=True)
    assert result["status"] == "passed"
    assert result["samples_verified"] == 3
    assert float(result["vacuum_reconstruction_max_abs"]) < 1.0e-10


def test_reference_inputs_are_pre_solve_and_non_constant() -> None:
    manifest = cast(dict[str, Any], json.loads((REFERENCE / "manifest.json").read_text()))
    inputs = np.load(REFERENCE / "inputs.npy", allow_pickle=False)
    names = [str(feature["name"]) for feature in manifest["features"]]
    assert inputs.shape == (3, 17)
    assert all(np.unique(inputs[:, index]).size == 3 for index in range(inputs.shape[1]))
    forbidden = ("psi_axis", "psi_x", "residual", "iteration", "converged")
    assert not any(token in name for token in forbidden for name in names)
    assert names[0] == "plasma_current_target_a"
    assert names[1:7] == [
        "coil_current_a.PF1U",
        "coil_current_a.PF1L",
        "coil_current_a.PF2U",
        "coil_current_a.PF2L",
        "coil_current_a.PF3U",
        "coil_current_a.PF3L",
    ]


def test_reference_diagnostics_close_physics_thresholds() -> None:
    manifest = cast(dict[str, Any], json.loads((REFERENCE / "manifest.json").read_text()))
    diagnostics = np.load(REFERENCE / "diagnostics.npy", allow_pickle=False)
    index = {name: i for i, name in enumerate(manifest["diagnostic_names"])}
    tolerance = manifest["tolerances"]
    assert np.all(diagnostics[:, index["converged"]] == 1.0)
    assert np.all(diagnostics[:, index["iterations"]] < manifest["solver"]["n_iter"])
    assert np.all(
        diagnostics[:, index["relative_gs_residual_rms"]]
        <= tolerance["relative_gs_residual_rms_max"]
    )
    assert np.all(
        diagnostics[:, index["plasma_current_relative_error"]]
        <= tolerance["plasma_current_relative_error_max"]
    )
    assert np.all(
        diagnostics[:, index["plasma_delta_max_abs_wb"]] >= tolerance["plasma_delta_max_abs_wb_min"]
    )


def test_verifier_rejects_authenticated_semantic_tamper(tmp_path: Path) -> None:
    copied = tmp_path / "dataset"
    shutil.copytree(REFERENCE, copied)
    manifest_path = copied / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["claims"]["facility_validated"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_machine_conditioned_dataset(copied, full_field_scan=True)
    assert result["status"] == "failed"
    assert (
        "claims.facility_validated must be false for this synthetic dataset" in result["failures"]
    )


def test_reference_source_provenance_matches_tracked_files() -> None:
    manifest = cast(dict[str, Any], json.loads((REFERENCE / "manifest.json").read_text()))
    for source in manifest["source"]["files"].values():
        path = REPO / source["path"]
        assert path.is_file()
        assert sha256_file(path) == source["sha256"]


def test_generation_contract_and_latin_hypercube_are_deterministic() -> None:
    spec = load_generation_spec(SPEC)
    ranges = feature_ranges(spec)
    first = latin_hypercube(11, len(ranges), seed=20260822)
    second = latin_hypercube(11, len(ranges), seed=20260822)
    assert np.array_equal(first, second)
    assert len(ranges) == 17
    for column in range(first.shape[1]):
        assert np.array_equal(np.sort(np.floor(first[:, column] * 11)), np.arange(11))


def test_verifier_cli_accepts_reference() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/verify_iter_machine_conditioned_dataset.py",
            "--dataset-dir",
            str(REFERENCE),
            "--full-field-scan",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"


def test_recoverable_store_resumes_only_from_durable_cursor(tmp_path: Path) -> None:
    partial = tmp_path / ".cohort.partial"
    recovery = tmp_path / ".cohort.recovery.json"
    shapes = {"inputs": (3, 2), "field": (3, 2, 2), "grid": (2,)}
    sample_arrays = frozenset({"inputs", "field"})
    contract = {"seed": 7, "candidate_design_sha256": "a" * 64}
    store = RecoverableNpyDataset.create(
        partial_dir=partial,
        recovery_path=recovery,
        shapes=shapes,
        sample_array_names=sample_arrays,
        run_contract=contract,
        initial_arrays={"grid": np.asarray([1.0, 2.0])},
    )
    store.write_sample(
        0,
        {
            "inputs": np.asarray([3.0, 4.0]),
            "field": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
        },
    )
    store.checkpoint(accepted_samples=1, next_candidate_index=2, rejection_counts={"cap": 1})
    store.write_sample(
        1,
        {
            "inputs": np.asarray([30.0, 40.0]),
            "field": np.asarray([[50.0, 60.0], [70.0, 80.0]]),
        },
    )

    resumed = RecoverableNpyDataset.resume(
        partial_dir=partial,
        recovery_path=recovery,
        shapes=shapes,
        sample_array_names=sample_arrays,
        run_contract=contract,
    )
    assert resumed.accepted_samples == 1
    assert resumed.next_candidate_index == 2
    assert resumed.rejection_counts == {"cap": 1}
    resumed.require_array_equal("grid", np.asarray([1.0, 2.0]))
    resumed.write_sample(
        1,
        {
            "inputs": np.asarray([9.0, 10.0]),
            "field": np.asarray([[11.0, 12.0], [13.0, 14.0]]),
        },
    )
    resumed.checkpoint(accepted_samples=2, next_candidate_index=3)
    assert np.array_equal(resumed.arrays["inputs"][1], np.asarray([9.0, 10.0]))


def test_recoverable_store_rejects_changed_contract_and_static_bytes(tmp_path: Path) -> None:
    partial = tmp_path / ".cohort.partial"
    recovery = tmp_path / ".cohort.recovery.json"
    shapes = {"inputs": (2, 1), "grid": (2,)}
    sample_arrays = frozenset({"inputs"})
    contract = {"seed": 11}
    store = RecoverableNpyDataset.create(
        partial_dir=partial,
        recovery_path=recovery,
        shapes=shapes,
        sample_array_names=sample_arrays,
        run_contract=contract,
        initial_arrays={"grid": np.asarray([1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="run contract"):
        RecoverableNpyDataset.resume(
            partial_dir=partial,
            recovery_path=recovery,
            shapes=shapes,
            sample_array_names=sample_arrays,
            run_contract={"seed": 12},
        )
    resumed = RecoverableNpyDataset.resume(
        partial_dir=partial,
        recovery_path=recovery,
        shapes=shapes,
        sample_array_names=sample_arrays,
        run_contract=contract,
    )
    with pytest.raises(ValueError, match="regenerated grid"):
        resumed.require_array_equal("grid", np.asarray([1.0, 3.0]))
    resumed.write_sample(0, {"inputs": np.asarray([5.0])})
    resumed.checkpoint(accepted_samples=1, next_candidate_index=1)
    resumed.arrays["inputs"][0] = np.asarray([6.0])
    if isinstance(resumed.arrays["inputs"], np.memmap):
        resumed.arrays["inputs"].flush()
    with pytest.raises(ValueError, match="sample chunk SHA-256"):
        RecoverableNpyDataset.resume(
            partial_dir=partial,
            recovery_path=recovery,
            shapes=shapes,
            sample_array_names=sample_arrays,
            run_contract=contract,
        )
