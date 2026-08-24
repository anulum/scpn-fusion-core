# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DeepONet Module Contract Tests
"""Direct public-contract tests complementing the end-to-end training suite."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest

from scpn_fusion.core.deeponet_training import (
    adamw_step,
    init_network,
    operator_forward,
    relative_field_objective,
    validation_objective,
)
from scpn_fusion.core.deeponet_training_contracts import TrainingConfig
from scpn_fusion.io.deeponet_training_recovery import serialize_network
from scpn_fusion.io.deeponet_training_report import running_report
from scpn_fusion.io.machine_conditioned_deeponet_cli import run_deeponet_cli
from scpn_fusion.io.machine_conditioned_surrogate_training import (
    MachineConditionedSplit,
    load_machine_conditioned_training_data,
)

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "validation/reference/iter_machine_conditioned_v2_n3_seed20260822_33x33"


def test_training_math_and_recovery_payload_are_deterministic() -> None:
    branch = init_network(random.PRNGKey(1), input_dim=3, hidden_sizes=(4,), output_dim=2)
    trunk = init_network(random.PRNGKey(2), input_dim=2, hidden_sizes=(4,), output_dim=2)
    params = {"branch": branch, "trunk": trunk}
    features = jnp.asarray([[0.25, -0.5, 0.75]], dtype=jnp.float32)
    coordinates = jnp.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)
    targets = jnp.asarray([[0.1, -0.2]], dtype=jnp.float32)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    prediction = operator_forward(params, features, coordinates)
    loss = relative_field_objective(params, features, coordinates, targets, weights)
    assert prediction.shape == (1, 2)
    assert np.isfinite(np.asarray(prediction)).all()
    assert float(loss) == pytest.approx(
        float(validation_objective(params, features, coordinates, targets, weights))
    )

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    updated, first_moment, second_moment, step_loss = adamw_step(
        params,
        zeros,
        zeros,
        features,
        coordinates,
        targets,
        weights,
        1.0e-3,
        1.0e-6,
        1.0,
        1,
    )
    assert float(step_loss) == pytest.approx(float(loss))
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(updated),
            strict=True,
        )
    )
    assert all(
        np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(first_moment)
    )
    assert all(
        np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(second_moment)
    )

    payload: dict[str, Any] = {}
    serialize_network(payload, "branch", branch)
    assert int(payload["branch_n_layers"][0]) == len(branch)
    assert payload["branch_0_W"].dtype == np.float64
    assert np.array_equal(payload["branch_0_W"], np.asarray(branch[0]["W"], dtype=np.float64))


def test_running_report_keeps_claims_closed_and_split_roles_explicit(tmp_path: Path) -> None:
    data = load_machine_conditioned_training_data(REFERENCE, full_field_scan=True)
    split = MachineConditionedSplit(
        training=np.asarray([0], dtype=np.int64),
        validation=np.asarray([1], dtype=np.int64),
        calibration=np.asarray([], dtype=np.int64),
        test=np.asarray([2], dtype=np.int64),
    )
    config = TrainingConfig(
        dataset_dir=REFERENCE,
        output_path=tmp_path / "candidate.npz",
        report_path=tmp_path / "report.json",
        checkpoint_dir=tmp_path / "checkpoints",
        steps=1,
        seed=7,
        validation_fraction=1.0 / 3.0,
        calibration_fraction=0.0,
        test_fraction=1.0 / 3.0,
        branch_hidden=(4,),
        trunk_hidden=(4,),
        basis_width=2,
        shot_batch_size=1,
        coordinate_batch_size=2,
        validation_probe_shots=1,
        validation_probe_coordinates=2,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        statistics_chunk_rows=1,
        evaluation_every=1,
        checkpoint_every=1,
        early_stopping_patience=1,
        resume=False,
    )
    report = running_report(
        data,
        split,
        {role: role * 8 for role in ("training", "validation", "calibration", "test")},
        config,
        "f" * 64,
        training_schema="test.training.v1",
    )
    assert report["status"] == "running"
    assert not any(
        report["claims"][name]
        for name in (
            "facility_validated",
            "cross_machine_validated",
            "experimental_shot_data",
            "free_boundary_prediction",
            "ida_or_efit_replacement",
        )
    )
    assert report["split"]["samples"] == {
        "training": 1,
        "validation": 1,
        "calibration": 0,
        "test": 1,
    }


def test_cli_adapter_preserves_the_public_argument_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    received: dict[str, Any] = {}

    def training_entrypoint(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"status": "completed_local_candidate_not_promoted"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_machine_conditioned_deeponet.py",
            "--dataset-dir",
            str(REFERENCE),
            "--out",
            str(tmp_path / "candidate.npz"),
            "--report",
            str(tmp_path / "report.json"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--steps",
            "7",
            "--basis-width",
            "5",
            "--resume",
        ],
    )
    with caplog.at_level(
        logging.INFO,
        logger="scpn_fusion.io.machine_conditioned_deeponet_cli",
    ):
        run_deeponet_cli(training_entrypoint, default_basis_width=3)
    assert caplog.messages[-1] == "completed_local_candidate_not_promoted"
    assert received["dataset_dir"] == REFERENCE
    assert received["steps"] == 7
    assert received["basis_width"] == 5
    assert received["resume"] is True
