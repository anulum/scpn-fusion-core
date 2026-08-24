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
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import DeepONetEquilibriumAccelerator as PublicDeepONetRuntime
from scpn_fusion.core.deeponet_equilibrium import DeepONetEquilibriumAccelerator


def _write_artifact(
    path: Path,
    *,
    schema: str = "scpn-fusion.equilibrium-deeponet.v1",
    field_scale: float = 2.0,
    manifest_sha256: str = "a" * 64,
    overrides: dict[str, np.ndarray[Any, Any]] | None = None,
    drop: frozenset[str] = frozenset(),
) -> None:
    payload = {
        "artifact_schema": np.asarray([schema]),
        "branch_n_layers": np.asarray([2], dtype=np.int64),
        "branch_0_W": np.ones((3, 4), dtype=np.float64) * 0.1,
        "branch_0_b": np.zeros(4, dtype=np.float64),
        "branch_1_W": np.ones((4, 2), dtype=np.float64) * 0.2,
        "branch_1_b": np.zeros(2, dtype=np.float64),
        "trunk_n_layers": np.asarray([2], dtype=np.int64),
        "trunk_0_W": np.ones((2, 4), dtype=np.float64) * 0.1,
        "trunk_0_b": np.zeros(4, dtype=np.float64),
        "trunk_1_W": np.ones((4, 2), dtype=np.float64) * 0.2,
        "trunk_1_b": np.zeros(2, dtype=np.float64),
        "input_mean": np.zeros(3, dtype=np.float64),
        "input_std": np.ones(3, dtype=np.float64),
        "coordinates_rz_m": np.asarray([[3.0, -1.0], [4.0, -1.0], [3.0, 1.0], [4.0, 1.0]]),
        "coordinate_mean": np.asarray([3.5, 0.0]),
        "coordinate_std": np.asarray([0.5, 1.0]),
        "field_mean": np.arange(4, dtype=np.float64),
        "field_scale": np.asarray([field_scale]),
        "basis_width": np.asarray([2], dtype=np.int64),
        "grid_nh": np.asarray([2], dtype=np.int64),
        "grid_nw": np.asarray([2], dtype=np.int64),
        "feature_names": np.asarray(["a", "b", "c"]),
        "dataset_manifest_sha256": np.asarray([manifest_sha256]),
    }
    if overrides:
        payload.update(overrides)
    np.savez(path, **{name: value for name, value in payload.items() if name not in drop})


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


def test_deeponet_rust_and_numpy_runtime_parity(tmp_path: Path) -> None:
    extension = pytest.importorskip("scpn_fusion_rs")
    if not hasattr(extension, "PyDeepOnetEquilibrium"):
        pytest.skip("installed Rust extension predates the DeepONet runtime")
    artifact = tmp_path / "deeponet.npz"
    _write_artifact(artifact)
    native = DeepONetEquilibriumAccelerator(prefer_rust=True)
    reference = DeepONetEquilibriumAccelerator(prefer_rust=False)
    native.load_weights(artifact)
    reference.load_weights(artifact)
    features = np.asarray([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]])
    assert native.backend == "rust"
    assert reference.backend == "numpy"
    np.testing.assert_allclose(
        native.predict_batch(features),
        reference.predict_batch(features),
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    extreme_artifact = tmp_path / "extreme_deeponet.npz"
    _write_artifact(
        extreme_artifact,
        field_scale=1.0e100,
        overrides={"field_mean": np.zeros(4, dtype=np.float64)},
    )
    native.load_weights(extreme_artifact)
    reference.load_weights(extreme_artifact)
    extreme_features = np.full((1, 3), -1000.0, dtype=np.float64)
    native_extreme = native.predict_batch(extreme_features)
    reference_extreme = reference.predict_batch(extreme_features)
    assert np.max(np.abs(reference_extreme)) < 1.0e-20
    np.testing.assert_allclose(native_extreme, reference_extreme, rtol=1.0e-14, atol=1.0e-30)


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
    with pytest.raises(ValueError, match="single-row"):
        runtime.predict(np.ones((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        runtime.predict_batch(np.ones((2, 4)))

    unstable_artifact = tmp_path / "unstable.npz"
    _write_artifact(
        unstable_artifact,
        overrides={"branch_0_W": np.full((3, 4), np.finfo(np.float64).max)},
    )
    runtime.load_weights(unstable_artifact)
    with pytest.raises(RuntimeError, match="non-finite output"):
        runtime.predict(np.ones(3))


@pytest.mark.parametrize(
    ("field_scale", "manifest_sha256", "message"),
    [
        (0.0, "a" * 64, "scales must be positive"),
        (2.0, "short", "manifest digest is invalid"),
    ],
)
def test_deeponet_runtime_rejects_invalid_artifact_contracts(
    tmp_path: Path, field_scale: float, manifest_sha256: str, message: str
) -> None:
    artifact = tmp_path / "invalid.npz"
    _write_artifact(artifact, field_scale=field_scale, manifest_sha256=manifest_sha256)
    with pytest.raises(ValueError, match=message):
        DeepONetEquilibriumAccelerator().load_weights(artifact)


def test_deeponet_runtime_rejects_missing_artifact_member(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.npz"
    _write_artifact(incomplete, drop=frozenset({"field_mean"}))
    with pytest.raises(ValueError, match="missing keys"):
        DeepONetEquilibriumAccelerator().load_weights(incomplete)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"branch_n_layers": np.asarray([0])}, "layer count"),
        ({"branch_0_b": np.zeros(3)}, "inconsistent dimensions"),
        ({"branch_1_W": np.ones((3, 2))}, "input width is inconsistent"),
        ({"trunk_0_W": np.full((2, 4), np.nan)}, "non-finite values"),
        ({"input_mean": np.zeros((1, 3))}, "input normalisation"),
        ({"feature_names": np.asarray(["a", "a", "c"])}, "feature contract"),
        ({"coordinates_rz_m": np.zeros((3, 2))}, "coordinate grid"),
        ({"coordinate_mean": np.zeros(3)}, "coordinate normalisation"),
        ({"field_mean": np.zeros(3)}, "field mean"),
        ({"branch_0_W": np.ones((4, 4))}, "network inputs"),
        ({"basis_width": np.asarray([3])}, "basis widths differ"),
        ({"input_mean": np.asarray([0.0, np.nan, 0.0])}, "non-finite arrays"),
    ],
)
def test_deeponet_runtime_rejects_structurally_corrupt_artifacts(
    tmp_path: Path, overrides: dict[str, np.ndarray[Any, Any]], message: str
) -> None:
    artifact = tmp_path / "corrupt.npz"
    _write_artifact(artifact, overrides=overrides)
    with pytest.raises(ValueError, match=message):
        DeepONetEquilibriumAccelerator().load_weights(artifact)
