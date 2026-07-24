# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused tests for fixed-reference operator decomposition helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

import validation.diagnose_ida_fixed_reference_operator as diagnostic
import validation.ida_fixed_reference_operator_contract as contract


def test_runtime_source_artifact_hashes_executed_module(tmp_path: Path) -> None:
    source_path = tmp_path / "boundary.py"
    source_bytes = b"def free_boundary():\\n    return True\\n"
    source_path.write_bytes(source_bytes)
    artifact = diagnostic._runtime_source_artifact(
        SimpleNamespace(__file__=str(source_path)),
        logical_path=contract.SOURCE_PATHS["freegs_boundary"],
        resource_name="boundary.py",
    )
    assert artifact == {
        "path": "python-package://freegs/boundary.py",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def test_runtime_source_artifact_rejects_uninspectable_module(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no inspectable"):
        diagnostic._runtime_source_artifact(
            SimpleNamespace(),
            logical_path=contract.SOURCE_PATHS["freegs_boundary"],
            resource_name="boundary.py",
        )
    wrong_path = tmp_path / "other.py"
    wrong_path.write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not resolve"):
        diagnostic._runtime_source_artifact(
            SimpleNamespace(__file__=str(wrong_path)),
            logical_path=contract.SOURCE_PATHS["freegs_boundary"],
            resource_name="boundary.py",
        )


def test_public_example_uses_bound_digest_and_rejects_local_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    public_path = tmp_path / contract.FREEGS_EXAMPLE_PATH
    source_bytes = b"# frozen public source\n"
    binding = {
        "public_example": {
            "path": contract.FREEGS_EXAMPLE_PATH,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    }
    assert (
        diagnostic._bound_public_example_artifact(
            binding,
            public_example_path=public_path,
        )
        == binding["public_example"]
    )

    public_path.parent.mkdir(parents=True)
    public_path.write_bytes(b"# drifted source\n")
    with pytest.raises(ValueError, match="bytes disagree"):
        diagnostic._bound_public_example_artifact(
            binding,
            public_example_path=public_path,
        )


def test_metric_binds_field_and_same_unit_reference_scale() -> None:
    field = np.asarray([3.0, 4.0], dtype=np.float64)
    scale = np.asarray([0.0, 10.0], dtype=np.float64)
    row = diagnostic._metric(field, reference_scale=scale)
    assert row["relative_l2_to_reference_scale"] == pytest.approx(0.5)
    assert row["rms"] == pytest.approx(np.sqrt(12.5))
    assert row["linf"] == 4.0
    assert len(row["field_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "scale"),
    [
        (np.ones(2), np.ones(3)),
        (np.asarray([np.nan]), np.ones(1)),
        (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
    ],
)
def test_metric_rejects_invalid_fields(
    field: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="matching finite arrays"):
        diagnostic._metric(field, reference_scale=scale)


def test_source_field_and_exact_algebraic_closure() -> None:
    current = np.arange(12, dtype=np.float64).reshape(3, 4)
    r_grid = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
    source = diagnostic._source_field(current=current, r_grid=r_grid, mu0=2.0)
    assert source == pytest.approx(-2.0 * r_grid[:, None] * current)

    baseline = np.full((2, 2), 1.0, dtype=np.float64)
    operator = np.full((2, 2), 2.0, dtype=np.float64)
    vacuum = np.full((2, 2), -0.5, dtype=np.float64)
    source_delta = np.full((2, 2), 0.25, dtype=np.float64)
    actual = baseline + operator + vacuum + source_delta
    assert (
        diagnostic._closure_max_abs(
            actual,
            (baseline, operator, vacuum, source_delta),
        )
        == 0.0
    )


def test_native_laplacian_star_matches_manufactured_quadratic_interior() -> None:
    r_grid = np.linspace(1.0, 2.0, 9, dtype=np.float64)
    z_grid = np.linspace(-0.5, 0.5, 9, dtype=np.float64)
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="ij")
    psi_rz = rr**2 + zz**2
    lhs = diagnostic._native_lhs(psi_rz, r_grid=r_grid, z_grid=z_grid)
    # Δ*(R² + Z²) = 2 - 2 + 2 = 2.
    assert lhs[1:-1, 1:-1] == pytest.approx(2.0, abs=1.0e-12)


def test_interior_and_source_shape_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-trivial"):
        diagnostic._interior(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="first dimension"):
        diagnostic._source_field(
            current=np.ones((2, 3), dtype=np.float64),
            r_grid=np.ones(3, dtype=np.float64),
            mu0=1.0,
        )
    with pytest.raises(ValueError, match="interior mask"):
        diagnostic._masked_interior(
            np.ones((4, 4), dtype=np.float64),
            np.zeros((2, 2), dtype=np.bool_),
        )


def test_real_diiid_fixed_reference_operator_decomposition() -> None:
    """The public 129² case must preserve the measured routing mechanism."""
    report = diagnostic.run_diagnostic(generated_at="2026-07-23T11:20:00Z")
    contract.validate_report(report)
    interior = report["interior_components"]
    coil_region = report["coil_region_diagnostic"]
    wall = report["wall_components"]
    assert report["routing"]["interior_dominant_component"] == "exact_source_convention"
    assert report["routing"]["wall_dominant_component"] == "exact_source_convention"
    assert interior["freegs_fourth_order_baseline"]["relative_l2_to_reference_scale"] < 1.0e-9
    assert interior["vacuum_discretisation"]["relative_l2_to_reference_scale"] < 1.0e-3
    assert coil_region["all_interior_vacuum_field"]["relative_l2_to_reference_scale"] > 1.0
    assert coil_region["outside_reference_support_l2_fraction"] > 0.999
    assert coil_region["coil_filaments_inside_domain"] == coil_region["coil_filament_count"]
    assert wall["coil_vacuum_convention"]["relative_l2_to_reference_scale"] < 1.0e-12
    assert max(report["closure"].values()) < contract.CLOSURE_MAX_ABS
