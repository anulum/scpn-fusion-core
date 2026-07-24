# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused tests for the IDA native-inverse operator-response diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import validation.diagnose_ida_operator_response as diagnostic
import validation.ida_operator_response_contract as contract
import validation.ida_operator_response_fields as fields


def test_zero_wall_sum_closure_and_forcing_metric_are_explicit() -> None:
    raw = np.arange(25, dtype=np.float64).reshape(5, 5)
    zero_wall = fields.zero_wall(raw, field="fixture")
    assert np.count_nonzero(zero_wall[[0, -1], :]) == 0
    assert np.count_nonzero(zero_wall[:, [0, -1]]) == 0
    assert np.array_equal(zero_wall[1:-1, 1:-1], raw[1:-1, 1:-1])

    first = np.ones((5, 5), dtype=np.float64)
    second = np.full((5, 5), 2.0, dtype=np.float64)
    total = fields.sum_fields((first, second))
    assert np.array_equal(total, np.full((5, 5), 3.0))
    assert fields.closure_max_abs(total, (first, second)) == 0.0
    metric = fields.forcing_metric(first, exact_residual=total)
    assert metric["relative_l2_to_exact_source_residual"] == pytest.approx(1.0 / 3.0)


def test_helpers_reject_shape_nonfinite_and_empty_decompositions() -> None:
    with pytest.raises(ValueError, match="finite non-trivial"):
        fields.finite_plane(np.ones((2, 2)), field="small")
    with pytest.raises(ValueError, match="finite non-trivial"):
        fields.zero_wall(np.full((3, 3), np.nan), field="nan")
    with pytest.raises(ValueError, match="must not be empty"):
        fields.sum_fields(())
    with pytest.raises(ValueError, match="matching finite"):
        fields.sum_fields(
            (
                np.ones((3, 3), dtype=np.float64),
                np.ones((4, 4), dtype=np.float64),
            )
        )


def test_native_inverse_uses_identity_wall_and_centered_delta_star() -> None:
    rhs = np.zeros((3, 3), dtype=np.float64)
    rhs[1, 1] = 1.0
    solution = fields.native_inverse(
        rhs,
        r_grid=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
        d_r=0.5,
        d_z=0.5,
        preconditioner=lambda value: jnp.asarray(value),
        x0_zr=np.zeros_like(rhs),
    )
    assert solution[1, 1] == pytest.approx(-0.0625, abs=1.0e-12)
    assert np.max(np.abs(solution[[0, -1], :])) <= 1.0e-12
    assert np.max(np.abs(solution[:, [0, -1]])) <= 1.0e-12


def test_operator_binding_checks_support_vector_digests() -> None:
    reference_current = np.zeros((5, 5), dtype=np.float64)
    reference_current[2, 2] = 1.0
    components = {
        name: np.full((5, 5), float(index + 1), dtype=np.float64)
        for index, name in enumerate(contract.COMPONENTS)
    }
    report_names = {
        "freegs_fourth_order_baseline": "freegs_fourth_order_baseline",
        "native_second_order_stencil": "second_order_operator",
        "coil_vacuum_discretisation": "vacuum_discretisation",
        "exact_source_convention": "exact_source_convention",
    }
    operator_report: dict[str, Any] = {
        "interior_components": {
            report_name: {
                "field_sha256": diagnostic._array_sha256(
                    np.asarray([components[name][2, 2]], dtype=np.float64)
                )
            }
            for name, report_name in report_names.items()
        }
    }
    fields.verify_operator_binding(
        components,
        reference_current_rz=reference_current,
        operator_report=operator_report,
    )
    operator_report["interior_components"]["second_order_operator"]["field_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reconstruction disagrees"):
        fields.verify_operator_binding(
            components,
            reference_current_rz=reference_current,
            operator_report=operator_report,
        )


def test_cli_rejects_invalid_existing_report_without_solver_execution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged.json"
    path.write_text(
        json.dumps({"schema_version": "forged"}, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level fields"):
        diagnostic.main(["--validate-report", str(path)])
