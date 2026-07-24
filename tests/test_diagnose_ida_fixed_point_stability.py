# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused tests for the IDA fixed-point stability diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import validation.diagnose_ida_fixed_point_stability as diagnostic


def test_vector_and_jvp_metrics_preserve_projection_and_gain() -> None:
    terminal = np.eye(3, dtype=np.float64)
    field = 0.5 * terminal
    vector = diagnostic._vector_metrics(field, terminal_error=terminal)
    assert vector["cosine_to_terminal_error"] == pytest.approx(1.0)
    assert vector["projection_on_terminal_error"] == pytest.approx(0.5)
    assert vector["relative_l2_to_terminal_error"] == pytest.approx(0.5)

    jvp = diagnostic._jvp_metrics(terminal, 2.0 * terminal)
    assert jvp["alignment_with_input"] == pytest.approx(1.0)
    assert jvp["gain_l2"] == pytest.approx(2.0)


def test_trajectory_row_uses_terminal_error_as_frozen_scale() -> None:
    reference = np.zeros((3, 3), dtype=np.float64)
    candidate = np.ones((3, 3), dtype=np.float64)
    midpoint = np.full((3, 3), 0.5, dtype=np.float64)
    row = diagnostic._trajectory_row(
        midpoint,
        step=1,
        reference=reference,
        candidate=candidate,
    )
    assert row["distance_to_reference_relative_to_terminal"] == pytest.approx(0.5)
    assert row["distance_to_candidate_relative_to_terminal"] == pytest.approx(0.5)
    assert row["projection_on_terminal_error"] == pytest.approx(0.5)


def test_helpers_fail_closed_on_shape_nonfinite_and_zero_direction() -> None:
    with pytest.raises(ValueError, match="finite non-trivial"):
        diagnostic._finite_plane(np.ones((2, 2)), field="small")
    with pytest.raises(ValueError, match="finite non-trivial"):
        diagnostic._finite_plane(np.full((3, 3), np.nan), field="nan")
    with pytest.raises(ValueError, match="non-zero norm"):
        diagnostic._jvp_metrics(np.zeros((3, 3)), np.ones((3, 3)))


def test_cli_rejects_invalid_existing_report_without_solver_execution(tmp_path: Path) -> None:
    path = tmp_path / "forged.json"
    path.write_text(
        json.dumps({"schema_version": "forged"}, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level fields"):
        diagnostic.main(["--validate-report", str(path)])
