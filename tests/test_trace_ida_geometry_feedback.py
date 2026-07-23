# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused runner tests for exact IDA compiled-trace evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import validation.ida_geometry_feedback_trace_metrics as metrics
import validation.trace_ida_geometry_feedback as runner


def test_trace_run_merges_exact_terminal_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dynamic terminal replaces a duplicate sparse row and remains unique."""

    def fake_checkpoint_row(**kwargs: Any) -> dict[str, Any]:
        return {
            "iteration_index": kwargs["iteration_index"],
            "terminal": kwargs["terminal"],
        }

    monkeypatch.setattr(metrics, "_checkpoint_row", fake_checkpoint_row)
    plane = np.ones((2, 2), dtype=np.float64)
    trace = SimpleNamespace(
        checkpoint_indices=(0, 4, 9),
        converged=np.array([False, False, False]),
        equilibrium=plane,
        fixed_point_residual=np.stack([plane, plane, plane]),
        ip_now=np.array([0.2, 1.0, 0.0]),
        iteration_count=5,
        psi_after=np.stack([plane, plane, plane]),
        psi_before=np.stack([plane, plane, plane]),
        recorded=np.array([True, True, False]),
        separatrix_refinement=np.array([0.0, 0.0, 0.0]),
        terminal_converged=True,
        terminal_fixed_point_residual=plane,
        terminal_ip_now=1.0,
        terminal_iteration_index=4,
        terminal_psi_after=plane,
        terminal_psi_before=plane,
        terminal_separatrix_refinement=0.0,
    )
    result = metrics.trace_run(
        run_name="cold",
        trace=trace,
        requested_indices=(0, 4, 9),
        iteration_cap=10,
        context={},
    )
    assert result["iteration_count"] == 5
    assert result["terminated_early"] is True
    assert result["checkpoints"] == [
        {"iteration_index": 0, "terminal": False},
        {"iteration_index": 4, "terminal": True},
    ]


def test_finite_array_rejects_shape_and_nonfinite_values() -> None:
    """Metric inputs must be finite two-dimensional planes."""
    assert metrics._finite_array(np.ones((2, 2)), field="plane").shape == (2, 2)
    with pytest.raises(ValueError, match="two-dimensional"):
        metrics._finite_array(np.ones(2), field="plane")
    with pytest.raises(ValueError, match="finite"):
        metrics._finite_array(np.array([[np.nan]]), field="plane")


def test_runner_rejects_invalid_existing_report_without_executing_solver(
    tmp_path: Path,
) -> None:
    """The CLI validation path must fail closed before any solver execution."""
    report_path = tmp_path / "trace.json"
    report_path.write_text(
        json.dumps({"schema_version": "forged"}, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level fields"):
        runner.main(["--validate-report", str(report_path)])
