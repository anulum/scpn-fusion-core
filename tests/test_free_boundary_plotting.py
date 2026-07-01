# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Plotting Tests
"""Tests for headless free-boundary supervisory plotting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pytest

from scpn_fusion.control._free_boundary_plotting import plot_free_boundary_control
from scpn_fusion.control._free_boundary_supervisory_types import FloatArray, FreeBoundaryTarget

matplotlib.use("Agg", force=True)


def _sample_plot_inputs() -> tuple[FloatArray, FloatArray, FloatArray, FreeBoundaryTarget]:
    """Return a small finite supervisory trace for real rendering tests."""
    time_axis = np.linspace(0.0, 0.2, 3, dtype=np.float64)
    states = cast(
        FloatArray,
        np.column_stack(
            (
                np.array([6.00, 6.01, 6.02], dtype=np.float64),
                np.array([0.00, -0.01, -0.02], dtype=np.float64),
                np.array([5.02, 5.01, 5.00], dtype=np.float64),
                np.array([-3.48, -3.49, -3.50], dtype=np.float64),
            ),
        ),
    )
    actions = cast(
        FloatArray,
        np.column_stack(
            (
                np.array([0.00, 0.04, 0.03], dtype=np.float64),
                np.array([0.00, -0.02, -0.01], dtype=np.float64),
            ),
        ),
    )
    target = FreeBoundaryTarget(6.01, -0.01, 5.01, -3.49)
    return time_axis, states, actions, target


def test_plot_free_boundary_control_writes_nested_headless_plot(tmp_path: Path) -> None:
    """Render a real PNG under Agg and create missing output directories."""
    time_axis, states, actions, target = _sample_plot_inputs()
    output_path = tmp_path / "plots" / "free_boundary.png"

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path=str(output_path),
    )

    assert saved is True
    assert error is None
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_free_boundary_control_rejects_mismatched_history_lengths(
    tmp_path: Path,
) -> None:
    """Reject traces where the time, state, and action rows cannot align."""
    time_axis, states, actions, target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=time_axis[:-1],
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == (
        "invalid free-boundary plot input: time_axis, states, and actions must "
        "have matching row counts."
    )


def test_plot_free_boundary_control_rejects_empty_time_axis(tmp_path: Path) -> None:
    """Reject empty simulation time axes before rendering."""
    _, states, actions, target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=np.array([], dtype=np.float64),
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: time_axis must be a non-empty 1-D array."


def test_plot_free_boundary_control_rejects_nonfinite_time_axis(
    tmp_path: Path,
) -> None:
    """Reject non-finite time samples before rendering."""
    time_axis, states, actions, target = _sample_plot_inputs()
    time_axis[1] = np.inf

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: time_axis must contain finite values."


def test_plot_free_boundary_control_rejects_state_history_with_too_few_columns(
    tmp_path: Path,
) -> None:
    """Reject geometry histories that cannot contain axis and X-point states."""
    time_axis, states, actions, target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states[:, :3],
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == (
        "invalid free-boundary plot input: states must be a non-empty 2-D array "
        "with at least 4 columns."
    )


def test_plot_free_boundary_control_rejects_nonfinite_state_history(
    tmp_path: Path,
) -> None:
    """Reject NaN and infinite geometry traces before calling Matplotlib."""
    time_axis, states, actions, target = _sample_plot_inputs()
    states[1, 2] = np.nan

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: states must contain finite values."


def test_plot_free_boundary_control_rejects_nonfinite_target(tmp_path: Path) -> None:
    """Reject target coordinates that would draw misleading reference lines."""
    time_axis, states, actions, _target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=FreeBoundaryTarget(np.nan, -0.01, 5.01, -3.49),
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: target coordinates must be finite."


def test_plot_free_boundary_control_rejects_empty_output_path() -> None:
    """Reject blank output destinations before importing Matplotlib."""
    time_axis, states, actions, target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path="   ",
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: output_path must not be empty."


def test_plot_free_boundary_control_rejects_directory_output_path() -> None:
    """Reject destinations that resolve to a directory root instead of a file."""
    time_axis, states, actions, target = _sample_plot_inputs()

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path="/",
    )

    assert saved is False
    assert error == "invalid free-boundary plot input: output_path must point to a file."


def test_plot_free_boundary_control_reports_matplotlib_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return a structured error when the optional plotting dependency is absent."""
    time_axis, states, actions, target = _sample_plot_inputs()
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error is not None
    assert error.startswith("matplotlib unavailable:")


def test_plot_free_boundary_control_closes_figure_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Close Matplotlib figures even when the output backend raises."""
    from matplotlib import pyplot as plt

    time_axis, states, actions, target = _sample_plot_inputs()
    closed: list[object] = []
    original_close = plt.close

    def _raise_savefig(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    def _record_close(fig: Any | None = None) -> None:
        closed.append(fig)
        original_close(fig)

    monkeypatch.setattr(plt, "savefig", _raise_savefig)
    monkeypatch.setattr(plt, "close", _record_close)

    saved, error = plot_free_boundary_control(
        time_axis=time_axis,
        states=states,
        target=target,
        actions=actions,
        output_path=str(tmp_path / "plot.png"),
    )

    assert saved is False
    assert error == "plot render failed: disk full"
    assert closed
    assert closed[-1] is not None
