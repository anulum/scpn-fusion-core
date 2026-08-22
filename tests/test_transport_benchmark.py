# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport Validation Benchmark Tests
"""Behavioural tests for the transport validation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from validation.benchmark_transport import (
    main,
    run_iter_scaling_benchmark,
    run_threshold_benchmark,
)


def test_critical_gradient_benchmark_separates_sub_and_supercritical_flux() -> None:
    """The analytic fallback stays closed below threshold and opens above it."""
    result = run_threshold_benchmark()

    assert result["pass"] is True
    assert result["chi_sub"] == 0.0
    assert result["chi_super"] > 0.0


def test_iter_scaling_benchmark_reports_finite_prediction_and_uncertainty() -> None:
    """The ITER scaling lane returns positive confinement and uncertainty values."""
    result = run_iter_scaling_benchmark()

    assert result["pass"] is True
    assert result["tau_predicted"] > 0.0
    assert result["uncertainty_sigma"] > 0.0


def test_main_writes_machine_and_human_readable_reports(tmp_path: Path) -> None:
    """The entry point writes consistent JSON and Markdown from real benchmark runs."""
    main(nr=20, report_dir=tmp_path)

    json_path = tmp_path / "transport_benchmark.json"
    markdown_path = tmp_path / "transport_benchmark.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["pure_diffusion"]["pass"] is True
    assert payload["threshold"]["pass"] is True
    assert payload["iter_scaling"]["pass"] is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Transport Validation Benchmark" in markdown
    assert "| Pure Diffusion |" in markdown
