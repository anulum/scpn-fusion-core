#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rotating BVP Acceptance Report Tests
"""Regression tests for the fail-closed FUS-C.1 rotating-BVP acceptance report."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_frc_rotating_bvp_acceptance.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "benchmark_frc_rotating_bvp_acceptance",
        MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frc_rotating_bvp_acceptance_report_is_fail_closed_and_cross_surface() -> None:
    module = _load_module()

    report = cast(dict[str, Any], module.build_report(run_rust=False))

    assert report["schema"] == "frc-rotating-bvp-acceptance.v1"
    assert report["accepted_full_fidelity_rotating_bvp"] is False
    assert report["status"] == "blocked_rotating_bvp_reference_missing_fail_closed_contract_passed"
    assert report["python_status"]["status"] == "blocked_missing_verified_steinhauer_rotating_closure"
    assert report["python_no_rotation_contract"]["passed"] is True
    assert report["python_nonzero_rotation_contract"]["passed"] is True
    assert report["steinhauer_reference_gate"]["passed"] is True
    assert report["steinhauer_reference_gate"]["download_status"] == "blocked_by_publisher_http_403"
    assert report["rust_status"]["status"] == "not_run"
    assert "Steinhauer 2011 Section II.B plus Figure 3 closure" in report["missing_requirements"]
