#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rotating BVP Acceptance Report Tests
"""Regression tests for the implemented FUS-C.1 rotating rigid-rotor acceptance report."""

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


def test_frc_rotating_bvp_acceptance_report_is_implemented_and_cross_surface() -> None:
    module = _load_module()

    report = cast(dict[str, Any], module.build_report(run_rust=False))

    assert report["schema"] == "frc-rotating-bvp-acceptance.v2"
    assert report["accepted_rotating_closure"] is True
    assert report["status"] == "implemented_rostoker_qerushi_rotating_closure_accepted"
    assert report["python_status"]["status"] == "implemented_rostoker_qerushi_1d_rotating_closure"
    assert report["python_status"]["rotating_bvp_implemented"] is True
    # No-rotation contract still solves byte-unchanged.
    assert report["python_no_rotation_contract"]["passed"] is True
    assert (
        report["python_no_rotation_contract"]["model"] == "steinhauer_2011_no_rotation_analytical"
    )
    # Rotating equilibrium validates, non-negative, sub-sonic, in tolerance.
    rotating = report["python_rotating_contract"]
    assert rotating["passed"] is True
    assert rotating["pressure_non_negative"] is True
    assert rotating["pressure_clipped_fraction"] == 0.0
    assert rotating["rotation_mach_number"] < 0.2
    # Reduction to the no-rotation contract with omega^2 scaling and bit-exact field.
    reduction = report["python_reduction_to_contract"]
    assert reduction["passed"] is True
    assert reduction["quadratic_scaling"] is True
    assert reduction["field_bit_exact_with_no_rotation"] is True
    # Steinhauer Figure 3 parity boundary is held (not claimed, payload unavailable).
    boundary = report["steinhauer_figure3_boundary"]
    assert boundary["passed"] is True
    assert boundary["figure3_parity_claimed"] is False
    assert boundary["has_verified_pdf_payload"] is False
    assert report["rust_parity"]["status"] == "not_run"
