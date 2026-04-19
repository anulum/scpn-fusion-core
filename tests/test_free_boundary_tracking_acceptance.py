# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Acceptance Tests
"""Tests for deterministic free-boundary acceptance validation.

Requires FusionKernel.build_coilset_from_config() — not yet ported to
scpn-fusion-core.  All tests in this file are skipped until the kernel
API is extended.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skip(reason="FusionKernel.build_coilset_from_config not yet ported")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "free_boundary_tracking_acceptance.py"
SPEC = importlib.util.spec_from_file_location("free_boundary_tracking_acceptance", MODULE_PATH)
assert SPEC and SPEC.loader
free_boundary_tracking_acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = free_boundary_tracking_acceptance
SPEC.loader.exec_module(free_boundary_tracking_acceptance)


@lru_cache(maxsize=1)
def _campaign_cached() -> dict[str, object]:
    return free_boundary_tracking_acceptance.run_campaign()


def test_campaign_is_deterministic() -> None:
    a = free_boundary_tracking_acceptance.run_campaign()
    b = free_boundary_tracking_acceptance.run_campaign()
    assert a["passes_thresholds"] is True
    assert b["passes_thresholds"] is True
    for scenario in (
        "nominal",
        "coil_kick",
        "measurement_fault_uncorrected",
        "measurement_fault_corrected",
        "measurement_latency_uncorrected",
        "measurement_latency_corrected",
        "x_point_divertor_kick",
        "x_point_divertor_measurement_latency_uncorrected",
        "x_point_divertor_measurement_latency_corrected",
        "x_point_divertor_supervisor_measurement_fault_uncorrected",
        "x_point_divertor_supervisor_measurement_fault_corrected",
        "x_point_divertor_supervisor_measurement_latency_uncorrected",
        "x_point_divertor_supervisor_measurement_latency_corrected",
        "x_point_divertor_combined_fault_uncorrected",
        "x_point_divertor_combined_fault_corrected",
        "x_point_divertor_measurement_fault_uncorrected",
        "x_point_divertor_measurement_fault_corrected",
        "supervisor_fallback_kick",
    ):
        assert a["scenarios"][scenario]["summary"]["final_tracking_error_norm"] == pytest.approx(
            b["scenarios"][scenario]["summary"]["final_tracking_error_norm"]
        )
        assert (
            a["scenarios"][scenario]["passes_thresholds"]
            == b["scenarios"][scenario]["passes_thresholds"]
        )
    for sweep in (
        "measurement_fault_scale",
        "measurement_fault_corrected_scale",
        "measurement_latency_steps",
        "measurement_latency_corrected_steps",
        "actuator_slew_limit",
        "coil_kick_scale",
        "topology_kick_scale",
        "topology_actuator_slew_limit",
        "topology_measurement_fault_scale",
        "topology_measurement_corrected_scale",
        "topology_supervisor_measurement_fault_scale",
        "topology_supervisor_measurement_corrected_scale",
    ):
        assert a["sweeps"][sweep]["passes_thresholds"] == b["sweeps"][sweep]["passes_thresholds"]


def test_campaign_thresholds_pass() -> None:
    out = _campaign_cached()
    assert out["passes_thresholds"] is True
    assert out["scenarios"]["nominal"]["passes_thresholds"] is True
    assert out["scenarios"]["coil_kick"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_fault_uncorrected"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_fault_corrected"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_latency_uncorrected"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_latency_corrected"]["passes_thresholds"] is True
    assert out["scenarios"]["x_point_divertor_kick"]["passes_thresholds"] is True
    assert (
        out["scenarios"]["x_point_divertor_measurement_latency_uncorrected"]["passes_thresholds"]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_measurement_latency_corrected"]["passes_thresholds"]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_supervisor_measurement_fault_uncorrected"][
            "passes_thresholds"
        ]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_supervisor_measurement_fault_corrected"][
            "passes_thresholds"
        ]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_supervisor_measurement_latency_uncorrected"][
            "passes_thresholds"
        ]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_supervisor_measurement_latency_corrected"][
            "passes_thresholds"
        ]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_combined_fault_uncorrected"]["passes_thresholds"] is True
    )
    assert (
        out["scenarios"]["x_point_divertor_combined_fault_corrected"]["passes_thresholds"] is True
    )
    assert (
        out["scenarios"]["x_point_divertor_measurement_fault_uncorrected"]["passes_thresholds"]
        is True
    )
    assert (
        out["scenarios"]["x_point_divertor_measurement_fault_corrected"]["passes_thresholds"]
        is True
    )
    assert out["scenarios"]["supervisor_fallback_kick"]["passes_thresholds"] is True
    assert out["sweeps"]["measurement_fault_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["measurement_fault_corrected_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["measurement_latency_steps"]["passes_thresholds"] is True
    assert out["sweeps"]["measurement_latency_corrected_steps"]["passes_thresholds"] is True
    assert out["sweeps"]["actuator_slew_limit"]["passes_thresholds"] is True
    assert out["sweeps"]["coil_kick_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["topology_kick_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["topology_actuator_slew_limit"]["passes_thresholds"] is True
    assert out["sweeps"]["topology_measurement_fault_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["topology_measurement_corrected_scale"]["passes_thresholds"] is True
    assert out["sweeps"]["topology_supervisor_measurement_fault_scale"]["passes_thresholds"] is True
    assert (
        out["sweeps"]["topology_supervisor_measurement_corrected_scale"]["passes_thresholds"]
        is True
    )


def test_measurement_fault_scenarios_expose_and_remove_gap() -> None:
    out = _campaign_cached()
    uncorrected = out["scenarios"]["measurement_fault_uncorrected"]
    corrected = out["scenarios"]["measurement_fault_corrected"]
    nominal = out["scenarios"]["nominal"]

    assert uncorrected["measured_true_gap"] >= uncorrected["thresholds"]["min_measured_true_gap"]
    assert (
        uncorrected["summary"]["max_abs_measurement_offset"]
        >= uncorrected["thresholds"]["min_measurement_offset"]
    )
    assert corrected["measured_true_gap"] <= corrected["thresholds"]["max_measured_true_gap"]
    assert (
        corrected["summary"]["max_abs_measurement_offset"]
        <= corrected["thresholds"]["max_measurement_offset"]
    )
    assert corrected["summary"]["final_tracking_error_norm"] == pytest.approx(
        nominal["summary"]["final_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )


def test_measurement_sweep_gap_and_offset_grow_monotonically() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["measurement_fault_scale"]
    gaps = [entry["measured_true_gap"] for entry in sweep["entries"]]
    offsets = [entry["max_abs_measurement_offset"] for entry in sweep["entries"]]

    assert sweep["checks"]["measured_true_gap_monotone"] is True
    assert sweep["checks"]["measurement_offset_monotone"] is True
    assert gaps[0] == pytest.approx(0.0)
    assert offsets[0] == pytest.approx(0.0)
    assert gaps[-1] > gaps[1]
    assert offsets[-1] > offsets[1]


def test_latency_scenarios_bound_delayed_and_corrected_observation_error() -> None:
    out = _campaign_cached()
    uncorrected = out["scenarios"]["measurement_latency_uncorrected"]
    corrected = out["scenarios"]["measurement_latency_corrected"]

    assert (
        uncorrected["delayed_observation_error_norm"]
        >= uncorrected["thresholds"]["min_delayed_observation_error_norm"]
    )
    assert (
        uncorrected["summary"]["final_true_tracking_error_norm"]
        <= uncorrected["thresholds"]["max_true_tracking_error_norm"]
    )
    assert corrected["summary"]["latency_compensation_enabled"] is True
    assert (
        corrected["estimated_observation_error_norm"]
        <= corrected["thresholds"]["max_estimated_observation_error_norm"]
    )
    assert (
        corrected["summary"]["final_true_tracking_error_norm"]
        <= corrected["thresholds"]["max_final_true_tracking_error_norm"]
    )
    assert (
        corrected["estimated_observation_error_norm"]
        <= uncorrected["delayed_observation_error_norm"] + 1.0e-3
    )
    assert (
        abs(
            corrected["summary"]["final_true_tracking_error_norm"]
            - uncorrected["summary"]["final_true_tracking_error_norm"]
        )
        <= 1.0e-6
    )


def test_latency_sweeps_activate_with_delay_and_stay_corrected() -> None:
    out = _campaign_cached()
    uncorrected = out["sweeps"]["measurement_latency_steps"]
    corrected = out["sweeps"]["measurement_latency_corrected_steps"]
    delayed_error = [entry["delayed_observation_error_norm"] for entry in uncorrected["entries"]]
    corrected_error = [entry["estimated_observation_error_norm"] for entry in corrected["entries"]]

    assert uncorrected["checks"]["delayed_observation_error_monotone"] is True
    assert uncorrected["checks"]["delayed_observation_error_active"] is True
    assert uncorrected["checks"]["final_true_tracking_error_bounded"] is True
    assert corrected["checks"]["estimated_observation_error_bounded"] is True
    assert corrected["checks"]["final_true_tracking_error_bounded"] is True
    assert delayed_error[0] == pytest.approx(0.0)
    assert delayed_error[1] > delayed_error[0]
    assert delayed_error[-1] >= delayed_error[1] - 1.0e-6
    assert (
        max(corrected_error)
        <= free_boundary_tracking_acceptance.LATENCY_CORRECTED_THRESHOLDS[
            "max_estimated_observation_error_norm"
        ]
    )


def test_actuator_slew_sweep_lag_grows_as_limits_tighten() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["actuator_slew_limit"]
    max_lag = [entry["max_abs_actuator_lag"] for entry in sweep["entries"]]
    mean_lag = [entry["mean_abs_actuator_lag"] for entry in sweep["entries"]]
    max_coil_current = [entry["max_abs_coil_current"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_abs_actuator_lag_monotone"] is True
    assert sweep["checks"]["mean_abs_actuator_lag_monotone"] is True
    assert sweep["checks"]["max_abs_coil_current_monotone"] is True
    assert max_lag[-1] > max_lag[0]
    assert mean_lag[-1] > mean_lag[0]
    assert max_coil_current[-1] < max_coil_current[0]


def test_corrected_measurement_sweep_keeps_gap_collapsed() -> None:
    out = _campaign_cached()
    corrected = out["sweeps"]["measurement_fault_corrected_scale"]
    gaps = [entry["measured_true_gap"] for entry in corrected["entries"]]
    offsets = [entry["max_abs_measurement_offset"] for entry in corrected["entries"]]
    tracking = [entry["final_tracking_error_norm"] for entry in corrected["entries"]]

    assert corrected["checks"]["max_measured_true_gap"] is True
    assert corrected["checks"]["max_measurement_offset"] is True
    assert corrected["checks"]["tracking_error_constant"] is True
    assert (
        max(gaps) <= free_boundary_tracking_acceptance.CORRECTED_THRESHOLDS["max_measured_true_gap"]
    )
    assert (
        max(offsets)
        <= free_boundary_tracking_acceptance.CORRECTED_THRESHOLDS["max_measurement_offset"]
    )
    assert (
        max(tracking) - min(tracking)
        <= free_boundary_tracking_acceptance.CORRECTED_THRESHOLDS["max_measured_true_gap"]
    )


def test_kick_scale_sweep_requires_more_coil_authority_but_stays_bounded() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["coil_kick_scale"]
    max_coil_current = [entry["max_abs_coil_current"] for entry in sweep["entries"]]
    final_tracking_error = [entry["final_tracking_error_norm"] for entry in sweep["entries"]]
    objective_converged = [entry["objective_converged"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_abs_coil_current_monotone"] is True
    assert sweep["checks"]["objective_converged_all"] is True
    assert sweep["checks"]["final_tracking_error_bounded"] is True
    assert max_coil_current[-1] > max_coil_current[1]
    assert all(objective_converged)
    assert (
        max(final_tracking_error)
        <= free_boundary_tracking_acceptance.NOMINAL_THRESHOLDS["max_final_tracking_error_norm"]
    )


def test_x_point_divertor_kick_tracks_topology_objectives() -> None:
    out = _campaign_cached()
    scenario = out["scenarios"]["x_point_divertor_kick"]
    summary = scenario["summary"]
    thresholds = scenario["thresholds"]

    assert scenario["checks"]["final_tracking_error_norm"] is True
    assert scenario["checks"]["shape_rms"] is True
    assert scenario["checks"]["x_point_position_error"] is True
    assert scenario["checks"]["x_point_flux_error"] is True
    assert scenario["checks"]["divertor_rms"] is True
    assert scenario["checks"]["divertor_max_abs"] is True
    assert scenario["checks"]["objective_converged"] is True
    assert summary["x_point_position_error"] is not None
    assert summary["x_point_flux_error"] is not None
    assert summary["divertor_rms"] is not None
    assert summary["divertor_max_abs"] is not None
    assert summary["x_point_position_error"] <= thresholds["max_x_point_position_error"]
    assert summary["x_point_flux_error"] <= thresholds["max_x_point_flux_error"]
    assert summary["divertor_rms"] <= thresholds["max_divertor_rms"]
    assert summary["divertor_max_abs"] <= thresholds["max_divertor_max_abs"]
    assert summary["objective_converged"] is True


def test_topology_measurement_fault_scenarios_expose_and_remove_topology_gap() -> None:
    out = _campaign_cached()
    uncorrected = out["scenarios"]["x_point_divertor_measurement_fault_uncorrected"]
    corrected = out["scenarios"]["x_point_divertor_measurement_fault_corrected"]

    assert (
        uncorrected["x_point_position_gap"] >= uncorrected["thresholds"]["min_x_point_position_gap"]
    )
    assert uncorrected["x_point_flux_gap"] >= uncorrected["thresholds"]["min_x_point_flux_gap"]
    assert uncorrected["divertor_rms_gap"] >= uncorrected["thresholds"]["min_divertor_rms_gap"]
    assert (
        uncorrected["divertor_max_abs_gap"] >= uncorrected["thresholds"]["min_divertor_max_abs_gap"]
    )
    assert (
        uncorrected["summary"]["max_abs_measurement_offset"]
        >= uncorrected["thresholds"]["min_measurement_offset"]
    )
    assert (
        uncorrected["summary"]["true_x_point_position_error"]
        <= uncorrected["thresholds"]["max_true_x_point_position_error"]
    )
    assert (
        uncorrected["summary"]["true_x_point_flux_error"]
        <= uncorrected["thresholds"]["max_true_x_point_flux_error"]
    )
    assert (
        uncorrected["summary"]["true_divertor_rms"]
        <= uncorrected["thresholds"]["max_true_divertor_rms"]
    )
    assert (
        uncorrected["summary"]["true_divertor_max_abs"]
        <= uncorrected["thresholds"]["max_true_divertor_max_abs"]
    )
    assert uncorrected["summary"]["objective_converged"] is False

    assert corrected["x_point_position_gap"] <= corrected["thresholds"]["max_x_point_position_gap"]
    assert corrected["x_point_flux_gap"] <= corrected["thresholds"]["max_x_point_flux_gap"]
    assert corrected["divertor_rms_gap"] <= corrected["thresholds"]["max_divertor_rms_gap"]
    assert corrected["divertor_max_abs_gap"] <= corrected["thresholds"]["max_divertor_max_abs_gap"]
    assert (
        corrected["summary"]["max_abs_measurement_offset"]
        <= corrected["thresholds"]["max_measurement_offset"]
    )
    assert (
        corrected["summary"]["x_point_position_error"]
        <= corrected["thresholds"]["max_x_point_position_error"]
    )
    assert (
        corrected["summary"]["x_point_flux_error"]
        <= corrected["thresholds"]["max_x_point_flux_error"]
    )
    assert corrected["summary"]["divertor_rms"] <= corrected["thresholds"]["max_divertor_rms"]
    assert (
        corrected["summary"]["divertor_max_abs"] <= corrected["thresholds"]["max_divertor_max_abs"]
    )
    assert corrected["summary"]["objective_converged"] is True


def test_topology_measurement_latency_scenarios_expose_gap_and_restore_tracking() -> None:
    out = _campaign_cached()
    uncorrected = out["scenarios"]["x_point_divertor_measurement_latency_uncorrected"]
    corrected = out["scenarios"]["x_point_divertor_measurement_latency_corrected"]

    assert (
        uncorrected["x_point_position_gap"] >= uncorrected["thresholds"]["min_x_point_position_gap"]
    )
    assert uncorrected["x_point_flux_gap"] >= uncorrected["thresholds"]["min_x_point_flux_gap"]
    assert uncorrected["divertor_rms_gap"] >= uncorrected["thresholds"]["min_divertor_rms_gap"]
    assert (
        uncorrected["divertor_max_abs_gap"] >= uncorrected["thresholds"]["min_divertor_max_abs_gap"]
    )
    assert (
        uncorrected["delayed_observation_error_norm"]
        >= uncorrected["thresholds"]["min_delayed_observation_error_norm"]
    )
    assert (
        uncorrected["summary"]["final_true_tracking_error_norm"]
        <= uncorrected["thresholds"]["max_true_tracking_error_norm"]
    )
    assert uncorrected["summary"]["objective_converged"] is False

    assert corrected["x_point_position_gap"] <= corrected["thresholds"]["max_x_point_position_gap"]
    assert corrected["x_point_flux_gap"] <= corrected["thresholds"]["max_x_point_flux_gap"]
    assert corrected["divertor_rms_gap"] <= corrected["thresholds"]["max_divertor_rms_gap"]
    assert corrected["divertor_max_abs_gap"] <= corrected["thresholds"]["max_divertor_max_abs_gap"]
    assert (
        corrected["estimated_observation_error_norm"]
        <= corrected["thresholds"]["max_estimated_observation_error_norm"]
    )
    assert (
        corrected["summary"]["final_true_tracking_error_norm"]
        <= corrected["thresholds"]["max_final_true_tracking_error_norm"]
    )
    assert corrected["summary"]["objective_converged"] is True


def test_topology_combined_fault_scenarios_expose_and_remove_topology_gap() -> None:
    out = _campaign_cached()
    baseline = out["scenarios"]["x_point_divertor_kick"]
    uncorrected = out["scenarios"]["x_point_divertor_combined_fault_uncorrected"]
    corrected = out["scenarios"]["x_point_divertor_combined_fault_corrected"]

    assert (
        uncorrected["x_point_position_gap"] >= uncorrected["thresholds"]["min_x_point_position_gap"]
    )
    assert uncorrected["x_point_flux_gap"] >= uncorrected["thresholds"]["min_x_point_flux_gap"]
    assert uncorrected["divertor_rms_gap"] >= uncorrected["thresholds"]["min_divertor_rms_gap"]
    assert (
        uncorrected["divertor_max_abs_gap"] >= uncorrected["thresholds"]["min_divertor_max_abs_gap"]
    )
    assert (
        uncorrected["summary"]["max_abs_measurement_offset"]
        >= uncorrected["thresholds"]["min_measurement_offset"]
    )
    assert (
        uncorrected["summary"]["true_x_point_flux_error"]
        <= uncorrected["thresholds"]["max_true_x_point_flux_error"]
    )
    assert (
        uncorrected["summary"]["true_divertor_rms"]
        <= uncorrected["thresholds"]["max_true_divertor_rms"]
    )
    assert (
        uncorrected["summary"]["true_divertor_max_abs"]
        <= uncorrected["thresholds"]["max_true_divertor_max_abs"]
    )
    assert uncorrected["summary"]["objective_converged"] is False

    assert corrected["summary"]["final_tracking_error_norm"] == pytest.approx(
        baseline["summary"]["final_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["x_point_flux_error"] == pytest.approx(
        baseline["summary"]["x_point_flux_error"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["divertor_rms"] == pytest.approx(
        baseline["summary"]["divertor_rms"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["divertor_max_abs"] == pytest.approx(
        baseline["summary"]["divertor_max_abs"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert (
        corrected["summary"]["max_abs_measurement_offset"]
        <= corrected["thresholds"]["max_measurement_offset"]
    )
    assert corrected["summary"]["objective_converged"] is True


def test_topology_supervisor_measurement_scenarios_stay_safe_and_remove_gap() -> None:
    out = _campaign_cached()
    baseline = out["scenarios"]["supervisor_fallback_kick"]
    uncorrected = out["scenarios"]["x_point_divertor_supervisor_measurement_fault_uncorrected"]
    corrected = out["scenarios"]["x_point_divertor_supervisor_measurement_fault_corrected"]

    assert uncorrected["checks"]["supervisor_intervention_count"] is True
    assert uncorrected["checks"]["fallback_active_steps"] is True
    assert uncorrected["checks"]["max_abs_actuator_lag"] is True
    assert uncorrected["checks"]["supervisor_active"] is True
    assert uncorrected["checks"]["supervisor_safe"] is True
    assert (
        uncorrected["x_point_position_gap"] >= uncorrected["thresholds"]["min_x_point_position_gap"]
    )
    assert uncorrected["x_point_flux_gap"] >= uncorrected["thresholds"]["min_x_point_flux_gap"]
    assert uncorrected["divertor_rms_gap"] >= uncorrected["thresholds"]["min_divertor_rms_gap"]
    assert (
        uncorrected["divertor_max_abs_gap"] >= uncorrected["thresholds"]["min_divertor_max_abs_gap"]
    )
    assert (
        uncorrected["summary"]["max_abs_measurement_offset"]
        >= uncorrected["thresholds"]["min_measurement_offset"]
    )
    assert (
        uncorrected["summary"]["true_x_point_flux_error"]
        <= uncorrected["thresholds"]["max_true_x_point_flux_error"]
    )
    assert (
        uncorrected["summary"]["true_divertor_rms"]
        <= uncorrected["thresholds"]["max_true_divertor_rms"]
    )
    assert (
        uncorrected["summary"]["true_divertor_max_abs"]
        <= uncorrected["thresholds"]["max_true_divertor_max_abs"]
    )
    assert uncorrected["summary"]["objective_converged"] is False

    assert corrected["checks"]["supervisor_intervention_count"] is True
    assert corrected["checks"]["fallback_active_steps"] is True
    assert corrected["checks"]["max_abs_actuator_lag"] is True
    assert corrected["checks"]["supervisor_active"] is True
    assert corrected["checks"]["supervisor_safe"] is True
    assert corrected["summary"]["final_tracking_error_norm"] == pytest.approx(
        baseline["summary"]["final_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["x_point_flux_error"] == pytest.approx(
        baseline["summary"]["x_point_flux_error"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["divertor_rms"] == pytest.approx(
        baseline["summary"]["divertor_rms"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert corrected["summary"]["divertor_max_abs"] == pytest.approx(
        baseline["summary"]["divertor_max_abs"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert (
        corrected["summary"]["max_abs_measurement_offset"]
        <= corrected["thresholds"]["max_measurement_offset"]
    )
    assert corrected["summary"]["objective_converged"] is True


def test_topology_supervisor_latency_scenarios_stay_safe_and_bound_combined_faults() -> None:
    out = _campaign_cached()
    uncorrected = out["scenarios"]["x_point_divertor_supervisor_measurement_latency_uncorrected"]
    corrected = out["scenarios"]["x_point_divertor_supervisor_measurement_latency_corrected"]

    assert uncorrected["checks"]["supervisor_intervention_count"] is True
    assert uncorrected["checks"]["fallback_active_steps"] is True
    assert uncorrected["checks"]["max_abs_actuator_lag"] is True
    assert uncorrected["checks"]["supervisor_active"] is True
    assert uncorrected["checks"]["supervisor_safe"] is True
    assert (
        uncorrected["x_point_position_gap"] >= uncorrected["thresholds"]["min_x_point_position_gap"]
    )
    assert uncorrected["x_point_flux_gap"] >= uncorrected["thresholds"]["min_x_point_flux_gap"]
    assert uncorrected["divertor_rms_gap"] >= uncorrected["thresholds"]["min_divertor_rms_gap"]
    assert (
        uncorrected["divertor_max_abs_gap"] >= uncorrected["thresholds"]["min_divertor_max_abs_gap"]
    )
    assert (
        uncorrected["delayed_observation_error_norm"]
        >= uncorrected["thresholds"]["min_delayed_observation_error_norm"]
    )
    assert uncorrected["summary"]["objective_converged"] is False

    assert corrected["checks"]["supervisor_intervention_count"] is True
    assert corrected["checks"]["fallback_active_steps"] is True
    assert corrected["checks"]["max_abs_actuator_lag"] is True
    assert corrected["checks"]["supervisor_active"] is True
    assert corrected["checks"]["supervisor_safe"] is True
    assert corrected["x_point_position_gap"] <= corrected["thresholds"]["max_x_point_position_gap"]
    assert corrected["x_point_flux_gap"] <= corrected["thresholds"]["max_x_point_flux_gap"]
    assert corrected["divertor_rms_gap"] <= corrected["thresholds"]["max_divertor_rms_gap"]
    assert corrected["divertor_max_abs_gap"] <= corrected["thresholds"]["max_divertor_max_abs_gap"]
    assert (
        corrected["estimated_observation_error_norm"]
        <= corrected["thresholds"]["max_estimated_observation_error_norm"]
    )
    assert (
        corrected["summary"]["final_true_tracking_error_norm"]
        <= corrected["thresholds"]["max_final_true_tracking_error_norm"]
    )
    assert corrected["summary"]["objective_converged"] is True


def test_supervisor_fallback_kick_reduces_lag_and_stays_safe() -> None:
    out = _campaign_cached()
    scenario = out["scenarios"]["supervisor_fallback_kick"]
    summary = scenario["summary"]
    reference = scenario["unsupervised_reference"]
    thresholds = scenario["thresholds"]

    assert scenario["checks"]["supervisor_intervention_count"] is True
    assert scenario["checks"]["fallback_active_steps"] is True
    assert scenario["checks"]["max_abs_actuator_lag"] is True
    assert scenario["checks"]["lag_reduction_factor"] is True
    assert scenario["checks"]["final_tracking_error_norm"] is True
    assert scenario["checks"]["x_point_position_error"] is True
    assert scenario["checks"]["x_point_flux_error"] is True
    assert scenario["checks"]["divertor_rms"] is True
    assert scenario["checks"]["divertor_max_abs"] is True
    assert scenario["checks"]["supervisor_active"] is True
    assert scenario["checks"]["supervisor_safe"] is True
    assert scenario["checks"]["objective_converged"] is True
    assert (
        summary["supervisor_intervention_count"] >= thresholds["min_supervisor_intervention_count"]
    )
    assert summary["fallback_active_steps"] >= thresholds["min_fallback_active_steps"]
    assert summary["supervisor_active"] is True
    assert summary["supervisor_safe"] is True
    assert summary["max_abs_actuator_lag"] <= thresholds["max_abs_actuator_lag"]
    assert scenario["lag_reduction_factor"] >= thresholds["min_lag_reduction_factor"]
    assert reference["max_abs_actuator_lag"] > summary["max_abs_actuator_lag"]
    assert summary["final_tracking_error_norm"] <= thresholds["max_final_tracking_error_norm"]
    assert summary["x_point_position_error"] is not None
    assert summary["x_point_flux_error"] is not None
    assert summary["divertor_rms"] is not None
    assert summary["divertor_max_abs"] is not None
    assert summary["x_point_position_error"] <= thresholds["max_x_point_position_error"]
    assert summary["x_point_flux_error"] <= thresholds["max_x_point_flux_error"]
    assert summary["divertor_rms"] <= thresholds["max_divertor_rms"]
    assert summary["divertor_max_abs"] <= thresholds["max_divertor_max_abs"]
    assert summary["objective_converged"] is True


def test_topology_kick_sweep_keeps_objectives_bounded() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_kick_scale"]
    x_point_flux_error = [entry["x_point_flux_error"] for entry in sweep["entries"]]
    divertor_rms = [entry["divertor_rms"] for entry in sweep["entries"]]
    divertor_max_abs = [entry["divertor_max_abs"] for entry in sweep["entries"]]
    max_coil_current = [entry["max_abs_coil_current"] for entry in sweep["entries"]]
    objective_converged = [entry["objective_converged"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_abs_coil_current_monotone"] is True
    assert sweep["checks"]["x_point_flux_error_monotone"] is True
    assert sweep["checks"]["divertor_rms_monotone"] is True
    assert sweep["checks"]["divertor_max_abs_monotone"] is True
    assert sweep["checks"]["objective_converged_all"] is True
    assert sweep["checks"]["topology_errors_bounded"] is True
    assert max_coil_current[-1] > max_coil_current[0]
    assert x_point_flux_error[-1] > x_point_flux_error[0]
    assert divertor_rms[-1] > divertor_rms[0]
    assert divertor_max_abs[-1] > divertor_max_abs[0]
    assert all(objective_converged)
    assert (
        max(divertor_max_abs)
        <= free_boundary_tracking_acceptance.TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
    )


def test_topology_actuator_slew_sweep_tracks_constraint_tradeoff() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_actuator_slew_limit"]
    max_abs_actuator_lag = [entry["max_abs_actuator_lag"] for entry in sweep["entries"]]
    x_point_flux_error = [entry["x_point_flux_error"] for entry in sweep["entries"]]
    divertor_rms = [entry["divertor_rms"] for entry in sweep["entries"]]
    divertor_max_abs = [entry["divertor_max_abs"] for entry in sweep["entries"]]
    max_abs_coil_current = [entry["max_abs_coil_current"] for entry in sweep["entries"]]
    objective_converged = [entry["objective_converged"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_abs_actuator_lag_monotone"] is True
    assert sweep["checks"]["x_point_flux_error_monotone"] is True
    assert sweep["checks"]["divertor_rms_monotone"] is True
    assert sweep["checks"]["divertor_max_abs_monotone"] is True
    assert sweep["checks"]["max_abs_coil_current_monotone"] is True
    assert sweep["checks"]["objective_converged_all"] is True
    assert sweep["checks"]["topology_errors_bounded"] is True
    assert max_abs_actuator_lag[-1] > max_abs_actuator_lag[0]
    assert x_point_flux_error[-1] > x_point_flux_error[0]
    assert divertor_rms[-1] > divertor_rms[0]
    assert divertor_max_abs[-1] > divertor_max_abs[0]
    assert max_abs_coil_current[-1] < max_abs_coil_current[0]
    assert all(objective_converged)
    assert (
        max(divertor_max_abs)
        <= free_boundary_tracking_acceptance.TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
    )


def test_topology_measurement_sweep_gaps_grow_monotonically() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_measurement_fault_scale"]
    x_point_position_gap = [entry["x_point_position_gap"] for entry in sweep["entries"]]
    x_point_flux_gap = [entry["x_point_flux_gap"] for entry in sweep["entries"]]
    divertor_rms_gap = [entry["divertor_rms_gap"] for entry in sweep["entries"]]
    divertor_max_abs_gap = [entry["divertor_max_abs_gap"] for entry in sweep["entries"]]
    measurement_offset = [entry["max_abs_measurement_offset"] for entry in sweep["entries"]]

    assert sweep["checks"]["x_point_position_gap_monotone"] is True
    assert sweep["checks"]["x_point_flux_gap_monotone"] is True
    assert sweep["checks"]["divertor_rms_gap_monotone"] is True
    assert sweep["checks"]["divertor_max_abs_gap_monotone"] is True
    assert sweep["checks"]["measurement_offset_monotone"] is True
    assert x_point_position_gap[0] == pytest.approx(0.0)
    assert x_point_flux_gap[0] == pytest.approx(0.0)
    assert divertor_rms_gap[0] == pytest.approx(0.0)
    assert divertor_max_abs_gap[0] == pytest.approx(0.0)
    assert measurement_offset[0] == pytest.approx(0.0)
    assert x_point_position_gap[-1] > x_point_position_gap[1]
    assert x_point_flux_gap[-1] > x_point_flux_gap[1]
    assert divertor_rms_gap[-1] > divertor_rms_gap[1]
    assert divertor_max_abs_gap[-1] > divertor_max_abs_gap[1]
    assert measurement_offset[-1] > measurement_offset[1]


def test_topology_corrected_measurement_sweep_keeps_gaps_collapsed() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_measurement_corrected_scale"]
    x_point_position_gap = [entry["x_point_position_gap"] for entry in sweep["entries"]]
    x_point_flux_gap = [entry["x_point_flux_gap"] for entry in sweep["entries"]]
    divertor_rms_gap = [entry["divertor_rms_gap"] for entry in sweep["entries"]]
    divertor_max_abs_gap = [entry["divertor_max_abs_gap"] for entry in sweep["entries"]]
    measurement_offset = [entry["max_abs_measurement_offset"] for entry in sweep["entries"]]
    objective_converged = [entry["objective_converged"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_x_point_position_gap"] is True
    assert sweep["checks"]["max_x_point_flux_gap"] is True
    assert sweep["checks"]["max_divertor_rms_gap"] is True
    assert sweep["checks"]["max_divertor_max_abs_gap"] is True
    assert sweep["checks"]["max_measurement_offset"] is True
    assert sweep["checks"]["objective_converged_all"] is True
    assert (
        max(x_point_position_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS[
            "max_x_point_position_gap"
        ]
    )
    assert (
        max(x_point_flux_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
    )
    assert (
        max(divertor_rms_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
    )
    assert (
        max(divertor_max_abs_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS[
            "max_divertor_max_abs_gap"
        ]
    )
    assert (
        max(measurement_offset)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
    )
    assert all(objective_converged)


def test_topology_supervisor_measurement_sweep_gaps_grow_monotonically_while_safe() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_supervisor_measurement_fault_scale"]
    x_point_position_gap = [entry["x_point_position_gap"] for entry in sweep["entries"]]
    x_point_flux_gap = [entry["x_point_flux_gap"] for entry in sweep["entries"]]
    divertor_rms_gap = [entry["divertor_rms_gap"] for entry in sweep["entries"]]
    divertor_max_abs_gap = [entry["divertor_max_abs_gap"] for entry in sweep["entries"]]
    measurement_offset = [entry["max_abs_measurement_offset"] for entry in sweep["entries"]]
    supervisor_active = [entry["supervisor_active"] for entry in sweep["entries"]]
    supervisor_safe = [entry["supervisor_safe"] for entry in sweep["entries"]]
    interventions = [entry["supervisor_intervention_count"] for entry in sweep["entries"]]
    fallback_steps = [entry["fallback_active_steps"] for entry in sweep["entries"]]
    max_abs_actuator_lag = [entry["max_abs_actuator_lag"] for entry in sweep["entries"]]

    assert sweep["checks"]["x_point_position_gap_monotone"] is True
    assert sweep["checks"]["x_point_flux_gap_monotone"] is True
    assert sweep["checks"]["divertor_rms_gap_monotone"] is True
    assert sweep["checks"]["divertor_max_abs_gap_monotone"] is True
    assert sweep["checks"]["measurement_offset_monotone"] is True
    assert sweep["checks"]["supervisor_active_all"] is True
    assert sweep["checks"]["supervisor_safe_all"] is True
    assert sweep["checks"]["intervention_all"] is True
    assert sweep["checks"]["fallback_all"] is True
    assert sweep["checks"]["lag_bounded_all"] is True
    assert x_point_position_gap[0] == pytest.approx(0.0)
    assert x_point_flux_gap[0] == pytest.approx(0.0)
    assert divertor_rms_gap[0] == pytest.approx(0.0)
    assert divertor_max_abs_gap[0] == pytest.approx(0.0)
    assert measurement_offset[0] == pytest.approx(0.0)
    assert x_point_position_gap[-1] > x_point_position_gap[1]
    assert x_point_flux_gap[-1] > x_point_flux_gap[1]
    assert divertor_rms_gap[-1] > divertor_rms_gap[1]
    assert divertor_max_abs_gap[-1] > divertor_max_abs_gap[1]
    assert measurement_offset[-1] > measurement_offset[1]
    assert all(supervisor_active)
    assert all(supervisor_safe)
    assert min(interventions) >= 1
    assert min(fallback_steps) >= 1
    assert (
        max(max_abs_actuator_lag)
        <= free_boundary_tracking_acceptance.SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
    )


def test_topology_supervisor_corrected_measurement_sweep_keeps_gaps_collapsed() -> None:
    out = _campaign_cached()
    sweep = out["sweeps"]["topology_supervisor_measurement_corrected_scale"]
    x_point_position_gap = [entry["x_point_position_gap"] for entry in sweep["entries"]]
    x_point_flux_gap = [entry["x_point_flux_gap"] for entry in sweep["entries"]]
    divertor_rms_gap = [entry["divertor_rms_gap"] for entry in sweep["entries"]]
    divertor_max_abs_gap = [entry["divertor_max_abs_gap"] for entry in sweep["entries"]]
    measurement_offset = [entry["max_abs_measurement_offset"] for entry in sweep["entries"]]
    objective_converged = [entry["objective_converged"] for entry in sweep["entries"]]
    supervisor_active = [entry["supervisor_active"] for entry in sweep["entries"]]
    supervisor_safe = [entry["supervisor_safe"] for entry in sweep["entries"]]
    interventions = [entry["supervisor_intervention_count"] for entry in sweep["entries"]]
    fallback_steps = [entry["fallback_active_steps"] for entry in sweep["entries"]]
    max_abs_actuator_lag = [entry["max_abs_actuator_lag"] for entry in sweep["entries"]]

    assert sweep["checks"]["max_x_point_position_gap"] is True
    assert sweep["checks"]["max_x_point_flux_gap"] is True
    assert sweep["checks"]["max_divertor_rms_gap"] is True
    assert sweep["checks"]["max_divertor_max_abs_gap"] is True
    assert sweep["checks"]["max_measurement_offset"] is True
    assert sweep["checks"]["objective_converged_all"] is True
    assert sweep["checks"]["supervisor_active_all"] is True
    assert sweep["checks"]["supervisor_safe_all"] is True
    assert sweep["checks"]["intervention_all"] is True
    assert sweep["checks"]["fallback_all"] is True
    assert sweep["checks"]["lag_bounded_all"] is True
    assert (
        max(x_point_position_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS[
            "max_x_point_position_gap"
        ]
    )
    assert (
        max(x_point_flux_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
    )
    assert (
        max(divertor_rms_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
    )
    assert (
        max(divertor_max_abs_gap)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS[
            "max_divertor_max_abs_gap"
        ]
    )
    assert (
        max(measurement_offset)
        <= free_boundary_tracking_acceptance.TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
    )
    assert all(objective_converged)
    assert all(supervisor_active)
    assert all(supervisor_safe)
    assert min(interventions) >= 1
    assert min(fallback_steps) >= 1
    assert (
        max(max_abs_actuator_lag)
        <= free_boundary_tracking_acceptance.SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
    )


def test_render_markdown_contains_sections() -> None:
    report = {
        "generated_at_utc": "2026-03-12T00:00:00+00:00",
        "runtime_seconds": 0.0,
        "free_boundary_tracking_acceptance": _campaign_cached(),
    }
    text = free_boundary_tracking_acceptance.render_markdown(report)
    assert "# Free-Boundary Tracking Acceptance" in text
    assert "## Scenarios" in text
    assert "## Sweeps" in text
    assert "nominal" in text
    assert "measurement_fault_uncorrected" in text
    assert "measurement_latency_uncorrected" in text
    assert "measurement_latency_corrected" in text
    assert "x_point_divertor_kick" in text
    assert "x_point_divertor_measurement_latency_uncorrected" in text
    assert "x_point_divertor_measurement_latency_corrected" in text
    assert "x_point_divertor_supervisor_measurement_fault_uncorrected" in text
    assert "x_point_divertor_supervisor_measurement_fault_corrected" in text
    assert "x_point_divertor_supervisor_measurement_latency_uncorrected" in text
    assert "x_point_divertor_supervisor_measurement_latency_corrected" in text
    assert "x_point_divertor_combined_fault_uncorrected" in text
    assert "x_point_divertor_combined_fault_corrected" in text
    assert "x_point_divertor_measurement_fault_uncorrected" in text
    assert "x_point_divertor_measurement_fault_corrected" in text
    assert "supervisor_fallback_kick" in text
    assert "measurement_fault_scale" in text
    assert "measurement_fault_corrected_scale" in text
    assert "measurement_latency_steps" in text
    assert "measurement_latency_corrected_steps" in text
    assert "actuator_slew_limit" in text
    assert "coil_kick_scale" in text
    assert "topology_kick_scale" in text
    assert "topology_actuator_slew_limit" in text
    assert "topology_measurement_fault_scale" in text
    assert "topology_measurement_corrected_scale" in text
    assert "topology_supervisor_measurement_fault_scale" in text
    assert "topology_supervisor_measurement_corrected_scale" in text
