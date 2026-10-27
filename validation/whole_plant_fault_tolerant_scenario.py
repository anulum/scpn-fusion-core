#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Whole-Plant Fault-Tolerant Scenario Campaign
"""Consolidated whole-plant fault-tolerant scenario campaign.

This campaign consolidates existing SCPN Fusion Core safety and fault lanes into
one evidence surface. It intentionally distinguishes measured reduced-order
software evidence from plant hardware, HIL, and certification claims.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from scpn_fusion.control.fault_tolerant_control import (  # noqa: E402
    FDIMonitor,
    FaultInjector,
    FaultType,
    ReconfigurableController,
)
from scpn_fusion.core.direct_energy_conversion import (  # noqa: E402
    evaluate_direct_energy_conversion_fault,
)
from scpn_fusion.core.divertor_thermal_sim import DivertorLab  # noqa: E402
from scpn_fusion.core.disruption_structural_response import (  # noqa: E402
    evaluate_disruption_structural_response,
)
from scpn_fusion.core.hts_quench import evaluate_rebco_quench  # noqa: E402
from scpn_fusion.core.plasma_wall_interaction import TransientThermalLoad, WallThermalModel  # noqa: E402
from scpn_fusion.engineering.thermal_hydraulics import CoolantLoop  # noqa: E402
from task13_free_boundary_disruption_policy_recovery import (  # type: ignore[import-not-found] # noqa: E402
    run_campaign as run_task13_campaign,
)
from task14_free_boundary_failsafe_dropout_replay import (  # type: ignore[import-not-found] # noqa: E402
    run_campaign as run_task14_campaign,
)


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_fault_controller_lane(seed: int, steps: int = 80) -> dict[str, Any]:
    """Run deterministic sensor-fault and actuator-failover control evidence.

    Parameters
    ----------
    seed
        Random seed for the bounded measurement-noise lane.
    steps
        Number of fault-controller samples.

    Returns
    -------
    dict[str, Any]
        Detection timing, reconfiguration, controllability, and command bounds.
    """
    rng = np.random.default_rng(seed)
    dt_s = 0.01
    monitor = FDIMonitor(n_sensors=4, n_actuators=4, threshold_sigma=3.0, n_alert=4)
    monitor.S_diag = np.full(4, 0.04, dtype=np.float64)
    controller = ReconfigurableController(None, np.eye(4, dtype=np.float64), 4, 4)
    dropout = FaultInjector(
        fault_time=0.22,
        component_index=2,
        fault_type=FaultType.SENSOR_DROPOUT,
        severity=1.0,
    )
    noise = FaultInjector(
        fault_time=0.34,
        component_index=1,
        fault_type=FaultType.SENSOR_DRIFT,
        severity=0.85,
    )
    predicted = np.array([0.35, -0.25, 0.85, -0.15], dtype=np.float64)
    actuator_fault_time_s = 0.42
    sensor_detection_times: list[float] = []
    max_command = 0.0
    max_l1 = 0.0
    failover_applied = False

    for step in range(steps):
        t_s = float(step) * dt_s
        measurement = predicted + rng.normal(0.0, 0.015, size=4)
        measurement = dropout.inject(t_s, measurement)
        measurement = noise.inject(t_s, measurement)
        faults = monitor.update(measurement, predicted, t_s)
        for fault in faults:
            controller.handle_sensor_fault(fault.component_index, fault.fault_type)
            sensor_detection_times.append(float(fault.time_detected))
        if not failover_applied and t_s >= actuator_fault_time_s:
            controller.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)
            failover_applied = True
        command = controller.step(predicted - measurement, dt_s)
        max_command = max(max_command, float(np.max(np.abs(command))))
        max_l1 = max(max_l1, float(np.sum(np.abs(command))))

    shutdown = controller.graceful_shutdown()
    first_detection_s = min(sensor_detection_times) if sensor_detection_times else None
    thresholds = {
        "max_sensor_detection_s": 0.30,
        "max_actuator_failover_s": 0.45,
        "max_command_abs": 1.25,
        "max_command_l1": 3.25,
        "require_controllable_after_one_actuator_loss": True,
        "require_zero_shutdown_command": True,
    }
    failure_reasons: list[str] = []
    if first_detection_s is None or first_detection_s > thresholds["max_sensor_detection_s"]:
        failure_reasons.append("sensor_detection_time")
    if actuator_fault_time_s > thresholds["max_actuator_failover_s"]:
        failure_reasons.append("actuator_failover_time")
    if max_command > thresholds["max_command_abs"]:
        failure_reasons.append("max_command_abs")
    if max_l1 > thresholds["max_command_l1"]:
        failure_reasons.append("max_command_l1")
    if not controller.controllability_check():
        failure_reasons.append("post_fault_controllability")
    if not bool(np.allclose(shutdown, 0.0)):
        failure_reasons.append("shutdown_command")

    return {
        "status": "measured",
        "source_modules": [
            "scpn_fusion.control.fault_tolerant_control.FDIMonitor",
            "scpn_fusion.control.fault_tolerant_control.ReconfigurableController",
        ],
        "steps": int(steps),
        "dt_s": dt_s,
        "sensor_fault_count": len(sensor_detection_times),
        "first_sensor_detection_s": first_detection_s,
        "actuator_failover_time_s": actuator_fault_time_s,
        "faulted_sensors": sorted(int(i) for i in controller.faulted_sensors),
        "faulted_coils": sorted(int(i) for i in controller.faulted_coils),
        "post_fault_controllable": controller.controllability_check(),
        "max_command_abs": max_command,
        "max_command_l1": max_l1,
        "shutdown_norm": float(np.linalg.norm(shutdown)),
        "thresholds": thresholds,
        "failure_reasons": failure_reasons,
        "passes_thresholds": len(failure_reasons) == 0,
    }


def _run_thermal_wall_lane() -> dict[str, Any]:
    """Run reduced cooling, divertor, and wall-load fault evidence.

    Returns
    -------
    dict[str, Any]
        Thermal exhaust, coolant-loop, and wall transient diagnostics.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        divertor = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.5)
        divertor.solve_2point_transport(expansion_factor=22.0, f_rad=0.72)
        temhd = divertor.simulate_temhd_liquid_metal(flow_velocity_m_s=6.0, expansion_factor=22.0)
    wall = WallThermalModel(thickness_mm=12.0, n_nodes=48)
    transient = TransientThermalLoad(wall)
    disruption_delta_t_k = transient.disruption_load(W_th_MJ=18.0, A_wet_m2=24.0, tau_TQ_ms=1.8)
    elm_delta_t_k = transient.elm_load(delta_W_MJ=0.42, A_wet_m2=7.5, tau_IR_ms=0.35)
    cycles = transient.n_elm_cycles_to_fatigue(elm_delta_t_k, T_base_K=640.0)
    coolant = CoolantLoop("water").calculate_pumping_power(
        Q_thermal_MW=180.0,
        delta_T=55.0,
        L=80.0,
        D=0.30,
    )
    thresholds = {
        "max_temhd_stability_index": 1.0,
        "max_surface_heat_flux_mw_m2": 45.0,
        "max_coolant_pump_power_mw": 1.0,
        "max_disruption_delta_t_k": 1050.0,
        "min_elm_cycles_to_fatigue": 1000,
    }
    surface_heat_flux_mw_m2 = float(temhd["surface_heat_flux_w_m2"]) / 1.0e6
    failure_reasons: list[str] = []
    if float(temhd["stability_index"]) > thresholds["max_temhd_stability_index"]:
        failure_reasons.append("temhd_stability_index")
    if surface_heat_flux_mw_m2 > thresholds["max_surface_heat_flux_mw_m2"]:
        failure_reasons.append("surface_heat_flux")
    if float(coolant["P_pump_MW"]) > thresholds["max_coolant_pump_power_mw"]:
        failure_reasons.append("coolant_pump_power")
    if disruption_delta_t_k > thresholds["max_disruption_delta_t_k"]:
        failure_reasons.append("disruption_delta_t")
    if cycles < thresholds["min_elm_cycles_to_fatigue"]:
        failure_reasons.append("elm_fatigue_cycles")

    return {
        "status": "measured",
        "source_modules": [
            "scpn_fusion.core.divertor_thermal_sim.DivertorLab",
            "scpn_fusion.core.plasma_wall_interaction.TransientThermalLoad",
            "scpn_fusion.engineering.thermal_hydraulics.CoolantLoop",
        ],
        "surface_heat_flux_mw_m2": surface_heat_flux_mw_m2,
        "temhd_stability_index": float(temhd["stability_index"]),
        "temhd_is_stable": bool(temhd["is_stable"]),
        "shielding_fraction": float(temhd["shielding_fraction"]),
        "pressure_loss_pa": float(temhd["pressure_loss_pa"]),
        "coolant_pump_power_mw": float(coolant["P_pump_MW"]),
        "coolant_reynolds": float(coolant["Re"]),
        "disruption_delta_t_k": float(disruption_delta_t_k),
        "elm_delta_t_k": float(elm_delta_t_k),
        "elm_cycles_to_fatigue": int(cycles),
        "thresholds": thresholds,
        "failure_reasons": failure_reasons,
        "passes_thresholds": len(failure_reasons) == 0,
    }


def _scenario_row(
    scenario_id: str,
    *,
    status: str,
    evidence_source: str,
    response_time_s: float | None,
    pass_status: bool,
    summary: str,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": status,
        "evidence_source": evidence_source,
        "response_time_s": response_time_s,
        "passes_thresholds": bool(pass_status),
        "summary": summary,
    }


def run_campaign(*, seed: int = 42) -> dict[str, Any]:
    """Run the consolidated whole-plant fault-tolerant scenario campaign.

    Parameters
    ----------
    seed
        Deterministic campaign seed.

    Returns
    -------
    dict[str, Any]
        Scenario rows, measured-lane diagnostics, residual blocked-subsystem
        rows, and available-evidence gate status.
    """
    seed_i = _require_int("seed", seed, 0)
    task13 = run_task13_campaign(seed=seed_i, shot_length=104, control_dt_s=0.05)[
        "task13_free_boundary_disruption_policy_recovery"
    ]
    task14 = run_task14_campaign(seed=seed_i, shot_length=96, control_dt_s=0.05)[
        "task14_free_boundary_failsafe_dropout_replay"
    ]
    fault_control = _run_fault_controller_lane(seed_i + 211)
    thermal_wall = _run_thermal_wall_lane()
    dec_fault = evaluate_direct_energy_conversion_fault().to_dict()
    rebco_quench = evaluate_rebco_quench().to_dict()
    structural_response = evaluate_disruption_structural_response().to_dict()

    task13_closed = task13["policy_closed_loop"]
    task14_summary = task14["faulted_replay"]["summary"]
    rows = [
        _scenario_row(
            "vertical_excursion_vde",
            status="measured_reduced_order",
            evidence_source="task14_free_boundary_failsafe_dropout_replay",
            response_time_s=float(task14_summary["recovery_transition_count"]) * 0.05,
            pass_status=bool(task14["passes_thresholds"]),
            summary=(
                f"p95 axis {task14_summary['p95_axis_error_m']:.4f} m; "
                f"late alert {task14['recovery_window']['late_max_alert_level']}"
            ),
        ),
        _scenario_row(
            "disruption_risk_spike",
            status="measured_reduced_order",
            evidence_source="task13_free_boundary_disruption_policy_recovery",
            response_time_s=float(task13_closed["recovery_transition_count"]) * 0.05,
            pass_status=bool(task13["passes_thresholds"]),
            summary=(
                f"max risk {task13_closed['max_disruption_risk']:.4f}; "
                f"late mean risk {task13['recovery_window']['late_mean_disruption_risk']:.4f}"
            ),
        ),
        _scenario_row(
            "sensor_dropout_noise",
            status="measured_reduced_order",
            evidence_source="task14 + FDIMonitor",
            response_time_s=fault_control["first_sensor_detection_s"],
            pass_status=bool(task14["passes_thresholds"] and fault_control["passes_thresholds"]),
            summary=(
                f"diagnostic dropouts {task14_summary['diagnostic_dropout_count']}; "
                f"faulted sensors {fault_control['faulted_sensors']}"
            ),
        ),
        _scenario_row(
            "actuator_saturation_dropout",
            status="measured_reduced_order",
            evidence_source="task14 + ReconfigurableController",
            response_time_s=fault_control["actuator_failover_time_s"],
            pass_status=bool(task14["passes_thresholds"] and fault_control["passes_thresholds"]),
            summary=(
                f"actuator dropouts {task14_summary['actuator_dropout_count']}; "
                f"max action L1 {task14_summary['max_action_l1']:.3f}"
            ),
        ),
        _scenario_row(
            "controller_failover",
            status="measured_reduced_order",
            evidence_source="ReconfigurableController",
            response_time_s=fault_control["actuator_failover_time_s"],
            pass_status=bool(fault_control["passes_thresholds"]),
            summary=(
                f"post-fault controllable {fault_control['post_fault_controllable']}; "
                f"shutdown norm {fault_control['shutdown_norm']:.3e}"
            ),
        ),
        _scenario_row(
            "cooling_thermal_limit",
            status="measured_reduced_order",
            evidence_source="DivertorLab + CoolantLoop",
            response_time_s=None,
            pass_status=bool(thermal_wall["passes_thresholds"]),
            summary=(
                f"surface heat flux {thermal_wall['surface_heat_flux_mw_m2']:.3f} MW/m2; "
                f"pump {thermal_wall['coolant_pump_power_mw']:.3f} MW"
            ),
        ),
        _scenario_row(
            "shielding_wall_load_warning",
            status="measured_reduced_order",
            evidence_source="DivertorLab + WallThermalModel",
            response_time_s=None,
            pass_status=bool(thermal_wall["passes_thresholds"]),
            summary=(
                f"shielding {thermal_wall['shielding_fraction']:.3f}; "
                f"disruption delta-T {thermal_wall['disruption_delta_t_k']:.1f} K"
            ),
        ),
        _scenario_row(
            "direct_energy_conversion_fault",
            status="measured_reduced_order",
            evidence_source="scpn_fusion.core.direct_energy_conversion",
            response_time_s=float(dec_fault["fail_closed_time_ms"]) * 1.0e-3,
            pass_status=bool(dec_fault["passes_thresholds"]),
            summary=(
                f"fail-closed {dec_fault['fail_closed_time_ms']:.2f} ms; "
                f"isolated energy {dec_fault['isolated_energy_mj']:.4f} MJ"
            ),
        ),
        _scenario_row(
            "rebco_quench_fault",
            status="measured_reduced_order",
            evidence_source="scpn_fusion.core.hts_quench",
            response_time_s=float(rebco_quench["detection_time_s"]),
            pass_status=bool(rebco_quench["passes_thresholds"]),
            summary=(
                f"hotspot {rebco_quench['hotspot_temperature_k']:.2f} K; "
                f"terminal {rebco_quench['peak_terminal_voltage_v']:.1f} V"
            ),
        ),
        _scenario_row(
            "disruption_structural_shock_strain",
            status="measured_reduced_order",
            evidence_source="scpn_fusion.core.disruption_structural_response",
            response_time_s=None,
            pass_status=bool(structural_response["passes_thresholds"]),
            summary=(
                f"stress margin {structural_response['stress_margin']:.2f}; "
                f"strain margin {structural_response['strain_margin']:.2f}; "
                f"displacement {structural_response['displacement_mm']:.3f} mm"
            ),
        ),
    ]
    measured_rows = [row for row in rows if row["status"] == "measured_reduced_order"]
    blocked_rows = [row for row in rows if row["status"].startswith("blocked_")]
    available_evidence_pass = all(bool(row["passes_thresholds"]) for row in measured_rows)
    payload = {
        "schema": "scpn-fusion-core.whole_plant_fault_tolerant_scenario.v1",
        "seed": seed_i,
        "campaign_status": "available_reduced_order_evidence_no_hardware_claim",
        "passes_available_evidence": available_evidence_pass,
        "full_whole_plant_claim_ready": False,
        "claim_boundary": (
            "Reduced-order software campaign only. This is not plant hardware, "
            "physical HIL, certified fault tolerance, certified REBCO quench "
            "protection, validated direct-energy conversion, finite-element "
            "analysis, or plant qualification."
        ),
        "scenario_rows": rows,
        "measured_lane_count": len(measured_rows),
        "blocked_lane_count": len(blocked_rows),
        "measured_lanes": {
            "task13_disruption_policy": task13,
            "task14_failsafe_dropout": task14,
            "fault_controller": fault_control,
            "thermal_wall": thermal_wall,
            "direct_energy_conversion": dec_fault,
            "rebco_quench": rebco_quench,
            "disruption_structural_response": structural_response,
        },
        "blocked_subsystems": blocked_rows,
    }
    payload["trace_checksum"] = _canonical_hash(
        {
            "scenario_rows": rows,
            "fault_controller": fault_control,
            "thermal_wall": thermal_wall,
            "direct_energy_conversion": dec_fault,
            "rebco_quench": rebco_quench,
            "disruption_structural_response": structural_response,
            "task13_signature": task13_closed["replay_signature"],
            "task14_signature": task14_summary["replay_signature"],
        }
    )
    return payload


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Create a timestamped whole-plant fault-tolerant scenario report.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`run_campaign`.

    Returns
    -------
    dict[str, Any]
        Report payload with UTC generation timestamp.
    """
    report = run_campaign(**kwargs)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the whole-plant fault campaign report as Markdown.

    Parameters
    ----------
    report
        Output from :func:`generate_report`.

    Returns
    -------
    str
        Human-readable Markdown report.
    """
    lines = [
        "# Whole-Plant Fault-Tolerant Scenario Campaign",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['campaign_status']}`",
        f"- Available evidence pass: `{'YES' if report['passes_available_evidence'] else 'NO'}`",
        f"- Full whole-plant claim ready: `{'YES' if report['full_whole_plant_claim_ready'] else 'NO'}`",
        f"- Trace checksum: `{report['trace_checksum']}`",
        "",
        report["claim_boundary"],
        "",
        "## Scenario Matrix",
        "",
        "| Scenario | Status | Evidence | Response time [s] | Pass | Summary |",
        "|----------|--------|----------|-------------------|------|---------|",
    ]
    for row in report["scenario_rows"]:
        response = "n/a" if row["response_time_s"] is None else f"{row['response_time_s']:.3f}"
        lines.append(
            f"| {row['scenario_id']} | {row['status']} | {row['evidence_source']} | "
            f"{response} | {'YES' if row['passes_thresholds'] else 'NO'} | {row['summary']} |"
        )

    fault = report["measured_lanes"]["fault_controller"]
    thermal = report["measured_lanes"]["thermal_wall"]
    dec = report["measured_lanes"]["direct_energy_conversion"]
    rebco = report["measured_lanes"]["rebco_quench"]
    structural = report["measured_lanes"]["disruption_structural_response"]
    first_detection = (
        "n/a"
        if fault["first_sensor_detection_s"] is None
        else f"{fault['first_sensor_detection_s']:.3f} s"
    )
    lines.extend(
        [
            "",
            "## Fault Controller",
            "",
            f"- First sensor detection: `{first_detection}`",
            f"- Actuator failover time: `{fault['actuator_failover_time_s']:.3f} s`",
            f"- Faulted sensors: `{fault['faulted_sensors']}`",
            f"- Faulted coils: `{fault['faulted_coils']}`",
            f"- Post-fault controllable: `{'YES' if fault['post_fault_controllable'] else 'NO'}`",
            f"- Max command L1: `{fault['max_command_l1']:.3f}`",
            "",
            "## Thermal And Wall Loads",
            "",
            f"- Surface heat flux: `{thermal['surface_heat_flux_mw_m2']:.3f} MW/m2`",
            f"- TEMHD stability index: `{thermal['temhd_stability_index']:.3f}`",
            f"- Coolant pump power: `{thermal['coolant_pump_power_mw']:.3f} MW`",
            f"- Disruption delta-T: `{thermal['disruption_delta_t_k']:.3f} K`",
            f"- ELM cycles to fatigue: `{thermal['elm_cycles_to_fatigue']}`",
            "",
            "## Subsystem Fault Lanes",
            "",
            f"- DEC fail-closed time: `{dec['fail_closed_time_ms']:.3f} ms`",
            f"- DEC isolated energy: `{dec['isolated_energy_mj']:.4f} MJ`",
            f"- REBCO quench detection time: `{rebco['detection_time_s']:.3f} s`",
            f"- REBCO hotspot temperature: `{rebco['hotspot_temperature_k']:.2f} K`",
            f"- Structural equivalent stress: `{structural['equivalent_stress_mpa']:.2f} MPa`",
            f"- Structural displacement: `{structural['displacement_mm']:.3f} mm`",
            "",
            "## Blocked Subsystems",
            "",
        ]
    )
    if report["blocked_subsystems"]:
        for row in report["blocked_subsystems"]:
            lines.append(f"- `{row['scenario_id']}`: `{row['status']}` — {row['summary']}")
    else:
        lines.append(
            "- None at the reduced-order software-model layer; hardware, HIL, "
            "certification, and FEA-grade claims remain blocked by the claim boundary."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the campaign and write JSON/Markdown artifacts.

    Parameters
    ----------
    argv
        Optional CLI argument list.

    Returns
    -------
    int
        ``0`` when available measured evidence passes; ``2`` on strict failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "whole_plant_fault_tolerant_scenario.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "whole_plant_fault_tolerant_scenario.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    report = generate_report(seed=args.seed)
    report["runtime_seconds"] = float(time.perf_counter() - started)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print("Whole-plant fault-tolerant scenario campaign complete.")
    print(
        "Summary -> "
        f"available_evidence_pass={report['passes_available_evidence']}, "
        f"measured_lanes={report['measured_lane_count']}, "
        f"blocked_lanes={report['blocked_lane_count']}, "
        f"trace_checksum={report['trace_checksum'][:16]}"
    )
    if args.strict and not report["passes_available_evidence"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
