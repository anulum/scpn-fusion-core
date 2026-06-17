# Whole-Plant Fault-Tolerant Scenario Campaign

- Generated: `2026-06-17T18:08:04.265427+00:00`
- Schema: `scpn-fusion-core.whole_plant_fault_tolerant_scenario.v1`
- Status: `partial_available_evidence_blocked_subsystems`
- Available evidence pass: `YES`
- Full whole-plant claim ready: `NO`
- Trace checksum: `e373a0c9fdc889b84853141062af48bcd478b97192b6c48c76b53bbe1f8b22fa`

Reduced-order software campaign only. This is not plant hardware, physical HIL, certified fault tolerance, REBCO quench protection, or direct-energy-conversion fault evidence.

## Scenario Matrix

| Scenario | Status | Evidence | Response time [s] | Pass | Summary |
|----------|--------|----------|-------------------|------|---------|
| vertical_excursion_vde | measured_reduced_order | task14_free_boundary_failsafe_dropout_replay | 0.150 | YES | p95 axis 0.0548 m; late alert 0 |
| disruption_risk_spike | measured_reduced_order | task13_free_boundary_disruption_policy_recovery | 0.100 | YES | max risk 0.2360; late mean risk 0.2066 |
| sensor_dropout_noise | measured_reduced_order | task14 + FDIMonitor | 0.250 | YES | diagnostic dropouts 16; faulted sensors [2] |
| actuator_saturation_dropout | measured_reduced_order | task14 + ReconfigurableController | 0.420 | YES | actuator dropouts 18; max action L1 0.780 |
| controller_failover | measured_reduced_order | ReconfigurableController | 0.420 | YES | post-fault controllable True; shutdown norm 0.000e+00 |
| cooling_thermal_limit | measured_reduced_order | DivertorLab + CoolantLoop | n/a | YES | surface heat flux 16.270 MW/m2; pump 0.230 MW |
| shielding_wall_load_warning | measured_reduced_order | DivertorLab + WallThermalModel | n/a | YES | shielding 0.980; disruption delta-T 950.1 K |
| direct_energy_conversion_fault | blocked_no_subsystem_model | none | n/a | NO | No direct-energy-conversion subsystem model exists in this repository. |
| rebco_quench_fault | blocked_no_subsystem_model | none | n/a | NO | No REBCO/HTS quench dynamics subsystem model exists in this repository. |

## Fault Controller

- First sensor detection: `0.250 s`
- Actuator failover time: `0.420 s`
- Faulted sensors: `[2]`
- Faulted coils: `[1]`
- Post-fault controllable: `YES`
- Max command L1: `0.879`

## Thermal And Wall Loads

- Surface heat flux: `16.270 MW/m2`
- TEMHD stability index: `0.535`
- Coolant pump power: `0.230 MW`
- Disruption delta-T: `950.148 K`
- ELM cycles to fatigue: `1869515`

## Blocked Subsystems

- `direct_energy_conversion_fault`: `blocked_no_subsystem_model` — No direct-energy-conversion subsystem model exists in this repository.
- `rebco_quench_fault`: `blocked_no_subsystem_model` — No REBCO/HTS quench dynamics subsystem model exists in this repository.