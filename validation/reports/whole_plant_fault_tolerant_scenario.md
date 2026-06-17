# Whole-Plant Fault-Tolerant Scenario Campaign

- Generated: `2026-06-17T21:27:01.724756+00:00`
- Schema: `scpn-fusion-core.whole_plant_fault_tolerant_scenario.v1`
- Status: `available_reduced_order_evidence_no_hardware_claim`
- Available evidence pass: `YES`
- Full whole-plant claim ready: `NO`
- Trace checksum: `4b0f4d2e1714a2dc5abb8c039c895e42c9de525a9e9511a7581f0514d126734a`

Reduced-order software campaign only. This is not plant hardware, physical HIL, certified fault tolerance, certified REBCO quench protection, validated direct-energy conversion, finite-element analysis, or plant qualification.

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
| direct_energy_conversion_fault | measured_reduced_order | scpn_fusion.core.direct_energy_conversion | 0.008 | YES | fail-closed 7.90 ms; isolated energy 0.0394 MJ |
| rebco_quench_fault | measured_reduced_order | scpn_fusion.core.hts_quench | 0.045 | YES | hotspot 20.00 K; terminal 810.0 V |
| disruption_structural_shock_strain | measured_reduced_order | scpn_fusion.core.disruption_structural_response | n/a | YES | stress margin 5.66; strain margin 7.76; displacement 0.920 mm |

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

## Subsystem Fault Lanes

- DEC fail-closed time: `7.900 ms`
- DEC isolated energy: `0.0394 MJ`
- REBCO quench detection time: `0.045 s`
- REBCO hotspot temperature: `20.00 K`
- Structural equivalent stress: `61.22 MPa`
- Structural displacement: `0.920 mm`

## Blocked Subsystems

- None at the reduced-order software-model layer; hardware, HIL, certification, and FEA-grade claims remain blocked by the claim boundary.
