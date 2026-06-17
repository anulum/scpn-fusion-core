# Subsystem Fault Hardening Campaign

- Generated: `2026-06-17T21:27:01.484344+00:00`
- Schema: `scpn-fusion-core.subsystem_fault_hardening.v1`
- Status: `reduced_order_software_evidence_no_hardware_claim`
- Available evidence pass: `YES`
- Full-fidelity claim ready: `NO`
- Trace checksum: `02ea136ca10bc12444e253390fa92e0501a16759e66407282a719d2236a91a16`

Reduced-order software campaign only. This is not certified quench protection, validated direct-energy conversion, finite-element analysis, hardware-in-the-loop evidence, or plant qualification.

## Scenario Matrix

| Scenario | Status | Evidence | Response time [s] | Pass | Summary |
|----------|--------|----------|-------------------|------|---------|
| rebco_quench_fault | measured_reduced_order | scpn_fusion.core.hts_quench.evaluate_rebco_quench | 0.0450 | YES | hotspot 20.00 K; terminal 810.0 V; detection 0.0023 V |
| direct_energy_conversion_fault | measured_reduced_order | scpn_fusion.core.direct_energy_conversion.evaluate_direct_energy_conversion_fault | 0.0079 | YES | fail-closed 7.90 ms; isolated energy 0.0394 MJ; bus overvoltage 0.0004 |
| disruption_structural_shock_strain | measured_reduced_order | scpn_fusion.core.disruption_structural_response.evaluate_disruption_structural_response | n/a | YES | equivalent stress 61.22 MPa; stress margin 5.66; strain margin 7.76 |

## Diagnostics

- REBCO current-sharing margin: `25.00 K`
- REBCO hotspot temperature: `20.00 K`
- DEC isolated energy: `0.0394 MJ`
- DEC peak dump power: `144.00 MW`
- Structural equivalent stress: `61.22 MPa`
- Structural displacement: `0.920 mm`

## Boundaries

- `rebco_quench_fault`: Reduced-order REBCO/HTS quench screen only; not a certified magnet protection design, hardware quench detector, or conductor qualification.
- `direct_energy_conversion_fault`: Reduced-order direct-energy-conversion fault boundary only; not a validated DEC subsystem, power-electronics design, or hardware interlock.
- `disruption_structural_shock_strain`: Reduced-order structural shock screen only; not finite-element analysis, vessel certification, or component stress qualification.