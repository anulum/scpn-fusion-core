# GK External Nonlinear Output Parity

Strict fail-closed GENE/CGYRO/GS2 nonlinear GK external-output conversion and native parity report.

- Schema: `gk-external-nonlinear-output-parity-report.v1`
- Status: `blocked_missing_external_output_manifest`
- Accepted full-fidelity ready: `False`
- Reference output ready: `False`
- Same-deck group ready: `False`
- Native same-case comparison ready: `False`
- Grid convergence ready: `False`
- Production-scale scaling ready: `False`
- Solver-family completeness ready: `False`
- Converted reference artefacts: `0`
- Same-deck group reason: `missing_solver_family_same_deck_rows`

## Solver-family rows

| Solver | Status | Reference output | Native comparison | Missing requirements |
|---|---|:---:|:---:|---|
| GENE | `blocked_missing_external_output_manifest` | `False` | `False` | same_deck_external_nonlinear_output, nonlinear_distribution_output, heat_flux_spectra_time_kx_ky_species, field_energy_history_phi_apar_bpar, zonal_flow_and_saturation_metrics, native_same_case_solver_output_comparison, grid_convergence_evidence, production_scale_scaling_evidence |
| CGYRO | `blocked_missing_external_output_manifest` | `False` | `False` | same_deck_external_nonlinear_output, nonlinear_distribution_output, heat_flux_spectra_time_kx_ky_species, field_energy_history_phi_apar_bpar, zonal_flow_and_saturation_metrics, native_same_case_solver_output_comparison, grid_convergence_evidence, production_scale_scaling_evidence |
| GS2 | `blocked_missing_external_output_manifest` | `False` | `False` | same_deck_external_nonlinear_output, nonlinear_distribution_output, heat_flux_spectra_time_kx_ky_species, field_energy_history_phi_apar_bpar, zonal_flow_and_saturation_metrics, native_same_case_solver_output_comparison, grid_convergence_evidence, production_scale_scaling_evidence |

## Solver-family completeness matrix

| Solver | Reference output | Required observables | Native comparison | Native thresholds |
|---|:---:|:---:|:---:|:---:|
| GENE | `False` | `False` | `False` | `False` |
| CGYRO | `False` | `False` | `False` | `False` |
| GS2 | `False` | `False` | `False` | `False` |

## Missing full-fidelity requirements

- same-deck external nonlinear distribution output for GENE, CGYRO, and GS2
- heat_flux_spectra_time_kx_ky_species for all required solver families
- field_energy_history_phi_apar_bpar for all required solver families
- zonal_flow_and_saturation_metrics for all required solver families
- shared benchmark_case_id and deck_physics_sha256 across GENE, CGYRO, and GS2
- native same-case nonlinear GK solver-output comparison
- grid-convergence evidence for converted public nonlinear GK outputs
- production-scale scaling evidence for converted public nonlinear GK outputs
