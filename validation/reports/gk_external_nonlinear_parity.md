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
- Evidence package ready: `False`
- Roadmap evidence surfaces ready: `False`
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

## Evidence package matrix

| Solver | Manifest | Provenance/license | Artefact | Metadata | Native thresholds | Grid | Scaling | Ready |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GENE | `False` | `False` | `False` | `False` | `False` | `False` | `False` | `False` |
| CGYRO | `False` | `False` | `False` | `False` | `False` | `False` | `False` | `False` |
| GS2 | `False` | `False` | `False` | `False` | `False` | `False` | `False` | `False` |

## Roadmap evidence surface matrix

| Solver | Surface | Ready | Required observables | Blockers |
|---|---|:---:|---|---|
| GENE | `nonlinear_distribution_output` | `False` | nonlinear_distribution_function, nonlinear_distribution_function_imag | missing_same_deck_external_observables, missing_observable:nonlinear_distribution_function, missing_observable:nonlinear_distribution_function_imag |
| GENE | `heat_flux_spectra_time_kx_ky_species` | `False` | ion_heat_flux_spectrum, electron_heat_flux_spectrum | missing_same_deck_external_observables, missing_observable:ion_heat_flux_spectrum, missing_observable:electron_heat_flux_spectrum |
| GENE | `field_energy_history_phi_apar_bpar` | `False` | electromagnetic_phi_energy, electromagnetic_apar_energy, electromagnetic_bpar_energy | missing_same_deck_external_observables, missing_observable:electromagnetic_phi_energy, missing_observable:electromagnetic_apar_energy, missing_observable:electromagnetic_bpar_energy |
| GENE | `zonal_flow_and_saturation_metrics` | `False` | zonal_flow_energy, saturated_phi_rms | missing_same_deck_external_observables, missing_observable:zonal_flow_energy, missing_observable:saturated_phi_rms |
| GENE | `native_same_case_solver_output_comparison` | `False` | - | missing_or_failed_native_same_case_thresholds |
| GENE | `grid_convergence_evidence` | `False` | - | missing_grid_convergence_evidence |
| GENE | `production_scale_scaling_evidence` | `False` | - | missing_production_scale_scaling_evidence |
| CGYRO | `nonlinear_distribution_output` | `False` | nonlinear_distribution_function, nonlinear_distribution_function_imag | missing_same_deck_external_observables, missing_observable:nonlinear_distribution_function, missing_observable:nonlinear_distribution_function_imag |
| CGYRO | `heat_flux_spectra_time_kx_ky_species` | `False` | ion_heat_flux_spectrum, electron_heat_flux_spectrum | missing_same_deck_external_observables, missing_observable:ion_heat_flux_spectrum, missing_observable:electron_heat_flux_spectrum |
| CGYRO | `field_energy_history_phi_apar_bpar` | `False` | electromagnetic_phi_energy, electromagnetic_apar_energy, electromagnetic_bpar_energy | missing_same_deck_external_observables, missing_observable:electromagnetic_phi_energy, missing_observable:electromagnetic_apar_energy, missing_observable:electromagnetic_bpar_energy |
| CGYRO | `zonal_flow_and_saturation_metrics` | `False` | zonal_flow_energy, saturated_phi_rms | missing_same_deck_external_observables, missing_observable:zonal_flow_energy, missing_observable:saturated_phi_rms |
| CGYRO | `native_same_case_solver_output_comparison` | `False` | - | missing_or_failed_native_same_case_thresholds |
| CGYRO | `grid_convergence_evidence` | `False` | - | missing_grid_convergence_evidence |
| CGYRO | `production_scale_scaling_evidence` | `False` | - | missing_production_scale_scaling_evidence |
| GS2 | `nonlinear_distribution_output` | `False` | nonlinear_distribution_function, nonlinear_distribution_function_imag | missing_same_deck_external_observables, missing_observable:nonlinear_distribution_function, missing_observable:nonlinear_distribution_function_imag |
| GS2 | `heat_flux_spectra_time_kx_ky_species` | `False` | ion_heat_flux_spectrum, electron_heat_flux_spectrum | missing_same_deck_external_observables, missing_observable:ion_heat_flux_spectrum, missing_observable:electron_heat_flux_spectrum |
| GS2 | `field_energy_history_phi_apar_bpar` | `False` | electromagnetic_phi_energy, electromagnetic_apar_energy, electromagnetic_bpar_energy | missing_same_deck_external_observables, missing_observable:electromagnetic_phi_energy, missing_observable:electromagnetic_apar_energy, missing_observable:electromagnetic_bpar_energy |
| GS2 | `zonal_flow_and_saturation_metrics` | `False` | zonal_flow_energy, saturated_phi_rms | missing_same_deck_external_observables, missing_observable:zonal_flow_energy, missing_observable:saturated_phi_rms |
| GS2 | `native_same_case_solver_output_comparison` | `False` | - | missing_or_failed_native_same_case_thresholds |
| GS2 | `grid_convergence_evidence` | `False` | - | missing_grid_convergence_evidence |
| GS2 | `production_scale_scaling_evidence` | `False` | - | missing_production_scale_scaling_evidence |

## Grid-convergence evidence matrix

| Solver | Case | Observable | Relative L2 | Limit | Ready | Reasons |
|---|---|---|---:|---:|:---:|---|

## Production-scaling evidence matrix

| Solver | Case | Device | Phase cells | Ranks | Wall time s | Ready | Reasons |
|---|---|---|---:|---:|---:|:---:|---|

## Published threshold contract

| Threshold | Observable | Metric | Comparator | Limit |
|---|---|---|:---:|---:|
| distribution_imaginary_relative_l2_max | `nonlinear_distribution_function_imag` | `relative_l2` | `<=` | 0.25 |
| distribution_relative_l2_max | `nonlinear_distribution_function` | `relative_l2` | `<=` | 0.25 |
| field_bpar_energy_relative_error_max | `electromagnetic_bpar_energy` | `relative_error` | `<=` | 0.2 |
| field_energy_relative_error_max | `electromagnetic_apar_energy` | `relative_error` | `<=` | 0.2 |
| field_phi_energy_relative_error_max | `electromagnetic_phi_energy` | `relative_error` | `<=` | 0.2 |
| heat_flux_relative_l2_max | `ion_heat_flux_spectrum` | `relative_l2` | `<=` | 0.15 |
| spectrum_relative_l2_max | `electron_heat_flux_spectrum` | `relative_l2` | `<=` | 0.2 |
| zonal_energy_relative_error_max | `zonal_flow_energy` | `relative_error` | `<=` | 0.2 |

## Missing full-fidelity requirements

- same-deck external nonlinear distribution output for GENE, CGYRO, and GS2
- heat_flux_spectra_time_kx_ky_species for all required solver families
- field_energy_history_phi_apar_bpar for all required solver families
- zonal_flow_and_saturation_metrics for all required solver families
- shared benchmark_case_id and deck_physics_sha256 across GENE, CGYRO, and GS2
- native same-case nonlinear GK solver-output comparison
- grid-convergence evidence for converted public nonlinear GK outputs
- production-scale scaling evidence for converted public nonlinear GK outputs
- complete checksum/provenance/threshold evidence package
