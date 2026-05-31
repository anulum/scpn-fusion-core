# Full-Fidelity End-to-End Campaign

This report keeps all declared full-fidelity blockers in one fail-closed gate.

- Schema: `full-fidelity-end-to-end-campaign.v1`
- Status: `not_full_fidelity`
- Acceptance passed: `False`
- Public source registry: `validation/reference_data/full_fidelity_public_sources.json`
- Public source download report: `validation/reports/full_fidelity_public_source_downloads.json`
- Public sources cached: `True`
- Public source cache root: `data/external/full_fidelity_public_sources`
- Public reference artifact conversion report: `validation/reports/full_fidelity_reference_artifact_conversion.json`
- Partial public output artifacts: `3`
- Accepted public reference artifacts: `0`
- DREAM execution report: `validation/reports/dream_reference_execution_request.json`
- DREAM settings deck generated: `True`
- DREAM reference output ready: `False`
- DREAM execution status: `blocked_missing_dream_backend`
- Aurora execution report: `validation/reports/aurora_reference_execution_artifact.json`
- Aurora artifact generated: `True`
- Aurora reference output ready: `True`
- Aurora execution status: `blocked_partial_public_atomic_artifact_not_transport_parity`
- GK deck inventory report: `validation/reports/gk_public_reference_deck_inventory.json`
- GK public decks indexed: `40`
- GK public outputs indexed: `21`
- GK deck inventory status: `blocked_public_gk_decks_indexed_missing_solver_output_parity`
- GK electromagnetic fidelity report: `validation/reports/gk_electromagnetic_fidelity.json`
- GK electromagnetic fidelity status: `blocked_missing_full_vlasov_maxwell_field_solve`
- GK electromagnetic compact closure ready: `True`
- GK electromagnetic external parity ready: `False`
- Production decomposition report: `validation/reports/production_decomposition_contract.json`
- Production decomposition contract pass: `True`
- Production-scale ready: `False`
- Production decomposition status: `blocked_contract_ready_missing_distributed_runtime_scaling`
- Free-boundary machine metadata report: `validation/reports/free_boundary_public_machine_metadata_inventory.json`
- Free-boundary machine metadata indexed: `23`
- Free-boundary machine metadata ready: `True`
- Free-boundary machine metadata status: `blocked_machine_metadata_indexed_missing_same_case_free_boundary_reconstruction`
- FreeGS public example reconstruction report: `validation/reports/freegs_public_example_reconstruction.json`
- FreeGS public example cases: `2`
- FreeGS public example vacuum comparison pass: `True`
- FreeGS public example external output ready: `True`
- FreeGS public example reconstruction status: `blocked_public_freegs_native_same_case_compared_missing_strict_threshold_grid_convergence_coil_sidecars`
- Local contracts ready: `True`
- Reference parity ready: `False`

| Lane | Status | Local contract ready | Reference parity ready | Sources | Next evidence |
| --- | --- | ---: | ---: | --- | --- |
| gene_cgyro_gs2_nonlinear_gk_parity | blocked_missing_external_output_manifest | True | False | GENE, CGYRO, GS2 | same-deck external nonlinear distribution output for GENE, CGYRO, and GS2<br>heat_flux_spectra_time_kx_ky_species for all required solver families<br>field_energy_history_phi_apar_bpar for all required solver families<br>zonal_flow_and_saturation_metrics for all required solver families<br>shared benchmark_case_id and deck_physics_sha256 across GENE, CGYRO, and GS2<br>native same-case nonlinear GK solver-output comparison<br>grid-convergence evidence for converted public nonlinear GK outputs<br>production-scale scaling evidence for converted public nonlinear GK outputs |
| full_maxwell_electromagnetic_fidelity | blocked_missing_full_vlasov_maxwell_field_solve | True | False | GENE, CGYRO, GS2 | full Faraday/displacement-current Maxwell field evolution<br>same-deck electromagnetic GENE/CGYRO/GS2 output artifacts<br>native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds<br>grid-convergence evidence for electromagnetic field-energy histories |
| production_scale_decomposition | blocked_contract_ready_missing_distributed_runtime_scaling | True | False | none | MPI or multi-GPU execution path over the declared rank tiles<br>large-grid cluster/GPU wall-time scaling report<br>same-physics convergence evidence across distributed decomposition shapes<br>hardware-specific multi-rank throughput and efficiency thresholds |
| dream_grade_runaway_electrons | blocked_missing_public_dream_artifacts | True | False | DREAM | public DREAM deck ingestion and production artifact parity<br>full momentum-pitch-radius Fokker-Planck evolution rather than 1D momentum projection artifact<br>validated synchrotron, bremsstrahlung, partial-screening, and transport operators against DREAM<br>distribution-function, current, and growth-rate RMSE thresholds against public DREAM output |
| aurora_strahl_grade_impurities | blocked_partial_public_atomic_artifact_not_transport_parity | True | False | Aurora | transported_charge_state_density_time_radius_charge<br>line_radiation_power_time_radius_charge<br>ionisation_recombination_source_sink_matrix_time_radius_charge_charge<br>total_impurity_inventory_closure<br>native_same_case_solver_output_comparison |
| free_boundary_equilibrium_strict_parity | blocked_public_freegs_native_same_case_compared_missing_strict_threshold_grid_convergence_coil_sidecars | True | False | FreeGS, FreeGSNKE | strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance<br>grid convergence across public example resolutions<br>coil/vacuum reconstruction linked to public machine current sidecars |
