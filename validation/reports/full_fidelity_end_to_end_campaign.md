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
- FreeGS public example reconstruction status: `blocked_public_freegs_external_psi_ready_missing_native_same_case_comparison`
- Local contracts ready: `True`
- Reference parity ready: `False`

| Lane | Status | Local contract ready | Reference parity ready | Sources | Next evidence |
| --- | --- | ---: | ---: | --- | --- |
| gene_cgyro_gs2_nonlinear_gk_parity | blocked_public_gk_decks_indexed_missing_solver_output_parity | True | False | GENE, CGYRO, GS2 | same-deck external nonlinear distribution output<br>heat_flux_spectra_time_kx_ky_species<br>field_energy_history_phi_apar_bpar<br>zonal_flow_and_saturation_metrics<br>native_same_case_solver_output_comparison<br>grid_convergence_and_production_scale_scaling_evidence |
| full_maxwell_electromagnetic_fidelity | blocked_missing_full_vlasov_maxwell_field_solve | True | False | GENE, CGYRO, GS2 | native nonlinear Ampere/Faraday closure beyond compact A_parallel/B_parallel contracts<br>GENE/CGYRO/GS2 electromagnetic field-energy and transport parity artifacts |
| production_scale_decomposition | blocked_contract_ready_missing_distributed_runtime_scaling | True | False | none | MPI or multi-GPU execution path over the declared rank tiles<br>halo exchange implementation and correctness tests<br>large-grid cluster/GPU wall-time scaling report<br>same-physics convergence evidence across decomposition shapes |
| dream_grade_runaway_electrons | blocked_missing_public_dream_artifacts | True | False | DREAM | public DREAM deck ingestion and production artifact parity<br>full momentum-pitch-radius Fokker-Planck evolution rather than 1D momentum projection artifact<br>validated synchrotron, bremsstrahlung, partial-screening, and transport operators against DREAM<br>distribution-function, current, and growth-rate RMSE thresholds against public DREAM output |
| aurora_strahl_grade_impurities | blocked_partial_public_atomic_artifact_not_transport_parity | True | False | Aurora | transported_charge_state_density_time_radius_charge<br>line_radiation_power_time_radius_charge<br>ionisation_recombination_source_sink_matrix_time_radius_charge_charge<br>total_impurity_inventory_closure<br>native_same_case_solver_output_comparison |
| free_boundary_equilibrium_strict_parity | blocked_public_freegs_external_psi_ready_missing_native_same_case_comparison | True | False | FreeGS, FreeGSNKE | native same-case free-boundary profile-source reconstruction against finite external FreeGS psi output<br>native-vs-FreeGS psi_N RMSE threshold<br>axis/X-point/boundary containment and q-profile thresholds<br>grid convergence across public example resolutions |
