# Full-Fidelity End-to-End Campaign

This report keeps all declared full-fidelity blockers in one fail-closed gate.

- Schema: `full-fidelity-end-to-end-campaign.v1`
- Status: `not_full_fidelity`
- Acceptance passed: `False`
- SAS dataset readiness report: `validation/reports/sas_dataset_readiness.json`
- SAS dataset readiness status: `blocked_missing_required_external_parity_datasets`
- SAS dataset available entries: `11`
- SAS dataset blocked entries: `38`
- SAS dataset checksum rows: `84`
- SAS dataset external parity outputs ready: `False`
- SAS dataset accepted full-fidelity ready: `False`
- Public source registry: `validation/reference_data/full_fidelity_public_sources.json`
- Public source download report: `validation/reports/full_fidelity_public_source_downloads.json`
- Public sources cached: `True`
- Public source cache root: `data/external/full_fidelity_public_sources`
- Public reference artifact conversion report: `validation/reports/full_fidelity_reference_artifact_conversion.json`
- Partial public output artifacts: `3`
- Accepted public reference artifacts: `1`
- DREAM execution report: `validation/reports/dream_reference_execution_request.json`
- DREAM settings deck generated: `True`
- DREAM reference output ready: `False`
- DREAM execution status: `blocked_missing_dream_backend`
- Runaway native kinetic operator evidence ready: `True`
- Runaway full momentum-pitch-radius operator ready: `False`
- Runaway DREAM same-case thresholds ready: `False`
- Runaway source-term budget evidence ready: `True`
- Runaway source-term DREAM same-case budget ready: `False`
- Runaway kinetic operator evidence status: `blocked_native_projection_artifact_not_full_dream_operator`
- Aurora execution report: `validation/reports/aurora_reference_execution_artifact.json`
- Aurora artifact generated: `True`
- Aurora reference output ready: `True`
- Aurora execution status: `blocked_partial_public_atomic_artifact_not_transport_parity`
- Impurity native transport evidence ready: `True`
- Impurity charge-state radial transport operator ready: `True`
- Impurity Aurora/STRAHL same-case comparison ready: `True`
- Impurity Aurora/STRAHL same-case threshold checks ready: `True`
- Impurity Aurora/STRAHL same-case thresholds passed: `True`
- Impurity Aurora/STRAHL same-case comparison status: `accepted_native_aurora_effective_transport_closure_thresholds`
- Impurity source/sink budget evidence ready: `True`
- Impurity source/sink Aurora/STRAHL same-case budget ready: `True`
- Impurity transport operator evidence status: `accepted_native_effective_transport_source_sink_closure`
- GK deck inventory report: `validation/reports/gk_public_reference_deck_inventory.json`
- GK public decks indexed: `40`
- GK public outputs indexed: `21`
- GK deck inventory status: `blocked_public_gk_decks_indexed_missing_solver_output_parity`
- GK electromagnetic fidelity report: `validation/reports/gk_electromagnetic_fidelity.json`
- GK electromagnetic fidelity status: `blocked_missing_external_em_parity_outputs`
- GK electromagnetic compact closure ready: `True`
- GK electromagnetic grid convergence ready: `True`
- GK electromagnetic Maxwell evolution ready: `True`
- GK electromagnetic native same-case thresholds ready: `True`
- GK electromagnetic self-consistent kinetic current ready: `False`
- GK electromagnetic external parity ready: `False`
- Production decomposition report: `validation/reports/production_decomposition_contract.json`
- Production decomposition contract pass: `True`
- Production decomposition halo-face integrity ready: `True`
- Production decomposition distributed halo exchange ready: `False`
- Production-scale ready: `False`
- Production decomposition status: `blocked_local_decomposition_ready_missing_distributed_runtime_scaling`
- Free-boundary machine metadata report: `validation/reports/free_boundary_public_machine_metadata_inventory.json`
- Free-boundary machine metadata indexed: `23`
- Free-boundary machine metadata ready: `True`
- Free-boundary machine metadata status: `blocked_machine_metadata_indexed_missing_same_case_free_boundary_reconstruction`
- FreeGS public example reconstruction report: `validation/reports/freegs_public_example_reconstruction.json`
- FreeGS public example cases: `2`
- FreeGS public example vacuum comparison pass: `True`
- FreeGS public example external output ready: `True`
- FreeGS public example reconstruction status: `blocked_public_freegs_native_same_case_compared_missing_grid_convergence_coil_sidecars_reference_output`
- Free-boundary strict threshold acceptance ready: `True`
- Free-boundary geometry containment ready: `True`
- Free-boundary boundary-containment metric ready: `True`
- Free-boundary grid convergence ready: `False`
- Free-boundary coil/vacuum sidecar ready: `False`
- Free-boundary same-case public reference output ready: `False`
- Free-boundary failed threshold checks: `0`
- Free-boundary strict parity status: `blocked_free_boundary_strict_parity`
- Free-boundary strict parity blockers: `grid_convergence_evidence_missing, public_external_coil_vacuum_sidecars_missing, same_case_public_reference_output_missing`
- Free-boundary strict parity report: `validation/reports/free_boundary_strict_parity_benchmark.json`
- Local contracts ready: `True`
- Reference parity ready: `False`

| Lane | Status | Local contract ready | Reference parity ready | Sources | Next evidence |
| --- | --- | ---: | ---: | --- | --- |
| sas_dataset_readiness | blocked_missing_required_external_parity_datasets | True | False | none | facility_raw_data<br>free_boundary_coil_current_sidecars<br>gyrokinetics_em_same_deck_outputs<br>gyrokinetics_same_deck_outputs<br>impurity_transport_same_deck_outputs<br>runaway_same_deck_outputs |
| gene_cgyro_gs2_nonlinear_gk_parity | blocked_missing_external_output_manifest | True | False | GENE, CGYRO, GS2 | same-deck external nonlinear distribution output for GENE, CGYRO, and GS2<br>heat_flux_spectra_time_kx_ky_species for all required solver families<br>field_energy_history_phi_apar_bpar for all required solver families<br>zonal_flow_and_saturation_metrics for all required solver families<br>shared benchmark_case_id and deck_physics_sha256 across GENE, CGYRO, and GS2<br>native same-case nonlinear GK solver-output comparison<br>grid-convergence evidence for converted public nonlinear GK outputs<br>production-scale scaling evidence for converted public nonlinear GK outputs<br>complete checksum/provenance/threshold evidence package |
| full_maxwell_electromagnetic_fidelity | blocked_missing_external_em_parity_outputs | True | False | GENE, CGYRO, GS2 | self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop<br>same-deck electromagnetic GENE/CGYRO/GS2 output artifacts<br>external electromagnetic phi/A_parallel/B_parallel same-case parity thresholds<br>same-deck external electromagnetic grid-convergence evidence |
| production_scale_decomposition | blocked_local_decomposition_ready_missing_distributed_runtime_scaling | True | False | none | cluster MPI scaling report over the declared rank tiles<br>multi-GPU distributed execution path over the declared rank tiles<br>large-grid cluster/GPU wall-time scaling report<br>same-physics convergence evidence across distributed MPI/multi-GPU decomposition shapes<br>hardware-specific multi-rank throughput and efficiency thresholds<br>accepted distributed scaling gate over required rank counts<br>accepted distributed run manifests with reproducibility fields and checksums |
| dream_grade_runaway_electrons | blocked_missing_public_dream_artifacts | True | False | DREAM | compiled DREAM iface/dreami same-case output<br>native coupled momentum-pitch-radius Fokker-Planck operator<br>radial transport operator on evolved radius grid<br>full pitch-angle scattering operator on evolved pitch grid<br>DREAM partial-screening operator parity<br>DREAM bremsstrahlung and synchrotron loss parity<br>distribution, current, and growth-rate threshold comparison against DREAM |
| aurora_strahl_grade_impurities | accepted_native_aurora_effective_transport_closure_thresholds | True | True | Aurora | independent mechanistic Aurora/STRAHL recycling validation beyond effective closure replay |
| free_boundary_equilibrium_strict_parity | blocked_free_boundary_strict_parity | True | False | FreeGS, FreeGSNKE | grid_convergence_evidence_missing<br>public_external_coil_vacuum_sidecars_missing<br>same_case_public_reference_output_missing |
