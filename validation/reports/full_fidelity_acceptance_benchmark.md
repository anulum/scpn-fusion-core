# Full-Fidelity Acceptance Benchmark

This report is a fail-closed diagnostic for native full-fidelity claims.
A surface passes only after public reference parity against the named production solver family is demonstrated.

- Schema: `full-fidelity-acceptance.v1`
- Gate mode: `diagnostic_fail_closed`
- Reference manifest: `validation/reference_data/full_fidelity_reference_cases.json`
- Artefact schema: `validation/reference_data/full_fidelity_artifact_schema.json`
- Acceptance passed: `False`

| Surface | Required reference equivalence | Status | Reference cases ready | Implemented dimensions | Missing requirements |
| --- | --- | --- | ---: | --- | --- |
| native_nonlinear_gyrokinetics | GENE/CGYRO/GS2 full nonlinear 5D Vlasov-Maxwell | not_full_fidelity | False | explicit_5d_phase_space_contract, electromagnetic_b_parallel_surface, electromagnetic_b_parallel_hamiltonian_coupling, electromagnetic_field_energy_accounting, electromagnetic_energy_history_export, heat_flux_spectrum_history_export, five_dimensional_delta_f_state, named_conservative_exb_term, nonlinear_invariant_history_export, jax_run_history_parity, nonlinear_exb_operator, kinetic_electron_surface, electromagnetic_a_parallel_surface, moment_conserving_collision_contract | public nonlinear GENE/CGYRO/GS2 benchmark deck parity<br>production-scale radial/toroidal domain decomposition and convergence evidence<br>Maxwell field solve parity beyond compact A_parallel contract<br>validated flux spectra, zonal-flow, and saturation parity against production GK outputs |
| runaway_electrons | DREAM kinetic/fluid runaway electron solver | not_full_fidelity | False | dreicer_source, avalanche_source, hot_tail_seed, fluid_density_balance, one_dimensional_momentum_fokker_planck_contract | multidimensional DREAM kinetic distribution parity<br>coupled radial-momentum-pitch kinetic grid with DREAM reference cases<br>synchrotron, bremsstrahlung, partial-screening, and transport parity gates<br>public DREAM deck ingestion and distribution-function RMSE thresholds |
| impurity_transport | Aurora/STRAHL collisional-operator impurity transport | not_full_fidelity | False | trace_radial_transport, edge_source_particle_conservation, neoclassical_pinch_contract, radiated_power_monotonicity | charge-state-resolved collisional-radiative operator parity<br>ADAS-backed ionisation/recombination/radiation coefficient ingestion<br>Aurora/STRAHL public case ingestion and density/radiation RMSE gates<br>multi-species source/sink matrix conservation across charge states |
