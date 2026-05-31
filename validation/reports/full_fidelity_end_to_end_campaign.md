# Full-Fidelity End-to-End Campaign

This report keeps all declared full-fidelity blockers in one fail-closed gate.

- Schema: `full-fidelity-end-to-end-campaign.v1`
- Status: `not_full_fidelity`
- Acceptance passed: `False`
- Public source registry: `validation/reference_data/full_fidelity_public_sources.json`
- Public source download report: `validation/reports/full_fidelity_public_source_downloads.json`
- Public sources cached: `True`
- Public source cache root: `data/external/full_fidelity_public_sources`
- Local contracts ready: `True`
- Reference parity ready: `False`

| Lane | Status | Local contract ready | Reference parity ready | Sources | Next evidence |
| --- | --- | ---: | ---: | --- | --- |
| gene_cgyro_gs2_nonlinear_gk_parity | blocked_missing_public_reference_artifacts | True | False | GENE, CGYRO, GS2 | public nonlinear GENE/CGYRO/GS2 benchmark deck parity<br>production-scale radial/toroidal domain decomposition and convergence evidence<br>Maxwell field solve parity beyond compact A_parallel contract<br>validated flux spectra, zonal-flow, and saturation parity against production GK outputs |
| full_maxwell_electromagnetic_fidelity | blocked_missing_full_vlasov_maxwell_field_solve | True | False | GENE, CGYRO, GS2 | native nonlinear Ampere/Faraday closure beyond compact A_parallel/B_parallel contracts<br>GENE/CGYRO/GS2 electromagnetic field-energy and transport parity artifacts |
| production_scale_decomposition | blocked_missing_cluster_scaling_evidence | True | False | none | radial/toroidal domain decomposition implementation<br>multi-GPU or cluster scaling reports on production-size grids<br>large-grid warm GPU throughput and convergence evidence |
| dream_grade_runaway_electrons | blocked_missing_public_dream_artifacts | True | False | DREAM | public DREAM deck ingestion and production artifact parity<br>full momentum-pitch-radius Fokker-Planck evolution rather than 1D momentum projection artifact<br>validated synchrotron, bremsstrahlung, partial-screening, and transport operators against DREAM<br>distribution-function, current, and growth-rate RMSE thresholds against public DREAM output |
| aurora_strahl_grade_impurities | blocked_missing_public_aurora_strahl_artifacts | True | False | Aurora | public Aurora/STRAHL artifact ingestion and production parity<br>licensed ADAS/Open-ADAS coefficient ingestion rather than parametric ADAS-style coefficients<br>validated charge-state transport, recycling, and radiation operators against Aurora/STRAHL<br>charge-state density, total-density, and radiation RMSE thresholds against public outputs |
| free_boundary_equilibrium_strict_parity | blocked_missing_external_coil_current_reference_artifacts | True | False | FreeGS, FreeGSNKE | public coil-current sidecars or machine coil metadata for GEQDSK rows<br>strict FreeGS backend convergence on public free-boundary cases<br>profile-source/free-boundary reconstruction parity artifacts |
