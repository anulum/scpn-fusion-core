# External Parity Score

- Schema: `scpn-fusion-core.external-parity-score.v1`
- Status: `blocked_external_parity_score`
- Acceptance passed: `False`
- Reproducibility score: `0.9`
- Parity score: `0.1`

Scores summarize current report readiness. They do not promote a lane whose underlying fail-closed report remains blocked.

## Lanes

| Lane | External code | Status | Reproducibility score | Parity score | Blocked requirements |
| --- | --- | --- | ---: | ---: | --- |
| torax_transport | TORAX | `blocked_same_physics_thresholds` | `1.0` | `0.2` | native_transport_model<br>sources_and_boundary_conditions<br>time_integration_contract |
| freegsnke_free_boundary | FreeGS/FreeGSNKE | `blocked_free_boundary_external_parity` | `0.8` | `0.0` | non_admitting_strict_parity_report |

## Source Reports

| Report | Status | File SHA-256 | Payload SHA-256 |
| --- | --- | --- | --- |
| `validation/reports/torax_real_parity.json` | `real_torax_reference_acquired_divergence_documented` | `392f8b42dfcd8527c62ec40eb366ac3ef854eb76f1d20a78486ae39948eb537e` | `14344c947c5287eadeee1705c03a15529a26f1bb8dc62ef66840594b6e74f60d` |
| `validation/reports/torax_same_physics_config_study.json` | `same_initial_profile_config_ready_thresholds_blocked` | `4de4e2cad4b0f2413a4b68f5a332615b3507fddb165aa86b4339b9a5f8c269d2` | `826f7c13afae4d2ed477c40a96b25deecbbbe8faf0c42a1097c11a9314d71061` |
| `validation/reports/torax_imas_interchange.json` | `torax_core_profiles_imas_fixture_ready` | `5cf1921d0acb2f6acd0692bec7183ec71f3410812f3286fef0e3dc7271a72f01` | `3b24dfdcfaf1092ce26f785ded6d618912cfad2c55e21953933391ed6b8864d9` |
| `validation/reports/free_boundary_strict_parity_benchmark.json` | `blocked_free_boundary_strict_parity` | `a351402424b1fda016b541b247664fd781fd60b97923301c2f07394816938d9b` | `e7e54de2d1ee4eed215f5dcc4d2d82ade8484186142bcc8cde16aac8264091b5` |
| `validation/reports/freegs_public_example_reconstruction.json` | `accepted_public_freegs_same_case_free_boundary_parity` | `73edc6c0ebc1d0acc249dd8d13dc14a668e3ecd3419e336e49397b322af663ed` | `39b624fba02e92e0e9d5a569d7ceb425405582ea18e2527dbb5defb3690f6928` |
| `validation/reports/free_boundary_public_machine_metadata_inventory.json` | `accepted_public_machine_metadata_with_same_case_free_boundary_reference` | `e0da36d9f1016abb0dbce77d313032d2710e325d3eb922d1908a9fb582c7e7e1` | `89dc932da5b2d2a2e48cc6179d19a1fb3ef2c320abbdae09a89ffac7c07acb48` |
