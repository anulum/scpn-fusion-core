# External Parity Score

- Schema: `scpn-fusion-core.external-parity-score.v1`
- Status: `blocked_external_parity_score`
- Acceptance passed: `False`
- Reproducibility score: `1.0`
- Parity score: `0.6`

Scores summarize current report readiness. They do not promote a lane whose underlying fail-closed report remains blocked.

## Lanes

| Lane | External code | Status | Reproducibility score | Parity score | Blocked requirements |
| --- | --- | --- | ---: | ---: | --- |
| torax_transport | TORAX | `blocked_same_physics_thresholds` | `1.0` | `0.2` | native_transport_model<br>sources_and_boundary_conditions<br>time_integration_contract |
| freegsnke_free_boundary | FreeGS/FreeGSNKE | `accepted_external_parity` | `1.0` | `1.0` | none |

## Source Reports

| Report | Status | File SHA-256 | Payload SHA-256 |
| --- | --- | --- | --- |
| `validation/reports/torax_real_parity.json` | `real_torax_reference_acquired_divergence_documented` | `7def6c1167eca2e3cb21d8afc1c04986ef6bfd520b0b0029e486e94565561bd4` | `9ef6d79b5d985dbfdd361db50dd0581a2fefbddacebf3df6e62b9d5d6ffe8a82` |
| `validation/reports/torax_same_physics_config_study.json` | `same_initial_profile_config_ready_thresholds_blocked` | `2c9a2d0216c07dd3756bd9441a505a52c8e5bf9cb62ba8f7b45b8250d63701dd` | `8220eaa73be0755e2fd1ea51a93c0a82714ffb58454079ba9f9ed39ee67dcf22` |
| `validation/reports/torax_imas_interchange.json` | `torax_core_profiles_imas_fixture_ready` | `157f35c837c0dca2683525711048a373bebce2f54a05ab1524f1b028fff2a7c0` | `9cbac602896c499982f265a6933a25d7ec11ab0e468e245e10654eb74ff947e5` |
| `validation/reports/free_boundary_strict_parity_benchmark.json` | `accepted_full_fidelity_free_boundary_parity` | `eae3b1065be4fab7b74803a05ef173369a772c3589d2e636980581f6a9c390f3` | `0633bfd9d0ca8d189e14b8220c2ae756fbd52b03b97657c7ede091014aa997be` |
| `validation/reports/freegs_public_example_reconstruction.json` | `accepted_public_freegs_same_case_free_boundary_parity` | `73edc6c0ebc1d0acc249dd8d13dc14a668e3ecd3419e336e49397b322af663ed` | `39b624fba02e92e0e9d5a569d7ceb425405582ea18e2527dbb5defb3690f6928` |
| `validation/reports/free_boundary_public_machine_metadata_inventory.json` | `accepted_public_machine_metadata_with_same_case_free_boundary_reference` | `e0da36d9f1016abb0dbce77d313032d2710e325d3eb922d1908a9fb582c7e7e1` | `89dc932da5b2d2a2e48cc6179d19a1fb3ef2c320abbdae09a89ffac7c07acb48` |
