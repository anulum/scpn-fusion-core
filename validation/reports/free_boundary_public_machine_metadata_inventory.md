# Free-Boundary Public Machine Metadata Inventory

This report indexes cached public FreeGSNKE machine metadata and FreeGS example
scripts for the strict free-boundary reconstruction lane. It is a provenance
and geometry inventory only; it is not accepted full-fidelity parity evidence.

- Schema: `free-boundary-public-machine-metadata-inventory-report.v1`
- Status: `accepted_public_machine_metadata_with_same_case_free_boundary_reference`
- Accepted full fidelity: `True`
- Machine metadata ready: `True`
- Reference output ready: `True`
- Machine config count: `23`
- FreeGS example count: `16`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/free_boundary_public_machine_metadata_inventory.json`
- Metadata: `validation/reference_data/full_fidelity_public_artifacts/free_boundary_public_machine_metadata_inventory.metadata.json`
- SHA256: `3da4f6ac2575517ea760302c5ffb6a0e2162a47fcad4ff979fe775f0222d5542`

## Machine configuration summaries

| Machine | Role | Top-level count | Geometry elements | Points | R bounds (m) | Z bounds (m) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ITER | `active_coils` | 10 | 14 | 20 | 1.687 to 11.9919 | -7.4665 to 7.5741 |
| ITER | `limiter` | 57 | 57 | 57 | 3.9421 to 8.3938 | -4.5743 to 4.7196 |
| ITER | `passive_coils` | 148 | 148 | 592 | 3.23239 to 9.71324 | -5.65707 to 5.68948 |
| ITER | `wall` | 57 | 57 | 57 | 3.9421 to 8.3938 | -4.5743 to 4.7196 |
| MAST-U | `active_coils` | 12 | 23 | 876 | 0.19475 to 1.92205 | -2.00405 to 2.00405 |
| MAST-U | `limiter` | 47 | 47 | 47 | 0.260841 to 1.8 | -1.5099 to 1.49874 |
| MAST-U | `magnetic_probes` | 2 | 0 | 0 | none | none |
| MAST-U | `passive_coils` | 138 | 138 | 552 | 0.229 to 2.12 | -2.22637 to 2.2204 |
| MAST-U | `wall` | 91 | 91 | 91 | 0.260841 to 1.8 | -2.06614 to 2.06041 |
| SPARC | `active_coils` | 11 | 22 | 1380 | 0.45862 to 3.954 | -2.5076 to 2.5076 |
| SPARC | `limiter` | 176 | 176 | 176 | 1.264 to 2.43 | -1.575 to 1.575 |
| SPARC | `passive_coils` | 18 | 18 | 368 | 1.12 to 2.85 | -1.765 to 1.765 |
| SPARC | `wall` | 176 | 176 | 176 | 1.264 to 2.43 | -1.575 to 1.575 |
| example | `active_coils` | 4 | 6 | 47 | 0.15 to 1.8 | -1.2 to 1.2 |
| example | `limiter` | 20 | 20 | 20 | 0.356819 to 1.35 | -0.797268 to 0.797268 |
| example | `magnetic_probes` | 2 | 0 | 0 | none | none |
| example | `passive_coils` | 8 | 8 | 20 | 0.25 to 2.1 | -1.4 to 1.4 |
| example | `wall` | 5 | 5 | 5 | 0.3 to 1.4 | -0.85 to 0.85 |
| test | `active_coils` | 12 | 23 | 878 | 0.19475 to 1.92205 | -2.00705 to 2.00705 |
| test | `limiter` | 47 | 47 | 47 | 0.260841 to 1.8 | -1.5099 to 1.49874 |
| test | `magnetic_probes` | 2 | 0 | 0 | none | none |
| test | `passive_coils` | 138 | 138 | 552 | 0.229 to 2.12 | -2.22637 to 2.2204 |
| test | `wall` | 91 | 91 | 91 | 0.260841 to 1.8 | -2.06614 to 2.06041 |

## Missing full-fidelity requirements

- None
