# GK Public Reference Deck Inventory

Public GENE/CGYRO/GS2 deck inventory for nonlinear GK parity. This is an input/output provenance artifact, not accepted full-fidelity parity.

- Schema: `gk-public-reference-deck-inventory-report.v1`
- Status: `blocked_public_gk_decks_indexed_missing_solver_output_parity`
- Decks indexed: `40`
- Output summaries indexed: `21`
- Web sources indexed: `6`
- Public output candidate ready: `False`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/gk_public_reference_deck_inventory.json`
- Metadata: `validation/reference_data/full_fidelity_public_artifacts/gk_public_reference_deck_inventory.metadata.json`
- SHA-256: `5cbb93ef19922e0a229aec1ea3a2a0e971db3e49203f31e34b3ace0c8dbc1bf6`
- Accepted full-fidelity ready: `False`

## Backend probe

- GS2 available: `False`
- CGYRO wrapper present: `True`
- GACODE runtime helper available: `False`

## Next action

Run GS2/CGYRO/GENE nonlinear decks with public outputs, then compare native 5D nonlinear GK heat-flux spectra, field energy, saturation, and convergence.

## Public output candidate matrix

| Solver | Status | Deck candidates | Output summaries | Web sources | Ready | Reasons |
|---|---|---:|---:|---:|:---:|---|
| GENE | `blocked_public_web_source_only_missing_redistributable_output_artifact` | 0 | 0 | 2 | `False` | missing_redistributable_same_deck_output_payload, missing_native_same_case_output_payload, missing_grid_convergence_output_evidence, missing_production_scaling_output_evidence |
| CGYRO | `blocked_public_decks_and_precision_snippets_not_same_deck_nonlinear_output` | 28 | 21 | 2 | `False` | missing_redistributable_same_deck_output_payload, missing_native_same_case_output_payload, missing_grid_convergence_output_evidence, missing_production_scaling_output_evidence, partial_numeric_output_not_full_observable_contract |
| GS2 | `blocked_public_decks_not_executed_output_artifact` | 12 | 0 | 2 | `False` | missing_redistributable_same_deck_output_payload, missing_native_same_case_output_payload, missing_grid_convergence_output_evidence, missing_production_scaling_output_evidence |