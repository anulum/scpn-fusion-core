# DREAM Reference Execution Request

Public DREAM 2kinetic reference execution request. Settings generation uses the external source cache when present and otherwise preserves tracked deck evidence; full reference output requires a compiled DREAM backend.

- Schema: `dream-reference-execution-request.v1`
- Status: `blocked_missing_dream_backend`
- Source commit: `ecdd5e146537c77602c9d7cc76b36100200e4b9a`
- Settings deck generated: `True`
- Settings deck: `data/external/full_fidelity_public_sources/repos/dream/examples/2kinetic/dream_settings.h5`
- Settings SHA-256: `0b185469ee2babaf25785cca406d4b2832b3ef10ac20975952632a18f8b62ed1`
- DREAM backend available: `False`
- Reference output ready: `False`
- Same-case comparison ready: `False`
- Comparison status: `blocked_missing_reference_output`
- Accepted full-fidelity ready: `False`

## Required DREAM output contract

- Schema: `dream-output-contract.v1`
- Coordinate axes: `time_s, radius_m, momentum_mec, pitch_cosine`
- Observables: `f_p_xi_t, runaway_current_t, avalanche_growth_rate_t, synchrotron_loss_power_t, partial_screening_drag_t, bremsstrahlung_loss_power_t`

## Next action

Install/build PETSc and compile DREAM with iface/dreami, then rerun this harness.
