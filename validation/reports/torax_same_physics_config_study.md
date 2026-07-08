# TORAX Same-Physics Configuration Study

Status: `same_initial_profile_config_ready_thresholds_blocked`
Same-physics ready: `False`
Threshold tightening ready: `False`
Physics equivalence claimed: `False`

This study proves the TORAX final profiles can initialize the native transport solver and identifies the remaining same-physics blockers. It does not claim TORAX/native physics equivalence.

## Native Solver Probe

- Config: `validation/iter_config.json`
- Profile points: `27`
- Transport model after update: `reduced_multichannel_analytic`
- Finite initial state: `True`
- Finite after one-step probe: `True`

## Same-Physics Matrix

| Component | Status | Blocks threshold tightening | Evidence |
|---|---|---:|---|
| `radial_grid` | `shared_ready` | `False` | TORAX rho_norm copied into native solver rho grid. |
| `initial_profiles` | `shared_ready` | `False` | TORAX Te, Ti, and ne copied into native solver units. |
| `native_transport_model` | `blocked_model_mismatch` | `True` | Native update uses reduced_multichannel_analytic; TORAX transport-model internals are not in the artifact. |
| `sources_and_boundary_conditions` | `blocked_missing_same_deck_controls` | `True` | The tracked TORAX artifact contains final profiles and config SHA, not source terms, boundary controls, or a native-compatible input deck. |
| `time_integration_contract` | `blocked_missing_same_step_trace` | `True` | No redistributable TORAX step-by-step state trace is tracked for matching native one-step or trajectory thresholds. |

## Threshold Blockers

- `native_transport_model`
- `sources_and_boundary_conditions`
- `time_integration_contract`
