# HANDOVER: H8-155 Transformer Input Shape Guards

Date: 2026-02-17
Scope: Harden disruption Transformer inference against malformed tensors by enforcing explicit forward-input contracts.

## Implemented
1. **src/scpn_fusion/control/disruption_predictor.py**
   - Added strict `DisruptionTransformer.forward(...)` guards:
     - requires rank-3 input (`[batch, seq, 1]`),
     - requires non-empty sequence (`seq >= 1`),
     - requires feature width of exactly 1.
   - Existing upper-bound guard (`seq <= configured seq_len`) remains in place.
   - Invalid shapes now fail fast with deterministic `ValueError` messages.

2. **tests/test_disruption_model_checkpoint.py**
   - Added `test_disruption_transformer_rejects_invalid_input_shapes` with parametrized malformed tensors:
     - rank-2 input,
     - zero-length sequence,
     - wrong feature dimension.
   - Confirms clear validation errors for each invalid shape class.

## Validation Run
- `python -m pytest tests/test_disruption_model_checkpoint.py -v --tb=short`
  - Result: `18 passed`
- `python tools/run_mypy_strict.py`
  - Result: `Success: no issues found in 23 source files`

## Notes
- Unrelated local workspace changes remained untouched:
  - `scpn-fusion-rs/crates/fusion-core/Cargo.toml`
  - `scpn-fusion-rs/crates/fusion-physics/Cargo.toml`
  - `mypy.ini.bak` (untracked)
