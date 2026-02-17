# HANDOVER: H8-150 Disruption Transformer Batch-First Hardening

Date: 2026-02-17
Scope: Remove nested-tensor performance warning and reduce shape-handling fragility in the disruption predictor Transformer path.

## Implemented
1. **src/scpn_fusion/control/disruption_predictor.py**
   - Updated `nn.TransformerEncoderLayer` to `batch_first=True`.
   - Removed manual `permute(1, 0, 2)` in `forward`.
   - Updated last-step extraction from `output[-1, :, :]` to `output[:, -1, :]`.
   - Preserved input/output tensor contract (`[batch, seq, features]` in, `[batch, 1]` out).

## Validation Run
- `python -m pytest tests/test_disruption_model_checkpoint.py -v --tb=short`
  - Result: `10 passed`
- `python -m pytest tests/test_gneu_02_anomaly.py -v --tb=short`
  - Result: `23 passed`
- `python tools/run_mypy_strict.py`
  - Result: `Success: no issues found in 23 source files`

## Notes
- Unrelated local workspace changes remained untouched:
  - `scpn-fusion-rs/crates/fusion-core/Cargo.toml`
  - `scpn-fusion-rs/crates/fusion-physics/Cargo.toml`
  - `mypy.ini.bak` (untracked)
