# HANDOVER: H8-154 Corrupted Checkpoint Fallback Hardening

Date: 2026-02-17
Scope: Prevent corrupted disruption-model checkpoints from crashing safe inference paths by enforcing structured fallback behavior.

## Implemented
1. **src/scpn_fusion/control/disruption_predictor.py**
   - Hardened `load_or_train_predictor(...)` checkpoint restore path.
   - Wrapped `torch.load(...)` + state restoration in `try/except`.
   - On restore failure with `allow_fallback=True`, now returns fallback metadata:
     - `fallback=True`
     - `reason=checkpoint_load_failed:<ExceptionType>`
     - preserves `model_path` and normalized `seq_len`.
   - Preserved strict behavior when `allow_fallback=False` (re-raises original exception).

2. **tests/test_disruption_model_checkpoint.py**
   - Added `test_load_or_train_returns_fallback_on_corrupted_checkpoint`.
   - Added `test_predict_disruption_risk_safe_falls_back_on_corrupted_checkpoint`.
   - These guard both direct loader behavior and end-user safe-risk API behavior.

## Validation Run
- `python -m pytest tests/test_disruption_model_checkpoint.py -v --tb=short`
  - Result: `15 passed`
- `python tools/run_mypy_strict.py`
  - Result: `Success: no issues found in 23 source files`

## Notes
- Unrelated local workspace changes remained untouched:
  - `scpn-fusion-rs/crates/fusion-core/Cargo.toml`
  - `scpn-fusion-rs/crates/fusion-physics/Cargo.toml`
  - `mypy.ini.bak` (untracked)
