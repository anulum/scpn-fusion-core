# HANDOVER: H8-147 Pytest Marker Hardening

Date: 2026-02-17
Scope: Remove `PytestUnknownMarkWarning` noise by registering the `slow` marker in project config while preserving current Python and Rust green baselines.

## Implemented
1. **pyproject.toml**
   - Added `[tool.pytest.ini_options]` section.
   - Registered `slow` marker:
     - `"slow: marks long-running regression tests"`
2. **No runtime logic changes**
   - This update is configuration-only and does not alter simulation, controller, physics, or Rust runtime behavior.

## Validation Run
- Python full suite baseline (pre-change):
  - `python -m pytest tests/ -v --tb=short`
  - Result: `1072 passed, 24 skipped, 5 warnings in 845.33s`
- Marker regression check (post-change):
  - `python -m pytest tests/test_physics_edge_cases.py -v --tb=short`
  - Result: `12 passed` (no unknown-mark warning)
- Typing gates:
  - Windows host run: `python -m mypy` -> reports platform-specific type errors.
  - CI-aligned run: `PYTHONPATH=src python -m mypy --platform linux` -> `Success: no issues found in 23 source files`
- Rust gates:
  - `cargo fmt --all -- --check` -> pass
  - `cargo clippy --all-targets --all-features -- -D warnings` -> pass
  - `cargo test --all-features` -> pass

## Notes
- Existing unrelated local workspace changes were left untouched:
  - `scpn-fusion-rs/crates/fusion-core/Cargo.toml`
  - `scpn-fusion-rs/crates/fusion-physics/Cargo.toml`
  - `mypy.ini.bak` (untracked)
