# HANDOVER: H8-149 Strict Pytest Marker Enforcement

Date: 2026-02-17
Scope: Harden test hygiene by converting unknown pytest markers from warnings into hard failures.

## Implemented
1. **pyproject.toml**
   - Added `addopts = "--strict-markers"` under `[tool.pytest.ini_options]`.
   - Existing custom marker registration (`slow`) remains unchanged.

## Validation Run
- `python -m pytest tests/test_physics_edge_cases.py -q`
  - Result: `12 passed`
- `python tools/run_mypy_strict.py`
  - Result: `Success: no issues found in 23 source files`

## Notes
- Unrelated local workspace changes were not modified:
  - `scpn-fusion-rs/crates/fusion-core/Cargo.toml`
  - `scpn-fusion-rs/crates/fusion-physics/Cargo.toml`
  - `mypy.ini.bak` (untracked)
