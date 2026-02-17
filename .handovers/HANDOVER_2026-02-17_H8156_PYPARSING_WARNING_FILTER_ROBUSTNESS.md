# HANDOVER: H8-156 Pyparsing Warning Filter Robustness

Date: 2026-02-17
Scope: Resolve Python 3.9 CI failure caused by pyparsing `oneOf` deprecation warnings bypassing the existing strict warning filter.

## Context
- Upstream commits integrated before this fix:
  - `120ef2c` (v2.0.0a1 baseline)
  - `b1d4ed8` (initial pyparsing filter)
  - `9daa2b0` (mypy strict + pyparsing filter v2)
- Failing CI run: `22094584757`
  - Python 3.9 failed in test stage due warning filter mismatch on pyparsing deprecation warning class/category.

## Implemented
1. **pyproject.toml**
   - Updated pytest warning filter from:
     - `"ignore:.*oneOf.*deprecated.*:DeprecationWarning"`
   - To:
     - `"ignore:.*oneOf.*[dD]eprecated.*:Warning"`
   - Rationale:
     - category broadened from `DeprecationWarning` to `Warning` to catch `PyparsingDeprecationWarning` variants,
     - message regex now case-tolerant for `deprecated`/`Deprecated`,
     - preserves global strict policy `error::DeprecationWarning`.

## Validation Run
- `python -m pytest tests/test_disruption_model_checkpoint.py -q`
  - Result: `18 passed`
- `python tools/run_mypy_strict.py`
  - Result: `Success: no issues found in 23 source files`

## Notes
- Unrelated local workspace items remained untouched:
  - `artifacts/` (untracked)
  - `mypy.ini.bak` (untracked)
