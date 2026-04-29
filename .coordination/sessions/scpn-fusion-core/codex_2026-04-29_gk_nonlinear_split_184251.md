# Session Log - Nonlinear GK Split

Date: 2026-04-29
Project: scpn-fusion-core
Branch: main

## Scope

- Continued the generated source P0 backlog after the nengo default-lane fix.
- Split `src/scpn_fusion/core/gk_nonlinear.py` into focused private modules:
  - `src/scpn_fusion/core/_gk_nonlinear_types.py`
  - `src/scpn_fusion/core/_gk_nonlinear_setup.py`
  - `src/scpn_fusion/core/_gk_nonlinear_operators.py`
  - `src/scpn_fusion/core/_gk_nonlinear_time.py`
- Kept `src/scpn_fusion/core/gk_nonlinear.py` as the public facade for `NonlinearGKConfig`, `NonlinearGKState`, `NonlinearGKResult`, and `NonlinearGKSolver`.
- Reworded the flagged collision-operator description to a domain-limited Sugama-like contract without changing the numerical operator.
- Added hardening linkage assertions for the new helper modules.
- Regenerated the underdeveloped register and source P0/P1 backlog artifacts.

## Verification

- `python -m py_compile src/scpn_fusion/core/gk_nonlinear.py src/scpn_fusion/core/_gk_nonlinear_types.py src/scpn_fusion/core/_gk_nonlinear_setup.py src/scpn_fusion/core/_gk_nonlinear_operators.py src/scpn_fusion/core/_gk_nonlinear_time.py src/scpn_fusion/core/jax_gk_nonlinear.py`
- `python -m pytest tests/test_gk_nonlinear.py -q`
- `python -m pytest tests/test_hardening_module_linkage.py -q`
- `python -m pytest tests/test_phase1_hardening.py -q`
- `python tools/generate_underdeveloped_register.py --check`
- `python tools/generate_source_p0p1_issue_backlog.py --check`
- `python tools/check_test_module_linkage.py`

## Result

- All GK split files are below the 500-line monolith threshold.
- The underdeveloped register decreased from 125 to 123 entries.
- The source P0/P1 backlog decreased from 26 to 25 issue seeds.

## Notes

- The target source and test files had pre-existing CRLF-only dirty state before this task.
- No push, pull, merge, or rebase was performed.
