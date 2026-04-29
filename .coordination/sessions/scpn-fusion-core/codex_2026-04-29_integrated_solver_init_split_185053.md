# Session Log - Integrated Solver Initialization Split

Date: 2026-04-29
Project: scpn-fusion-core
Branch: main

## Scope

- Continued the generated source P0 backlog after the nonlinear GK split.
- Extracted the `TransportSolver` initialization block from `src/scpn_fusion/core/integrated_transport_solver.py` into `src/scpn_fusion/core/_integrated_transport_solver_init.py`.
- Kept public neoclassical functions, Rust fast-path flags, fallback exception tuples, and compatibility exports in the public module so existing monkeypatch and lazy-resolution contracts still work.
- Added hardening linkage coverage for the initialization mixin.
- Regenerated the underdeveloped register and source P0/P1 backlog artifacts.

## Verification

- `python -m py_compile src/scpn_fusion/core/integrated_transport_solver.py src/scpn_fusion/core/_integrated_transport_solver_init.py`
- `python -m pytest tests/test_integrated_transport_solver.py::TestNeoclassical::test_hmode_with_neoclassical_records_eped_domain_contract -q`
- `python -m pytest tests/test_integrated_transport_solver.py tests/test_integrated_transport_solver_runtime.py tests/test_integrated_transport_runtime_physics_hardening.py -q`
- `python -m pytest tests/test_hardening_module_linkage.py -q`
- `python tools/generate_underdeveloped_register.py --check`
- `python tools/generate_source_p0p1_issue_backlog.py --check`
- `python tools/check_test_module_linkage.py`

## Result

- `src/scpn_fusion/core/integrated_transport_solver.py` is down to 489 lines.
- `src/scpn_fusion/core/_integrated_transport_solver_init.py` is 168 lines.
- The underdeveloped register decreased from 123 to 122 entries.
- The source P0/P1 backlog decreased from 25 to 24 issue seeds.

## Notes

- The public `EpedPedestalModel` re-export was preserved because the pedestal mixin resolves it through the public module.
- No push, pull, merge, or rebase was performed.
