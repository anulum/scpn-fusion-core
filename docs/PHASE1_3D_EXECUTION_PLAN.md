<!--
SCPN Fusion Core — Phase 1 3D Execution Plan
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Phase 1 3D Execution Plan (v1.1 Baseline)

Source: `SCPN Fusion Core with Enhanced 3D Capabilities.md`

## Objectives

- Stabilize immediate 3D baseline around LCFS mesh generation.
- Convert high-level roadmap into executable deliverables with tests.
- Produce explicit 3D gap inventory to drive Phase 2 implementation.

## Sprint Deliverables (Executed Now)

| ID | Task | Status | Output |
|---|---|---|---|
| P1-001 | Harden 3D geometry core | Done | `src/scpn_fusion/core/geometry_3d.py` |
| P1-002 | Add deterministic 3D unit tests | Done | `tests/test_geometry_3d.py` |
| P1-003 | Add "run in 5 mins" 3D quickstart | Done | `examples/run_3d_flux_quickstart.py` |
| P1-004 | Document 3D extension gaps | Done | `docs/3d_gaps.md` |
| P1-005 | Define sequenced Phase 1 plan | Done | `docs/PHASE1_3D_EXECUTION_PLAN.md` |

## Phase 1 Follow-Up Items (Executed)

| ID | Task | Status | Output |
|---|---|---|---|
| P1-006 | Add ITER/SPARC RMSE dashboard script | Done | `validation/rmse_dashboard.py` |
| P1-007 | Add cProfile + flamegraph scripts for kernel/3D path | Done | `profiling/profile_kernel.py`, `profiling/profile_geometry_3d.py` |
| P1-008 | Add CI job for geometry quickstart smoke test | Done | `.github/workflows/ci.yml` |
| P1-009 | Add docs quickstart section in README + Sphinx | Done | `README.md`, `docs/sphinx/` updates |
| P1-010 | Add RMSE dashboard regression test coverage | Done | `tests/test_rmse_dashboard.py` |
| P1-011 | Publish Sphinx page for 3D/validation/profiling workflows | Done | `docs/sphinx/workflows.rst` |
| P1-012 | Add CI smoke for RMSE dashboard generation | Done | `.github/workflows/ci.yml` |
| P1-013 | Add simple 3D visual preview workflow (PNG) | Done | `src/scpn_fusion/core/geometry_3d.py`, `examples/run_3d_flux_quickstart.py` |

## Next Queue (Phase 1)

All currently scoped Phase 1 items (`P1-001` through `P1-013`) are complete.
Phase 2 execution is tracked in `docs/NEXT_SPRINT_EXECUTION_QUEUE.md` and normalized by `docs/3d_gaps.md`.

## Verification Commands

```bash
python -m pytest tests/test_geometry_3d.py -v
python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24
```

## Success Criteria for v1.1 3D Baseline

- Geometry pipeline is test-covered and deterministic.
- Mesh export path works without notebook dependencies.
- 3D technical debt is explicit and prioritized for Phase 2.
