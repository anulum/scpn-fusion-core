# Source P0/P1 Issue Backlog

- Generated at: `2026-04-29T16:31:53.878102+00:00`
- Generator: `tools/generate_source_p0p1_issue_backlog.py`
- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity

## Summary

| Metric | Value |
|---|---:|
| Source issue seeds | 26 |
| P0 seeds | 4 |
| P1 seeds | 22 |
| Domains represented | 3 |

## Marker Distribution

| Marker | Count |
|---|---:|
| `SIMPLIFIED` | 23 |
| `MONOLITH` | 4 |

## Domain Distribution

| Domain | Count |
|---|---:|
| `core_physics` | 20 |
| `control` | 5 |
| `diagnostics_io` | 1 |

## Auto-generated Issue Seeds

_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._

### 1. [P0] Harden `src/scpn_fusion/core/gk_nonlinear.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`, `SIMPLIFIED`
- **Trigger Lines**: `1, 19`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 2. [P0] Harden `src/scpn_fusion/core/integrated_transport_solver.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 85.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 3. [P0] Harden `src/scpn_fusion/core/tglf_interface.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 4. [P0] Harden `src/scpn_fusion/io/tokamak_archive.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `diagnostics_io`
- **Owner Hint**: Diagnostics/IO WG
- **Priority Score**: `99`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 79.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 5. [P1] Harden `src/scpn_fusion/control/burn_controller.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `139`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 80.0% (tools/coverage_guard.py).

### 6. [P1] Harden `src/scpn_fusion/control/controller_tuning.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `62`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 80.0% (tools/coverage_guard.py).

### 7. [P1] Harden `src/scpn_fusion/control/density_controller.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `216`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 80.0% (tools/coverage_guard.py).

### 8. [P1] Harden `src/scpn_fusion/control/mu_synthesis.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `149`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 80.0% (tools/coverage_guard.py).

### 9. [P1] Harden `src/scpn_fusion/control/realtime_efit.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `98`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 80.0% (tools/coverage_guard.py).

### 10. [P1] Harden `src/scpn_fusion/core/alfven_eigenmodes.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `254`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 11. [P1] Harden `src/scpn_fusion/core/disruption_sequence.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `114, 211`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 12. [P1] Harden `src/scpn_fusion/core/elm_model.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `28, 56, 148`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 13. [P1] Harden `src/scpn_fusion/core/gk_eigenvalue.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `204`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 14. [P1] Harden `src/scpn_fusion/core/gk_geometry.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `120, 126`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 15. [P1] Harden `src/scpn_fusion/core/gk_online_learner.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `137`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 16. [P1] Harden `src/scpn_fusion/core/gk_species.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `12, 156, 165, 178`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 17. [P1] Harden `src/scpn_fusion/core/impurity_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `173`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 18. [P1] Harden `src/scpn_fusion/core/integrated_scenario.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `38, 248, 258`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 19. [P1] Harden `src/scpn_fusion/core/lh_transition.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `51, 152`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 20. [P1] Harden `src/scpn_fusion/core/locked_mode.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `35, 75`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 21. [P1] Harden `src/scpn_fusion/core/momentum_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `113`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 22. [P1] Harden `src/scpn_fusion/core/neoclassical.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `213`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 23. [P1] Harden `src/scpn_fusion/core/neural_turbulence.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `180, 227`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 24. [P1] Harden `src/scpn_fusion/core/plasma_wall_interaction.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `31, 47, 181`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 25. [P1] Harden `src/scpn_fusion/core/stellarator_geometry.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `154`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 26. [P1] Harden `src/scpn_fusion/core/vmec_lite.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `52`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).
