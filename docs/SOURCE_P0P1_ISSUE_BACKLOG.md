# Source P0/P1 Issue Backlog

- Generated at: `2026-03-14T18:53:02.367953+00:00`
- Generator: `tools/generate_source_p0p1_issue_backlog.py`
- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity

## Summary

| Metric | Value |
|---|---:|
| Source issue seeds | 20 |
| P0 seeds | 7 |
| P1 seeds | 13 |
| Domains represented | 3 |

## Marker Distribution

| Marker | Count |
|---|---:|
| `SIMPLIFIED` | 13 |
| `MONOLITH` | 7 |

## Domain Distribution

| Domain | Count |
|---|---:|
| `core_physics` | 14 |
| `control` | 5 |
| `diagnostics_io` | 1 |

## Auto-generated Issue Seeds

_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._

### 1. [P0] Harden `src/scpn_fusion/control/free_boundary_supervisory_control.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `110`
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

### 2. [P0] Harden `src/scpn_fusion/control/free_boundary_tracking.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `110`
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

### 3. [P0] Harden `src/scpn_fusion/core/integrated_transport_solver_model.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `109`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 4. [P0] Harden `src/scpn_fusion/core/neural_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `109`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 5. [P0] Harden `src/scpn_fusion/core/integrated_transport_solver.py`

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

### 6. [P0] Harden `src/scpn_fusion/core/tglf_interface.py`

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
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 7. [P0] Harden `src/scpn_fusion/io/tokamak_archive.py`

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

### 8. [P1] Harden `src/scpn_fusion/control/density_controller.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
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

### 9. [P1] Harden `src/scpn_fusion/control/mu_synthesis.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `146`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 10. [P1] Harden `src/scpn_fusion/control/realtime_efit.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `86`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `99`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).

### 11. [P1] Harden `src/scpn_fusion/core/alfven_eigenmodes.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `251`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 12. [P1] Harden `src/scpn_fusion/core/disruption_sequence.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `111, 208`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 13. [P1] Harden `src/scpn_fusion/core/elm_model.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `25, 53, 145`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 14. [P1] Harden `src/scpn_fusion/core/impurity_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `170`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 15. [P1] Harden `src/scpn_fusion/core/lh_transition.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `48, 149`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 16. [P1] Harden `src/scpn_fusion/core/locked_mode.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `32, 72`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 17. [P1] Harden `src/scpn_fusion/core/momentum_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `110`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 18. [P1] Harden `src/scpn_fusion/core/neural_turbulence.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `177, 224`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 19. [P1] Harden `src/scpn_fusion/core/plasma_wall_interaction.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `28, 44, 178`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).

### 20. [P1] Harden `src/scpn_fusion/core/vmec_lite.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `85`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `49`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
