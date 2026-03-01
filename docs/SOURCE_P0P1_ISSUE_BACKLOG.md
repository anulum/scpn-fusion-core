# Source P0/P1 Issue Backlog

- Generated at: `2026-03-01T01:36:17.207890+00:00`
- Generator: `tools/generate_source_p0p1_issue_backlog.py`
- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity

## Summary

| Metric | Value |
|---|---:|
| Source issue seeds | 15 |
| P0 seeds | 1 |
| P1 seeds | 14 |
| Domains represented | 4 |

## Marker Distribution

| Marker | Count |
|---|---:|
| `SIMPLIFIED` | 14 |
| `DEPRECATED` | 1 |
| `EXPERIMENTAL` | 1 |
| `NOT_VALIDATED` | 1 |

## Domain Distribution

| Domain | Count |
|---|---:|
| `core_physics` | 10 |
| `control` | 2 |
| `nuclear` | 2 |
| `other` | 1 |

## Auto-generated Issue Seeds

_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._

### 1. [P0] Harden `src/scpn_fusion/core/fno_turbulence_suppressor.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `104`
- **Markers**: `DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`
- **Trigger Lines**: `8, 10, 21, 154`

**Proposed Actions**
- Add real-data validation campaign and publish error bars.
- Replace default path or remove lane before next major release.
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Remove deprecated runtime-default path or replace with validated default lane.
- [ ] Publish real-data validation artifact and link it from RESULTS/claims map.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 2. [P1] Harden `src/scpn_fusion/cli.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `other`
- **Owner Hint**: Architecture WG
- **Priority Score**: `88`
- **Markers**: `EXPERIMENTAL`
- **Trigger Lines**: `75, 76, 77, 80, 83, 84, 85, 121, 135, 136, 256, 280, 295`

**Proposed Actions**
- Gate behind explicit flag and define validation exit criteria.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Ensure release lane remains experimental-excluded unless explicitly opted in.

### 3. [P1] Harden `src/scpn_fusion/control/fokker_planck_re.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `84`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `123`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 4. [P1] Harden `src/scpn_fusion/control/spi_ablation.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `84`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `15`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 5. [P1] Harden `src/scpn_fusion/core/eped_pedestal.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `9, 104`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 6. [P1] Harden `src/scpn_fusion/core/fusion_ignition_sim.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `52, 423`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 7. [P1] Harden `src/scpn_fusion/core/global_design_scanner.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `68`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 8. [P1] Harden `src/scpn_fusion/core/integrated_transport_solver.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `766, 797`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 9. [P1] Harden `src/scpn_fusion/core/jax_transport_solver.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `37`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 10. [P1] Harden `src/scpn_fusion/core/neural_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `128`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 11. [P1] Harden `src/scpn_fusion/core/rf_heating.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `89`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 12. [P1] Harden `src/scpn_fusion/core/sandpile_fusion_reactor.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `95`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 13. [P1] Harden `src/scpn_fusion/core/stability_mhd.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `83`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `375, 377`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 14. [P1] Harden `src/scpn_fusion/nuclear/blanket_neutronics.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `nuclear`
- **Owner Hint**: Nuclear WG
- **Priority Score**: `82`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `75`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.

### 15. [P1] Harden `src/scpn_fusion/nuclear/nuclear_wall_interaction.py`

- **Labels**: `hardening`, `underdeveloped`, `p1`, `nuclear`
- **Owner Hint**: Nuclear WG
- **Priority Score**: `82`
- **Markers**: `SIMPLIFIED`
- **Trigger Lines**: `256, 271`

**Proposed Actions**
- Upgrade with higher-fidelity closure or tighten domain contract.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.
- [ ] Document model-domain limits and tighten contract checks for out-of-domain inputs.
