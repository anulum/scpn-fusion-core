<!--
SCPN Fusion Core — Phase 2 Advanced RFC Tracker
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Phase 2 Advanced RFC Tracker

Use together with `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md`.

Legend:
- `Pending` = not ready for implementation
- `Ready` = RFC approved and queued
- `In Progress` = active implementation
- `Done` = implemented and validated

| Task ID | Status | Path Mapping | Data + License | Metrics Protocol | CI Cost | Owner |
|---|---|---|---|---|---|---|
| GNEU-01 | Done | Done (`docs/rfc/GNEU-01_RFC.md`, `validation/gneu_01_benchmark.py`) | Done (synthetic-only v1 scope) | Done (implemented + tested) | Done (smoke config under threshold) | Unassigned |
| GNEU-02 | Done | Done (`src/scpn_fusion/control/disruption_predictor.py`, `scpn-fusion-rs/crates/fusion-ml/src/disruption.rs`) | Done (synthetic campaign scope) | Done (pytest + Rust tests/proptest) | Done (bounded smoke runtime) | Unassigned |
| GNEU-03 | Done | Done (`src/scpn_fusion/control/fueling_mode.py`, `validation/gneu_03_fueling_mode.py`) | Done (reduced ITER-like synthetic mode) | Done (final abs density error `<=1e-3`) | Done (deterministic smoke under CI budget) | Unassigned |
| GAI-01 | Done | Done (`docs/rfc/GAI-01_RFC.md`, `src/scpn_fusion/core/gyro_swin_surrogate.py`, `validation/gai_01_turbulence_surrogate.py`) | Done (synthetic-only v1 scope) | Done (RMSE `<=10%`, speedup `>=1000x`) | Done (smoke profile under CI budget) | Unassigned |
| GAI-02 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GAI-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GMVR-01 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GMVR-02 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GMVR-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GDEP-01 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GDEP-02 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GDEP-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GDEP-05 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-01 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-02 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-04 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-05 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-06 | Pending | Pending | Pending | Pending | Pending | Unassigned |
