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
| GAI-02 | Done | Done (`docs/rfc/GAI-02_RFC.md`, `src/scpn_fusion/control/torax_hybrid_loop.py`, `validation/gai_02_torax_hybrid.py`, `run_realtime_simulation.py`) | Done (synthetic NSTX-U-like scenarios) | Done (avoidance `>=0.90`, parity `>=95%`, p95 latency `<=1.0ms`) | Done (deterministic smoke profile) | Unassigned |
| GAI-03 | Done | Done (`docs/rfc/GAI-03_RFC.md`, `src/scpn_fusion/core/heat_ml_shadow_surrogate.py`, `src/scpn_fusion/core/global_design_scanner.py`, `validation/gai_03_heat_ml_shadow.py`) | Done (synthetic HEAT-ML scope) | Done (RMSE `<=10%`, inference `<1s`, mean reduction `>=8%`) | Done (smoke profile under CI budget) | Unassigned |
| GMVR-01 | Done | Done (`docs/rfc/GMVR-01_RFC.md`, `src/scpn_fusion/core/global_design_scanner.py`, `validation/gmvr_01_compact_constraints.py`) | Done (synthetic compact-scan scope) | Done (R `1.2..1.5m`, `Q>5`, divertor/Zeff/HTS caps) | Done (deterministic smoke profile) | Unassigned |
| GMVR-02 | Done | Done (`docs/rfc/GMVR-02_RFC.md`, `src/scpn_fusion/core/divertor_thermal_sim.py`, `validation/gmvr_02_temhd_divertor.py`) | Done (synthetic TEMHD scope) | Done (pressure/evap ratios + toroidal stability threshold) | Done (deterministic smoke profile) | Unassigned |
| GMVR-03 | Done | Done (`docs/rfc/GMVR-03_RFC.md`, `src/scpn_fusion/core/geometry_3d.py`, `validation/gmvr_03_stellarator_extension.py`) | Done (synthetic stellarator extension scope) | Done (SNN improvement + VMEC++ proxy parity thresholds) | Done (deterministic smoke profile) | Unassigned |
| GDEP-01 | Done | Done (`docs/rfc/GDEP-01_RFC.md`, `src/scpn_fusion/control/digital_twin_ingest.py`, `validation/gdep_01_digital_twin_hook.py`) | Done (synthetic NSTX-U/SPARC stream scope) | Done (scenario-planning success/risk/latency thresholds) | Done (deterministic smoke profile) | Unassigned |
| GDEP-02 | Done | Done (`docs/rfc/GDEP-02_RFC.md`, `src/scpn_fusion/core/gpu_runtime.py`, `validation/gdep_02_gpu_integration.py`) | Done (synthetic GPU-sim scope) | Done (multigrid/SNN latency + speedup thresholds) | Done (deterministic smoke profile) | Unassigned |
| GDEP-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GDEP-05 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-01 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-02 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-03 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-04 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-05 | Pending | Pending | Pending | Pending | Pending | Unassigned |
| GPHY-06 | Pending | Pending | Pending | Pending | Pending | Unassigned |
