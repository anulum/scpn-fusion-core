<!--
SCPN Fusion Core — 3D Gap Audit
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# 3D Gap Audit (Phase 1 Baseline)

Current 3D support exists primarily in `geometry_3d.py` and visualization-oriented workflows.  
The following gaps block full 3D physics fidelity.

## Identified Extension Points

| Gap ID | Current State | Limitation | Required Extension |
|---|---|---|---|
| G3D-01 | LCFS extracted from 2D axisymmetric equilibrium and revolved | No true non-axisymmetric equilibrium | Add VMEC-like 3D equilibrium interface and native 3D flux coordinates |
| G3D-02 | Ray-based LCFS tracing over 2D grid | No field-line topology diagnostics | Add field-line tracing and Poincare map generation |
| G3D-03 | Transport is radial (1.5D) with low-order `n!=0` closure in `fusion-core/src/transport.rs` | Closure model only (no resolved `(rho, phi)` PDE transport) | Extend from reduced spectral closure to native toroidal transport solve |
| G3D-04 | FNO turbulence is 2D | Missing 3D spectral coupling | Add 3D FFT/FNO path with toroidal harmonics |
| G3D-05 | Neutronics simplified wall flux methods | No volumetric 3D blanket transport | Add 3D mesh-based neutronics surrogate |
| G3D-06 | Divertor models are reduced geometry | No 3D strike-point asymmetry mapping | Add 3D heat-flux projection from flux geometry |
| G3D-07 | Control logic built on 2D state summaries | No 3D instability observables | Add toroidal asymmetry features (ELM/RMP indicators) |
| G3D-08 | CI validates 2D-heavy paths | No dedicated 3D smoke/regression pipeline | Add CI smoke test generating and validating OBJ mesh artifacts |

## Priority Order

1. G3D-08 (CI safety for 3D path) - completed
2. G3D-03 (reduced low-order toroidal transport coupling) - completed
3. G3D-04 (3D turbulence surrogate)
4. G3D-06 (3D divertor heat flux)
5. G3D-01 (native 3D equilibrium)

## Phase 2 Entry Criteria

- G3D-08 complete and stable in CI.
- At least one physics module upgraded from 2D/1.5D to 3D-aware coupling.
- Benchmark suite includes explicit 3D accuracy and runtime metrics.
