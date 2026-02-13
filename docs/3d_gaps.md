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
| G3D-01 | LCFS extracted from 2D axisymmetric equilibrium and revolved, plus reduced VMEC-like flux-coordinate interface in `src/scpn_fusion/core/equilibrium_3d.py` | Reduced closure only (not a full 3D force-balance VMEC solve) | Extend reduced VMEC-like interface to a full 3D equilibrium solve with force-balance residual closure |
| G3D-02 | Ray-based LCFS tracing over 2D grid with reduced 3D field-line and Poincare diagnostics in `src/scpn_fusion/core/fieldline_3d.py` | Reduced closure only (does not yet solve full 3D field-line integration from self-consistent 3D MHD fields) | Extend reduced tracer to full field solve on self-consistent 3D magnetic geometry |
| G3D-03 | Transport is radial (1.5D) with low-order `n!=0` closure in `fusion-core/src/transport.rs` | Closure model only (no resolved `(rho, phi)` PDE transport) | Extend from reduced spectral closure to native toroidal transport solve |
| G3D-04 | FNO turbulence is 2D with reduced toroidal-harmonic spectral closure in `fusion-physics/src/fno.rs` | No native `(rho, theta, phi)` 3D spectral tensor evolution | Extend to full 3D FFT/FNO transport with explicit toroidal coordinate state |
| G3D-05 | Neutronics simplified wall flux methods | No volumetric 3D blanket transport | Add 3D mesh-based neutronics surrogate |
| G3D-06 | Divertor model includes reduced 3D strike-point asymmetry projection in `fusion-nuclear/src/divertor.rs` | Projection closure only (not a full 3D SOL/target transport solve) | Extend to self-consistent 3D SOL transport and wall-coupled strike-point evolution |
| G3D-07 | Control logic built on 2D state summaries | No 3D instability observables | Add toroidal asymmetry features (ELM/RMP indicators) |
| G3D-08 | CI validates 2D-heavy paths | No dedicated 3D smoke/regression pipeline | Add CI smoke test generating and validating OBJ mesh artifacts |

## Priority Order

1. G3D-08 (CI safety for 3D path) - completed
2. G3D-03 (reduced low-order toroidal transport coupling) - completed
3. G3D-04 (reduced toroidal-harmonic turbulence coupling) - completed
4. G3D-06 (reduced 3D strike-point asymmetry mapping) - completed
5. G3D-01 (reduced VMEC-like native 3D flux coordinates) - completed
6. G3D-02 (reduced field-line tracing + Poincare diagnostics) - completed
7. G3D-05 (3D blanket neutronics surrogate)

## Phase 2 Entry Criteria

- G3D-08 complete and stable in CI.
- At least one physics module upgraded from 2D/1.5D to 3D-aware coupling.
- Benchmark suite includes explicit 3D accuracy and runtime metrics.
