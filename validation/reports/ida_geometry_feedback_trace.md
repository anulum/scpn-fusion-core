# IDA geometry/source feedback trace

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `f43d360ca542a6e0bb3baddc6ff6aee135603984974088b84e1ef0ed386f0a4a`
- Same-case terminal parity: `false`
- Facility/control/PCS/safety/scientific admission: `false`

| Run | Iteration | Phase | Refinement | Production TV | Ref-boundary TV | FP rel L2 | Terminal |
|---|---:|---|---:|---:|---:|---:|---|
| cold | 0 | ip_ramp | 0.000 | 0.870184109 | 0.857727193 | 0.291801609 | false |
| cold | 14 | ip_ramp | 0.000 | 0.959097678 | 0.907798286 | 0.370437744 | false |
| cold | 29 | ip_ramp | 0.000 | 0.523674301 | 0.479292096 | 0.484217257 | false |
| cold | 49 | pre_separatrix | 0.000 | 0.862373976 | 0.892890055 | 0.0610183655 | false |
| cold | 79 | pre_separatrix | 0.000 | 0.906692188 | 0.936816498 | 0.0441126716 | false |
| cold | 99 | pre_separatrix | 0.000 | 0.943954475 | 0.967549403 | 1.38811999e-05 | false |
| cold | 100 | separatrix_homotopy | 0.050 | 0.945071784 | 0.96755047 | 0.00494995514 | false |
| cold | 109 | separatrix_homotopy | 0.500 | 0.95945549 | 0.970664343 | 0.0196034864 | false |
| cold | 119 | separatrix_homotopy | 1.000 | 0.975056094 | 0.975627246 | 0.0256275941 | false |
| cold | 129 | post_homotopy | 1.000 | 0.980264665 | 0.977444497 | 0.0254953637 | false |
| cold | 149 | post_homotopy | 1.000 | 0.98783456 | 0.982693995 | 0.00136481388 | false |
| cold | 159 | post_homotopy | 1.000 | 0.987921478 | 0.982839782 | 2.77706068e-10 | true |
| warm | 0 | warm_polish | 1.000 | 0.987921478 | 0.982839782 | 2.77706068e-10 | false |
| warm | 1 | warm_polish | 1.000 | 0.987921478 | 0.982839782 | 2.00160244e-10 | true |

- Largest sparse TV increase: `0.338699675` into `pre_separatrix`
- Next ratcheting target: `compiled_trace_parity_failure`

This trace routes one engineering correction; it is not physical validation.
