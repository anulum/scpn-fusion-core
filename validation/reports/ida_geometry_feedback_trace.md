# IDA geometry/source feedback trace

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `4520d424547db3294086b964266021efbc8dd86e80336fc19b0b704e34a9245c`
- Same-case terminal parity: `true`
- Facility/control/PCS/safety/scientific admission: `false`

| Run | Iteration | Phase | Refinement | Production TV | Ref-boundary TV | FP rel L2 | Terminal |
|---|---:|---|---:|---:|---:|---:|---|
| cold | 0 | ip_ramp | 0.000 | 0.870184109 | 0.857727193 | 0.291801609 | false |
| cold | 14 | ip_ramp | 0.000 | 0.950958222 | 0.906254832 | 0.512359119 | false |
| cold | 29 | ip_ramp | 0.000 | 0.933741092 | 0.898832252 | 2.32721682 | false |
| cold | 49 | pre_separatrix | 0.000 | 0.43866241 | 0.355012796 | 0.00532217421 | false |
| cold | 79 | pre_separatrix | 0.000 | 0.441251996 | 0.35985253 | 0 | false |
| cold | 99 | pre_separatrix | 0.000 | 0.441251996 | 0.35985253 | 0 | false |
| cold | 100 | separatrix_homotopy | 0.050 | 0.438155977 | 0.35985253 | 0.00629332026 | false |
| cold | 109 | separatrix_homotopy | 0.500 | 0.412183442 | 0.359852661 | 0.0573208898 | false |
| cold | 119 | separatrix_homotopy | 1.000 | 0.387916321 | 0.360442401 | 0.102737081 | false |
| cold | 129 | post_homotopy | 1.000 | 0.284178009 | 0.246083492 | 0.0128612805 | false |
| cold | 149 | post_homotopy | 1.000 | 0.251353196 | 0.220638753 | 5.22139502e-10 | true |
| warm | 0 | warm_polish | 1.000 | 0.251353196 | 0.220638753 | 5.22139502e-10 | false |
| warm | 1 | warm_polish | 1.000 | 0.251353196 | 0.220638753 | 3.71439263e-10 | true |

- Largest sparse TV increase: `0.0807741128` into `ip_ramp`
- Next ratcheting target: `ip_ramp_geometry_source_feedback`

This trace routes one engineering correction; it is not physical validation.
