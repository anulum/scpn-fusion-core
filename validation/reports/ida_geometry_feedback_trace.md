# IDA geometry/source feedback trace

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `5d98c93fdcb59d914aac2282fefa1ad105677214546cb204959a0d85bf9ca28c`
- Same-case terminal parity: `true`
- Facility/control/PCS/safety/scientific admission: `false`

| Run | Iteration | Phase | Refinement | Production TV | Ref-boundary TV | FP rel L2 | Terminal |
|---|---:|---|---:|---:|---:|---:|---|
| cold | 0 | ip_ramp | 0.000 | 0.870184109 | 0.857727193 | 0.291801609 | false |
| cold | 14 | ip_ramp | 0.000 | 0.950958222 | 0.906254832 | 0.512359119 | false |
| cold | 29 | ip_ramp | 0.000 | 0.933741092 | 0.898832252 | 2.32721682 | false |
| cold | 49 | pre_separatrix | 0.000 | 0.770998768 | 0.781225948 | 0.2706786 | false |
| cold | 79 | pre_separatrix | 0.000 | 0.44363853 | 0.363829658 | 0.000846013273 | false |
| cold | 99 | pre_separatrix | 0.000 | 0.441251996 | 0.35985253 | 1.27294663e-11 | false |
| cold | 100 | separatrix_homotopy | 0.050 | 0.438155977 | 0.35985253 | 0.00629332026 | false |
| cold | 109 | separatrix_homotopy | 0.500 | 0.412183442 | 0.359852661 | 0.0573208898 | false |
| cold | 119 | separatrix_homotopy | 1.000 | 0.387916419 | 0.360442502 | 0.102737042 | false |
| cold | 129 | post_homotopy | 1.000 | 0.405799274 | 0.373581947 | 0.073910861 | false |
| cold | 149 | post_homotopy | 1.000 | 0.371876662 | 0.351081174 | 0.0369165463 | false |
| cold | 179 | post_homotopy | 1.000 | 0.25135327 | 0.220638793 | 1.65808742e-08 | true |
| warm | 0 | warm_polish | 1.000 | 0.251353216 | 0.220638763 | 4.32921808e-09 | false |
| warm | 1 | warm_polish | 1.000 | 0.251353218 | 0.220638764 | 3.86530896e-09 | false |
| warm | 4 | warm_polish | 1.000 | 0.251353215 | 0.220638763 | 3.72095489e-09 | false |
| warm | 6 | warm_polish | 1.000 | 0.251353197 | 0.220638753 | 8.93875696e-10 | true |

- Largest sparse TV increase: `0.0807741128` into `ip_ramp`
- Next ratcheting target: `ip_ramp_geometry_source_feedback`

This trace routes one engineering correction; it is not physical validation.
