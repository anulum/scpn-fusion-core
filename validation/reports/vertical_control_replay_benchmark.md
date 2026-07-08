# Vertical Control Replay Benchmark

- Schema: `vertical-control-replay-benchmark.v1`
- Status: `blocked_pending_multi_profile_release_gate`
- Schema version: `1.0.0`
- Deterministic replay pass: `YES`
- Overall pass: `YES`
- Release gate status: `blocked_pending_multi_profile_release_gate`
- Full PCS production-grade ready: `NO`
- Steps: `240`
- dt: `0.001000 s`
- State trace checksum: `8b6bfb8fe534fd895f66ca6a51a811dd3ef2746c7fbcd7023c43cebc00103865`

| Controller | P95 |z| (m) | Final |z| (m) | Max |u| | Max slew | Pass |
|------------|-------------|---------------|---------|----------|------|
| pid | 0.005285 | 0.000130 | 0.164572 | 0.035000 | YES |
| super_twisting | 0.005212 | 0.000791 | 0.175000 | 0.035000 | YES |
| sliding_mode_vertical | 0.005212 | 0.000791 | 0.175000 | 0.035000 | YES |
| no_control | 0.009209 | 0.009458 | 0.000000 | 0.000000 | NO |

## Post-disturbance relaxation

- Primary controllers must reduce vertical displacement after the disturbance window ends.
- Maximum accepted final/start ratio: `0.750`

| Controller | Start |z| (m) | Final/start ratio | Pass |
|------------|----------------|-------------------|------|
| pid | 0.000789 | 0.165219 | YES |
| super_twisting | 0.001941 | 0.407205 | YES |
| sliding_mode_vertical | 0.001941 | 0.407205 | YES |
| no_control | 0.007187 | 1.315892 | NO |

## Uncertainty

- Cases: `32`
- Max uncertain |z|: `0.005997 m`
