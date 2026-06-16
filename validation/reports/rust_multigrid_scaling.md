# Rust Multigrid Scaling Validation

- Generated: `2026-06-16T23:34:32+00:00`
- Status: `passed`
- Claim boundary: `convergence instrumentation and local non-isolated scaling evidence only`
- Benchmark context: `single-run local workstation; not CPU-isolated; not a production speedup claim`
- Command: `cargo run --quiet -p fusion-math --example multigrid_scaling`
- CPU: `11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz`
- Kernel: `6.17.0-35-generic`
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`
- Cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
- Load average: `12.53 11.81 8.09 8/6434 1624497`

| Grid | Points | Cycles | Converged | Initial residual | Final residual | Contraction | Wall time ms |
|------|--------|--------|-----------|------------------|----------------|-------------|--------------|
| 33x33 | 1089 | 19 | true | 1.000000e+00 | 8.520544e-09 | 8.520544e-09 | 64.877 |
| 65x65 | 4225 | 20 | true | 1.000000e+00 | 7.762878e-09 | 7.762878e-09 | 285.688 |
| 129x129 | 16641 | 21 | true | 1.000000e+00 | 4.988578e-09 | 4.988578e-09 | 1457.553 |

This report validates convergence instrumentation and local scaling shape only.
It is not an isolated release-performance claim and must not be cited as a production speedup.
