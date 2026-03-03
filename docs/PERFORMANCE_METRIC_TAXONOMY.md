# Performance Metric Taxonomy

This page defines the latency metric IDs used across README, HONEST_SCOPE, and
competitive analysis to avoid mixing kernel-only and end-to-end timings.

## Control Latency Metrics

| Metric ID | Scope | Includes | Excludes | Current headline value |
|---|---|---|---|---|
| `control.pid_kernel_step_us` | Rust PID kernel microbenchmark | Core control kernel compute step in Rust | Python orchestration, plant integration, report/log overhead | 0.52 us P50 |
| `control.closed_loop_step_us` | End-to-end closed-loop control step | Controller step including orchestration/interop overhead used in campaign loops | Full transport/equilibrium solve wall-clock | 11.9 us P50 / 23.9 us P99 |
| `control.hil_loop_us` | Hardware-in-the-loop loop latency | HIL integration path as reported in `RESULTS.md` | Offline synthetic microbench shortcuts | 11.3 us P50 (latest HIL lane) |

## Clock Separation

- **Fast control clock (1-10 kHz):** controller tick (`control.pid_kernel_step_us`, `control.closed_loop_step_us`).
- **Slow physics clock (10-100 Hz):** equilibrium/transport state updates and inverse solves.
- **Surrogate bridge:** reduced-order/neural surrogates can approximate the slow lane at higher update rates; they are reported separately from high-fidelity solves.

## Rules

1. Never compare two numbers without stating the metric ID.
2. Kernel metrics (`control.pid_kernel_step_us`) and closed-loop metrics
   (`control.closed_loop_step_us`) must be reported separately.
3. Any new latency claim must include:
   - metric ID
   - artifact/script source
   - hardware/runtime context
