# Performance Metric Taxonomy

This page defines the latency metric IDs used across README, public method notes, and
competitive analysis to avoid mixing kernel-only and end-to-end timings.

## Control Latency Metrics

| Metric ID | Scope | Includes | Excludes | Current headline value |
|---|---|---|---|---|
| `control.pid_kernel_step_us` | Rust PID kernel microbenchmark | Core reduced-order control kernel compute step in Rust | Python orchestration, plant integration, report/log overhead, Grad-Shafranov equilibrium solving | 0.52 us P50 in the committed Rust-extension campaign artifact |
| `control.closed_loop_step_us` | End-to-end closed-loop control step | Sensor preprocessing, controller call, actuator lag, and explicitly selected plant update mode | Claims that mix reduced-order surrogate timing with full Grad-Shafranov equilibrium timing | PID p95: 0.012 ms surrogate mode / 0.047 ms full mode in `validation/reports/scpn_end_to_end_latency.md` |
| `control.hil_loop_us` | Hardware-in-the-loop loop latency | HIL integration path as reported in `RESULTS.md` | Offline synthetic microbench shortcuts | 11.3 us P50 (latest HIL lane) |

## Rust Equilibrium Metrics

| Metric ID | Scope | Includes | Excludes | Current local value |
|---|---|---|---|---|
| `equilibrium.rust_gs_solve_us` | Full-order Rust Grad-Shafranov equilibrium component | `fusion-core` Picard/SOR or Picard/multigrid solve on the configured grid | Reduced-order flight simulator, Python orchestration, EFIT inverse reconstruction | 412.83 us SOR / 844.59 us multigrid on 33x33 grid |
| `equilibrium.rust_vacuum_field_us` | Full-order Rust vacuum-field component | Vacuum flux from six ITER-like coils on the configured grid | Plasma current iteration and inverse reconstruction | 139.55 us on 33x33 grid / 488.86 us on 65x65 grid |

## Clock Separation

- **Fast control clock (1-10 kHz):** controller tick (`control.pid_kernel_step_us`, `control.closed_loop_step_us`).
- **Slow physics clock (10-100 Hz):** equilibrium/transport state updates and inverse solves.
- **Surrogate bridge:** reduced-order/neural surrogates can approximate the slow lane at higher update rates; they are reported separately from high-fidelity solves.
- **Cross-fidelity ratios:** a reduced-order Rust surrogate step divided by a
  Python Grad-Shafranov step is a mixed-fidelity throughput ratio, not a
  same-work Rust-versus-Python speedup.

## Rules

1. Never compare two numbers without stating the metric ID.
2. Kernel metrics (`control.pid_kernel_step_us`) and closed-loop metrics
   (`control.closed_loop_step_us`) must be reported separately.
3. Any new latency claim must include:
   - metric ID
   - artifact/script source
   - hardware/runtime context
