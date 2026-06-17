# Performance Metric Taxonomy

## Context for performance reporting

SCPN Fusion Core publishes performance numbers from different execution layers.
This file gives the shared vocabulary needed so a reader can compare values
without mixing benchmark classes.

Facts to use this page correctly:

- A metric row is a contract, not a marketing summary.
- Every number in benchmark docs should reference one of these IDs and the
  associated artifact or benchmark script.
- The same metric can only be compared across runs when hardware, runtime, and
  execution mode are explicitly recorded in the same artifact.

This page defines the latency metric IDs used across README, public method notes, and
competitive analysis to avoid mixing kernel-only and end-to-end timings.

## Purpose and interpretation

This taxonomy is the public index for performance claims. It separates
control-kernel timing from closed-loop and hardware-in-the-loop timing so
benchmark comparisons remain reproducible and claim-ready across reporting layers.

## Evidence boundary for timing claims

Metric IDs here are the project-wide contract for benchmark comparisons. Any
public speed claim must name one or more IDs and the exact artifact or script
that produced the value.

This avoids mixing kernel-only throughput with full-loop transport or equilibrium
benchmarks and keeps control-timing and physics-timing claims auditable.

## Interpretation rule

The same metric ID can only be interpreted across runs when the recorded artifact
shows equal hardware, driver/runtime, and input deck context. This allows readers
to compare trend changes and regressions without attributing unrelated stack
changes to algorithm quality.

## Control Latency Metrics

| Metric ID | Scope | Includes | Excludes | Current headline value |
|---|---|---|---|---|
| `control.pid_kernel_step_us` | Rust PID kernel microbenchmark | Core reduced-order control kernel compute step in Rust | Python orchestration, plant integration, report/log overhead, Grad-Shafranov equilibrium solving | 0.52 us P50 in the committed Rust-extension campaign artifact |
| `control.closed_loop_step_us` | End-to-end closed-loop control step | Sensor preprocessing, controller call, actuator lag, and explicitly selected plant update mode | Claims that mix reduced-order surrogate timing with full Grad-Shafranov equilibrium timing | PID p95: 0.012 ms surrogate mode / 0.047 ms full mode in `validation/reports/scpn_end_to_end_latency.md` |
| `control.hil_loop_us` | Hardware-in-the-loop or simulated-HIL loop latency | HIL integration path as reported in `RESULTS.md`; simulated host ADC/DAC scaffold rows when explicitly labelled as simulated | Offline synthetic microbench shortcuts; physical HIL, FPGA, CODAC, or actuator-hardware claims unless a physical report names the device | Legacy collect-results row: 24.5 us P50 in `RESULTS.md`; current simulated 256-actuator scaffold: 167.140500 us P50 / 232.522000 us P95 in `validation/reports/scpn_end_to_end_latency.md` |

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
