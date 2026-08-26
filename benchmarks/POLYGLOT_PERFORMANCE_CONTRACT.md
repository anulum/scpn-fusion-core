<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Public Polyglot Performance Contract -->

# Polyglot performance comparison contract

## Getting started

Every promoted polyglot backend must publish a side-by-side comparison against
the canonical implementation. A timing row is valid only when both backends
pass the same numerical acceptance gates on the same input deck.

Use the machine-readable schema at
[`validation/polyglot/performance_comparison.schema.json`](../validation/polyglot/performance_comparison.schema.json)
for new reports.

## Required comparison scope

A report must record:

- identical input data, boundary conditions, precision, step count, and output
  metrics;
- the complete timed boundary, including construction, preprocessing, FFI,
  transfers, execution, and result readback;
- language, implementation, package/toolchain version, and build profile for
  each backend;
- one cold sample, discarded warmups, paired raw warm samples, median, P05, and
  P95;
- CPU, operating system, affinity, governor, thread settings, process count,
  and load average before and after the cohort;
- every correctness threshold, failure, and exclusion.

Alternate backend order across paired warm samples. Do not compare a
reduced-order kernel with a full-physics runtime under one speedup label.

## Usage

The cylindrical transport reference implementation compares release Rust via
PyO3 with the canonical NumPy Crank-Nicolson path:

```bash
VIRTUAL_ENV="$PWD/.venv" maturin develop --release \
  --manifest-path scpn-fusion-rs/crates/fusion-python/Cargo.toml
.venv/bin/python benchmarks/bench_transport_polyglot.py
```

The command writes:

- `validation/reports/transport_polyglot_comparison.json`
- `validation/reports/transport_polyglot_comparison.md`

The reconciled differentiable transport rollout compares JAX/XLA with its
canonical NumPy tier:

```bash
.venv/bin/python benchmarks/bench_transport_jax.py
```

The command writes:

- `validation/reports/transport_jax_comparison.json`
- `validation/reports/transport_jax_comparison.md`

For small device-backed workloads, transfer and synchronization costs can make
an accelerator slower than NumPy. The automatic runtime order must follow the
retained same-scope result; callers can still request an available reconciled
backend explicitly when they require its differentiability or device semantics.

## Interpretation

The report may state the observed ratio on the disclosed machine. It must not
convert a loaded, shared, unpinned, or otherwise non-isolated cohort into a
portable performance guarantee. Readers can rerun the command on their own
hardware and compare the raw distributions.

Public reports contain methodology and measured results only. Project strategy,
competitive targets, unpublished gaps, and backlog ordering remain private.
