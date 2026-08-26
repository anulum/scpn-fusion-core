<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Transport Polyglot Comparison -->

# Rust/PyO3 and NumPy cylindrical transport comparison

## Outcome

On the disclosed local machine, the NumPy/Rust median wall-time ratio was 72.9563 for this case. The final profiles differ by 3.330669e-16 keV.

The ratio is the observed result on the disclosed local machine. It is not a portable performance guarantee.

## Side-by-side timing

| Backend | Language | Build profile | Cold (s) | Warm P05 (s) | Warm median (s) | Warm P95 (s) | Samples |
|---|---|---|---:|---:|---:|---:|---:|
| NumPy | Python | CPython/NumPy | 0.004279457 | 0.003866177 | 0.004028503 | 0.004454075 | 31 |
| Rust/PyO3 | Rust via Python | release | 0.000149619 | 0.000044460 | 0.000055218 | 0.000067666 | 31 |

## Numerical checks

| Check | Result | Limit |
|---|---:|---:|
| Maximum Rust/NumPy profile difference | 3.330669e-16 keV | 2.000000e-14 keV |
| Analytic Bessel RMSE | 1.050048e-06 keV | 2.282931e-06 keV |
| Exact outer edge | True | `true` |
| Finite positive profiles | True | `true` |

## Timed scope

- Grid: 129 radial nodes, float64.
- Evolution: 10 steps at dt=0.001 s.
- Included: construct backend state, transfer inputs, execute 10 public steps, and transfer/read the final profile.
- Excluded: extension compilation and module import.
- Pair order: alternating NumPy-Rust and Rust-NumPy pairs.
- Discarded paired warmups: 10.

## Environment

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz
- Logical CPUs: 12
- Affinity: 12 CPUs (`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`)
- Governors: `{'powersave': 12}`
- Process count: 930
- Load average before: `[12.82275390625, 13.92041015625, 14.27685546875]`
- Load average after: `[12.82275390625, 13.92041015625, 14.27685546875]`
- Platform: Linux-7.0.0-28-generic-x86_64-with-glibc2.39
- Python / NumPy / SciPy: 3.12.3 / 1.26.4 / 1.15.3
- Rust: rustc 1.96.0 (ac68faa20 2026-05-25)
- Thread environment: `{}`

## Reproduce

```bash
VIRTUAL_ENV="$PWD/.venv" maturin develop --release \
  --manifest-path scpn-fusion-rs/crates/fusion-python/Cargo.toml
.venv/bin/python benchmarks/bench_transport_polyglot.py
```

The JSON companion retains every raw warm sample and the source hashes.
