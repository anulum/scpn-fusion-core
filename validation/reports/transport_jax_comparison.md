<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — JAX Transport Comparison -->

# JAX and NumPy cylindrical transport comparison

## Outcome

On the disclosed local machine, the NumPy/JAX median wall-time ratio was 4.5366 for this case. The final profiles differ by 1.443290e-15 keV.
For this workload, the observed JAX median was 0.2204 times the NumPy median, so automatic dispatch retains NumPy.

The ratio is the observed result on the disclosed local machine. It is not a portable performance guarantee.

## Side-by-side timing

| Backend | Language | Build profile | Cold (s) | Warm P05 (s) | Warm median (s) | Warm P95 (s) | Samples |
|---|---|---|---:|---:|---:|---:|---:|
| NumPy | Python | CPython/NumPy | 0.004049168 | 0.003960042 | 0.004008580 | 0.004071496 | 31 |
| JAX | JAX/XLA via Python | JAX float64/cpu | 0.382114836 | 0.000754504 | 0.000883605 | 0.001044889 | 31 |

## Numerical checks

| Check | Result | Limit |
|---|---:|---:|
| Maximum JAX/NumPy profile difference | 1.443290e-15 keV | 2.000000e-14 keV |
| Analytic Bessel RMSE | 1.050048e-06 keV | 2.282931e-06 keV |
| Source-gradient relative error | 5.925568e-11 | 1.000000e-02 |
| Exact outer edge | True | `true` |
| Finite positive profile | True | `true` |

## Timed scope

- Grid: 129 radial nodes, float64.
- Evolution: 10 steps at dt=0.001 s.
- Included: array construction, transfers, solver calls, JAX synchronization, and readback.
- Excluded: module import.
- Pair order: alternating NumPy-JAX and JAX-NumPy pairs.
- Discarded paired warmups: 10.

## Environment

- CPU: INTEL(R) XEON(R) PLATINUM 8573C
- Logical CPUs: 4
- Affinity: 4 CPUs (`[0, 1, 2, 3]`)
- Governors: `{'schedutil': 4}`
- Process count: 162
- Load average before: `[0.8583984375, 0.31689453125, 0.11767578125]`
- Load average after: `[0.8583984375, 0.31689453125, 0.11767578125]`
- Platform: Linux-7.0.0-1019-azure-x86_64-with-glibc2.43
- Python / NumPy / SciPy: 3.12.16 / 2.5.4 / 1.17.1
- JAX / jaxlib: 0.7.1 / 0.7.1
- JAX backend and devices: cpu / `[{'platform': 'cpu', 'device_kind': 'cpu', 'id': 0}]`
- Thread environment: `{}`

## Reproduce

```bash
.venv/bin/python benchmarks/bench_transport_jax.py
```

The JSON companion retains every raw warm sample, gradient values, and source hashes.
