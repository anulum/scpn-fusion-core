<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — JAX Transport Comparison -->

# JAX and NumPy cylindrical transport comparison

## Outcome

On the disclosed local machine, the NumPy/JAX median wall-time ratio was 4.5990 for this case. The final profiles differ by 1.443290e-15 keV.
For this workload, the observed JAX median was 0.2174 times the NumPy median, so automatic dispatch retains NumPy.

The ratio is the observed result on the disclosed local machine. It is not a portable performance guarantee.

## Side-by-side timing

| Backend | Language | Build profile | Cold (s) | Warm P05 (s) | Warm median (s) | Warm P95 (s) | Samples |
|---|---|---|---:|---:|---:|---:|---:|
| NumPy | Python | CPython/NumPy | 0.005586122 | 0.005578871 | 0.005651208 | 0.006487462 | 31 |
| JAX | JAX/XLA via Python | JAX float64/cpu | 0.452520326 | 0.001078656 | 0.001228788 | 0.001561921 | 31 |

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

- CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
- Logical CPUs: 4
- Affinity: 4 CPUs (`[0, 1, 2, 3]`)
- Governors: `{'performance': 4}`
- Process count: 163
- Load average before: `[0.853515625, 0.30322265625, 0.11083984375]`
- Load average after: `[0.853515625, 0.30322265625, 0.11083984375]`
- Platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
- Python / NumPy / SciPy: 3.12.15 / 1.26.4 / 1.17.1
- JAX / jaxlib: 0.7.1 / 0.7.1
- JAX backend and devices: cpu / `[{'platform': 'cpu', 'device_kind': 'cpu', 'id': 0}]`
- Thread environment: `{}`

## Reproduce

```bash
.venv/bin/python benchmarks/bench_transport_jax.py
```

The JSON companion retains every raw warm sample, gradient values, and source hashes.
