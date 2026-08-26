<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — JAX Transport Comparison -->

# JAX and NumPy cylindrical transport comparison

## Outcome

On the disclosed local machine, the NumPy/JAX median wall-time ratio was 0.1376 for this case. The final profiles differ by 0.000000e+00 keV.
For this workload, the observed JAX median was 7.2690 times the NumPy median, so automatic dispatch retains NumPy.

The ratio is the observed result on the disclosed local machine. It is not a portable performance guarantee.

## Side-by-side timing

| Backend | Language | Build profile | Cold (s) | Warm P05 (s) | Warm median (s) | Warm P95 (s) | Samples |
|---|---|---|---:|---:|---:|---:|---:|
| NumPy | Python | CPython/NumPy | 0.004188955 | 0.004255571 | 0.004644339 | 0.005839449 | 31 |
| JAX | JAX/XLA via Python | JAX float64/gpu | 0.943007164 | 0.030790580 | 0.033759743 | 0.038697059 | 31 |

## Numerical checks

| Check | Result | Limit |
|---|---:|---:|
| Maximum JAX/NumPy profile difference | 0.000000e+00 keV | 2.000000e-14 keV |
| Analytic Bessel RMSE | 1.050048e-06 keV | 2.282931e-06 keV |
| Source-gradient relative error | 5.925595e-11 | 1.000000e-02 |
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

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz
- Logical CPUs: 12
- Affinity: 12 CPUs (`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`)
- Governors: `{'powersave': 12}`
- Process count: 933
- Load average before: `[9.70751953125, 9.28955078125, 10.9189453125]`
- Load average after: `[10.37158203125, 9.4345703125, 10.95703125]`
- Platform: Linux-7.0.0-28-generic-x86_64-with-glibc2.39
- Python / NumPy / SciPy: 3.12.3 / 1.26.4 / 1.15.3
- JAX / jaxlib: 0.7.1 / 0.7.1
- JAX backend and devices: gpu / `[{'platform': 'gpu', 'device_kind': 'NVIDIA GeForce GTX 1060 6GB', 'id': 0}]`
- Thread environment: `{}`

## Reproduce

```bash
.venv/bin/python benchmarks/bench_transport_jax.py
```

The JSON companion retains every raw warm sample, gradient values, and source hashes.
