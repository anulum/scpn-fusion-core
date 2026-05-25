<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - Persistent GPU SOR Benchmark -->

# Persistent GPU SOR Benchmark

- Host: UpCloud `fi-hel2`, NVIDIA L4, driver `595.71.05`
- Timing: upload once, warm-up, repeated synchronised `solve()`, one final `download()` measurement.
- Iterations per solve: `20`
- Omega: `1.3`

| Grid | Runs | Persistent solve median ms | Persistent solve P95 ms | Download ms |
|---|---:|---:|---:|---:|
| `129x129` | 100 | 0.760128 | 2.940710 | 0.053754 |
| `257x257` | 100 | 0.764012 | 2.897592 | 0.165949 |
| `513x513` | 50 | 0.861687 | 3.009115 | 0.343303 |
