<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core - CUDA-JAX Nonlinear GK Benchmark -->

# Nonlinear GK Solver Comparison

- Benchmark: `gk_nonlinear_solver_comparison`
- Python: `3.12.3`
- Platform: `Linux-6.8.0-117-generic-x86_64-with-glibc2.39`
- JAX available: `True`
- Diagnostic contract: adiabatic-electron cases report `chi_e = 0.5 * chi_i`; kinetic-electron cases compute `chi_e` from the electron distribution moment.

| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |
|---|---:|---:|---:|---:|---:|---:|
| krook | numpy | 0.020947 | True | 6.771828e-11 | 3.385914e-11 | 1.709088e-03 |
| krook | jax | 4.051154 | True | 6.771828e-11 | 3.385914e-11 | 1.732038e-03 |
| sugama | numpy | 0.022762 | True | 6.771828e-11 | 3.385914e-11 | 1.709094e-03 |
| sugama | jax | 1.648905 | True | 6.771828e-11 | 3.385914e-11 | 1.732044e-03 |
| sugama_electromagnetic_kinetic | numpy | 0.045141 | True | 6.138569e-07 | -6.507130e-07 | 3.146493e-03 |
| sugama_electromagnetic_kinetic | jax | 1.662950 | True | 6.138569e-07 | -6.507130e-07 | 3.170644e-03 |

## Sugama Moment Residuals

| Moment | Max abs residual |
|---|---:|
| density | 1.610094e-22 |
| parallel_momentum | 5.918823e-23 |
| energy | 3.176374e-22 |

## Nonlinear E x B Invariant Contract

The diagnostic contracts the dealiased nonlinear bracket with the 5D distribution. In an undriven collisionless nonlinear step this term must not inject free energy, and all bracket energy outside the 2/3 spectral mask must remain zero.

| Case | Free-energy production | Relative production | High-k max abs | Pass |
|---|---:|---:|---:|---:|
| krook | 2.611359e-12 | 1.512070e-02 | 0.000000e+00 | True |
| sugama | 2.611359e-12 | 1.512070e-02 | 0.000000e+00 | True |
| sugama_electromagnetic_kinetic | 2.212299e-11 | 2.642871e-02 | 0.000000e+00 | True |
