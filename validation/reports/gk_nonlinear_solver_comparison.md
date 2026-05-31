# Nonlinear GK Solver Comparison

- Benchmark: `gk_nonlinear_solver_comparison`
- Python: `3.12.3`
- Platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- JAX available: `True`
- Diagnostic contract: adiabatic-electron cases report `chi_e = 0.5 * chi_i`; kinetic-electron cases compute `chi_e` from the electron distribution moment.

| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |
|---|---:|---:|---:|---:|---:|---:|
| krook | numpy | 0.115710 | True | 6.771828e-11 | 3.385914e-11 | 1.709088e-03 |
| krook | jax | 11.773385 | True | 6.771828e-11 | 3.385914e-11 | 1.732038e-03 |
| sugama | numpy | 0.152750 | True | 6.771828e-11 | 3.385914e-11 | 1.709094e-03 |
| sugama | jax | 4.604102 | True | 6.771828e-11 | 3.385914e-11 | 1.732044e-03 |
| sugama_electromagnetic_kinetic | numpy | 0.215034 | True | 6.137781e-07 | -6.506029e-07 | 3.146440e-03 |
| sugama_electromagnetic_kinetic | jax | 4.285303 | True | 6.138569e-07 | -6.507130e-07 | 3.170644e-03 |

## Sugama Moment Residuals

| Moment | Max abs residual |
|---|---:|
| density | 1.674096e-22 |
| parallel_momentum | 6.277859e-23 |
| energy | 3.220188e-22 |

## Nonlinear E x B Invariant Contract

The diagnostic contracts the dealiased nonlinear bracket with the 5D distribution. In an undriven collisionless nonlinear step this term must not inject free energy, and all bracket energy outside the 2/3 spectral mask must remain zero.

| Case | Free-energy production | Relative production | High-k max abs | Pass |
|---|---:|---:|---:|---:|
| krook | 2.611359e-12 | 1.512070e-02 | 0.000000e+00 | True |
| sugama | 2.611359e-12 | 1.512070e-02 | 0.000000e+00 | True |
| sugama_electromagnetic_kinetic | 2.212299e-11 | 2.642871e-02 | 0.000000e+00 | True |

## Compact Electromagnetic Closure Diagnostics

These residuals verify the native compact A_parallel and B_parallel closures against their algebraic field solves. They are not full Faraday/displacement-current Vlasov-Maxwell parity evidence.

| Residual | Value |
|---|---:|
| Ampere A_parallel L_inf | 0.000000e+00 |
| Pressure-balance B_parallel L_inf | 0.000000e+00 |
| Compact closure finite | True |
| Compact closure passes | True |
| Full Faraday/displacement-current supported | False |
| Full Vlasov-Maxwell parity ready | False |
