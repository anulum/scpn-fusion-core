# Nonlinear GK Solver Comparison

- Benchmark: `gk_nonlinear_solver_comparison`
- Python: `3.12.3`
- Platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- JAX available: `True`
- Diagnostic contract: adiabatic-electron cases report `chi_e = 0.5 * chi_i`; kinetic-electron cases compute `chi_e` from the electron distribution moment.

| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |
|---|---:|---:|---:|---:|---:|---:|
| krook | numpy | 0.038414 | True | 6.771828e-11 | 3.385914e-11 | 1.709088e-03 |
| krook | jax | 5.568772 | True | 6.546531e-11 | 3.273266e-11 | 1.732038e-03 |
| sugama | numpy | 0.053960 | True | 6.771828e-11 | 3.385914e-11 | 1.709094e-03 |
| sugama | jax | 2.831624 | True | 6.546528e-11 | 3.273264e-11 | 1.732044e-03 |
| sugama_electromagnetic_kinetic | numpy | 0.090440 | True | 6.138569e-07 | -6.507130e-07 | 3.146493e-03 |
| sugama_electromagnetic_kinetic | jax | 2.719143 | True | 6.111619e-07 | -6.341059e-07 | 3.185728e-03 |

## Sugama Moment Residuals

| Moment | Max abs residual |
|---|---:|
| density | 1.610094e-22 |
| parallel_momentum | 5.918823e-23 |
| energy | 3.176374e-22 |
