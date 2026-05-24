# Nonlinear GK Solver Comparison

- Benchmark: `gk_nonlinear_solver_comparison`
- Python: `3.12.3`
- Platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- JAX available: `True`

| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |
|---|---:|---:|---:|---:|---:|---:|
| krook | numpy | 0.046570 | True | 6.771828e-11 | 3.385914e-11 | 1.709088e-03 |
| krook | jax | 3.268715 | True | 6.569139e-11 | 3.284570e-11 | 1.732038e-03 |
| sugama | numpy | 0.064599 | True | 6.771828e-11 | 3.385914e-11 | 1.709094e-03 |
| sugama | jax | 1.683652 | True | 6.569137e-11 | 3.284568e-11 | 1.732044e-03 |
| sugama_electromagnetic_kinetic | numpy | 0.121162 | True | 6.138569e-07 | 3.069285e-07 | 3.146493e-03 |
| sugama_electromagnetic_kinetic | jax | 1.519801 | True | 6.130765e-07 | 3.065383e-07 | 3.174583e-03 |

## Sugama Moment Residuals

| Moment | Max abs residual |
|---|---:|
| density | 1.610094e-22 |
| parallel_momentum | 5.918823e-23 |
| energy | 3.176374e-22 |
