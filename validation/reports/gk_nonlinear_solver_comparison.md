# Nonlinear GK Solver Comparison

- Benchmark: `gk_nonlinear_solver_comparison`
- Python: `3.12.3`
- Platform: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- JAX available: `True`

| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |
|---|---:|---:|---:|---:|---:|---:|
| krook | numpy | 0.087492 | True | 6.771828e-11 | 3.385914e-11 | 1.709088e-03 |
| krook | jax | 7.604175 | True | 6.569139e-11 | 3.284570e-11 | 1.732038e-03 |
| sugama | numpy | 0.181311 | True | 6.771828e-11 | 3.385914e-11 | 1.709094e-03 |
| sugama | jax | 4.465272 | True | 6.569137e-11 | 3.284568e-11 | 1.732044e-03 |

## Sugama Moment Residuals

| Moment | Max abs residual |
|---|---:|
| density | 1.610094e-22 |
| parallel_momentum | 5.918823e-23 |
| energy | 3.176374e-22 |
