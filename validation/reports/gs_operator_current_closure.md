# Grad-Shafranov Operator Current Closure Benchmark

Manufactured contract: `psi(R, Z) = c Z^2`, `Delta*psi = 2c`, `J_phi = -2c / (mu0 R)`.

## Local machine

- `platform`: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- `system`: `Linux`
- `release`: `6.17.0-29-generic`
- `machine`: `x86_64`
- `processor`: `x86_64`
- `python`: `3.12.3`
- `numpy`: `2.2.6`
- `cpu_count`: `12`

## Results

| Grid | coeff | elapsed s | max Delta* abs error | max J rel error | total current rel error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17x19 | -0.25 | 2.400550e-04 | 5.662137e-15 | 1.138333e-14 | 1.523927e-16 |
| 33x35 | -0.125 | 1.734650e-04 | 7.827072e-15 | 3.140702e-14 | 4.276956e-16 |

Pass threshold: `{'delta_star_max_abs_error': 1e-10, 'current_density_max_relative_error': 1e-12, 'total_current_relative_error': 1e-12}`.
Overall status: `PASS`.
