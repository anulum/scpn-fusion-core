# Grad-Shafranov Operator Current Closure Benchmark

Manufactured contracts: `psi(R, Z) = a R^4 + b Z^2`, `Delta*psi = 8aR^2 + 2b`, and `J_phi = -Delta*psi / (mu0 R)`.

For radial-quartic terms the centered second-order stencil has exact discrete truncation `-2a dR^2`; the benchmark reports both the discrete-contract error and the analytic truncation magnitude.

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

| Case | Grid | a | b | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vertical_quadratic | 17x19 | 0 | -0.25 | 2.063400e-04 | 5.662137e-15 | 5.662137e-15 | 1.138333e-14 | 1.523927e-16 |
| radial_quartic | 33x35 | 0.03125 | -0.125 | 1.800380e-04 | 2.380318e-13 | 2.441406e-04 | 1.901481e-13 | 0.000000e+00 |

Pass threshold: `{'delta_star_max_abs_error': 1e-10, 'current_density_max_relative_error': 1e-12, 'total_current_relative_error': 1e-12}`.
Overall status: `PASS`.
