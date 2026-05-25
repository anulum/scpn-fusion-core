# Grad-Shafranov Operator Current Closure Benchmark

Manufactured contracts: `psi(R, Z) = a R^4 + b Z^2 + c R^2 Z^2`, `Delta*psi = 8aR^2 + 2b + 2cR^2`, and `J_phi = -Delta*psi / (mu0 R)`.

For radial-quartic terms the centered second-order stencil has exact discrete truncation `-2a dR^2`; the benchmark reports both the discrete-contract error and the analytic truncation magnitude.

## Local machine

- `platform`: `Linux-6.8.0-117-generic-x86_64-with-glibc2.39`
- `system`: `Linux`
- `release`: `6.8.0-117-generic`
- `machine`: `x86_64`
- `processor`: `x86_64`
- `python`: `3.12.3`
- `numpy`: `2.4.6`
- `cpu_count`: `8`

## Results

| Case | Grid | a | b | c | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vertical_quadratic | 17x19 | 0 | -0.25 | 0 | 1.649230e-04 | 5.662137e-15 | 5.662137e-15 | 1.138333e-14 | 1.523927e-16 |
| radial_quartic_17 | 17x19 | 0.03125 | -0.125 | 0 | 9.521900e-05 | 3.330669e-14 | 9.765625e-04 | 4.338971e-14 | 0.000000e+00 |
| radial_quartic_33 | 33x35 | 0.03125 | -0.125 | 0 | 1.089770e-04 | 2.380318e-13 | 2.441406e-04 | 1.901481e-13 | 0.000000e+00 |
| radial_quartic_65 | 65x67 | 0.03125 | -0.125 | 0 | 1.621490e-04 | 1.027178e-12 | 6.103516e-05 | 3.028695e-12 | 8.311853e-16 |
| mixed_solovev | 29x31 | 0 | -0.125 | 0.05 | 8.919000e-05 | 5.545564e-14 | 5.545564e-14 | 1.119700e-12 | 9.237473e-16 |

## Radial-quartic convergence

The radial-quartic analytic error measures the expected centered-stencil truncation against the continuous `Delta*` operator. The measured order from the two finest radial grids is `2.000000`.

The radial-quartic total-current closure stability gate reports the worst relative current error across the refinement sequence: `8.311853e-16`.

Pass threshold: `{'delta_star_max_abs_error': 1e-10, 'current_density_max_relative_error': 1e-11, 'total_current_relative_error': 1e-12, 'radial_total_current_relative_error_max': 1e-12}`.
Overall status: `PASS`.
