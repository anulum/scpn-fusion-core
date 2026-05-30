# Grad-Shafranov Operator Current Closure Benchmark

Benchmark ID: `gs_operator_current_closure`
Schema: `gs-operator-current-closure.v2`
Scope: `native_grad_shafranov_operator_current_closure`
Solver mode: `manufactured_flux_operator_current_closure`
Gates passed: `3/3`

Manufactured contracts: `psi(R, Z) = a R^4 + b Z^2 + c R^2 Z^2`, `Delta*psi = 8aR^2 + 2b + 2cR^2`, and `J_phi = -Delta*psi / (mu0 R)`.

For radial-quartic terms the centered second-order stencil has exact discrete truncation `-2a dR^2`; the benchmark reports both the discrete-contract error and the analytic truncation magnitude.

## Local machine

- `platform`: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- `system`: `Linux`
- `release`: `6.17.0-29-generic`
- `machine`: `x86_64`
- `processor`: `x86_64`
- `python`: `3.12.3`
- `numpy`: `1.26.4`
- `cpu_count`: `12`

## Results

| Case | Grid | a | b | c | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vertical_quadratic | 17x19 | 0 | -0.25 | 0 | 3.375161e-04 | 5.662137e-15 | 5.662137e-15 | 1.138333e-14 | 1.523927e-16 |
| radial_quartic_17 | 17x19 | 0.03125 | -0.125 | 0 | 2.874980e-04 | 3.330669e-14 | 9.765625e-04 | 4.338971e-14 | 0.000000e+00 |
| radial_quartic_33 | 33x35 | 0.03125 | -0.125 | 0 | 3.539780e-04 | 2.380318e-13 | 2.441406e-04 | 1.901481e-13 | 0.000000e+00 |
| radial_quartic_65 | 65x67 | 0.03125 | -0.125 | 0 | 5.379200e-04 | 1.078249e-12 | 6.103516e-05 | 3.028695e-12 | 8.311853e-16 |
| mixed_solovev | 29x31 | 0 | -0.125 | 0.05 | 2.412869e-04 | 5.545564e-14 | 5.545564e-14 | 1.119700e-12 | 9.237473e-16 |

## Radial-quartic convergence

The radial-quartic analytic error measures the expected centered-stencil truncation against the continuous `Delta*` operator. The measured order from the two finest radial grids is `2.000000`.

The radial-quartic total-current closure stability gate reports the worst relative current error across the refinement sequence: `8.311853e-16`.

Pass threshold: `{'delta_star_max_abs_error': 1e-10, 'current_density_max_relative_error': 1e-11, 'total_current_relative_error': 1e-12, 'radial_total_current_relative_error_max': 1e-12}`.
Overall status: `PASS`.
