# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Analytical Contract

# Field-Reversed Configuration Rigid-Rotor Analytical Contract

This page documents the accepted FRC equilibrium surface currently implemented
in SCPN Fusion Core. The implemented contract is the Steinhauer no-rotation
analytical axial-field profile. It is a full numerical implementation of that
contract, not a reduced-order surrogate for it. It is not a claim that the full
rotating rigid-rotor boundary-value problem is complete.

## Implemented equation

For radial coordinate `r`, external axial field `B_ext`, separatrix radius
`R_s`, and layer thickness `delta`, the accepted analytical field is:

```text
B_z(r) = -B_ext * tanh((r^2 - R_s^2) / (2 R_s delta))
```

The solver requires radial samples on both sides of `R_s`, evaluates the
cylindrical flux from the closed-form primitive of `r * B_z`, reports finite
pressure and energy diagnostics, locates the magnetic null, validates that the
interpolated zero-crossing matches the configured separatrix radius, verifies
that the axial field reverses sign across the separatrix, exposes toroidal
diamagnetic current density from the closed-form derivative of the analytical
field, and records the normalised radial ideal-MHD force-balance residual:

```text
psi(r) = -B_ext R_s delta * (log(cosh(a(r))) - log(cosh(a(0))))
a(r) = (r^2 - R_s^2) / (2 R_s delta)
```

The flux derivative residual compares the analytical primitive against an
independent second-order finite-difference derivative on the active grid:

```text
F_r = dpsi/dr - r B_z
```

```text
R_r = dp/dr - (J x B)_r
```

For this no-rotation axial-field slice, the implemented current density is:

```text
J_theta = B_ext * (1 - tanh(a)^2) * r / (mu_0 R_s delta)
a = (r^2 - R_s^2) / (2 R_s delta)
```

The Ampere closure residual compares that analytical current against the
second-order finite-difference derivative on the active grid:

```text
A_r = mu_0 J_theta + dB_z/dr
```

The quality-of-equilibrium parameter is computed from the local thermal-ion
gyroradius profile, not from the separatrix-layer shortcut:

```text
s = (1 / R_s) * integral_0^R_s r / rho_i(r) dr
rho_i(r) = sqrt(2 m_i T_i) / (e * abs(B_z(r)))
```

The implementation evaluates the equivalent finite integrand
`r * e * abs(B_z(r)) / sqrt(2 m_i T_i)` and inserts an interpolated
separatrix endpoint when the grid does not land exactly on `R_s`.

## Production boundary

Accepted:

- Python NumPy reference implementation in `scpn_fusion.core.frc_rigid_rotor`.
- Rust implementation in `fusion-physics::frc`.
- PyO3 exposure through `scpn_fusion_rs.py_solve_frc_equilibrium` when the
  Rust extension is built.
- Cross-surface parity tests for the exposed Python and Rust/PyO3 paths.
- Explicit `J_theta` current-density and Ampere closure residual diagnostics
  for the accepted axial-field slice, with the residual kept as an independent
  grid diagnostic instead of a self-cancelling derivative reuse.
- Explicit separatrix target, interpolated radius error, and field-reversal
  diagnostics so validation compares against the configured `R_s` rather than
  only against a recomputed state value.
- Closed-form cylindrical flux primitive and a finite-grid derivative closure
  gate so `psi` is mathematically tied to the accepted axial-field equation
  instead of only being produced by numerical quadrature.
- Finite-grid convergence diagnostics for the implemented no-rotation scalar
  invariants: null radius, Eq. 27 `s`, energy per metre, and pressure-balance
  ratio.
- Benchmark artifact generation in `validation/reports/frc_rigid_rotor_benchmark.json`.

Fail-closed:

- Nonzero `theta_dot` rotating rigid-rotor cases.
- Full FRC rotating BVP.
- Full FRC kinetic or transport evolution.
- Go, Julia, and Lean parity rows until those languages expose equivalent FRC
  solver logic.

## Reproducibility

Run the Python/Rust benchmark and regenerate the tracked JSON report:

```bash
PYTHONPATH=src python benchmarks/bench_frc_rigid_rotor.py
```

Run focused Python tests:

```bash
PYTHONPATH=src python -m pytest tests/test_frc_rigid_rotor.py tests/test_frc_rigid_rotor_rust_parity.py
```

Run focused Rust tests:

```bash
cd scpn-fusion-rs
cargo test -p fusion-physics frc
```

Run Criterion benchmarks:

```bash
cd scpn-fusion-rs
cargo bench -p fusion-physics --bench frc_rigid_rotor_bench
```

## Evidence interpretation

The benchmark report compares scalar diagnostics and weighted numerical
checksums for `B_z`, `J_theta`, `psi`, pressure, and the Eq. 27 `s` value. It
also records separatrix radius error, field reversal, flux derivative residual,
Ampere residual, and peak-current diagnostics plus finite-grid convergence
against the finest tracked radial grid for the scalar invariants, separatrix
error, flux derivative residual, and the independent Ampere residual accepted
in this contract. Blocked or
not-applicable rows are recorded instead of promoting missing surfaces to
parity evidence. This is intentional: the accepted claim is limited to the
explicit no-rotation analytical FRC contract.
