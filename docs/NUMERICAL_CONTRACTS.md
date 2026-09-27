<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Numerical input and output contracts

## Shared Grad–Shafranov physical case

Python, Julia, Go, Rust and Lean use a conforming TOML 1.0 parser and the same
physical case. The only root is `grad_shafranov`; its thirteen required fields
are `R_min`, `R_max`, `Z_min`, `Z_max`, `NR`, `NZ`, `Ip_target`, `mu0`,
`n_picard`, `n_jacobi`, `alpha`, `omega_j` and `beta_mix`. Unknown, missing
and duplicate definitions fail. Valid quoted, dotted and inline table syntax
has the same meaning as a normal table. There is no default permeability.

| Fields | Units and admitted values |
|---|---|
| `R_min`, `R_max` | metres; finite, `0 < R_min < R_max` |
| `Z_min`, `Z_max` | metres; finite, `Z_min < Z_max` |
| `Ip_target` | amperes; finite, including negative and zero current |
| `mu0` | H/m; finite and positive |
| `NR`, `NZ` | TOML integers, from 3 through 1025 |
| `n_picard`, `n_jacobi` | TOML integers, from 1 through 10000 |
| `alpha` | finite, `0 < alpha <= 1` |
| `omega_j` | finite, `0 < omega_j < 2` |
| `beta_mix` | finite, `0 <= beta_mix <= 1` |

Integers use the signed 64-bit TOML domain. Real fields also accept integers
that convert exactly to binary64. Booleans, strings, integral floats for
counts, nonfinite numbers and inexact integer conversions fail. Before
allocation, `NR * NZ * n_picard * n_jacobi` must be at most 100000000.
Generated coordinates must be finite and strictly increasing with positive
representable spacing. These are software admission limits, not accuracy guarantees.

The same validation applies to public direct case construction and solves.
Explicit caller conversion into a statically typed floating field is the
caller's responsibility. Invalid CLI input exits nonzero, writes diagnostics
to stderr and emits no flux CSV. A numerical failure also fails; it never
produces successful nonfinite CSV. Valid output has `NZ` rows and `NR` columns.

| Language | Public loader / solver |
|---|---|
| Python | `physical_case.case_from_toml`, `GradShafranovCase`, `jax_gs_solver.gs_solve_np` |
| Go | `gssolver.CaseFromTOML`, `gssolver.Case`, `gssolver.Solve` |
| Julia | `case_from_toml`, `GradShafranovCase`, `solve_grad_shafranov` |
| Rust | `fusion_polyglot::load_case` / `parse_case`, `GradShafranovCase`, `solve_grad_shafranov` |
| Lean | `SCPNFusionSolvers.caseFromToml`, `GradShafranovCase`, `solveGradShafranov` |

The reference deck is `validation/polyglot/gs_picard_reference.toml`. Shared
accepted and refused decks with checksums live in `validation/polyglot/case_contract/`.
Use the native `gs_picard_csv` command in each language project. The comparison
driver below invokes all five actual implementations and records parity and timing.

## NumPy and Rust magnetic sensing / full multigrid solve

These contracts apply to direct functions, registered NumPy/Rust dispatch
providers and the optional Rust compatibility wrappers:

- `synthetic_sensors.measure_magnetics` / `scpn_fusion_rs.measure_magnetics`;
- `multigrid_solve.multigrid_solve` / `scpn_fusion_rs.multigrid_vcycle`.

The Rust `multigrid_vcycle` binding performs a **full solve**. The Python
helper of that name performs one cycle and has a separate internal interface.

Each array must already be a rank-two NumPy ndarray with native-endian
`float64` dtype and exact shape `(nz, nr)`. Lists, float32/integer/complex/object
arrays and byte-swapped float64 arrays raise `TypeError`; wrong rank, shape or
nonfinite data raise `ValueError`. No public function or provider converts an
invalid array automatically. All logical cells, including boundary cells,
are checked. Array subclasses are read through base ndarray storage.
Readonly, Fortran, transposed, unaligned and valid positive/negative/zero-stride
views are accepted without input mutation.

Migration is explicit at the caller:

```python
psi = np.array(existing_data, dtype=np.float64, order="C", copy=True)
```

Counts accept builtin and NumPy signed/unsigned integer scalars, excluding
booleans and integral floats. Real scalars accept builtin floats, NumPy floating
scalars of up to 64 bits, and exactly representable integers. Strings, Decimal,
complex values, zero-dimensional arrays and wider NumPy floating types raise
`TypeError`. Inexact, nonfinite or out-of-domain values raise `ValueError`.
Builtin and NumPy scalar subclasses are read through base scalar storage;
caller conversion hooks and overridden dtype attributes are not used.
Counts and binary64 byte products must fit signed native address space before
copies or mesh construction.

Sensor dimensions are at least two on each axis. Bounds may include nonpositive
radius, but must be increasing with finite positive nominal spacing. The
twenty endpoint-inclusive wall probes retain truncation toward zero and clamped
boundary behavior. The free function is deterministic and adds no noise.
`SensorSuite` retains its explicit application conversion and noise layer.

Full multigrid dimensions are at least three; radius is strictly positive.
`tol > 0` and `max_cycles >= 1`, with defaults `1e-6` and `500`. NumPy-only
controls require `1 <= omega < 2`, nonnegative integer `pre_smooth` and
`post_smooth`, and integer `min_grid >= 3`; they are checked even on an already
converged field. Actual coordinates and all active fine/coarse stencil
denominators, reciprocals and coefficients must be representable. Both public
backends validate both mesh-generation paths. Accepted tiny positive radii
are used directly in the GS operator; there is no `1e-10` radius floor.
Python FusionKernel helper callers explicitly retain their legacy `1e-10`
radius policy independently at each coarse level. This is separate from the
strict public full solve. Internal native fixed-cycle kernel callers may retain
`tol=0` at their own
typed interface; this does not alter the public Python contract.

| Outcome | Public Python result |
|---|---|
| Wrong delivered kind/dtype | `TypeError` |
| Invalid domain, shape, geometry or address-space product | `ValueError` |
| Accepted inputs cause nonfinite numerical arithmetic | `RuntimeError` |
| Valid inputs cannot allocate ordinary working storage | `MemoryError` |
| Finite cycle budget exhausted | finite tuple with `converged=False` |
| Optional extension unavailable | existing `None`/`ImportError` availability path |

Input or numerical errors propagate; they do not select another backend.
Sensing returns shape `(20,)`. Full multigrid returns
`(psi, residual, n_cycles, converged)`, with native float64 C-contiguous `psi`
of shape `(nz, nr)`, finite nonnegative Python float residual, Python int cycle
count and Python bool status. Inputs and boundary values are preserved.
Results own independent storage, survive input deletion and do not alias
inputs or other calls. Native capsule ownership is valid even if NumPy's
`OWNDATA` flag is false.

Convergence means the actual final interior GS residual is strictly less than
`tol`. An initially converged field returns a fresh copy and zero cycles;
equality is nonconvergence. Finite nonconvergence is separate from numerical
failure. Algorithms need not follow identical cycle histories.

## Reproduce the comparison

```bash
PYTHONPATH=src python benchmarks/polyglot_gs_solver_comparison.py
```

Recorded results: [JSON](../validation/reports/polyglot_gs_solver_comparison.json)
and [Markdown](../validation/reports/polyglot_gs_solver_comparison.md).
Native CLI times include process startup; Go/Rust compilation is excluded.
These measurements on a shared host are regression evidence, not isolated
production throughput claims. Case admission does not establish scientific
accuracy or qualify a free-boundary reconstruction.
