# SCPN Fusion Core — Benchmark Comparison

Comparison of SCPN Fusion Core against established fusion simulation codes.

> **Transparency note:** Timings labelled "Rust" use the compiled Rust backend
> with `opt-level = 3` and fat LTO. Timings labelled "Python" use the pure
> NumPy/SciPy path. "Projected" values are estimates, not measurements.
> Community code timings are from published literature (see references below).
> We encourage independent reproduction — see [`benchmarks/`](../benchmarks/).

## UpCloud L4 Native Solver Benchmark Bundle (2026-05-25)

Fresh GPU-host run:

- Provider/zone: UpCloud `fi-hel2`
- GPU: NVIDIA L4, driver `595.71.05`, `23034 MiB`
- CPU: `8x AMD EPYC 9575F 64-Core Processor`
- RAM: `62 GiB`
- OS: Linux `6.8.0-117-generic` x86_64
- Evidence bundle: [`validation/reports/upcloud_l4_native_solver_benchmarks.md`](../validation/reports/upcloud_l4_native_solver_benchmarks.md)
- Machine-readable bundle: [`validation/reports/upcloud_l4_native_solver_benchmarks.json`](../validation/reports/upcloud_l4_native_solver_benchmarks.json)

### Rust CPU/GPU Grad-Shafranov SOR, apples-to-apples

Tracked benchmark target:

`cd scpn-fusion-rs && cargo bench -p fusion-gpu --bench gpu_sor_bench -- --sample-size 10`

The CPU and GPU rows use the same grids, sinusoidal source, `20` SOR
iterations, and `omega = 1.3`. These numbers are end-to-end `solve_full`
GPU timings, including upload, synchronised compute, and download.

| Grid | CPU SOR median | GPU SOR `solve_full` median | Status |
|---|---:|---:|---|
| `33x33` | `45.215 us` | `965.68 us` | CPU faster; GPU launch/readback dominated |
| `65x65` | `177.11 us` | `965.96 us` | CPU faster; GPU launch/readback dominated |
| `129x129` | `709.97 us` | `984.41 us` | Near crossover, but CPU still faster |

This benchmark is the official tracked GPU baseline. It does **not** support a
GPU speedup claim at these grid sizes; it shows the workload is too small to
amortise launch and transfer overhead. Larger grids and persistent-buffer
timing remain required before making throughput claims.

### Native Rust solver kernels

| Benchmark | Median |
|---|---:|
| `picard_gs_solve/sor_33x33` | `194.08 us` |
| `picard_multigrid_solve/multigrid_33x33` | `408.26 us` |
| `vacuum_field_33x33_6coils` | `73.153 us` |
| `vacuum_field_65x65_6coils` | `277.52 us` |
| `sor_step_33x33` | `1.8483 us` |
| `sor_solve_33x33_500iter` | `921.00 us` |
| `sor_solve_65x65_500iter` | `3.7896 ms` |
| `sor_residual_65x65` | `8.0632 us` |
| `gmres_33x33` | `203.79 us` |
| `gmres_65x65` | `1.2071 ms` |
| `multigrid_33x33` | `404.59 us` |
| `multigrid_65x65` | `1.5946 ms` |
| `fft2_real_64x64` | `18.660 us` |
| `ifft2_real_64x64` | `21.421 us` |
| `cfft2_cifft2_complex_64x64` | `41.566 us` |
| `inverse_reconstruct_analytic_60probes` | `42.133 us` |
| `finite_difference_60probes` | `520.29 us` |
| `analytical_60probes` | `338.89 us` |
| `geqdsk_flux_profile_interpolation/second_order_33x33` | `16.742 us` |
| `geqdsk_flux_profile_interpolation/current_conserving_33x33` | `44.082 us` |
| `geqdsk_flux_profile_interpolation/second_order_65x65` | `60.039 us` |
| `geqdsk_flux_profile_interpolation/current_conserving_65x65` | `170.17 us` |
| `geqdsk_profile_source_components/source_components_33x33` | `107.13 us` |
| `geqdsk_profile_source_components/source_components_65x65` | `403.90 us` |
| `geqdsk_source_convention_adapter/select_adapter_33x33` | `44.608 us` |
| `geqdsk_source_convention_adapter/select_adapter_65x65` | `188.28 us` |
| `transport_step/lmode_single_step` | `754.06 ns` |
| `transport_step/hmode_single_step` | `866.43 ns` |
| `transport_step/hmode_neoclassical_single_step` | `3.4128 us` |
| `chang_hinton_chi_50pts` | `1.3872 us` |
| `bench_hall_mhd_step_64` | `864.60 us` |
| `bench_hall_mhd_step_128` | `5.2128 ms` |
| `bench_hall_mhd_run_100_64` | `82.088 ms` |

### Polyglot Grad-Shafranov scaling

The polyglot benchmark executes independent Python, Julia, Go, Rust, and Lean
implementations, not wrappers. These timings include the benchmark driver's
process invocation cost for CLI implementations, so Julia and Lean are
startup-dominated in this mode.

| Grid | Python | Go | Rust | Julia | Lean |
|---|---:|---:|---:|---:|---:|
| `17x17` | `1.983 ms` | `1.771 ms` | `1.257 ms` | `1387.005 ms` | `2145.238 ms` |
| `33x33` | `2.393 ms` | `2.761 ms` | `1.527 ms` | `990.421 ms` | `645.246 ms` |
| `65x65` | `4.030 ms` | `10.574 ms` | `3.018 ms` | `997.984 ms` | `670.504 ms` |

Numerical parity stayed near machine precision:

- `33x33`: Rust relative L2 `6.18e-16`, Go `1.69e-16`, Julia `1.06e-16`, Lean `2.45e-14`
- `65x65`: Rust relative L2 `5.02e-16`, Go `3.74e-16`, Julia `7.89e-17`, Lean `3.68e-14`

Startup-excluded in-process Lean timing is still missing; without that surface,
the Lean process-startup row above must not be interpreted as steady-state
solver throughput.

### Polyglot warm-throughput timing

Warm-throughput timing excludes language/tool startup for Python, Go, Rust, and
Julia by running `100` solves in a single long-lived process after `5`
warm-up solves on the `65x65` case.

| Language | Median | P95 |
|---|---:|---:|
| Rust | `1.302658 ms` | `1.878885 ms` |
| Julia | `1.663381 ms` | `3.752034 ms` |
| Python | `3.680793 ms` | `4.109378 ms` |
| Go | `4.022329 ms` | `4.808413 ms` |
| Lean | `1503.000000 ms` | `1593.000000 ms` |

Lean was measured with a single `lake env lean --run` process and `100` solves
inside that process. It is startup-excluded but remains much slower than the
compiled Rust, Julia, Python, and Go runtime surfaces.

### Persistent-buffer GPU SOR timing

Persistent-buffer timing uploads the source once, warms the GPU, then measures
synchronised `solve()` calls separately from a final download. This isolates
GPU compute dispatch from per-run host/device transfer.

| Grid | Runs | Persistent solve median | Persistent solve P95 | Final download |
|---|---:|---:|---:|---:|
| `129x129` | `100` | `0.760128 ms` | `2.940710 ms` | `0.053754 ms` |
| `257x257` | `100` | `0.764012 ms` | `2.897592 ms` | `0.165949 ms` |
| `513x513` | `50` | `0.861687 ms` | `3.009115 ms` | `0.343303 ms` |

The persistent-buffer result is the correct GPU throughput baseline. The
`solve_full` rows above remain the correct end-to-end latency baseline.

### CUDA-JAX nonlinear gyrokinetic benchmark

CUDA-enabled JAX was installed and detected one `CudaDevice(id=0)`.

| Case | NumPy elapsed | JAX/CUDA elapsed | Converged |
|---|---:|---:|---:|
| `krook` | `0.020947 s` | `4.051154 s` | true |
| `sugama` | `0.022762 s` | `1.648905 s` | true |
| `sugama_electromagnetic_kinetic` | `0.045141 s` | `1.662950 s` | true |

The CUDA-JAX rows are currently slower because this benchmark is tiny and
includes compilation/dispatch overhead. A separate warm JIT timing loop is
required before any CUDA throughput claim.

### Equilibrium reconstruction gates

| Contract | Status | Evidence |
|---|---|---|
| Free-boundary coil/vacuum benchmark | PASS | `validation/reports/free_boundary_benchmark.json` |
| Solov'ev manufactured-source FreeGS fallback | PASS | `artifacts/freegs_benchmark.json` on the GPU host |
| Strict FreeGS backend comparison | FAIL | FreeGS 0.8.2 scalar-derivative compatibility was patched in the benchmark harness; the benchmark now reaches FreeGS solve setup but still fails with no O-points or Picard non-convergence in all five configured cases |
| EFIT/GEQDSK raw profile-source gate | FAIL | `0/18` rows under `psi_N RMSE <= 0.05`; worst `jet/jet_lmode_2MA.geqdsk` at `10.626997` |
| Public operator-source gate | PASS | `8/8` public rows under `psi_N RMSE <= 1e-6` |
| Adapted profile-source gate | PASS | `4/4` accepted adapter rows under `psi_N RMSE <= 0.05` |
| Native operator/current closure | PASS | radial convergence order `2.000000`, worst radial current closure `8.31e-16` |

The raw profile-source and strict FreeGS failures are open benchmark blockers.
They are not CI or harness failures and must not be hidden by fallback rows.

## Solver Performance

| Metric | SCPN Fusion Core (Rust) | SCPN (Python) | TORAX | DIII-D (PCS) |
|--------|------------------------|---------------|-------|---------|
| **Control loop freq** | **10–30 kHz (Verified)** | 100 Hz | 50 Hz | 4–10 kHz (physics loops) |
| **Step compute time** | **0.3 μs** | 10 ms | ~1 ms | 100–250 μs |
| **Equilibrium solver** | Picard + SOR / Multigrid | Jacobi + Picard | JAX autodiff | rtEFIT |
| **Turbulence model** | JAX-FNO (synthetic-data surrogate) | FNO (Legacy) | QLKNN | N/A |
| **Language** | Rust + Python | Python | Python/JAX | C / Fortran |

## Feature Comparison

| Feature | SCPN Fusion Core | TORAX | PROCESS | FREEGS |
|---------|-----------------|-------|---------|--------|
| Grad-Shafranov equilibrium | Yes (multigrid) | Yes (spectral) | No | Yes (Picard) |
| Free-boundary solve | Yes | Partial | No | Yes |
| H-mode pedestal profiles | Yes (mtanh) | Yes (NN) | IPB98 scaling | No |
| Transport solver | 1.5D coupled | 1D flux-driven | 0-D | No |
| Disruption prediction | ML (transformer) | No | No | No |
| SPI mitigation | Yes | No | No | No |
| Neutronics / TBR | Yes (1-D slab) | No | Yes | No |
| Divertor thermal | Eich λ_q model | No | Eich model | No |
| RF heating (ICRH/ECRH) | Ray-tracing | No | Power balance | No |
| Neuro-symbolic control | SNN compiler | No | No | No |
| FNO turbulence | Yes (synthetic surrogate) | No | QLKNN | No |
| Sawtooth / MHD | Kadomtsev model | No | No | No |
| Digital twin | Real-time | No | No | No |
| Compact reactor optimizer | MVR-0.96 | No | Yes (DEMO) | No |
| GEQDSK I/O | Read + validate | No | No | Read + write |
| Rust acceleration | Native (10 crates) | No | JAX/XLA | No |
| GPU support | Yes (JAX XLA + wgpu) | Yes (JAX) | No | No |
| Autodifferentiation | Yes (JAX GS solver) | Yes (JAX) | No | No |
| RL environment | Yes (Gymnasium) | Gym-TORAX | No | No |
| Research validation (opt-in) | SPARC, ITPA, JET | DIII-D | ITER, DEMO | JET |

## Validation Accuracy

### IPB98(y,2) Confinement Scaling

Validation against the ITPA H-mode confinement database (20 entries, 10 machines):

| Machine | Shots | τ_E measured (s) | τ_E predicted (s) | Error (%) |
|---------|-------|-----------------|-------------------|-----------|
| JET | 3 | 0.15–0.85 | 0.14–0.82 | 5–8% |
| DIII-D | 3 | 0.10–0.18 | 0.09–0.17 | 6–10% |
| ASDEX-U | 3 | 0.05–0.12 | 0.05–0.11 | 4–9% |
| C-Mod | 2 | 0.02–0.04 | 0.02–0.04 | 3–7% |
| SPARC | 8 GEQDSK/EQDSK | B=12.2 T, I_p=8.7 MA | 16 public-case gated rows pass at 65x65 and 129x129 | ψ NRMSE, axis metadata, boundary containment, signed-q finite profile |
| DIII-D/JET synthetic GEQDSK | 10 GEQDSK | synthetic Solov'ev references | diagnostic rows, not public EFIT gate | ψ NRMSE and GEQDSK scalar/contour diagnostics |

> **Note on confinement accuracy:** The JET/DIII-D/ASDEX-U/C-Mod error
> percentages above are computed by the IPB98(y,2) scaling law implementation
> against the ITPA H-mode dataset. These are **scaling law errors**, not
> full-profile RMSE comparisons. The SPARC validation checks point-wise ψ NRMSE
> on the bundled public GEQDSK/EQDSK grids and GEQDSK compatibility invariants:
> finite non-degenerate ψ, declared-axis consistency, boundary containment, and
> finite signed-q profiles. The DIII-D/JET GEQDSK files in this repository are
> synthetic diagnostics; they are reported by
> `validation/benchmark_sparc_geqdsk_rmse.py` but are not counted as public
> EFIT parity gates.
> The aggregate EFIT report also records this curation per row:
> `reference_class=public_efit_reference` and `reference_role=gate` for SPARC,
> versus `reference_class=synthetic_proxy_reference` and
> `reference_role=diagnostic` for bundled DIII-D/JET proxy GEQDSK files.
> The aggregate report also carries summary counts so public gates and proxy
> diagnostics cannot be mixed accidentally: current local
> `reference_role_counts={'gate': 8, 'diagnostic': 10}` and
> `reference_class_counts={'public_efit_reference': 8,
> 'synthetic_proxy_reference': 10}`.
> It also emits `gate_row_count` and `gate_pass_count`, so public EFIT parity
> numerators are machine-readable without counting synthetic proxy diagnostics.
> Public-gate worst-row evidence is likewise separate via
> `gate_worst_file` and `gate_worst_psi_rmse_norm`.
> Adapter evidence is also split with
> `gate_source_convention_adapter_pass_count` and
> `gate_adapted_profile_pass_count`, preventing accepted public SPARC adapter
> rows from being conflated with synthetic diagnostics.
> Operator-source solver evidence is split the same way with
> `gate_operator_source_pass_count`, `gate_operator_source_worst_file`, and
> `gate_operator_source_worst_psi_rmse_norm`.
> Source-residual diagnosis also separates public rows through
> `gate_worst_source_alignment_file` and `gate_worst_source_residual_l2`.
> It also reports `solver_mode_counts` for the three profile-source lanes:
> raw GEQDSK profile-source fixed-boundary, operator-source fixed-boundary, and
> adapted GEQDSK profile-source fixed-boundary each have `18` labelled rows.

### GEQDSK current-closure diagnostics

`validation/psi_pointwise_rmse.py` now records toroidal-current closure from
two independent sources: the discrete operator current
`j_phi = -Delta*psi/(mu0 R)` and the profile-derived current from pprime/FFprime.
This is diagnostic evidence, not an EFIT-grade inverse-reconstruction claim.
On the current local run, 5 of 18 aggregate rows close operator-derived current
within 5% of the declared GEQDSK current. Four high-current public SPARC EQDSK
rows close within `6.4e-5` relative error, while the profile-source RMSE gate
still fails and remains documented as debt in the benchmark report.

### Native Grad-Shafranov operator-current closure

`validation/benchmark_gs_operator_current_closure.py` validates the native
non-reduced cylindrical operator contract on manufactured fields:
`psi(R, Z) = a R^4 + b Z^2 + c R^2 Z^2`,
`Delta*psi = 8aR^2 + 2b + 2cR^2`, and
`J_phi = -Delta*psi / (mu0 R)`. The radial-quartic case explicitly exercises
the cylindrical `-(1/R)dpsi/dR` term, and the mixed Solov'ev-style case verifies
the radial cancellation in `R^2 Z^2`. This benchmark is separate from EFIT
inverse reconstruction: it proves the native operator/current diagnostic obeys
the Grad-Shafranov current relation on the local grid.

Local run on this machine:

- Platform: Linux 6.17.0-29-generic x86_64
- CPU count: 12
- Python: 3.12.3
- NumPy: 2.2.6

| Case | Grid | a | b | c | elapsed s | max discrete Delta* abs error | analytic Delta* error | max J rel error | total current rel error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vertical_quadratic | 17x19 | 0 | -0.25 | 0 | 2.400220e-04 | 5.662137e-15 | 5.662137e-15 | 1.138333e-14 | 1.523927e-16 |
| radial_quartic_17 | 17x19 | 0.03125 | -0.125 | 0 | 1.361470e-04 | 3.330669e-14 | 9.765625e-04 | 4.338971e-14 | 0.000000e+00 |
| radial_quartic_33 | 33x35 | 0.03125 | -0.125 | 0 | 1.663470e-04 | 2.380318e-13 | 2.441406e-04 | 1.901481e-13 | 0.000000e+00 |
| radial_quartic_65 | 65x67 | 0.03125 | -0.125 | 0 | 3.110590e-04 | 1.027178e-12 | 6.103516e-05 | 3.028695e-12 | 8.311853e-16 |
| mixed_solovev | 29x31 | 0 | -0.125 | 0.05 | 1.633400e-04 | 5.545564e-14 | 5.545564e-14 | 1.119700e-12 | 9.237473e-16 |

The radial-quartic analytic error equals the expected second-order centered
stencil truncation `2a dR^2`. The measured order from the two finest grids is
`1.999999977`; the discrete-contract error remains near machine precision.
The radial refinement sequence now also gates total-current closure stability:
worst radial-quartic total-current relative error is `8.311852886233243e-16`
against threshold `1e-12`.

Status: PASS against thresholds `Delta* <= 1e-10`, `J_rel <= 1e-11`,
`I_total_rel <= 1e-12`, and radial current-closure stability `<= 1e-12`.

### GEQDSK Grad-Shafranov source contract

The DIII-D/JET proxy GEQDSK validation runner now consumes the same shared source construction used by the point-wise RMSE benchmark: second-order flux-normalized profile interpolation, current-conserving weighted integral correction, and explicit zeroing of boundary rows/columns outside the physical plasma source domain. These rows remain diagnostic proxy references, but their source residuals no longer use a separate weaker linear interpolation path.

`validation/benchmark_sparc_geqdsk_rmse.py` now gates bundled GEQDSK/EQDSK
references on the native Grad-Shafranov PDE relation, not only point-wise
`psi` RMSE. For every public reference, the benchmark evaluates the centered
cylindrical operator and checks the EFIT profile-source convention:

`Delta*psi = -mu0 R^2 p'(psi_N) - FF'(psi_N)`.

The gate records magnetic-axis location error, boundary containment, finite
profile/q arrays, in-plasma source sample count, absolute source residual,
source-relative L2 residual, and best-fit convention attribution for global
source scaling (`canonical`, sign flips, `2π`, and inverse-`2π`). The current
source-rel-L2 threshold is `5e-2`.
The source contract is available as a strict gate via
`python validation/benchmark_sparc_geqdsk_rmse.py --strict-source-contract`.
The default benchmark records these metrics without failing mixed-convention
public files, so profile-inconsistent references are visible but do not hide
the standard point-wise `psi` benchmark status.

Local source-convention attribution on the bundled public SPARC rows shows
four high-current cases (`sparc_1305`, `sparc_1310`, `sparc_1315`,
`sparc_1349`) are explained by a near-`2π` global source scale
(`best_fit_scale = 6.16–6.25`, best-fit relative L2 `0.067–0.139`). The
`lmode_*` rows and `sparc_1300` remain unclassified, so they are not accepted
as strict native source-contract evidence.

The aggregate EFIT/GEQDSK point-wise RMSE report
(`validation/psi_pointwise_rmse.py`) records the same convention classifier in
each row as `source_best_fit_convention`, keeping the strict aggregate and
SPARC benchmark reports aligned. Latest local aggregate run:
`PYTHONPATH=src python validation/psi_pointwise_rmse.py --mode benchmark
--reference-root validation/reference_data --output-json
artifacts/efit_nrmse_benchmark.json --output-md
artifacts/efit_nrmse_benchmark.md`. Result: strict EFIT/GEQDSK gate remains
`FAIL`, with `0/18` rows below `psi_N RMSE <= 0.05`, worst row
`jet/jet_lmode_2MA.geqdsk` at `10.626997`, and `18/18` rows classified as
`profile_source_mismatch`. The new convention classifier is report evidence
for diagnosing those failures, not a relaxation of the native source contract.
The tracked JSON report is now schema `efit-nrmse-benchmark.v2` and declares
`benchmark_scope = profile_source_fixed_boundary_reconstruction` with explicit
raw-profile, operator-source, and adapted-profile `solver_mode` fields at both
top level and row level. These labels keep the strict profile-source gate,
operator-source elliptic-solver gate, and adapted-profile reconstruction gate
machine-separable from free-boundary coil/vacuum reconstruction benchmarks.
The SPARC point-wise JSON report likewise declares `benchmark_id =
sparc-pointwise-rmse`, the same fixed-boundary profile-source scope, and keeps
raw GEQDSK profile-source metrics separate from the explicitly requested
public-SPARC convention adapter metrics. Raw canonical source metrics remain
strict and unchanged. The report now also emits
`geqdsk_adapted_source_contract_pass`,
`geqdsk_adapted_source_convention_adapter`, and
`geqdsk_adapted_source_rel_l2` per row, plus top-level
`adapted_source_contract_row_count`, `adapted_source_contract_pass_count`, and
`gate_adapted_source_contract_pass_count`. The adapted gate can be enforced by
running `python validation/benchmark_sparc_geqdsk_rmse.py
--strict-adapted-source-contract`; it is an explicit convention-normalised
fixed-boundary reconstruction contract, not an operator-source,
free-boundary, or reduced-order surrogate result.
The same report now carries a separate operator-source elliptic-solver gate:
`18/18` rows reproduce `Delta*psi_ref` below `psi_N RMSE <= 1e-6`, with worst
operator-source row `sparc/lmode_vh.geqdsk` at `2.05e-14`. This isolates the
current benchmark debt to profile-source/GEQDSK convention compatibility rather
than the SOR elliptic solve itself.
The diagnostic source ranking now evaluates explicit scale candidates
(`2*pi`, `1/(2*pi)`, and physical-flux-span transforms). Latest candidate
distribution across all 18 rows: `profile_source_scaled_by_2pi=5`,
`profile_source_scaled_by_minus_2pi=3`, `pressure_only=4`,
`negated_profile_source_over_flux_span=2`,
`profile_source_over_flux_span=1`, `profile_source=1`,
`pressure_plus_negated_ffprime=1`, and
`negated_profile_source_times_flux_span=1`. On the public high-current SPARC
EQDSK rows, the explicit `2*pi` profile-source candidate is now selected
directly with relative L2 `0.067-0.139`, while `sparc_1300` remains a
flux-span candidate with relative L2 `4.043` and is not accepted as strict
native source-contract evidence.
An explicit GEQDSK convention adapter contract is now part of the aggregate
schema. It accepts only named transforms, not fitted least-squares scales, and
uses residual threshold `0.15`. Latest local report:
`source_convention_adapter_pass_count=4/18`, with accepted rows limited to
`sparc_1305`, `sparc_1310`, `sparc_1315`, and `sparc_1349`, all under the
explicit `scaled_by_2pi` adapter. Adapter counts were
`scaled_by_2pi=5`, `scaled_by_minus_2pi=3`, `over_flux_span=1`,
`negated_over_flux_span=4`, `canonical=4`, and
`negated_times_flux_span=1`; only the four high-current SPARC `scaled_by_2pi`
rows are below the adapter residual threshold. The raw canonical gate remains
strict and failing.
Reference-case curation is enforced in the same schema: every row declares its
dataset id, provenance class, gate role, expected contract, and expected
source convention. Public SPARC files are gate rows; synthetic DIII-D/JET
GEQDSK files are diagnostic rows only and cannot be accidentally counted as
public EFIT parity evidence.
The reference-data provenance manifest now emits the same curation fields for
GEQDSK/EQDSK equilibrium files, so benchmark inputs are also separable before
they are consumed by the EFIT/GEQDSK reports.
The FreeGS/Solov'ev manufactured-solution report now includes a separate
`manufactured_solovev_gs_source_grid_convergence` contract that checks
monotonic finite-difference GS source convergence across `33`, `65`, and
`129` point grids, requiring observed order at least `1.5` and fine/coarse
source residual ratio no worse than `0.35`.
Latest local result in `artifacts/freegs_benchmark.json`: observed order
`1.998225`, fine/coarse ratio `0.062654`, residuals `2.7931e-4`, `6.986e-5`,
and `1.75e-5`; the convergence contract passed.

The same report now includes an adapted profile-source reconstruction gate for
rows where that explicit named adapter is accepted. This is not a replacement
for raw canonical mode: rows without a passing adapter stay diagnostic-only for
this gate, and the raw `psi_N` RMSE gate remains unchanged. Latest local result:
`adapted_profile_pass_count=4/4` accepted adapter rows at threshold
`psi_N RMSE <= 0.05`, worst accepted row `sparc/sparc_1315.eqdsk` at
`0.012696`. Each adapted row also reports axis error, boundary contour
containment, boundary-flux RMSE, SOR residual, and q-profile sanity so the
result is a reconstruction contract rather than a scale-factor diagnostic.
The q-profile sanity gate is machine-readable rather than boolean-only:
accepted-adapter rows report finite fraction `1.000`, minimum `|q|` in
`[0.9939342073, 1.002669677]`, zero sign changes, and monotonic fraction in
`[0.95, 1.00]`.
Profile-source interpolation now uses a second-order flux-normalized
quadratic stencil, preserves the masked current-relevant weighted integral of
the established linear GEQDSK profile contract, and explicitly masks boundary
rows and columns before source assembly. This is a numerical interpolation and
boundary-treatment hardening; it does not relax raw canonical GEQDSK convention
semantics.

Polyglot status: the native Julia, Go, Rust, and Lean solver packages expose
the same operator-current surfaces. Julia, Go, and Rust package tests validate
manufactured `Z^2`, `R^4 + Z^2`, and mixed `R^2 Z^2` closure; Lean builds the corresponding
`deltaStar`, `toroidalCurrentDensityFromFlux`, and
`totalToroidalCurrentFromFlux` definitions as part of `lake build`.
Rust `fusion-core::source` also exposes the same accepted GEQDSK
source-convention adapter contract as the Python benchmark path: canonical,
negated, `2*pi`, inverse-`2*pi`, and flux-span transforms are ranked as named
transforms only, with no fitted scale accepted as a pass. Local verification:
`cargo test -p fusion-core geqdsk_source_convention --lib` (`5 passed`).
The Rust surface now strictly round-trips the same GEQDSK convention labels
used by the Python reports and rejects fitted/unknown convention names instead
of accepting an implicit scale. It also exposes residual-ranked executable
candidate rows so Rust audits can distinguish evaluated named transforms from
non-executable fitted-scale diagnostics; flux-span candidates are emitted only
when the physical flux span is finite and non-zero.
Rust `fusion-core::source` now also exposes native second-order and current-conserving flux-profile interpolation helpers matching the Python profile-source construction contract: local quadratic GEQDSK profile interpolation, masked weighted-integral preservation against the linear GEQDSK contract, finite input guards, shape guards, and non-negative masked weight enforcement.
It also exposes `compute_geqdsk_profile_source_components`, which assembles pressure, FFprime, and total Grad-Shafranov source arrays with plasma-mask reporting, explicit boundary zeroing, and source-norm diagnostics matching the Python profile-source path.
Free-boundary coil/vacuum parity is intentionally narrower: Python and Rust
now expose native circular-filament Green-function reconstruction contracts,
while Go, Julia, and Lean currently expose fixed-boundary/operator-current
surfaces only. No Go/Julia/Lean free-boundary wrappers are claimed as parity
until those languages grow equivalent native coil/vacuum solver logic.

Local verification commands:

- `julia --project=scpn-fusion-jl scpn-fusion-jl/test/runtests.jl`
- `go test ./gssolver`
- `cargo test -p fusion-polyglot operator_current_closure`
- `lake build`

### Native Grad-Shafranov mesh-convergence contract

`validation/mesh_convergence_study.py` now reports an explicit solver-fidelity
contract for the fixed-boundary manufactured Solov'ev solve. The benchmark
solves the elliptic Grad-Shafranov equation on successively refined grids with
analytic Dirichlet boundaries and rejects regressions unless at least two
adjacent-grid transitions remain second-order within the configured floor.

Local run on this machine:

- Platform: Linux-6.17.0-29-generic-x86_64-with-glibc2.39
- CPU count: 12
- Python: 3.12.3
- NumPy: 2.2.6

| Grid | h | NRMSE | adjacent-grid rate | time s | iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17x17 | 1.2500e-01 | 8.1756e-05 | N/A | 0.1133 | 801 |
| 33x33 | 6.2500e-02 | 1.9818e-05 | 2.04 | 0.8425 | 2601 |
| 65x65 | 3.1250e-02 | 4.8780e-06 | 2.02 | 6.3576 | 10001 |
| 129x129 | 1.5625e-02 | 1.2256e-06 | 1.99 | 33.2110 | 25000 |

Status: PASS. The measured minimum adjacent-grid rate is `1.992859` against the
required floor `1.80`, with `3` rated transitions against the required `2`.

### Transport Source Power-Balance Contract

Auxiliary-heating source normalisation (MW -> volumetric W/m^3 -> keV/s)
is benchmarked with deterministic reconstruction checks:

| Metric | Value | Command |
|--------|-------|---------|
| Cases | 8 (single-ion + multi-ion, 4 powers) | `python validation/benchmark_transport_power_balance.py` |
| Max relative power-balance error | 2.4e-16 | same |
| Threshold | <= 1e-6 | same |

### Vertical-Control Replay Contract

The vertical-control replay benchmark is a deterministic replay scaffold
exercising reduced-order RZIP-backed vertical-axis plant dynamics across
PID, super-twisting, and repository sliding-mode controller lanes, with a
`no_control` diagnostic lane retained to prove that the acceptance gate is
sensitive to missing control action.

| Contract | Acceptance evidence | Command |
|----------|---------------------|---------|
| Deterministic replay | Repeat controller traces and RZIP state trajectory checksums must match | `python validation/vertical_control_replay_benchmark.py --strict` |
| Actuator bounds | Commands are clipped after controller output and must respect amplitude and slew limits | same |
| Post-disturbance relaxation | Primary controllers must reduce vertical displacement after the disturbance window ends | same |
| Fault and saturation paths | High-growth and low-actuator uncertainty cases must remain bounded and deterministic; a `no_control` lane remains diagnostic-only and must fail acceptance | same |
| Multi-profile replay | ITER-like, DIII-D-like, and compact-tokamak reduced-order plant profiles must pass | `python validation/vertical_control_replay_benchmark.py --all-profiles --strict` |
| Uncertainty envelope | Growth, damping, actuator, sensor-bias, and one-step latency perturbations are replayed | same |

### Vertical-Control Replay Release Gate

Before this lane is labeled production-grade, all of the following must pass and be
recorded in the latest strict run:

- `python validation/vertical_control_replay_benchmark.py --strict`
- `python validation/vertical_control_replay_benchmark.py --all-profiles --strict`
- strict JSON schema validation in `tests/test_vertical_control_replay_benchmark.py`
- deterministic replay checksums (`deterministic_replay_pass == true` and deterministic trajectory checksums in both JSON payloads)
- uncertainty envelope checks (`passes_thresholds == true`, `all_profiles_pass == true`)
- explicit saturation/fault semantics (`no_control` remains diagnostic-only and fails acceptance)
- CI provenance gate in `.github/workflows/ci.yml` `benchmark-provenance-smoke`
- report review of generated Markdown artifacts under `artifacts/vertical_control_replay_benchmark.md` and `artifacts/vertical_control_replay_profiles.md`

Until all items above are green in CI, the lane remains a deterministic
reduced-order replay scaffold and is not presented as a full PCS production
control claim.

### Equilibrium Solver Convergence

**Current production path (Picard + Red-Black SOR):**

| Grid | Solver | Picard Iters | Inner SOR Iters | Time (Rust release) |
|------|--------|-------------|-----------------|---------------------|
| 33x33 | Picard+SOR | 3–5 | 50/iter | ~50 ms |
| 65x65 | Picard+SOR | 5–8 | 50/iter | ~100 ms |
| 128x128 | Picard+SOR | 8–12 | 50/iter | ~1 s |

**Multigrid V-cycle (wired into kernel, selectable via `set_solver_method("multigrid")`):**

| Grid | Solver | V-cycles | Residual | Time (projected) |
|------|--------|---------|----------|-----------------|
| 33x33 | Multigrid | 5–8 | < 1e-8 | ~2 ms |
| 65x65 | Multigrid | 8–12 | < 1e-6 | ~15 ms |
| 128x128 | Multigrid | 10–15 | < 1e-4 | ~95 ms |

> **Note:** The multigrid path is now available from Python via
> `kernel.set_solver_method("multigrid")`. Run `validation/benchmark_solvers.py`
> to compare SOR vs multigrid end-to-end on your hardware.

## Inverse Reconstruction Performance

The full-kernel Levenberg-Marquardt inverse solver calls the forward
Grad-Shafranov equilibrium solver 9 times per iteration: 1 baseline solve plus
8 finite-difference Jacobian columns for the mtanh pressure and FF profile
parameters. The forward solve dominates wall time;
Tikhonov regularisation, Huber robust loss, and per-probe σ-weighting add
negligible overhead.

The normalized profile-space inverse still exposes a closed-form mtanh
Jacobian. The physical `(R, Z)` kernel inverse does not use that reduced
response model; both accepted Jacobian modes route to full nonlinear
forward-solve finite differences.

| Configuration | Overhead per LM iter | Notes |
|---------------|---------------------|-------|
| Full-kernel default | 9 forward solves + damped least squares | baseline |
| + Tikhonov (α=0.1) | same + N_PARAMS additions | negligible overhead |
| + Huber (δ=0.1) | same + IRLS weights | negligible overhead |
| + σ weights | same + per-probe division | negligible overhead |
| **Total (1 LM iter, 65×65, release)** | **~0.8 s** | dominated by forward solve |
| **Full reconstruction (5 iters)** | **~4 s** | see EFIT comparison below |

### vs EFIT (Literature Comparison)

> **Note:** EFIT timings are from Lao et al. (1985) and are not direct
> measurements on equivalent hardware. This is an order-of-magnitude
> comparison for context, not a head-to-head benchmark.

| Metric | SCPN Fusion Core (Rust) | EFIT (literature) |
|--------|------------------------|------|
| Forward solve (65×65) | ~0.1 s | ~50 ms |
| 1 LM iteration | ~0.8 s | ~0.4 s (Picard) |
| Full reconstruction | ~4 s | ~2 s |
| Regularisation | Tikhonov + Huber + σ | Von-Hagenow smoothing |
| Profile model | mtanh (7 params) | Spline knots (~20 params) |

SCPN is currently ~2× slower than reported EFIT timings. The gap is expected
to close when the multigrid solver replaces Picard+SOR in the kernel.

*Reference: Lao, L.L. et al. (1985). Nucl. Fusion 25, 1611.*

## Neural Transport Surrogate

MLP surrogate for fast transport coefficient estimation. Pure NumPy inference
— no TensorFlow/PyTorch overhead.

Two weight sets are shipped:
- `weights/neural_transport_qlknn.npz` — 14→1024→512→256→6 gated MLP trained on
  QLKNN-10D (van de Plassche 2020, Zenodo DOI 10.5281/zenodo.3497066).
  test_rel_L2 = 0.201, trained on NVIDIA L40S 48GB.
- `weights/fno_turbulence_jax.npz` — 4-layer JAX FNO (modes=24, width=128) trained
  on 2000 QLKNN-oracle spatial equilibria. val_rel_L2 = 0.356.

**Latency measurements (synthetic weights, Criterion benchmark):**

| Method | Single-point | 100-pt profile | 1000-pt profile |
|--------|-------------|----------------|-----------------|
| Critical-gradient (numpy) | ~2 µs | ~0.2 ms | ~2 ms |
| MLP surrogate (numpy, H=64) | ~5 µs | ~0.05 ms | ~0.3 ms |

**Literature reference (not direct comparison):**

| Method | Single-point | Source |
|--------|-------------|--------|
| QuaLiKiz (gyrokinetic) | ~1 s | van de Plassche 2020 |
| QLKNN (TensorFlow) | ~10 µs | van de Plassche 2020 |

The latency gap between an MLP surrogate and a first-principles gyrokinetic
solver is expected to be very large (orders of magnitude), but this is
inherent to the surrogate approach — speed is traded for fidelity. The
accuracy of the surrogate depends entirely on training data quality and
has not been validated against gyrokinetic output in this repository.

Key properties:
- Vectorised `predict_profile()` gives ~100× speedup over point-by-point loop
- SHA-256 weight checksums for reproducibility tracking
- Transparent degradation to analytic model when no weights are available

*Reference: van de Plassche, K.L. et al. (2020). Phys. Plasmas 27, 022310.*

## Native Nonlinear Gyrokinetic Contract

The native nonlinear gyrokinetic benchmark is a bounded 5D delta-f NumPy/JAX
contract, not a replacement for production GENE or CGYRO turbulence campaigns.
`benchmarks/gk_solver_comparison.py` records transport diagnostics, Sugama
moment residuals, and the nonlinear E x B invariant diagnostic. The invariant
requires zero high-k leakage outside the 2/3 dealiased spectral mask and no
free-energy injection from the undriven collisionless nonlinear bracket.

Latest local results are written to
`validation/reports/gk_nonlinear_solver_comparison.md`.

## Extended Community Baseline Comparison

Comparison against established equilibrium, transport, and integrated
modelling codes used in the fusion community. Runtimes are representative
single-shot values on contemporary hardware (2024–2025 publications).

| Code | Category | Solver | Transport | Grid | Typical Runtime | Language |
|------|----------|--------|-----------|------|-----------------|----------|
| **EFIT** | Reconstruction | Current-filament Picard | N/A | 65×65 | ~2 s | Fortran |
| **P-EFIT** | Reconstruction | GPU-accelerated EFIT | N/A | 65×65 | <1 ms | Fortran+OpenACC |
| **CHEASE** | Equilibrium | Fixed-boundary, cubic Hermite | N/A | 257×257 | ~5 s | Fortran |
| **HELENA** | Equilibrium | Fixed-boundary, isoparametric | N/A | 201 flux, 257 pol | ~10 s | Fortran |
| **JINTRAC** | Integrated | HELENA + QLKNN + NEMO | 1.5D flux-driven | 100 radial | ~10 min/shot | Fortran/Python |
| **TORAX** | Integrated | JAX spectral | 1D QLKNN | Spectral | ~30 s (GPU) | Python/JAX |
| **GENE** | Gyrokinetic | Nonlinear δf | 5D Vlasov | 128³×64v² | ~10⁶ CPU-h | Fortran/MPI |
| **CGYRO** | Gyrokinetic | Nonlinear | 5D continuum | 256 radial | ~10⁵ CPU-h | Fortran/MPI |
| **DREAM** | Disruption | RE kinetic + fluid | 0D–1D | 100 radial | ~1 s | C++ |
| **SCPN (Rust)** | Full-stack | Picard+SOR + LM inverse | 1.5D + crit-gradient | 65×65 | ~4 s recon | Rust+Python |
| **SCPN (Python)** | Full-stack | Picard + Jacobi | 1.5D + crit-gradient | 65×65 | ~40 s recon | Python |

**References:**
- Lao, L.L. et al. (1985). *Nucl. Fusion* 25, 1611 (EFIT).
- Sabbagh, S.A. et al. (2023). GPU-accelerated EFIT (P-EFIT).
- Lütjens, H. et al. (1996). *Comput. Phys. Commun.* 97, 219 (CHEASE).
- Huysmans, G.T.A. et al. (1991). *Proc. CP90 Conf. Comput. Physics* (HELENA).
- Romanelli, M. et al. (2014). *Plasma Fusion Res.* 9, 3403023 (JINTRAC).
- Jenko, F. et al. (2000). *Phys. Plasmas* 7, 1904 (GENE).
- Belli, E.A. & Candy, J. (2008). *Phys. Plasmas* 15, 092510 (CGYRO).
- Hoppe, M. et al. (2021). *Comput. Phys. Commun.* 268, 108098 (DREAM).
- van de Plassche, K.L. et al. (2020). *Phys. Plasmas* 27, 022310 (QLKNN).

## Computational Power Metrics

Estimated FLOPS, memory footprint, and energy for each solver component.
Energy estimated at ~15 pJ/FLOP (AMD Zen 4 core, ~5 W at 300 GFLOP/s).

### FLOP and Memory Estimates

| Component | Grid/Size | FLOP count | Memory (MB) | Est. Energy (mJ) | Notes |
|-----------|-----------|-----------|-------------|-------------------|-------|
| SOR step (65×65) | 4,225 pts | ~0.1 MFLOP | 0.26 | ~0.002 | 5-pt stencil, 4 FLOP/pt |
| Multigrid V-cycle (65×65) | 4 levels | ~2 MFLOP | 0.7 | ~0.03 | 3+3 smoothing + restrict + prolong |
| Full equilibrium (65×65, 12 cycles) | — | ~24 MFLOP | 0.7 | ~0.4 | 12 V-cycles × 2 MFLOP |
| Full equilibrium (128×128, 15 cycles) | — | ~120 MFLOP | 2.5 | ~2 | Dominated by SOR sweeps |
| Inverse LM iter (65×65) | 8 fwd solves | ~192 MFLOP | 1.5 | ~3 | + Cholesky ~0.01 MFLOP |
| MLP inference (H=64/32) | 10→64→32→3 | ~5 KFLOP | <0.01 | <0.001 | 2 matmul + 2 ReLU + softplus |
| MLP profile (1000-pt) | batch×10→3 | ~5 MFLOP | 0.08 | ~0.08 | Single batched matmul path |
| Critical-gradient (1000-pt) | 1000 pts | ~0.02 MFLOP | 0.06 | ~0.0003 | Vectorised numpy |

### Memory Bandwidth Utilisation

| Component | Data moved (KB) | BW utilisation |
|-----------|----------------|----------------|
| SOR step 65×65 | 132 KB (2 arrays × 4225 × 8B × 2 pass) | <1% of 50 GB/s |
| Multigrid V-cycle | ~300 KB (multi-level) | <1% |
| MLP 1000-pt batch | 160 KB (input + output + weights) | <1% |

All current workloads are compute-bound rather than memory-bound at these
grid sizes. Bandwidth becomes significant at 512×512 and above.

## GPU Offload Roadmap

Status and projected targets for GPU acceleration. Tracked in issue #12.
Implementation strategy uses the `wgpu` crate (cross-platform
Vulkan/Metal/D3D12/WebGPU) to avoid CUDA lock-in.
See `SCPN_FUSION_CORE_COMPREHENSIVE_STUDY.md` Section 28 for full details.

### Target Status

| Target | Backend | Expected Speedup | Priority | Status |
|--------|---------|-----------------|----------|--------|
| SOR red-black sweep | wgpu compute shader | 20–50× (65×65), 100–200× (256×256) | P0 | Targeted |
| Multigrid V-cycle | wgpu + host orchestration | 10–30× | P1 | Targeted |
| Vacuum field (elliptic integrals) | rayon (CPU) → wgpu | 5–10× | P2 | rayon done |
| MLP batch inference | wgpu or cuBLAS | 2–5× (small H) | P3 | Targeted |
| FNO turbulence (FFT) | cuFFT / wgpu FFT | 50–100× (64×64) | P3 | Targeted |

### Projected Timings (GPU, RTX 4090-class)

| Component | CPU Rust (release) | GPU projected | Source |
|-----------|-------------------|---------------|--------|
| Equilibrium 65×65 | 100 ms | ~2 ms | Section 28 study |
| Equilibrium 256×256 | ~10 s | ~50 ms | Extrapolated |
| P-EFIT reference (65×65) | — | <1 ms | Sabbagh 2023 |
| Full inverse reconstruction | ~4 s | ~200 ms | 8× GPU fwd solve |
| MLP 1000-pt profile | 0.3 ms | ~0.05 ms | Batch matmul |

Implementation path: `wgpu` crate targeting Vulkan/Metal/D3D12/WebGPU,
with CPU SIMD alternate path for systems without GPU support.

## Adaptive Grids & 3D Transport Roadmap

### Current State & Targets

| Feature | Current State | Target | Effort | Prerequisite |
|---------|--------------|--------|--------|-------------|
| Uniform multigrid | Production (V-cycle, 4 grid sizes) | — | Done | — |
| AMR (h-refinement) | Not implemented | Quadtree, error-based tagging | ~4 weeks | Multigrid |
| AMR error estimator | Not implemented | Gradient-jump + curvature indicators | ~1 week | AMR structure |
| 3D equilibrium (stellarator) | Not applicable (tokamak only) | VMEC-like 3D | ~3 months | — |
| 3D transport | 1.5D radial only | Toroidal mode coupling (n=0,1,2) | ~6 weeks | 3D geometry |
| FNO 3D turbulence | 2D proof-of-concept | 3D fftn + toroidal modes | ~4 weeks | Training data |
| 3D geometry physics | Visualization only (OBJ export) | Field-line tracing, Poincaré maps | ~3 weeks | 3D equilibrium |

### AMR Comparison with Community Codes

| Code | AMR Type | Application |
|------|---------|-------------|
| NIMROD | Block-structured | 3D MHD |
| JOREK | Bézier elements, h-p | 3D nonlinear MHD |
| BOUT++ | Field-aligned, block | Edge turbulence |
| SCPN (targeted) | Quadtree, gradient-based | 2D GS + 3D extension |

The targeted quadtree AMR is simpler than JOREK's h-p adaptivity but
sufficient for equilibrium and transport applications where steep gradients
are localised near the pedestal and X-point regions.

## Rust Full-Order Equilibrium Benchmarks

Rust full-order equilibrium timings are reported separately from the
reduced-order Rust flight-simulator control kernel. The control kernel is a
linearised plant surrogate for fast controller-loop studies; the benchmarks
below exercise Grad-Shafranov and vacuum-field work in `fusion-core`.

Local run on 2026-05-24:

| Parameter | Value |
|-----------|-------|
| **CPU** | Intel Core i5-11600K, 6C/12T |
| **RAM** | 31.1 GB |
| **OS** | Linux 6.17.0-29-generic x86_64, glibc 2.39 |
| **Rust** | 1.95.0 |
| **Python** | 3.12.3 (`/home/anulum/.local/bin/python`) |
| **GPU** | NVIDIA GeForce GTX 1060 6GB, driver 580.159.03 |
| **Command** | `cargo bench -p fusion-core --bench picard_bench`; `cargo bench -p fusion-core --bench vacuum_bench` |

| Benchmark | Physics scope | Grid | Criterion centre estimate |
|-----------|---------------|------|---------------------------|
| `picard_gs_solve/sor_33x33` | Full-order Grad-Shafranov SOR solve, 10 Picard iterations | 33x33 | 412.83 us |
| `picard_multigrid_solve/multigrid_33x33` | Full-order Grad-Shafranov Picard multigrid solve, 10 Picard iterations | 33x33 | 844.59 us |
| `vacuum_field_33x33_6coils` | Vacuum flux from six ITER-like coils | 33x33 | 121.47 us |
| `vacuum_field_65x65_6coils` | Vacuum flux from six ITER-like coils | 65x65 | 543.69 us |

Vacuum-field contract: Python and Rust now use the same circular-filament
Green's-function convention for external free-boundary coupling. Observation
points exactly on a source filament return zero to exclude coil self-inductance
from the vacuum boundary map; self-inductance belongs in a separate coil-circuit
model. Local contract checks passed with
`python -m pytest tests/test_coil_optimization.py -q` (`33 passed`) and
`cargo test -p fusion-core vacuum::tests --lib` (`12 passed`).
The vacuum benchmark row was rerun locally with
`cargo bench -p fusion-core --bench vacuum_bench -- --sample-size 10` after the
self-observation fix; Criterion centre estimates were `121.47 us` for 33x33 and
`543.69 us` for 65x65.

Free-boundary contour reconstruction now has a native Green-function gate:
`validation/benchmark_free_boundary.py` reconstructs flux on a named boundary
point set directly from coil currents and the circular-filament response
matrix, without replaying fixed Dirichlet values. The same gate now samples
limiter, magnetic-axis, and X-point metadata with the same vacuum response.
The tracked JSON report declares `benchmark_id =
free_boundary_coil_vacuum_reconstruction`, `benchmark_scope =
free_boundary_reconstruction`, and per-lane `physics_scope`/`solver_mode`
fields so it cannot be confused with fixed-boundary profile-source or
operator-source solves. It also emits a top-level `passes` boolean and a
fail-closed `gate_summary` (`6/6` gates passed locally) so diagnostic rows
cannot be mistaken for the aggregate benchmark decision; the CLI exits
non-zero when any named gate is missing or failed.
Latest local result: boundary Green reconstruction RMSE `0.00e+00`, max
absolute error `0.00e+00`, response rank `1/1` coils over `5` contour points,
`4` limiter points, `2` X-points, axis flux `2.589381e-01`, minimum limiter
clearance `0.380789 m`, limiter containment fraction `1.000`, symmetric
X-point pair flux residual `0.00e+00`, status `PASS`. The shape-control current-inversion gate recovers three bounded external coil currents from five boundary-flux target points with response rank `3/3`, condition number `26.035`, current relative L2 error `2.96e-15`, and flux relative RMSE `8.44e-17`, status `PASS`.
The actual `solve_free_boundary` path now returns the same coil/vacuum
reconstruction diagnostic on its computational boundary; latest local solver
contract result: vacuum-boundary absolute error `0.00e+00` over `256` boundary
points in `1` outer iteration, with `4` limiter points, `2` X-points, and
axis flux `2.589381e-01`, computational-boundary containment fraction `0.000`
(diagnostic-only because the computational wall is outside the limiter; the
machine-readable `boundary_containment_contract_role` is
`diagnostic_computational_wall_outside_limiter` and
`limiter_containment_required=false`),
symmetric X-point pair flux residual `0.00e+00`, status `PASS`.
Rust `fusion-core::vacuum` now exposes the same native boundary-flux
reconstruction contract through `reconstruct_boundary_flux_from_coils` and
`reconstruct_boundary_flux_from_coils_with_metadata`, including optional target
residual, RMSE, max absolute error, point count, coil count, limiter flux,
minimum limiter clearance, axis flux, X-point flux, X-point flux span, and
symmetric X-point pair flux residual. This is a full Rust implementation using
the same circular-filament Green function, not a wrapper around the Python
benchmark path.
The Rust surface also exposes `reconstruct_shape_currents_from_boundary_flux`,
which solves the native Green-function response system for bounded coil
currents from boundary-flux targets and reports recovered currents,
reconstructed flux, residual RMSE, relative flux RMSE, response rank,
condition number, and active current bounds. This ports the accepted Python
shape-current inversion contract into Rust rather than leaving the benchmark
as Python-only evidence. The Python `solve_free_boundary(..., optimize_shape=True)`
path now also reports integrated shape-optimization diagnostics: latest local
result recovered three bounded coil currents from five target-flux points with
current relative L2 error `2.96e-15`, flux relative RMSE `8.44e-17`,
vacuum-boundary absolute error `5.55e-17`, and response rank `3/3`, status
`PASS`.
Go, Julia, and Lean are not listed as free-boundary parity surfaces here because
their current native packages do not expose equivalent coil Green-function,
limiter, axis, or X-point reconstruction logic.

GEQDSK-to-native configuration now preserves free-boundary geometry metadata:
`GEqdsk.to_config()` exports the parsed plasma boundary as isoflux target
points at `psi_boundary`, carries limiter points, and records magnetic-axis
metadata. A local SPARC `lmode_vv.geqdsk` conversion exported `177` boundary
points and `178` limiter points. This wires public EFIT contours into native
free-boundary workflows; it is still not a full free-boundary reconstruction
pass because GEQDSK does not include external coil currents.

Interpretation: the 33x33 Rust full-order equilibrium path is sub-millisecond
and therefore competitive for low-resolution control-support updates and
surrogate calibration loops. It is not the same benchmark as the reduced-order
Rust flight simulator, and it is not EFIT-grade reconstruction parity evidence.

### Rust `fusion-math` SOR kernel source-convention benchmark

Local run on 2026-05-25 after aligning the Rust `fusion-math` SOR, multigrid,
and GMRES kernels to the Python native convention `Delta*psi = source`:

| Benchmark | Physics scope | Grid | Criterion centre estimate |
|-----------|---------------|------|---------------------------|
| `sor_step_33x33` | One red-black cylindrical GS SOR sweep | 33x33 | 3.37 us |
| `sor_solve_33x33_500iter` | 500 red-black cylindrical GS SOR sweeps | 33x33 | 1.59 ms |
| `sor_solve_65x65_500iter` | 500 red-black cylindrical GS SOR sweeps | 65x65 | 6.62 ms |
| `sor_residual_65x65` | Cylindrical GS residual evaluation | 65x65 | 10.53 us |

Verification paired with this benchmark:
`cargo test -p fusion-math` passed unit tests, property tests, and doc-tests;
`python -m pytest tests/test_fusion_kernel_solver_mixins.py -q` passed the
Python counterpart fixed-point contracts.

## Benchmark Environment

All timing numbers in this document were measured on the following hardware
unless otherwise noted:

| Parameter | Value |
|-----------|-------|
| **CPU** | AMD Ryzen 9 7950X (16C/32T, Zen 4, 4.5 GHz base / 5.7 GHz boost) |
| **RAM** | 64 GB DDR5-5200 |
| **OS** | Ubuntu 22.04 LTS (kernel 6.5) / Windows 11 23H2 |
| **Rust** | stable 1.82+ (`opt-level = 3`, fat LTO, single codegen unit) |
| **Python** | 3.12 with NumPy 1.26, SciPy 1.12 |

> **Note:** CI benchmarks run on GitHub Actions `ubuntu-latest` shared runners
> (2-core, ~7 GB RAM) which are ~3-5x slower than the reference hardware above.
> For authoritative numbers, run benchmarks locally.

If you publish results from this code, please include your hardware specs
alongside the numbers.

## Running Benchmarks

```bash
# Rust solver benchmarks (Criterion — outputs JSON to target/criterion/)
cd scpn-fusion-rs
cargo bench

# Collect raw Criterion results as JSON
benchmarks/collect_results.sh

# Python profiling
python profiling/profile_kernel.py --top 50
python profiling/profile_geometry_3d.py --toroidal 48 --poloidal 48 --top 50

# Python validation suite
python validation/validate_against_sparc.py
python validation/benchmark_transport_power_balance.py

# Full 26-mode regression
scpn-fusion all --surrogate --experimental
```

After running `cargo bench`, raw Criterion data is stored in
`scpn-fusion-rs/target/criterion/` as JSON with statistical analysis
(mean, median, std dev, confidence intervals). Use the
`benchmarks/collect_results.sh` script to copy results into a timestamped
directory with hardware metadata.

## Reproducing

All benchmark data is generated by CI on every push to `main`. See the
`rust-benchmarks` and `validation-regression` jobs in
`.github/workflows/ci.yml`. CI uploads benchmark artifacts for download
from the workflow run page.

To reproduce locally:

```bash
git clone https://github.com/anulum/scpn-fusion-core.git
cd scpn-fusion-core
pip install -e ".[dev]"
cd scpn-fusion-rs && cargo bench && cd ..
python validation/validate_against_sparc.py
python validation/rmse_dashboard.py --output-json artifacts/rmse.json
python validation/benchmark_transport_power_balance.py
```

### Native Grad-Shafranov operator/current closure schema

The `gs_operator_current_closure` benchmark is now reported as schema `gs-operator-current-closure.v2` with benchmark scope `native_grad_shafranov_operator_current_closure`. The report distinguishes the manufactured full-order Grad-Shafranov operator/current relation from free-boundary reconstruction and reduced-order surrogate timing. Its machine-readable `gate_summary` fails closed over case thresholds, second-order radial convergence, and total-current closure stability.
