<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Benchmark Comparison -->

# SCPN Fusion Core — Benchmark Comparison


## Benchmark purpose

This comparison page is the contract for reproducible solver and runtime evidence. It separates accepted measurements from blocked lanes so external reference-parity requirements are not silently promoted.

## Evidence role

`BENCHMARKS.md` is the project-level acceptance index for cross-module quality.
It is intended for external review before any public full-fidelity claim is
presented and is the entry point for checking whether each solver lane is
production-open or still blocked due to missing parity inputs.

Comparison of SCPN Fusion Core against established fusion simulation codes.

> **Transparency note:** Timings labelled "Rust" use the compiled Rust backend
> with `opt-level = 3` and fat LTO. Timings labelled "Python" use the pure
> NumPy/SciPy path. "Projected" values are estimates, not measurements.
> Community code timings are from published literature (see references below).
> We encourage independent reproduction — see [`benchmarks/`](../benchmarks/).

## Current full-fidelity reproducibility status

The tracked production-parity campaign is intentionally fail-closed. Local
contracts are actionable, but the solver is not marked full-fidelity until
same-case external reference artefacts and quantitative comparisons exist.

| Lane | Current status | Reproducibility command |
|---|---|---|
| GENE/CGYRO/GS2 nonlinear GK parity | Blocked: missing redistribution-permitted same-deck nonlinear external outputs and native same-case comparisons; the report now exposes both a fail-closed roadmap evidence-surface matrix and an evidence-package matrix covering manifest fields, public provenance/license, source checksums, converted JSON/NPZ artefacts, metadata checksums, native thresholds, grid convergence, and scaling so absent GENE, CGYRO, or GS2 outputs cannot be promoted to parity evidence | `python tools/gk_external_output_parity.py` |
| Full electromagnetic / Maxwell fidelity | Blocked: compact `A_parallel`/`B_parallel` closure, local source-free Faraday/Ampere-Maxwell evolution, native same-case EM replay thresholds, local compact-EM grid-convergence evidence, and the EM evidence-gate matrix pass; sourced 5D kinetic-current coupling and external same-deck EM parity remain missing | `python validation/benchmark_gk_electromagnetic_fidelity.py` |
| Production-scale decomposition | Blocked: deterministic radial/toroidal decomposition, reciprocal rank-neighbour graph checks, rank communication contracts, local halo-face integrity, executable local rank-tile reductions, local process-isolated CPU execution, optional real local 2D MPI face-and-corner halo execution, optional CUDA rank-tile reductions, local large-grid CPU evidence over `9,437,184` phase cells, declared distributed halo-volume accounting, explicit distributed scaling gate, and distributed-run acceptance manifest pass; cluster MPI and multi-GPU scaling evidence is missing | `python validation/benchmark_production_decomposition_contract.py` |
| DREAM-grade runaway electrons | Blocked: public DREAM settings deck evidence plus native source-term budget diagnostics exist; PETSc/compiled `dreami` backend output and same-case source-budget parity are missing | `python tools/run_dream_reference_artifact.py --no-execute-backend` |
| Aurora/STRAHL-grade impurities | Partially accepted: Aurora/Open-ADAS argon artefact, coefficient sidecars, finite-volume radial-transport budget diagnostics, native same-case effective source/recycling closure, and time-resolved source-sink matrix parity pass density, radiation, inventory, particle-conservation, and source-sink thresholds; independent mechanistic recycling validation remains missing | `python tools/run_aurora_reference_artifact.py` |
| Free-boundary equilibrium strict parity | Accepted locally: dedicated strict-parity gate consumes the FreeGS public-example reconstruction and machine-metadata reports; same-case nonlinear output, native profile-source comparison, strict thresholds, geometry containment, grid convergence, public coil/vacuum sidecars, and same-case public reference output all pass | `python validation/benchmark_free_boundary_strict_parity.py --strict` |
| SAS dataset readiness | Blocked: SAS now holds public or locally authorised source snapshots and reference inputs under `DATASETS/SCPN-FUSION-CORE`, but missing same-deck external parity outputs remain explicit blocked rows rather than accepted evidence | `python validation/benchmark_sas_dataset_manifest.py` |

Source acquisition and conversion commands:

```bash
python tools/download_full_fidelity_public_sources.py --allow-failures
python tools/convert_full_fidelity_reference_artifacts.py --check
python validation/benchmark_sas_dataset_manifest.py
python validation/benchmark_full_fidelity_acceptance.py
python validation/full_fidelity_end_to_end_campaign.py
```

Published reports must retain blocker statuses when external artefacts are
missing. Do not substitute synthetic, reduced-order, or partial diagnostic
outputs for accepted full-fidelity parity evidence.

## FRC rigid-rotor no-rotation analytical benchmark

The accepted FRC analytical lane is benchmarked separately from Grad-Shafranov,
gyrokinetic, and free-boundary evidence. It covers the Steinhauer no-rotation
axial-field contract only:

```bash
PYTHONPATH=src python benchmarks/bench_frc_rigid_rotor.py
```

Tracked report: [`validation/reports/frc_rigid_rotor_benchmark.json`](../validation/reports/frc_rigid_rotor_benchmark.json)

The tracked report is local regression evidence unless its
`benchmark_evidence.classification` states otherwise. Current committed FRC
timing rows record command, CPU affinity, and host-load context, but they are
not isolated-core production throughput claims.

The report compares Python NumPy, Rust `fusion-physics`, and optional PyO3
surfaces on `65`, `129`, `257`, and `513` point radial grids using null radius,
configured separatrix target, separatrix radius error, field reversal,
Steinhauer Eq. 27 S-parameter, energy, local pressure balance, thermal-pressure
consistency, force-balance, and weighted numerical checksums for `B_z`,
`J_theta`, `psi`, `psi_N`, and pressure. It also records peak toroidal current density,
analytical flux-primitive derivative residuals, pressure-balance residuals,
analytical pressure-gradient residuals, Ampere closure residuals, `psi_N`
axis/separatrix closure, `psi_N` monotonicity and bounds gates, separatrix pressure-energy inventory,
magnetic-deficit inventory, energy-closure relative error, separatrix
current-sheet field-gradient/current-density closure, resolved sheet-current
integral closure, and a finite-grid convergence block for null radius,
separatrix radius error, Eq. 27 `s`, energy
per metre, pressure-balance ratio, pressure-balance residual,
analytical pressure-gradient residual, `psi_N` closure, flux derivative residual, current-sheet closure, and the independent
Ampere residual against the finest tracked grid. It also records a deterministic
16-case MIF/FRC no-rotation parameter cohort that spans accepted layer
thicknesses, external axial fields, separatrix radii, and grid sizes across
Python, Rust `fusion-physics`, and optional PyO3. Go, Julia, and Lean are
recorded as `not_applicable_no_frc_surface` until those languages expose
equivalent solver logic. Nonzero-rotation FRC cases remain fail-closed and are
not benchmarked as accepted physics.

## MIF/FRC Faraday recovery benchmark

The FUS-C.7 recovery lane is benchmarked separately from external Slough
acceptance evidence:

```bash
PYTHONPATH=src python benchmarks/bench_faraday_recovery.py
```

Tracked report:
[`validation/reports/faraday_recovery_benchmark.json`](../validation/reports/faraday_recovery_benchmark.json)

The report records local non-isolated regression rows for the exact classical
Faraday relation over supplied trajectories. It now also includes internal
FUS-C.6 supplied-current and voltage-driven pulsed-compression sidecar rows:
each row converts compression states to `(t, R_s, B_ext, dR_s/dt)`, carries the
final `compression_work_J`, and evaluates the energy-budget gate as `passed` or
`failed` instead of marking the work sidecar missing. Voltage-driven rows also
carry final coil-circuit `source_work_J` and evaluate `source_budget_claim_status`
separately. FUS-C.6 coupled rows now also carry the upstream compression
flux-budget sidecar and publish `compression_flux_budget_claim_status`, while
plain supplied-trajectory rows remain
`blocked_missing_compression_flux_budget` and
`blocked_missing_compression_trajectory_diagnostics`.

Schema `scpn-fusion-core.faraday_recovery_benchmark.v6` also records the
sampled Faraday-law closure diagnostic
`finite_difference(Phi) + EMF/N_turns = 0`. Python rows publish
`flux_derivative_residual_linf`, `flux_derivative_residual_l2`, and
`flux_derivative_closure_passed`, plus maximum field-ramp, radial-motion, and
total flux-rate terms; Rust Criterion rows publish
`flux_derivative_closure_status` and `flux_rate_term_status`. Analytical
supplied-trajectory Rust timing rows assert closure before emitting Criterion
estimates; coupled FUS-C.6 rows compute the diagnostic without upgrading it to
an external acceptance claim.
FUS-C.6 coupled rows now also carry the validated compression trajectory
diagnostic sidecar: minimum radius, compression ratio, maximum absolute radial
acceleration, radius-floor contact count, radial turning-point count, and
all-flux-budgets-passed status. Rust coupled Criterion rows assert the same
trajectory diagnostics inside the native harness.

External Slough same-case parity remains blocked until a public digitised
trajectory, compression-work sidecar, and compatible upstream flux-budget and
trajectory-quality evidence are available with provenance and checksums.

The public C-2U positive-net-heating table from Baltz et al., Scientific
Reports 7, 6425 (2017), is tracked in
`validation/reference_data/frc_public/c2u_optometrist_positive_heating_shots.csv`
with metadata and loader tests. It is a public FRC performance reference, not a
time-resolved pulsed-compression trajectory benchmark and not a substitute for
Slough Fig. 5 parity.

## Type-checking non-regression gate

The Python CI preflight runs a MyPy expansion guard before strict MyPy:

```bash
python tools/mypy_expansion_guard.py
python tools/run_mypy_strict.py
```

The guard compares `[tool.mypy]` in `pyproject.toml` against
`tools/mypy_expansion_baseline.json`. It permits the typed file cohort to grow,
but fails closed if a previously typed file is removed, `strict = true` is
disabled, global `ignore_missing_imports = false` is loosened, an exclude rule
hides files from the typed cohort, configured typed files disappear, or a new
`ignore_missing_imports` override appears without an explicit baseline update.
This prevents untyped code creep while the strict MyPy cohort expands
incrementally across source, validation, benchmark, and test surfaces.

## Provider-neutral cloud benchmark bundles

The tracked cloud runners are provider-neutral execution bundles. They write
timestamped output under `benchmark_runs/<RUN_ID>/`, record per-command logs,
snapshot generated reports, and archive the full artifact tree as
`benchmark_runs/<RUN_ID>.tar.gz`. Operational launchers, credentials, provider
account identifiers, and live instance IDs are internal-only material and must
not be committed to public paths.

Focused H-infinity/control diagnostic bundle:

```bash
RUN_ID=control_diag_$(date -u +%Y%m%dT%H%M%SZ) \
STRESS_EPISODES=20 \
STRESS_SHOT_DURATION=5 \
STRESS_CONTROLLERS='PID,H-infinity,LQR' \
bash scripts/cloud/gpu_benchmark_bundle.sh
```

Full-fidelity diagnostic bundle:

```bash
RUN_ID=full_fidelity_diag_$(date -u +%Y%m%dT%H%M%SZ) \
BENCHMARK_PROFILE=diagnostic \
bash scripts/cloud/full_fidelity_benchmark_bundle.sh
```

Native solver diagnostic bundle:

```bash
RUN_ID=solver_diag_$(date -u +%Y%m%dT%H%M%SZ) \
SOLVER_PROFILE=diagnostic \
bash scripts/cloud/solver_benchmark_bundle.sh
```

These bundles are reproducibility harnesses, not claim promotion mechanisms.
Benchmark numbers become public claims only after the archived artifacts are
retrieved, checksummed, reviewed for failed or blocked rows, and copied into
tracked reports with matching documentation updates.

Distributed production-decomposition measurements can be supplied to
`validation/benchmark_production_decomposition_contract.py` with
`SCPN_PRODUCTION_DECOMPOSITION_DISTRIBUTED_RUNS_JSON=/path/to/runs.json`. The
sidecar must be a JSON list whose rows include rank count, wall time, parallel
and weak-scaling efficiencies, owned phase cells per rank, halo bytes per step,
decomposition-invariant pass status, hardware metadata, command, and artifact
checksum. Missing or incomplete rows remain blocked.

The optional runtime dependency contract pins base NumPy below 2 and gates MPI/GPU lanes through `mpi4py>=4.1`, `cupy-cuda12x>=13.6,<14.0`, and `nvidia-cuda-nvrtc-cu12>=12.0,<13.0` so accelerator setup does not destabilise the base test environment.

Latest local large-grid CPU decomposition evidence is tracked in [`validation/reports/production_decomposition_contract.md`](../validation/reports/production_decomposition_contract.md): `large_cpu_96x48_6x4` executed `9,437,184` 5D phase cells over `24` local rank tiles in `1.557183 s` (`6.060419e6` cells/s) with zero reconstruction error and invariant relative errors below `1e-12`. This is single-process CPU evidence only; it does not satisfy the distributed MPI or multi-GPU scaling requirement.

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

WGPU benchmark rows are accepted as GPU evidence only when the adapter reports
`DiscreteGpu` or `IntegratedGpu`. Software Vulkan adapters such as `llvmpipe`
are rejected by default and require `SCPN_FUSION_GPU_ALLOW_CPU_ADAPTER=1` only
for local development smoke tests. A 2026-06-01 JarvisLabs L4 probe with driver
`580.126.20` exposed CUDA through `nvidia-smi`, but WGPU/Vulkan enumerated only
`llvmpipe`; its WGPU timings are therefore blocked as CPU-backed software
adapter evidence, not published GPU throughput.

### WGPU versus CUDA/JAX backend alternatives

The fail-closed dual-backend benchmark measures the two GPU paths separately:

```bash
python validation/benchmark_gpu_backend_alternatives.py
```

Cloud wrapper:

```bash
scripts/cloud/measure_gpu_backend_alternatives.sh RUN_ID=gpu_backend_alternatives_$(date -u +%Y%m%dT%H%M%SZ)
```

The report writes `validation/reports/gpu_backend_alternatives.json` and
`validation/reports/gpu_backend_alternatives.md` for local tracked runs, or
`benchmark_runs/<RUN_ID>/reports/` for cloud runs. A WGPU lane is publishable
only when the Vulkan/WGPU device is physical. A CUDA/JAX lane is publishable
only when JAX reports a CUDA device and the deterministic kernel result carries
a SHA-256 checksum. The two lanes are never merged into a generic GPU result.
The 2026-06-01 JarvisLabs L4 cloud report is tracked in [`validation/reports/gpu_backend_alternatives_jarvis_l4.md`](../validation/reports/gpu_backend_alternatives_jarvis_l4.md). It reports CUDA/JAX `passed` on `cuda:0` with median `8.9e-05 s` for the deterministic `256x256` JAX workload and WGPU `blocked_cpu_adapter` because Vulkan/WGPU exposed only `llvmpipe`, not the physical NVIDIA L4.

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
| `geqdsk_flux_profile_interpolation/second_order_33x33` | `17.266 us` |
| `geqdsk_flux_profile_interpolation/current_conserving_33x33` | `46.203 us` |
| `geqdsk_flux_profile_interpolation/second_order_65x65` | `59.550 us` |
| `geqdsk_flux_profile_interpolation/current_conserving_65x65` | `180.66 us` |
| `geqdsk_profile_source_components/source_components_33x33` | `113.78 us` |
| `geqdsk_profile_source_components/source_components_65x65` | `461.19 us` |
| `geqdsk_source_convention_adapter/select_adapter_33x33` | `45.674 us` |
| `geqdsk_source_convention_adapter/select_adapter_65x65` | `193.43 us` |
| `geqdsk_operator_current_domains/full_domain_current_33x33` | `5.6648 us` |
| `geqdsk_operator_current_domains/plasma_domain_current_33x33` | `6.6529 us` |
| `geqdsk_operator_current_domains/trapezoidal_full_domain_current_33x33` | `6.6539 us` |
| `geqdsk_operator_current_domains/full_domain_current_65x65` | `24.067 us` |
| `geqdsk_operator_current_domains/plasma_domain_current_65x65` | `27.543 us` |
| `geqdsk_operator_current_domains/trapezoidal_full_domain_current_65x65` | `29.637 us` |
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
| GEQDSK source-domain action attribution | FAIL | Aggregate report counts and failure reasons show `4/8` public rows require free-boundary coil/vacuum reconstruction directly and `4/8` require profile-source repair before free-boundary reconstruction |
| GEQDSK debug trace coverage | PASS | Every benchmark row records `attribute`, `normalise`, `solve`, `residual`, `classify`, and `blockers` stages plus first-blocker aggregate counts |
| Native operator/current closure | PASS | radial convergence order `2.000000`, worst radial current closure `8.31e-16` |

The raw profile-source and strict FreeGS failures are open benchmark blockers.
Accepted named-adapter rows are no longer counted as unresolved profile-source
blockers, but they remain blocked on free-boundary coil/vacuum reconstruction.
These are not CI or harness failures and must not be hidden by fallback rows.

### Full-fidelity public source acquisition

Public upstream source snapshots needed for the full-fidelity parity campaign
are cached under gitignored `data/external/full_fidelity_public_sources/`.
The tracked provenance report is
[`validation/reports/full_fidelity_public_source_downloads.md`](../validation/reports/full_fidelity_public_source_downloads.md),
with machine-readable checksums and revisions in
[`validation/reports/full_fidelity_public_source_downloads.json`](../validation/reports/full_fidelity_public_source_downloads.json).

The current acquisition covers GENE public pages, CGYRO/GACODE, GS2, DREAM,
Aurora, FreeGS, and FreeGSNKE. These raw snapshots are not benchmark parity
artifacts; production parity still requires schema-valid JSON/NPZ reference
artifacts, licenses, thresholds, observables, and solver-output comparisons.
The nonlinear GK external-output gate additionally requires a complete
`gk_external_nonlinear_full_fidelity_evidence_package_v1` evidence package:
every GENE, CGYRO, and GS2 row must provide a same-deck manifest entry, public
provenance, redistribution license, source checksum, converted artefact
checksum, metadata checksum, native same-case threshold evaluation,
grid-convergence row, and production-scaling row before readiness can pass.
Grid-convergence rows are accepted only when the linked observable reports
`relative_l2 <= 0.15` on a strict fine-grid refinement. Production-scaling rows
are accepted only when the linked device/grid/rank/timing record is finite,
positive, covers at least `64` phase cells, and reports
`wall_time_s <= 86400`; row presence alone is diagnostic-only.

The SAS dataset readiness gate is tracked in
[`validation/reports/sas_dataset_readiness.md`](../validation/reports/sas_dataset_readiness.md).
It reads `DATASETS/SCPN-FUSION-CORE/manifests/dataset_manifest.json`
from the shared data volume by auto-discovery, or
`SCPN_FUSION_DATASET_ROOT/manifests/dataset_manifest.json` when an alternate
dataset root is provided, plus its checksum sidecar. Public source trees, web snapshots, Aurora-bundled
ADAS data, FreeGS metadata, and repository-curated reference inputs are
accepted only as acquisition evidence. Missing GENE/CGYRO/GS2 same-deck
outputs, electromagnetic outputs, DREAM production outputs, Aurora/STRAHL
transport outputs, public coil-current sidecars, and restricted facility raw
data remain blocked rows.

The public-output conversion pass is tracked in
[`validation/reports/full_fidelity_reference_artifact_conversion.md`](../validation/reports/full_fidelity_reference_artifact_conversion.md).
It exports one accepted external reference artefact and three finite,
checksummed partial artefacts:

- Aurora argon transport output to
  `validation/reference_data/full_fidelity_public_artifacts/aurora_argon_transport_public.npz`

- DREAM avalanche HDF5 data to
  `validation/reference_data/full_fidelity_public_artifacts/dream_avalanche_public_raw.npz`
- FreeGSNKE static inverse baseline arrays to
  `validation/reference_data/full_fidelity_public_artifacts/freegsnke_static_inverse_baseline_public.npz`
- FreeGSNKE MAST-U-like coil-current sidecars to
  `validation/reference_data/full_fidelity_public_artifacts/freegsnke_mastu_current_sidecars_public.json`

Accepted external reference artefacts are now `1`: the Aurora argon transport
payload satisfies the manifest observable, coordinate, provenance, checksum,
and redistribution contract. The native same-case Aurora comparison now runs
on the tracked artefact axes and validates the declared threshold checks, but
it fails all four current thresholds with relative mismatch `1.0`; the lane is
therefore `blocked_native_aurora_same_case_threshold_mismatch`, not full
Aurora/STRAHL parity. The DREAM and FreeGSNKE payloads remain partial
provenance-backed public outputs and conversion smoke tests, not accepted
parity evidence. Clean checkouts without the gitignored external cache use the
tracked artefacts as a provenance-preserving fallback so CI does not erase
public-output evidence.

The Aurora execution lane is tracked in
[`validation/reports/aurora_reference_execution_artifact.md`](../validation/reports/aurora_reference_execution_artifact.md).
It runs the cached Aurora/Open-ADAS argon atomic-data path and exports
`validation/reference_data/full_fidelity_public_artifacts/aurora_argon_fractional_abundance_public.npz`
with source-data checksums and metadata. This is an ADAS-backed fractional
abundance artifact, not a full Aurora/STRAHL radial transport parity result.

The nonlinear GK deck inventory is tracked in
[`validation/reports/gk_public_reference_deck_inventory.md`](../validation/reports/gk_public_reference_deck_inventory.md).
It indexes public GS2 nonlinear input decks, CGYRO nonlinear input decks, CGYRO
regression precision output snippets, and public GENE/GS2/CGYRO web-source
hashes into
`validation/reference_data/full_fidelity_public_artifacts/gk_public_reference_deck_inventory.json`.
The same artifact now publishes a GENE/CGYRO/GS2 public-output candidate
matrix. The matrix is intentionally fail-closed: public decks, web pages, and
CGYRO precision snippets are acquisition candidates only, while accepted
nonlinear parity requires schema-valid same-deck output payloads with the full
distribution, heat-flux, field-energy, zonal/saturation, convergence, scaling,
and native-comparison contract.
On this runner it records `40` public decks and `21` CGYRO precision-output
summaries, but GS2 is not installed and the cached CGYRO wrapper lacks the
GACODE runtime helper. This inventory is therefore a reproducibility input, not
a full nonlinear 5D solver-output parity result.
The downstream external-output parity manifest is fail-closed by solver
family: accepted readiness requires one redistribution-permitted GENE, CGYRO,
and GS2 nonlinear output row sharing the same `benchmark_case_id` and
`deck_physics_sha256`, plus grid-convergence and production-scaling evidence
for every required family. Candidate output rows with private `file://`
provenance, placeholder provenance, unknown licensing, proprietary licensing,
or explicitly non-redistributable licensing are blocked before artifact
conversion. Native same-case comparison rows are also checksum-gated with
`native_output_sha256` before threshold evaluation.
For top-level NPZ payloads, the converter classifies keys by the declared
coordinate and observable contracts before metadata publication.
Grid-convergence and production-scaling rows must link to the converted
same-case `case_id` for each solver family.
Those rows now publish their own evidence matrices and acceptance thresholds:
grid convergence is capped at `relative_l2 <= 0.15`, while scaling requires
finite positive rank/grid/timing metadata, at least `64` phase cells, and
`wall_time_s <= 86400`.

The DREAM execution lane is tracked in
[`validation/reports/dream_reference_execution_request.md`](../validation/reports/dream_reference_execution_request.md).
It generates the public `examples/2kinetic/dream_settings.h5` deck from the
cached DREAM source when available; clean checkouts preserve the committed
settings-deck checksum as tracked evidence when the external cache is absent.
The report records a fail-closed backend blocker on this machine: PETSc and the
compiled DREAM `iface/dreami` executable are not installed. Once a runner
provides that backend, rerun
`uv run --no-sync python tools/run_dream_reference_artifact.py` to execute the
same deck and produce a candidate output HDF5 for conversion.

The production-scale decomposition contract is tracked in
[`validation/reports/production_decomposition_contract.md`](../validation/reports/production_decomposition_contract.md).
It validates deterministic radial/toroidal tiling for 5D nonlinear GK storage.
Current local rows include a `256 x 128 x 32 x 32 x 16` phase-space case split
over `8 x 4` rank tiles with exact owned-cell balance and halo overhead
reported. The local CPU gate also runs serial reference halo exchange,
owned-state reconstruction, and decomposition-invariant inventory/free-energy
checks on reproducible 5D phase-space payloads across multiple decomposition
shapes. The local invariant surface now also preserves a normalized-`vpar`
parallel-flow moment across decomposed rank reductions. The tracked report now
includes per-rank halo-face integrity evidence for radial/toroidal face
payloads, plus same-physics shape-convergence evidence across `4x2`, `8x1`,
and `2x4` radial/toroidal rank shapes with maximum inventory relative deviation
`0.0`, maximum free-energy relative deviation `3.3306658974988877e-16`,
maximum parallel-moment relative deviation `0.0`, and maximum owned-state
reconstruction error `0.0`. This is still not distributed runtime evidence;
MPI or multi-GPU execution, cluster timing, and hardware-specific scaling
thresholds remain required.

The free-boundary public machine-metadata inventory is tracked in
[`validation/reports/free_boundary_public_machine_metadata_inventory.md`](../validation/reports/free_boundary_public_machine_metadata_inventory.md).
It indexes cached FreeGSNKE machine configuration metadata for active coils,
passive structures, limiter/wall contours, and magnetic probes, with checksums
and guarded geometry summaries, plus FreeGS example-script checksums. This is a
reconstruction input inventory only; strict parity remains blocked until those
metadata are linked to same-case public equilibria, native coil/vacuum
reconstruction outputs, and FreeGS/FreeGSNKE solver-output comparisons.

The FreeGS public-example reconstruction attempt is tracked in
[`validation/reports/freegs_public_example_reconstruction.md`](../validation/reports/freegs_public_example_reconstruction.md).
It reconstructs the public FreeGS example machine coils after their control
constraints and compares native Green-function vacuum flux against FreeGS on
the same sample points. Current local rows pass the vacuum convention check and
the recorded Picard sweep produces finite external `psi(R,Z)` output for both
public examples. The same report now runs a native fixed-boundary profile-source
comparison on the finite FreeGS psi grid and publishes `psi_N` RMSE, magnetic
axis error, boundary error, sampled X-point constraint error, and current
closure, plus finite signed-q profile sanity from the solved public FreeGS
equilibrium. It also publishes machine-readable geometry-containment evidence
for source X-points, isoflux endpoints, native/external magnetic axes, and
boundary-containment metric readiness, plus per-case threshold checks,
failed-check counts, and readiness booleans for strict threshold acceptance,
grid convergence, public coil/vacuum sidecars, and same-case public reference
output. Grid-convergence evidence is explicit and fail-closed: the
public-example report records the `33x33`, `65x65`, and `129x129` ladder and
requires monotone non-increasing residual metrics for both public cases. Clean
CI checkouts preserve tracked machine-metadata, SAS-readiness, and
reconstruction reports when gitignored external caches are absent, so the
integrated campaign cannot erase prior public evidence during full-suite test
order.

The dedicated strict gate is tracked in
[`validation/reports/free_boundary_strict_parity_benchmark.md`](../validation/reports/free_boundary_strict_parity_benchmark.md).
It consumes the FreeGS public-example reconstruction report and public machine
metadata inventory, then accepts full-fidelity free-boundary parity only if
same-case external output, native profile-source comparison, strict thresholds,
geometry containment, grid convergence, public coil/vacuum sidecars, and
same-case public reference output are all ready. Current local result is
`accepted_full_fidelity_free_boundary_parity` with zero blockers and zero failed
threshold checks.

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
| SPARC | 8 GEQDSK/EQDSK | B=12.2 T, I_p=8.7 MA | Operator-source public rows pass; profile-source/free-boundary gate remains open | ψ NRMSE, axis metadata, boundary containment, signed-q finite profile |
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
Each benchmark row also emits a deterministic GEQDSK debug trace over
`attribute`, `normalise`, `solve`, `residual`, `classify`, and `blockers`
stages. The aggregate report counts the first blocking stage so source-unit
normalisation failures, reconstruction failures, residual-budget failures, and
unclassified rows can be triaged without row-by-row log inspection.
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
Accepted GEQDSK source-convention adapters now also report adapted-source
plasma/vacuum residuals in the EFIT benchmark rows, so a passing global adapter
cannot hide whether the residual is concentrated inside the plasma domain or in
the vacuum/source-free domain.
The aggregate report now classifies the effective source residual domain after
adapter selection. Current local refresh: `14/18` rows are
`plasma_and_vacuum_source_mismatch`, while the `4/18` accepted-adapter rows are
`vacuum_source_free_operator_residual`, meaning the named adapter aligns the
plasma-domain source but leaves a vacuum/free-boundary operator residual that
profile-only fixed-boundary reconstruction does not model.
Rust `fusion-core::source` now also exposes native second-order and current-conserving flux-profile interpolation helpers matching the Python profile-source construction contract: local quadratic GEQDSK profile interpolation, masked weighted-integral preservation against the linear GEQDSK contract, finite input guards, shape guards, and non-negative masked weight enforcement.
It also exposes `compute_geqdsk_profile_source_components`, which assembles pressure, FFprime, and total Grad-Shafranov source arrays with plasma-mask reporting, explicit boundary zeroing, source-norm diagnostics, and signed source-sum diagnostics matching the Python profile-source path. The signed sums are retained because source norms alone can hide pressure/FFprime sign regressions or cancelling assembly errors, and the EFIT/GEQDSK benchmark schema now requires those signed sums in every row.
The aggregate EFIT/GEQDSK report also gates the signed identity
`total_source_sum == pressure_source_sum + ffprime_source_sum` through
`source_sum_identity_max_abs_error` and `source_sum_identity_pass`.
Rows that require free-boundary reconstruction now also report whether the
GEQDSK file contains boundary, limiter, and magnetic-axis metadata and whether
external coil currents are available. Current local public SPARC rows carry
usable boundary/limiter/axis metadata, but all `8/8` public rows remain blocked
for native free-boundary reconstruction by
`external_coil_currents_missing_from_geqdsk`; GEQDSK alone is therefore not a
complete coil/vacuum reconstruction input.
The reconstruction contract now accepts an explicit
`external-coil-sidecar.v1` companion input with Ampere currents, metre
positions, positive turn counts, current-limit checks, finite-value guards,
unique coil names, strict unit labels, and provenance. GEQDSK-only public rows
still report `free_boundary_external_coil_sidecar_present = false` and
`free_boundary_external_coil_count = 0`; a row is allowed to report
`ready_for_free_boundary_reconstruction` only after valid sidecar data is
supplied.
The same aggregate report counts operator-current closure with
`operator_current_closure_pass_count` and
`gate_operator_current_closure_pass_count`, so the discrete Delta*psi current
closure row contract cannot silently regress while profile-source RMSE remains
unchanged.
Current local refresh: operator-current closure passes `5/18` aggregate rows
and `5/8` public gate rows. That is now exposed as an aggregate failure reason
instead of being buried in per-row diagnostics.
The report also identifies the worst operator and profile current-closure rows
through `operator_current_worst_relative_error`,
`gate_operator_current_worst_relative_error`, and
`profile_current_worst_relative_error`, so closure severity is visible without
manual row scanning.
Operator-current closure now carries domain attribution: every EFIT/GEQDSK row
reports full computational-domain current, plasma-domain current, trapezoidal
full-domain current, and the best-domain current residual through
`operator_current_best_domain`,
`operator_current_best_relative_error`, and aggregate best-domain pass counts.
The original `operator_current_closure_pass` remains the strict full-domain
contract, so plasma-domain near-closure is evidence rather than a hidden
threshold relaxation.
Rust parity now includes native masked current integration through
`fusion_core::kernel::total_toroidal_current_from_flux_masked` and
`fusion_polyglot::total_toroidal_current_from_flux_masked`, allowing the same
full-domain/plasma-domain current comparison outside Python without wrapper
delegation.
Rust also exposes
`fusion_core::kernel::total_toroidal_current_from_flux_trapezoidal` and
`fusion_polyglot::total_toroidal_current_from_flux_trapezoidal`, matching the
Python full-domain trapezoidal diagnostic. Current local GEQDSK refresh shows
this does not close the public current failures: best-domain operator-current
closure remains `5/8` public rows, so the remaining failures are source-domain
and free-boundary reconstruction blockers rather than a full-domain quadrature
artefact.
The same masked current-domain contract is also exposed natively in Go, Julia,
and Lean through their existing Grad-Shafranov operator-current implementations;
their native trapezoidal current integrations mirror the same diagnostic
surface. These are implementation parity surfaces, not wrappers around Python
or Rust.
Current local worst rows: operator current closure `sparc/sparc_1300.eqdsk`
with relative error `2.184689e+00`, and profile current closure
`jet/jet_lmode_2MA.geqdsk` with relative error `4.168623e+01`.
Profile-current closure is also threshold-counted with
`profile_current_closure_threshold`, `profile_current_closure_pass_count`, and
`gate_profile_current_closure_pass_count`; rows above the 5% current-closure
threshold are now aggregate gate failures rather than visual-only diagnostics.
Current local profile-current closure result: `0/18` aggregate rows and `0/8`
public gate rows pass the 5% threshold. Each report row now carries `operator_current_ratio_to_declared`,
`profile_current_ratio_to_declared`, `adapted_profile_current_ratio_to_declared`,
`pressure_current_ratio_to_declared`, `ffprime_current_ratio_to_declared`,
`profile_current_closure_pass`, `adapted_profile_current_closure_pass`,
`current_limited_adapted_profile_pass`, `effective_profile_current_closure_pass`, and
`profile_current_closure_failure_class` so downstream gates can distinguish
finite profile-current diagnostics from rows that satisfy the closure threshold
and can identify whether the pressure or FFprime source dominates an under- or
over-closed declared plasma current. The aggregate report also includes raw, adapted, current-limited adapted, and effective profile-current
closure pass counts plus `profile_current_closure_failure_class_counts`, so the
failure mode distribution is visible without row scanning. This is intentionally exposed as a public-gate benchmark failure because raw
profile-source reconstruction still has unresolved convention/source mismatch.
Diagnostic-only synthetic rows remain visible in counts and row reports, but do
not decide public gate failure reasons.
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
| Multi-profile replay | ITER-like, DIII-D-like, and compact-tokamak reduced-order plant profiles must pass and emit `accepted_reduced_order_replay_release_gate` | `python validation/vertical_control_replay_benchmark.py --all-profiles --strict` |
| Uncertainty envelope | Growth, damping, actuator, sensor-bias, and one-step latency perturbations are replayed | same |

### Benchmark Regression Gate

CI now runs a dedicated benchmark regression guard after benchmark artefact
generation in `.github/workflows/ci.yml` `Benchmark Provenance Smoke`.

The guard is configured by `tools/benchmark_regression_thresholds.json` and
executed with:

```bash
python tools/benchmark_regression_guard.py \
  --thresholds tools/benchmark_regression_thresholds.json \
  --summary-json artifacts/benchmark_regression_guard_summary.json
```

It fails closed when any required benchmark artefact is missing, any configured
metric path is absent, any boolean acceptance flag flips, or any numeric metric
crosses its configured bound. The current CI gate covers:

The threshold file is schema-versioned as
`benchmark-regression-thresholds.v2`. The guard rejects duplicate report IDs,
duplicate metric paths, non-finite numeric bounds, invalid `min`/`max` ranges,
and metrics that mix exact equality checks with numeric bounds. Report rows may
also set `expected_schema`, `expected_benchmark_id`, and `max_age_seconds`.
CI-generated benchmark reports use these identity and freshness checks so a
wrong, stale, local, or cached artefact cannot satisfy the gate only because it
contains similarly named metric fields.

- single-profile vertical-control replay acceptance and deterministic replay
- vertical-control profile-suite acceptance across the configured machine profiles
- disruption transfer-generalisation recall, false-positive rate, and transfer efficiency
- fallback-budget guard acceptance

This is separate from benchmark publication. The guard prevents silent
regression of already-generated benchmark artefacts; it does not convert
diagnostic-only rows into full-fidelity parity evidence.

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
- report review of generated Markdown artifacts under `validation/reports/vertical_control_replay_benchmark.md` and `validation/reports/vertical_control_replay_profiles.md`

The latest profile-suite JSON now includes an explicit `release_gate` object.
When it reports `accepted_reduced_order_replay_release_gate`, the accepted claim
is still limited to deterministic reduced-order RZIP replay. The JSON also
keeps `full_pcs_production_grade_ready == false`; this lane is not a full PCS
production-control claim.

### GPU Phase 1 Readiness Gate

The GPU Phase 1 gate verifies the Rust/wgpu SOR implementation surfaces and
then blocks until a tracked hardware benchmark artifact is available:

```bash
python validation/benchmark_gpu_phase1_readiness.py
```

The gate reports static implementation readiness separately from
`production_scaling_ready`, which remains `false` until current GPU benchmark
artifacts include device metadata, solver identity, and output checksums.
The artefact must also prove a physical WGPU adapter (`DiscreteGpu` or
`IntegratedGpu`) or a CUDA/JAX lane with a detected CUDA device and result
checksum; CPU software adapters do not satisfy the gate.

### Free-boundary Strict Acceptance Matrix

The strict free-boundary gate now emits both an acceptance contract and an
acceptance matrix:

```bash
python validation/benchmark_free_boundary_strict_parity.py
```

The gate accepts only when same-case public reference output, native
profile-source comparison, strict threshold metrics, grid convergence,
coil/vacuum sidecars, and machine metadata are all present. Missing rows remain
blocked rather than inferred.

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
moment residuals, kx/ky heat-flux spectra, and the nonlinear E x B invariant
diagnostic. The invariant requires zero high-k leakage outside the 2/3
dealiased spectral mask and no free-energy injection from the undriven
collisionless nonlinear bracket. The native run result now exports those
invariant histories plus zonal-flow energy histories so saturation windows can
be audited against the same discrete nonlinear-operator contract used by the
benchmark.

The native runaway-electron surface now exports a DREAM-style
`time_s x radius_m x momentum_mec x pitch_cosine` artifact contract from the
1D momentum Fokker-Planck kernel, with `f_p_xi_t`, runaway-current,
avalanche-growth, synchrotron-loss, partial-screening-drag, and bremsstrahlung
observables. The runaway benchmark now validates those axes, shapes,
non-negativity, finiteness, and a deterministic artifact checksum. It also
exports fail-closed native kinetic-operator evidence: momentum advection,
diffusion, Dreicer source, avalanche growth, and synchrotron force evidence are
present in the native path. The report also publishes native-only source-term
budget diagnostics for avalanche growth, synchrotron loss, partial-screening
drag, and bremsstrahlung loss channels, plus bounded pitch-cosine moment
evidence over the exported artifact pitch axis. Full pitch-angle scattering, radial
transport, DREAM partial-screening, DREAM bremsstrahlung-loss parity,
same-case distribution/current/growth/source-budget thresholds, and coupled
momentum-pitch-radius operator parity remain blocked. This is a reference-gate
artifact contract only; it does not replace public DREAM deck ingestion or
full momentum-pitch-radius kinetic operator parity.

The native impurity surface now keeps two explicit paths: the general impurity
transport solver used for ongoing native-model development, and a separate
`AuroraParityImpuritySolver` used only for Aurora/STRAHL same-case threshold
gates. The parity solver is a native implementation, not an Aurora wrapper; it
consumes an Aurora-compatible case contract with `time_s`, `radius_m`,
`charge_state`, tracked density/temperature profiles, diffusion/convection
tables, Open-ADAS-derived ionisation, recombination, and line-radiation
coefficient tables, and the optional `effective_source_m3_s_t_r_z` closure
sidecar. The closure sidecar is derived as the residual density-rate needed
after the native finite-volume transport and neighbouring charge-state CR
predictor to reproduce the public Aurora density trajectory. It is diagnostic
same-case source/recycling closure, not a mechanistic Aurora/STRAHL source or
recycling model.

The native impurity surface also exports an Aurora/STRAHL-style
`time_s x radius_m x charge_state` artifact contract with total impurity density
closure, line-radiation power, and finite ionisation/recombination source-sink
matrices. The artifact validates `time_s x radius_m x charge_state` densities,
`time_s x radius_m x charge_state x charge_state` conservative transfer
matrices, per-charge line-radiation power, inventory history, finite shapes, and
a deterministic checksum. The benchmark also publishes native-only source/sink
budget diagnostics for conservative charge-state transfer matrices,
ionisation/recombination source budgets, line-radiation power, and inventory
history.

The accepted Aurora argon transport artefact is used by the separate
Aurora-compatible native parity solver with matching `time_s x radius_m x
charge_state` axes. That comparator is structurally ready, checksum-gated, and
now consumes the tracked density/temperature profiles, diffusion and convection
tables, Open-ADAS-derived ionisation/recombination/radiation coefficient tables,
and effective source/recycling closure sidecar. The current same-case report passes charge-state-density, total-density,
radiated-power, inventory, particle-conservation, and time-resolved source-sink
matrix thresholds against the public Aurora artefact. The report also exposes an
explicit finite-volume radial-transport budget diagnostic on evolved
charge-state density, so the native radial operator is no longer reported as
absent. Full impurity parity remains blocked because the density threshold pass
still depends on a residual effective closure and independent mechanistic
recycling validation beyond effective closure replay is not yet available.

Latest local results are written to
`validation/reports/gk_nonlinear_solver_comparison.md`.

## Full-Fidelity Acceptance Contract

`validation/benchmark_full_fidelity_acceptance.py` is a fail-closed diagnostic
for native full-order claims across the three remaining high-fidelity physics
frontiers:

- Nonlinear gyrokinetics must demonstrate public GENE/CGYRO/GS2 parity for a
  full nonlinear 5D Vlasov-Maxwell campaign.
- Runaway electrons must demonstrate DREAM kinetic/fluid parity beyond the
  current scalar balance, 1D momentum Fokker-Planck kernel, and
  multidimensional DREAM-style artifact-export contract.
- Impurity transport must demonstrate Aurora/STRAHL collisional-operator parity
  beyond the current trace radial transport, charge-state artifact/source-sink
  conservation contract, and public reference-case manifest.

The native nonlinear GK surface now exposes an explicit
`species x kx x ky x theta x vpar x mu` phase-space contract plus a named
conservative pseudo-spectral ExB term with de-aliasing diagnostics. The
run results now export the actual `kx`, `ky`, `theta`, `vpar`, and `mu`
coordinate axes used by the solver, so saved spectra carry machine-readable
grid metadata for future GENE/CGYRO/GS2 artifact comparison. Public GS2/CGYRO
deck hashes are now tracked, but external nonlinear outputs and native same-case
comparisons remain required. The electromagnetic case now also reports compact
Ampere `A_parallel` and perpendicular pressure-balance `B_parallel` residual
histories through `validation/reports/gk_electromagnetic_fidelity.md`; these
prove internal algebraic-closure consistency and are gated separately from the
local source-free Maxwell evolution evidence. The report includes a
machine-readable Maxwell equation contract, explicit electromagnetic evidence-gate
matrix, blocked sourced-Maxwell contract, time-resolved 5D current/charge moment histories, spectral continuity-proxy evidence, and native J_parallel E_parallel exchange diagnostics, and explicit sourced-field evolution blockers for dE_parallel/dt, sourced Faraday, sourced Ampere-Maxwell, and field-particle energy balance while self-consistent sourced field coupling remains blocked,
and native source-free spectral Faraday, Ampere-Maxwell displacement-current,
inductive parallel electric-field
evolution, and perpendicular magnetic-divergence diagnostics. Full
Vlasov-Maxwell parity remains blocked until those fields are coupled to
self-consistent 5D kinetic current moments and same-deck electromagnetic
GENE/CGYRO/GS2 outputs. The electromagnetic gate now publishes a fail-closed
external-EM parity evidence matrix for GENE, CGYRO, and GS2 over the required
`phi`, `A_parallel`, `B_parallel`, heat-flux, and nonlinear-distribution
observables; every row remains blocked until same-deck external artefacts,
native same-case comparisons, and external grid-convergence evidence exist.
The latest local Maxwell evolution evidence reports
maximum relative total-field-energy drift `5.090958569120036e-16` with zero
Faraday, Ampere-Maxwell, inductive parallel electric-field, and magnetic
divergence residuals under tolerance `1.0e-12`. Native same-case EM replay
thresholds now gate `phi`, `A_parallel`, `B_parallel`, and total field-energy
histories with maximum absolute and relative error `0.0` under absolute
tolerance `1.0e-18` and relative tolerance `1.0e-15`; this is native
deterministic replay evidence, not external-code parity. The same report now
includes local compact-EM grid-convergence evidence for algebraic
field-energy histories over
`4x4x8x5x4`, `6x6x10x5x4`, and `8x8x12x5x4` retained
`kx x ky x theta x vpar x mu` grids. The latest local benchmark status is
`accepted_local_compact_em_grid_convergence`, with maximum relative
total-energy drift `5.494182e-03` under tolerance `5.0e-01`; this is still
local compact-closure evidence, not external Vlasov-Maxwell parity. The result surface
also exposes a JSON-compatible reference-artifact export with coordinates,
units, observable axes, heat-flux spectra, particle/field energy spectra, and
saturation diagnostics. The acceptance harness includes a quantitative
reference-artifact comparator for declared `absolute_error`, `relative_error`,
and `relative_l2` thresholds; it fails closed on missing observables, non-finite
payloads, unsupported contracts, and shape mismatches. The transport
diagnostics now include saved kx/ky ion and electron heat-flux spectra that
close exactly to the scalar flux histories and saved electrostatic zonal-flow
energy bounded by the total electrostatic field energy. Late-window
saturation summaries now report scalar `phi` RMS, averaged ion/electron
heat-flux spectra, zonal-flow energy, and electromagnetic energy components
from the same saved histories used by the benchmark. The electromagnetic state
contract now carries `phi`, `A_parallel`, and
`B_parallel` field components, and `B_parallel` now enters the Hamiltonian
gradient drive through the magnetic-moment compression term. The energy
diagnostics now account for particle free energy and electromagnetic field
energy separately before reporting total energy, and `run()` exports particle,
`phi`, `A_parallel`, `B_parallel`, and total-energy histories for saturation
and invariant analysis. The same NumPy and JAX run paths now also export
species-resolved particle free-energy spectra plus electromagnetic `phi`,
`A_parallel`, and `B_parallel` field-energy spectra over the retained kx/ky
grid; each spectrum closes to the corresponding scalar component energy, and
late-window saturated spectra are reported from those saved histories. It also
exports nonlinear ExB free-energy production, relative production, dealiased
high-k leakage, and per-save invariant-pass histories from both NumPy and JAX
run paths. This is necessary infrastructure for full nonlinear 5D parity, but
it is not sufficient to claim GENE/CGYRO/GS2 equivalence.

### Nonlinear GK parity item 1 execution checklist

- Current fail state: all GENE/CGYRO/GS2 external-output rows remain blocked on
  `blocked_missing_external_output_manifest` because same-deck external payloads
  are not yet redistributable.
- Required acceptance evidence for this lane:
  - strict `gk-nonlinear-external-output` payloads (coordinates, observables,
    redistributable provenance, and checksums)
  - native same-case comparison rows with thresholds on distribution RMSE,
    heat-flux spectra RMSE, field-energy history RMSE, and saturation/zonal-flow
    metrics
  - linked coarse-vs-fine grid-convergence row
  - linked production-scale row with hardware/device metadata
- Evidence refresh sequence (run after any new raw output):
  - `python tools/inventory_gk_public_reference_decks.py`
  - `python tools/gk_external_output_parity.py`
  - `python validation/benchmark_full_fidelity_acceptance.py`
- Internal acquisition plan is tracked in
  `docs/internal/gk_same_deck_external_output_acquisition_todo_2026-06-01.md`.

The current acceptance report is
`validation/reports/full_fidelity_acceptance_benchmark.md`. The integrated
end-to-end campaign report is
`validation/reports/full_fidelity_end_to_end_campaign.md`; it keeps the
GENE/CGYRO/GS2, full Maxwell/EM, production-scale decomposition, DREAM,
Aurora/STRAHL, and free-boundary blockers in one fail-closed gate. Public
source acquisition targets are declared in
`validation/reference_data/full_fidelity_public_sources.json`. Required public
reference artefacts and quantitative thresholds are declared in
`validation/reference_data/full_fidelity_reference_cases.json`; accepted
artefacts must also satisfy
`validation/reference_data/full_fidelity_artifact_schema.json` with provenance,
redistribution/license status, checksum, required observable keys, numeric
finite payload contracts, unit-labelled observable and coordinate/grid
contracts, observable-to-coordinate axis contracts, and explicit quantitative
threshold contracts linked to declared observables and supported metric
families. For multi-solver nonlinear GK, the manifest must also prove
same-deck identity across GENE, CGYRO, and GS2 and provide per-family
convergence/scaling evidence. Candidate artifacts must carry public HTTP(S)
provenance and an explicit redistribution-permitted license before conversion.
Native same-case comparison artefacts must carry a matching checksum before any
threshold can be marked ready.
For NPZ inputs, top-level keys are separated into coordinate and observable
metadata by the declared contracts before converted artefacts are reported.
Grid-convergence and production-scale evidence rows are accepted only when they
reference the converted same-case output row for the corresponding solver
family.
The benchmark intentionally does not pass
full-fidelity acceptance until those public reference gates exist and their
artefacts are present.
For nonlinear GK, the native artifact keeps the full species/kx/ky/theta/vpar/mu
distribution grid and serializes the complex spectral state as required real
and imaginary distribution components; it does not collapse the comparison to a
magnitude, heat-flux-only, or saturation-only diagnostic. Electromagnetic GK is
gated separately from electrostatic GK and now carries local source-free
Faraday/Ampere-Maxwell evolution evidence plus native deterministic same-case
thresholds while still requiring phi, `A_parallel`, and `B_parallel`
field-energy histories for future GENE/CGYRO/GS2 same-case parity.
The production-scale lane now distinguishes a passing decomposition contract
and executable local rank-tile reductions, including inventory, free-energy,
and normalized parallel-moment invariants, from actual distributed runtime
readiness: `production_scale_ready` remains false until MPI or multi-GPU
execution and scaling reports exist.

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
pass because GEQDSK does not include external coil currents. Public acquisition
materials for FreeGS and NSTX-U/MDSplus have been cached locally under ignored
`data/external/`; direct NSTX-U MDSplus and Princeton DataSpace payload access
were blocked from this workstation, so no public coil-current array has been
accepted into the benchmark gate yet.

Interpretation: the 33x33 Rust full-order equilibrium path is sub-millisecond
and therefore competitive for low-resolution control-support updates and
surrogate calibration loops. It is not the same benchmark as the reduced-order
Rust flight simulator, and it is not EFIT-grade reconstruction parity evidence.

### Rust `fusion-math` SOR kernel source-convention benchmark

### FRC rigid-rotor analytical benchmark

The tracked FRC benchmark report is `frc_rigid_rotor_no_rotation_analytical`.
It covers the accepted Steinhauer no-rotation axial-field contract only. The
input deck is pressure matched so `n0 * (T_i + T_e) * e` equals the
magnetic-pressure-balance peak for the configured `B_ext`. Reported scalar
diagnostics include null radius, separatrix error, Eq. 27 `s`, energy per
metre, pressure-balance ratio, pressure residual, analytical pressure-gradient residual,
solved peak density, input central density, central-density residual, central-density relative error,
beta peak, separatrix-averaged beta, particle line density, input thermal
pressure, separatrix pressure-energy inventory, magnetic-deficit inventory,
energy-closure relative error, separatrix field-gradient/current-density
closure, thermal-pressure ratio, flux residual, Ampere residual,
force-balance residual, and weighted checksums for `B_z`, `J_theta`, `psi`,
pressure, density, and beta. Python, Rust, and PyO3 rows must agree on the
same pressure-density-beta-energy-current-sheet closure contract; Go, Julia, and Lean remain
`not_applicable_no_frc_surface` until equivalent native FRC solver logic exists.
The same tracked report now includes a deterministic 16-case MIF/FRC
no-rotation parameter cohort for Python, Rust `fusion-physics`, and PyO3
checksum parity. These rows are parameter-contract evidence, not rotating-BVP
or kinetic-transport validation.

Reproduce locally:

```bash
PYTHONPATH=src python benchmarks/bench_frc_rigid_rotor.py
```

### FRC pulsed Hall-MHD flux-carrier benchmark

The tracked FUS-C.2 benchmark report is
`validation/reports/hall_mhd_pulsed_benchmark.json`. It covers the accepted
axisymmetric Ono Eq. 8 flux carrier:

$$\partial_t\psi=-\psi/\tau_\psi+R_{\rm null}E_\theta-\eta J_\theta.$$

The report includes Python timing rows, Rust Criterion rows, source checksums,
and blocked external-reference rows for `gkeyll_axisymmetric_small_hall` and
`ono_1997_fig4_flux_decay`. It does not claim full 2D two-fluid Hall-MHD,
Gkeyll/BOUT++ same-case parity, WGPU execution, or Ono figure reproduction.

Latest local non-isolated regression rows:

| Row | Grid | Steps | Mean seconds |
|---|---:|---:|---:|
| Python `python_64_grid_256_steps` | 64 | 256 | 0.0528046082 |
| Python `python_256_grid_256_steps` | 256 | 256 | 0.0620382902 |
| Python `python_1024_grid_256_steps` | 1,024 | 256 | 0.0689889970 |
| Rust `rust_64_grid_256_steps` | 64 | 256 | 0.0002328276 |
| Rust `rust_256_grid_256_steps` | 256 | 256 | 0.0008544832 |
| Rust `rust_1024_grid_256_steps` | 1,024 | 256 | 0.0031026612 |

Reproduce locally:

```bash
cargo bench --manifest-path scpn-fusion-rs/Cargo.toml -p fusion-physics --bench hall_mhd_pulsed_bench -- --sample-size 10
PYTHONPATH=src python benchmarks/bench_hall_mhd_pulsed.py
```

### FRC non-adiabatic current-diffusion carrier benchmark

The tracked FUS-C.3 benchmark report is
`validation/reports/current_diffusion_nonadiabatic_benchmark.json`. It covers
the accepted one-dimensional Ono-style carrier used by downstream pulsed
compression:

$$\partial_t\psi=-\psi/\tau_\psi+R_{\rm null}E_\theta-\eta J_\theta.$$

The report includes Python timing rows, Rust Criterion rows from
`fusion-core::current_diffusion`, and the local discrete update residual for
the exact source/damping budget
`psi[n+1] = psi[n] - damping_decrement[n] + source_increment[n]`. The report is
local non-isolated regression evidence, not production throughput evidence and
not Ono Fig. 4 same-case parity.

Reproduce locally:

```bash
cargo bench --manifest-path scpn-fusion-rs/Cargo.toml -p fusion-core --bench current_diffusion_nonadiabatic_bench -- --sample-size 10
PYTHONPATH=src python benchmarks/bench_current_diffusion_nonadiabatic.py
```

### FRC n=1 tilt-mode diagnostic benchmark

The tracked FUS-C.5 benchmark report is
`validation/reports/tilt_mode_frc_benchmark.json`. It covers the conservative
MHD Alfvén-time diagnostic and the FUS-C.6 supplied-current trajectory adapter:

$$\gamma_{\rm tilt}=C V_A/(E R_s).$$

The coupled adapter uses the self-similar projection
`s(t)=s0*(R/R0)*(B/B0)*sqrt(T_i0/T_i)` and recomputes growth from the
instantaneous compression state. The report includes Python timing rows, Rust
Criterion rows, FUS-C.6 coupled trajectory rows, source checksums, and an
external-reference row for `belova_2001_table1_tilt_stability` with status
`blocked_missing_public_digitised_reference`. It does not claim full Belova
hybrid eigenvalue parity or Table I reproduction.
Schema `scpn-fusion-core.tilt_mode_frc_benchmark.v4` also records the
FUS-C.6 trajectory growth exposure `G(t)=integral(gamma_tilt dt)`, finite
perturbation amplification, and overflow-limited flag for Python coupled rows.
Rust coupled Criterion rows publish `growth_integral_status` because the
timing harness asserts finite cumulative growth and amplification before
emitting estimates.

Latest local non-isolated regression rows:

| Row | Workload | Mean seconds |
|---|---:|---:|
| Python `python_1000_reports` | 1,000 reports | 0.0155345608 |
| Python `python_10000_reports` | 10,000 reports | 0.1576017428 |
| Python `python_100000_reports` | 100,000 reports | 1.4554153114 |
| Python `python_fus_c6_coupled_64_intervals` | 64 intervals / 65 states | 0.0012223596 |
| Python `python_fus_c6_coupled_256_intervals` | 256 intervals / 257 states | 0.0055735386 |
| Rust `rust_1000_reports` | 1,000 reports | 0.0000290052 |
| Rust `rust_10000_reports` | 10,000 reports | 0.0002804347 |
| Rust `rust_100000_reports` | 100,000 reports | 0.0027860003 |
| Rust `rust_fus_c6_coupled_64_intervals` | 64 intervals / 65 states | 0.0000030257 |
| Rust `rust_fus_c6_coupled_256_intervals` | 256 intervals / 257 states | 0.0000126976 |

The interval count excludes the initial state; each coupled row evaluates
`intervals + 1` compression states. Use the tracked JSON for exact local values
because these workstation rows are non-isolated regression evidence.

Reproduce locally:

```bash
cargo bench --manifest-path scpn-fusion-rs/Cargo.toml -p fusion-physics --bench tilt_mode_frc_bench -- --sample-size 10
PYTHONPATH=src python benchmarks/bench_tilt_mode_frc.py
```

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

## MIF/FRC pulsed compression

Tracked report:
[`validation/reports/pulsed_compression_benchmark.json`](../validation/reports/pulsed_compression_benchmark.json)

The report records local non-isolated regression rows for the FUS-C.6
pulsed-compression path. Supplied-current rows exercise the pressure-balance
trajectory, adiabatic heating, compression-work sidecar, and Ono
non-adiabatic flux-carrier coupling. The Python rows now publish the carrier
source-increment checksum, damping-decrement checksum, maximum absolute update
residual, `flux_budget_claim_status`, final force-balance
`radial_acceleration_m_s2`, the maximum absolute trajectory acceleration,
minimum radius, compression ratio, radius-floor contact count, radial
turning-point count, and an all-flux-budgets-passed diagnostic; Rust Criterion
rows assert the same budget and trajectory gates inside the native benchmark
harness. Voltage-driven rows add the exact lumped R-L coil-current contract over
the declared bank-voltage limit and record the coil-circuit energy residual
before feeding the same compression path.

Schema `scpn-fusion-core.pulsed_compression_benchmark.v4` records these
trajectory diagnostics for Python supplied-current and voltage-driven rows.
Rust rows publish `trajectory_diagnostics_status` because the native Criterion
harness asserts the trajectory gates before emitting estimates.

This is not a production throughput claim and not external Slough Fig. 5
parity. The Slough row remains blocked until a public digitised trajectory with
compatible radius, temperature, field, current, provenance, and checksum data
is available.

The C-2U positive-net-heating supplementary table is available as public FRC
performance context under `validation/reference_data/frc_public/`. It is not
used to pass this pulsed-compression benchmark because it does not provide the
time-resolved Slough radius, temperature, and field trajectory required by the
acceptance row.

## MIF/FRC MRTI growth spectrum

Tracked report:
[`validation/reports/mrti_benchmark.json`](../validation/reports/mrti_benchmark.json)

The report records local non-isolated regression rows for the analytical
MRTI growth-rate and spectrum tracker. It also includes internal FUS-C.6
supplied-current pulsed-compression rows: each row consumes the real
`PulsedCompressionState` history, projects the explicit FUS-C.6
`radial_acceleration_m_s2` sidecar into the MRTI interface-normal convention,
and advances the MRTI spectrum over the compression intervals.
Schema `scpn-fusion-core.mrti_benchmark.v3` records log-amplitude evolution
diagnostics for the tracker. Python rows publish `max_log_amplitude` and
`amplitude_overflow_limited`, and coupled rows publish the acceleration source
plus final and maximum effective acceleration; Rust Criterion rows publish
`log_amplitude_status` because the timing harness asserts finite physical and
log-amplitude state before emitting Criterion estimates.

This is same-codebase coupling evidence, not external nonlinear MRTI parity.
Redistributable pulsed-power MRTI image/diagnostic references remain blocked
until public same-case outputs with provenance and license terms are available.

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

CI generates regression and provenance artefacts on every push to `main`. See
the `rust-benchmarks` and `validation-regression` jobs in
`.github/workflows/ci.yml`. External-source caches, cloud-GPU runs, and
production solver backends are not regenerated on every CI run; their tracked
reports preserve provenance, checksums, hardware metadata, and fail-closed
blocker status.

To reproduce locally:

```bash
git clone https://github.com/anulum/scpn-fusion-core.git
cd scpn-fusion-core
pip install -e ".[dev]"
cd scpn-fusion-rs && cargo bench && cd ..
python validation/validate_against_sparc.py
python validation/rmse_dashboard.py --output-json artifacts/rmse.json
python validation/benchmark_transport_power_balance.py
python validation/benchmark_full_fidelity_acceptance.py
python validation/full_fidelity_end_to_end_campaign.py
```

### Native Grad-Shafranov operator/current closure schema

The `gs_operator_current_closure` benchmark is now reported as schema `gs-operator-current-closure.v2` with benchmark scope `native_grad_shafranov_operator_current_closure`. The report distinguishes the manufactured full-order Grad-Shafranov operator/current relation from free-boundary reconstruction and reduced-order surrogate timing. Its machine-readable `gate_summary` fails closed over case thresholds, second-order radial convergence, and total-current closure stability.
