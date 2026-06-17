# SCPN Fusion Core

**Evidence-bounded neuro-symbolic tokamak control, native plasma-solver research, and fail-closed fusion-validation infrastructure for control, equilibrium, transport, gyrokinetic, runaway-electron, impurity, and free-boundary campaigns.**

<p align="center">
  <img src="docs/assets/repo_header.png" alt="SCPN Fusion Core -- Neuro-Symbolic Tokamak Control">
</p>

[![CI](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/ci.yml)
[![Docs](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml/badge.svg)](https://github.com/anulum/scpn-fusion-core/actions/workflows/docs.yml)
[![Coverage](https://codecov.io/gh/anulum/scpn-fusion-core/branch/main/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-fusion-core)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://anulum.github.io/scpn-fusion-core/)
[![PyPI](https://img.shields.io/pypi/v/scpn-fusion)](https://pypi.org/project/scpn-fusion/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-fusion.svg)](https://pypi.org/project/scpn-fusion/)
[![All-time Downloads](https://static.pepy.tech/badge/scpn-fusion)](https://pepy.tech/project/scpn-fusion)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18820864.svg)](https://doi.org/10.5281/zenodo.18820864)
[![License](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Commercial License](https://img.shields.io/badge/Commercial_license-available-success.svg)](mailto:protoscience@anulum.li?subject=SCPN%20Fusion%20Core%20Commercial%20License)
![Version](https://img.shields.io/badge/Version-3.9.10-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/anulum/scpn-fusion-core/badge)](https://scorecard.dev/viewer/?uri=github.com/anulum/scpn-fusion-core)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12163/badge)](https://www.bestpractices.dev/projects/12163)

## Dual Licensing

SCPN Fusion Core uses a deliberate dual-licensing model:

- **Open-source path:** AGPL-3.0-or-later for research, review, education,
  reproducible validation, public derivative work, and network-deployed AGPL
  services.
- **Commercial path:** separate commercial licences are available for
  proprietary reactor programs, internal deployments, closed operational
  workflows, commercial support, and integrations that cannot use AGPL
  reciprocal terms.

Commercial licensing contact:
[protoscience@anulum.li](mailto:protoscience@anulum.li?subject=SCPN%20Fusion%20Core%20Commercial%20License).

**Financing the full-fidelity parity campaign:** the public GitHub Pages landing page is [anulum.github.io/scpn-fusion-core](https://anulum.github.io/scpn-fusion-core/). It separates current evidence from blocked GENE/CGYRO/GS2, DREAM, Aurora/STRAHL, FreeGS, and production-scale GPU/cluster validation work.

## At a Glance

SCPN Fusion Core is a control-first fusion software laboratory. It helps teams
move from a control idea to a reproducible evidence package without pretending
that partial research kernels are already full production reference solvers.

| Question | Answer |
|---|---|
| What does it build? | Neuro-symbolic plasma-control loops, native solver kernels, validation reports, and benchmark artefacts. |
| Who is it for? | Fusion-control researchers, validation engineers, accelerator teams, formal-methods contributors, and investors evaluating reproducible fusion software infrastructure. |
| What is validated today? | Local controller contracts, reduced-order replay lanes, selected Grad-Shafranov/operator-source checks, native kernel benchmarks, formal proof slices, and fail-closed benchmark gates. |
| What remains blocked? | Full GENE/CGYRO/GS2 nonlinear turbulence parity, full Vlasov-Maxwell parity, DREAM kinetic parity, Aurora/STRAHL transport parity, strict FreeGS/free-boundary parity, and production MPI/multi-GPU scaling. |
| How should readers judge claims? | Follow the linked reports, commands, checksums, thresholds, and accepted/blocked row status. |
| Is it safety certified? | No. The repository publishes an [IEC 61508 functional-safety roadmap](docs/IEC_61508_ROADMAP.md) for selected control surfaces, but it does not claim IEC 61508 or SIL certification. |

## Why This Matters

Fusion control software needs three properties at once: physical realism,
real-time execution, and auditable safety boundaries. Most projects optimize one
of those surfaces and leave the others informal. SCPN Fusion Core makes the
interfaces explicit: controllers compile from inspectable Petri-net contracts,
solver lanes publish evidence rows, and incomplete parity evidence stays blocked
instead of being hidden in prose.

This makes the repository useful before plant deployment: it is a place to test
controller architectures, benchmark kernel choices, prepare same-case reference
solver campaigns, and build the documentation discipline needed for later
hardware-in-the-loop and safety-assurance work.

Most fusion codes are physics-first — solve equations, then bolt on control.
SCPN Fusion Core inverts this: **control-first**. Express plasma control logic
as stochastic Petri nets, compile to spiking neural networks, execute at
**10 kHz+** against physics-informed plant models. Pure Python with optional
Rust acceleration for reduced-order control kernels; latency claims are
metric-scoped and are not same-work Rust-versus-Python physics speedups.

> **Compact control package:** [`scpn-control`](https://github.com/anulum/scpn-control)
> is the smaller controller-facing package for Petri-net compilation, SNN
> control, NMPC, runtime contracts, differentiable tuning facades, and
> hardware-in-the-loop/replay surfaces. This repository remains the broader
> physics and research suite for solver development, Rust/polyglot kernels,
> validation campaigns, and full-stack plant modelling.

## Relationship to scpn-control

`scpn-fusion-core` is the canonical SCPN physics and solver laboratory. It
contains the wider equilibrium, transport, gyrokinetic, 3D, neural-surrogate,
Rust/polyglot, benchmark, and validation surfaces.

[`scpn-control`](https://github.com/anulum/scpn-control) is the compact
control-grade package derived from this stack. It focuses on installation,
controller-loop integration, fail-closed runtime boundaries, replay metadata,
NMPC, differentiable tuning facades, and hardware/control contracts.

The projects are developed in parallel: broad physics kernels mature here,
while `scpn-control` exposes the subset needed for control loops through stable
facades and evidence-bounded validation reports.

## Current Release Snapshot

Version `3.9.10` is a documentation, release-readiness, and FUS-C evidence
traceability update. The package now presents the recent FUS-C.6 pulsed
compression trajectory diagnostics, FUS-C.7 Faraday flux and trajectory-quality
gates, and FUS-C.4 MRTI acceleration coupling as current local contract
evidence across Python, Rust, benchmarks, and public method documentation.

This release does not promote the solver to completed end-to-end full-fidelity
parity. The full GENE/CGYRO/GS2, DREAM, Aurora/STRAHL, FreeGS, electromagnetic,
and distributed GPU/cluster lanes remain accepted only when their tracked rows
carry same-case external outputs, provenance, checksums, thresholds, grid or
scaling evidence, and native comparisons.

<!-- capability-snapshot:start -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Generated by tools/capability_manifest.py; do not edit counts by hand. -->

### SCPN Fusion Core Capability Inventory

| Surface | Current inventory |
|---|---:|
| Package version | 3.9.10 |
| Public API exports | 2 |
| Python capability source modules | 274 |
| Python capability classes | 534 |
| Capability documentation pages | 54 |
| Rust workspace crates | 13 |
| Optional extras | 12 |
| Python test files | 420 |
| Public documentation pages | 54 |
| GitHub Actions workflows | 12 |

Evidence boundary: this snapshot is a static inventory. Performance, coverage, hardware, and scientific-fidelity claims require their own committed evidence artifacts.
<!-- capability-snapshot:end -->

## What Is It?

SCPN Fusion Core is a research-grade software stack for building and
validating tokamak control algorithms against physics-informed plant models.
It combines four surfaces that are usually split across separate tools:

- **Control representation:** stochastic Petri nets, spiking neural network
  execution, classical controllers, NMPC, replay, and hardware-in-the-loop
  contracts.
- **Native physics kernels:** Grad-Shafranov equilibrium, radial transport,
  gyrokinetic research operators, electromagnetic diagnostics, runaway-electron
  and impurity contracts, and free-boundary validation harnesses.
- **Reference-code integration:** fail-closed adapters and benchmark manifests
  for GENE, CGYRO, GS2, DREAM, Aurora, STRAHL, FreeGS, GEQDSK, IMAS/OMAS, and
  public reference datasets.
- **Evidence publication:** tracked JSON/Markdown reports, checksum-backed
  artefact manifests, GitHub Pages documentation, notebooks, and release gates
  that separate local contracts from accepted production-parity evidence.
- **Safety-assurance roadmap:** an explicit
  [IEC 61508 functional-safety roadmap](docs/IEC_61508_ROADMAP.md) for future
  assessment of selected controller, replay, interlock, and telemetry surfaces.

The intended users are fusion-control researchers, simulation engineers,
validation teams, accelerator/GPU engineers, formal-methods contributors, and
industrial groups evaluating how real-time control software can be hardened
before plant deployment.

## Documentation Map

| Reader goal | Start here |
|---|---|
| Understand the product and evidence boundary | [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) |
| Install and run the first commands | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) |
| Find public APIs and extension points | [`docs/API_OVERVIEW.md`](docs/API_OVERVIEW.md) |
| Understand applications and market value | [`docs/APPLICATIONS_AND_MARKET.md`](docs/APPLICATIONS_AND_MARKET.md) |
| Reproduce benchmark claims | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`RESULTS.md`](RESULTS.md) |
| Track full-fidelity blockers | [`validation/reports/full_fidelity_end_to_end_campaign.md`](validation/reports/full_fidelity_end_to_end_campaign.md) |
| Use notebooks | [`docs/notebooks/README.md`](docs/notebooks/README.md) |
| Read generated Sphinx docs | [`docs/sphinx/index.rst`](docs/sphinx/index.rst) |

## Capability Surface

| Layer | Modules | Capability |
|-------|---------|-----------|
| **Core Physics** | 118 | Grad-Shafranov, transport (1.5D + QLKNN + FNO), GK three-path (native + 5 external codes), MHD stability (7 criteria), neoclassical, disruption chain, ELM/MARFE/L-H transition, runaway electrons, pellet injection, plasma-wall interaction, 3D equilibrium |
| **Control** | 54 | PID, H-infinity, NMPC-JAX, SNN (Petri net compiler), gain-scheduled, fault-tolerant, safe RL (PPO), free-boundary tracking, burn control, RZIP, RWM feedback, mu-synthesis, detachment, density, volt-second management, state estimation (EKF) |
| **Phase Dynamics** | 10 | Kuramoto UPDE solver, adaptive K_nm coupling, GK-to-UPDE bridge, plasma K_nm, Lyapunov guard, real-time monitoring, WebSocket streaming |
| **Diagnostics** | 5 | Synthetic sensors, tomographic inversion, forward models |
| **Engineering** | 4 | Balance of plant, CAD raytrace, thermal hydraulics |
| **Nuclear** | 5 | Blanket neutronics, PWI erosion, wall interaction |
| **SCPN Compiler** | 12 | Petri net structure, compiler, contracts, safety interlocks, artifact packaging |
| **I/O** | 15 | IMAS/OMAS adapter, GEQDSK, tokamak archive, logging |
| **Rust Backend** | 11 crates | GS kernel (0.52 us), transport, control, ML inference, PyO3 bindings |

The generated capability inventory above is the source of truth for public
package, test, documentation, workflow, and Rust workspace counts.

## Try in 45 Seconds

```bash
pip install -e .
scpn-fusion kernel          # Grad-Shafranov equilibrium
scpn-fusion flight          # Tokamak flight simulator
pytest tests/ -x -q          # release suite
```

```bash
python examples/minimal.py --grid 17 --equilibrium-iters 4
```

Or run the **Golden Base** hero notebook — formal proofs, closed-loop control,
shot replay, all in one:
[`examples/neuro_symbolic_control_demo_v2.ipynb`](examples/neuro_symbolic_control_demo_v2.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/scpn-fusion-core/blob/main/examples/neuro_symbolic_control_demo_v2.ipynb)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/anulum/scpn-fusion-core/main?labpath=examples%2Fneuro_symbolic_control_demo_v2.ipynb)

```bash
docker compose up --build    # Streamlit dashboard at localhost:8501
```

## Key Results

| Metric | Value | Reproducibility |
|--------|-------|-----------------|
| Rust PID kernel latency | **0.52 us P50** | `validation/verify_10khz_rust.py` |
| Closed-loop HIL latency | **10.5 us P50** | `python validation/collect_results.py` |
| Taskset-affinity CPU sensor-to-control latency | **0.053408 ms P95** on logical CPUs 10,11; not shielded cpuset evidence | `validation/reports/scpn_end_to_end_latency.md` records the exact command |
| Simulated 256-actuator HIL scaffold | **232.522 us P95** host ADC/DAC loop; not physical HIL rig timing | `python validation/scpn_end_to_end_latency.py --strict` |
| Rust full-order GS solve | **413 us** SOR / **845 us** multigrid (33x33 local Criterion; multigrid is slower in this low-grid case) | `cargo bench -p fusion-core --bench picard_bench` |
| Rust vacuum field solve | **140 us** (33x33) / **489 us** (65x65) | `cargo bench -p fusion-core --bench vacuum_bench` |
| QLKNN-10D transport surrogate | test rel_L2 = **0.094** | `weights/neural_transport_qlknn.metrics.json` |
| FNO turbulence surrogate | val rel_L2 = **0.055** | `weights/fno_turbulence_jax.metrics.json` |
| Disruption rate (1,000-shot sim campaign) | **0%** (Rust-PID) | `validation/stress_test_campaign.py` |
| ITPA H-mode confinement | 53 shots / 24 machines | `validation/reference_data/itpa/` |
| SPARC GEQDSK validation | 8 public EFIT equilibria; operator-source gate passes; row-level debug traces expose profile-source/free-boundary blockers | `validation/benchmark_sparc_geqdsk_rmse.py`, `validation/psi_pointwise_rmse.py` |
| Q >= 10 operating point | Q = 15 (0D power balance) | `RESULTS.md` |
| TBR | 1.14 (0D 3-group blanket) | `RESULTS.md` |
| Free-boundary equilibrium validation | Public operator-source GEQDSK gate passes; FreeGS public-example vacuum comparison passes; native same-case profile-source metrics, finite signed-q sanity, and machine-readable strict threshold checks are published; strict parity remains blocked on thresholds, grid convergence, and public coil/vacuum sidecars | `validation/benchmark_sparc_geqdsk_rmse.py`, `validation/psi_pointwise_rmse.py`, `validation/benchmark_freegs_public_example_reconstruction.py` |
| Full-fidelity end-to-end campaign | `not_full_fidelity`; local contracts ready, `0` accepted full-fidelity reference artefacts, and all six production-parity lanes remain fail-closed until external same-case evidence exists | `validation/reports/full_fidelity_end_to_end_campaign.md` |

Latency taxonomy: `control.pid_kernel_step_us` (Rust reduced-order kernel),
`control.closed_loop_step_us` (controller loop with explicit surrogate/full
physics mode), and `control.hil_loop_us` (hardware-in-the-loop integration
path). Current local end-to-end benchmark: PID surrogate p95 0.012 ms, PID
full-mode p95 0.047 ms; full definitions:
[`docs/PERFORMANCE_METRIC_TAXONOMY.md`](docs/PERFORMANCE_METRIC_TAXONOMY.md).
The 256-actuator HIL scaffold row is a measured host-side simulated ADC/DAC
contract, not a physical rig, FPGA bitstream, CODAC, or actuator-hardware
timing claim.
The current sensor-to-control timing artifact records a taskset-affinity run on
logical CPUs `10,11`, including command, affinity, host load before/after, CPU
governor/frequency context, and concurrent-heavy-job note. It is not a shielded
cpuset or dedicated-runner claim.

Full numbers: [`RESULTS.md`](RESULTS.md) — re-run `python validation/collect_results.py` to reproduce.

## Gyrokinetic Three-Path Architecture

SCPN exposes three gyrokinetic transport lanes with explicit fidelity limits:

| Path | Fidelity | Speed | Module |
|------|----------|-------|--------|
| **A: External GK** | Reference when installed | minutes to CPU-hours | `gk_tglf`, `gk_gene`, `gk_gs2`, `gk_cgyro`, `gk_qualikiz` |
| **B: Native GK** | Linear transport plus nonlinear 5D operator contracts | ~0.3 s/surface for linear eigenvalue; local nonlinear benchmark in `validation/reports/gk_nonlinear_solver_comparison.md` | `gk_eigenvalue`, `gk_quasilinear`, `gk_nonlinear` |
| **C: Hybrid Surrogate+GK** | Adaptive | ~24 ns/point | `gk_ood_detector` + `gk_corrector` + `gk_scheduler` + `gk_online_learner` |

The hybrid layer validates QLKNN surrogates against GK spot-checks in real time.
External GENE/GS2/CGYRO adapters now carry explicit linear versus nonlinear
electrostatic/electromagnetic model metadata and can emit nonlinear 5D deck
requests for installed production solvers; SCPN does not bundle or replace
those solvers.

## Evidence Boundary

At minimum, this is **not** a replacement for TRANSP, JINTRAC, or GENE; the same boundary applies to CGYRO, GS2, DREAM, Aurora, STRAHL, and EFIT. It is a
**control-algorithm development and validation framework** with explicit fidelity boundaries,
fast controller-support models, native research solver contracts, and
fail-closed production-parity gates. See
[`validation/reports/full_fidelity_end_to_end_campaign.md`](validation/reports/full_fidelity_end_to_end_campaign.md),
[`validation/reports/full_fidelity_acceptance_benchmark.md`](validation/reports/full_fidelity_acceptance_benchmark.md),
and `python tools/generate_claims_evidence_map.py --check` for public claim
evidence.

Top limitations:
- No GENE/CGYRO-class full nonlinear 5D turbulence campaign in-loop; native nonlinear GK is a bounded NumPy/JAX research solver with explicit invariant benchmarks.
- No full 3D nonlinear MHD stack in-loop (external coupling required for that fidelity).
- Free-boundary equilibrium/inverse reconstruction is not yet EFIT-grade; public SPARC GEQDSK operator-source rows pass, while profile-source/free-boundary reconstruction and FreeGS strict-backend parity remain open evidence gates with row-level debug traces.

Full-fidelity acceptance status for native nonlinear GK, runaway electrons, and
impurity transport is tracked by
[`validation/reports/full_fidelity_acceptance_benchmark.md`](validation/reports/full_fidelity_acceptance_benchmark.md).
The current diagnostic is fail-closed: these surfaces are not marked full-order
until public GENE/CGYRO/GS2, DREAM, and Aurora/STRAHL parity gates are met.
Required public artefacts and thresholds are declared in
[`validation/reference_data/full_fidelity_reference_cases.json`](validation/reference_data/full_fidelity_reference_cases.json);
accepted artefacts must also satisfy
[`validation/reference_data/full_fidelity_artifact_schema.json`](validation/reference_data/full_fidelity_artifact_schema.json).
The integrated six-lane campaign, including public source acquisition targets
and production-scale blockers, is tracked by
[`validation/reports/full_fidelity_end_to_end_campaign.md`](validation/reports/full_fidelity_end_to_end_campaign.md).
Public upstream source snapshots for GENE, CGYRO/GACODE, GS2, DREAM, Aurora,
FreeGS, and FreeGSNKE are cached under gitignored `data/external/` and
summarised in
[`validation/reports/full_fidelity_public_source_downloads.md`](validation/reports/full_fidelity_public_source_downloads.md).
Those snapshots are acquisition inputs only; they are not accepted parity
artefacts until converted into schema-valid repository-local JSON/NPZ evidence.
The public conversion pass exports three finite public payloads: DREAM
avalanche HDF5 data, FreeGSNKE static inverse baselines, and FreeGSNKE
MAST-U-like current sidecars into tracked artifacts with metadata and checksums:
[`validation/reports/full_fidelity_reference_artifact_conversion.md`](validation/reports/full_fidelity_reference_artifact_conversion.md).
The Aurora execution lane separately exports an Aurora/Open-ADAS
argon fractional-abundance artifact:
[`validation/reports/aurora_reference_execution_artifact.md`](validation/reports/aurora_reference_execution_artifact.md).
These are partial diagnostic artefacts, not full-fidelity acceptance artefacts,
because required transport observables and same-case solver-output comparisons
are still missing.
The DREAM execution lane now also generates the upstream `examples/2kinetic`
settings deck and records backend readiness in
[`validation/reports/dream_reference_execution_request.md`](validation/reports/dream_reference_execution_request.md).
On this local runner the deck generation succeeds; clean CI checkouts without
the gitignored external cache preserve the tracked settings-deck evidence
instead of rewriting the report as missing. DREAM execution remains blocked
until PETSc and the compiled `iface/dreami` backend are available. The native
runaway benchmark now also publishes fail-closed kinetic-operator evidence:
the exported `time_s x radius_m x momentum_mec x pitch_cosine` artifact is
finite and schema-valid, but radius and pitch are not yet evolved operator
axes, same-case DREAM thresholds are not ready, and full coupled
momentum-pitch-radius Fokker-Planck parity remains blocked. It also publishes
native-only source-term budget evidence for avalanche growth, synchrotron loss,
partial-screening drag, and bremsstrahlung loss channels; DREAM same-case
source-budget parity remains blocked until compiled `iface/dreami` output is
available.
The Aurora execution lane runs a cached Aurora/Open-ADAS atomic-data path and
exports normalized argon charge-state fractions, but remains blocked for full
Aurora/STRAHL parity until public radial transport output, source/sink matrices,
radiation observables, and native same-case comparisons are present. The native
artifact gate validates charge-state density, conservative source-sink transfer
matrices, per-charge line-radiation power, and total impurity inventory history
as local contracts only. The impurity benchmark now also publishes fail-closed
native transport-operator evidence: trace radial transport, edge-source
conservation, neoclassical pinch, charge-state source/sink matrices, line
radiation, and inventory closure are local evidence, while charge-state
resolved radial transport parity, external ADAS transport coefficients,
same-case Aurora/STRAHL outputs, and quantitative thresholds remain blocked.
It also publishes native-only source/sink budget evidence for conservative
charge-state transfer matrices, ionisation/recombination source budgets,
line-radiation power, and inventory history; Aurora/STRAHL same-case
source-budget parity remains blocked.
The nonlinear GK lane now indexes public GS2 nonlinear decks, CGYRO nonlinear
decks, CGYRO regression precision outputs, and GENE/GS2/CGYRO public web-source
hashes in
[`validation/reports/gk_public_reference_deck_inventory.md`](validation/reports/gk_public_reference_deck_inventory.md).
That inventory now publishes a per-solver public-output candidate matrix for
GENE, CGYRO, and GS2. It records deck/source candidates separately from
accepted nonlinear output artifacts and keeps every row blocked until the
required same-deck `gk-nonlinear-external-output.v1` payload exists with
distribution, heat-flux, field-energy, zonal/saturation, convergence, scaling,
and native comparison evidence.
It also has a strict external-output conversion/comparison contract in
[`validation/reports/gk_external_nonlinear_parity.md`](validation/reports/gk_external_nonlinear_parity.md).
Missing redistributable same-deck GENE, CGYRO, or GS2 nonlinear outputs produce
blocked rows rather than placeholder parity. Full GENE/CGYRO/GS2 parity remains
blocked until nonlinear distribution outputs with real and imaginary spectral
components, heat-flux spectra, phi/A_parallel/B_parallel field-energy
histories, zonal/saturation metrics, grid-convergence evidence,
production-scale scaling evidence, and native same-case comparisons are
present for the required solver families. The external-output manifest is
strictly same-deck: all three solver-family rows must share one
`benchmark_case_id` and one `deck_physics_sha256`, and convergence/scaling
evidence must cover GENE, CGYRO, and GS2 before readiness can pass. Candidate
external outputs are rejected before conversion if provenance is private or the
license is unknown, proprietary, restricted, or otherwise non-redistributable.
Native same-case comparison rows must also provide `native_output_sha256`
before quantitative thresholds are evaluated.
Top-level NPZ output keys are classified by declared coordinate and observable
contracts before converted artefact metadata is emitted.
Grid-convergence and production-scaling evidence must reference the converted
same-case solver-family output rows before those gates can pass.
Grid-convergence rows are thresholded with `relative_l2 <= 0.15` against a
strict refinement contract. Production-scaling rows must declare a non-empty
device, positive integer grid/rank metadata, at least `64` phase cells, and
`wall_time_s <= 86400`; presence-only timing rows are not accepted.
The parity report now also emits a roadmap evidence-surface matrix plus an evidence-package matrix. Each solver-family
row must carry manifest completeness, public provenance/license readiness,
source and converted artefact checksums, converted metadata checksums, native
same-case threshold results, grid convergence, and production-scaling evidence
before the full-fidelity GK lane can be accepted.
The free-boundary lane now indexes public FreeGSNKE machine metadata for active
coils, passive structures, limiter/wall contours, and magnetic probes, plus
FreeGS example-script checksums, in
[`validation/reports/free_boundary_public_machine_metadata_inventory.md`](validation/reports/free_boundary_public_machine_metadata_inventory.md).
It also attempts same-case FreeGS public-example reconstruction in
[`validation/reports/freegs_public_example_reconstruction.md`](validation/reports/freegs_public_example_reconstruction.md):
native and FreeGS vacuum Green-function flux agree on the public machine coils,
and the external FreeGS nonlinear examples now produce finite `psi(R,Z)` output
under the recorded Picard iteration sweep. The report now also publishes native
fixed-boundary profile-source comparison metrics for `psi_N` RMSE, magnetic
axis error, boundary error, sampled X-point constraint error, and current
closure on the finite FreeGS grid, plus finite signed-q profile sanity from
the solved public FreeGS equilibrium. The benchmark also emits per-case
geometry-containment evidence for source X-points, isoflux endpoints,
native/external magnetic axes, and boundary-containment metric readiness,
alongside strict threshold checks and failed-check counts, so the blocker is
explicit rather than inferred from summary prose. Clean CI checkouts preserve the
tracked metadata and reconstruction evidence when gitignored public-source
caches are absent, so full-suite report generation remains deterministic
without promoting those partial artefacts to parity. Strict free-boundary parity remains
blocked until strict thresholds, grid convergence, and public coil/vacuum
sidecars are accepted.
The electromagnetic GK diagnostic now reports compact Ampere and perpendicular
pressure-balance residuals in
[`validation/reports/gk_nonlinear_solver_comparison.md`](validation/reports/gk_nonlinear_solver_comparison.md).
Those residuals are now also exported as nonlinear GK time histories and gated
separately from electrostatic GK in
[`validation/reports/gk_electromagnetic_fidelity.md`](validation/reports/gk_electromagnetic_fidelity.md).
They verify the native compact `A_parallel`/`B_parallel` closure, but they are
explicitly not same-deck external Vlasov-Maxwell parity.
The report now includes a machine-readable Maxwell equation contract, an explicit
electromagnetic evidence gate matrix, a blocked sourced-Maxwell contract plus time-resolved 5D current/charge moment histories, spectral continuity-proxy evidence, and native J_parallel E_parallel exchange diagnostics, and explicit sourced-field evolution blockers for dE_parallel/dt, sourced Faraday, sourced Ampere-Maxwell, and field-particle energy balance while self-consistent sourced field coupling remains blocked, and local source-free spectral Maxwell
evolution evidence for Faraday induction,
displacement-current Ampere-Maxwell evolution, and the inductive parallel
electric-field relation. The latest local Maxwell evidence reports maximum
relative total-field-energy drift `5.090958569120036e-16` with zero Faraday,
Ampere-Maxwell, and inductive parallel electric-field residuals under tolerance
`1.0e-12`.
Native same-case EM replay thresholds are also tracked for `phi`,
`A_parallel`, `B_parallel`, and total field-energy histories. The latest local
threshold gate passes with maximum absolute and relative error `0.0` for every
observable under absolute tolerance `1.0e-18` and relative tolerance `1.0e-15`.
It also records local compact-EM grid-convergence evidence for the algebraic
field-energy histories across `4x4x8`, `6x6x10`, and `8x8x12` spectral/theta
grids. The latest local run passes that compact-grid contract with maximum
relative total-energy drift `5.494182e-03` under tolerance `5.0e-01`, while
full Vlasov-Maxwell parity remains blocked on self-consistent 5D kinetic
current coupling, same-deck external electromagnetic outputs, and external
same-case parity thresholds.
Production-scale decomposition now has a deterministic radial/toroidal
partition contract in
[`validation/reports/production_decomposition_contract.md`](validation/reports/production_decomposition_contract.md).
The contract covers rank tiling, serial reference halo exchange, owned-state
reconstruction, decomposition-invariant inventory/free-energy checks, local CPU
timing metadata, rank-neighbour/halo-face payload-shape contracts, and
executable local rank-tile reductions across multiple decomposition shapes.
It now also publishes per-rank halo-face integrity evidence comparing radial and
toroidal halo faces against the serial reference payload, local process-isolated
CPU rank execution, real local 2D radial/toroidal MPI face-and-corner halo exchange when mpi4py/mpiexec are
available, and CUDA rank-tile reductions when CuPy can access a GPU. Cluster MPI
scaling and multi-GPU scaling remain blocked until measured artefacts exist. The
optional runtime dependency contract pins base NumPy below 2 and gates MPI/GPU
lanes through `mpi4py>=4.1` and `cupy-cuda12x>=13.6,<14.0` so accelerator
setup does not destabilise the base test environment.
The latest local run records same-physics shape convergence across `4x2`,
`8x1`, and `2x4` radial/toroidal rank shapes with maximum inventory relative
deviation `0.0`, maximum free-energy relative deviation
`3.3306658974988877e-16`, and maximum owned-state reconstruction error `0.0`.
It also records local large-grid CPU decomposition evidence for
`9,437,184` 5D phase cells over `24` local rank tiles in `1.557183 s`
(`6.060419e6` cells/s), with zero reconstruction error and invariant relative
errors below `1e-12`. Production scaling remains blocked until cluster MPI scaling, multi-GPU execution,
and hardware-specific timing evidence exist.

## Competitive Position

| Capability | SCPN Fusion Core | TORAX | FUSE | FreeGS | DREAM |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Free-boundary GS solve | Public GEQDSK operator-source gate passes; FreeGSNKE public machine metadata is indexed; FreeGS public-example vacuum convention passes; profile-source/free-boundary reconstruction gate remains open; not EFIT-grade inverse reconstruction | N | N | Y | N |
| 1.5D coupled transport | **Y** | Y | Y | N | N |
| Neural transport surrogate | **Y** (QLKNN-10D) | N | N | N | N |
| Native GK solver | Linear eigenvalue plus nonlinear 5D operator/invariant benchmarks; not GENE/CGYRO-class production turbulence | N | N | N | N |
| External GK coupling (5 codes) | **Y** | TGLF only | TGLF only | N | N |
| Neuro-symbolic SNN compiler | **Y** | N | N | N | N |
| Real-time control (<1 us) | **Y** (0.52 us Rust) | N | N | N | N |
| H-infinity robust control | **Y** | N | N | N | N |
| Free-boundary tracking | Direct kernel + supervisor; not EFIT/LiUQE-grade inverse reconstruction | N | N | N | N |
| Disruption chain (TQ+CQ+RE+halo) | Reduced chain with 0D runaway rates | N | N | N | Y |
| ELM model + RMP suppression | Peeling-ballooning proxy; no nonlinear MHD ELM simulation | N | Y | N | N |
| Runaway electron dynamics | DREAM-style fluid balance, 1D momentum Fokker-Planck, multidimensional artifact-export contract, fail-closed kinetic-operator evidence, and native-only source-term budget diagnostics; no public DREAM kinetic-distribution parity or coupled momentum-pitch-radius operator parity | N | N | N | Y |
| Pellet injection (Parks-Turnbull) | **Y** | N | N | N | N |
| Impurity transport (neoclassical) | Trace radial transport with source conservation, neoclassical pinch, charge-state artifact/source-sink contract, fail-closed native transport evidence, and native-only source/sink budget diagnostics; no public Aurora/STRAHL collisional-operator parity or same-case transport thresholds | N | N | N | N |
| Momentum transport (ExB shearing) | **Y** | N | partial | N | N |
| MHD stability (7 criteria) | **Y** | N | N | N | N |
| Digital twin + HIL testing | **Y** | N | N | N | N |
| Deterministic replay (RZIP reduced-order scaffold) | **Y** | N | N | N | N |
| SCPN phase dynamics (Kuramoto/UPDE) | **Y** | N | N | N | N |
| JAX autodiff transport | **Y** | Y | N | N | N |

Full analysis: [`docs/competitive_analysis.md`](docs/competitive_analysis.md)

## Core Innovation: Neuro-Symbolic Compiler

```
Petri Net (places + transitions + contracts)
    |
    v  compiler.py -- structure-preserving mapping
Stochastic LIF Network (neurons + synapses + thresholds)
    |
    v  controller.py -- closed-loop execution
Real-Time Plasma Control (sub-ms latency, deterministic replay)
    |
    v  artifact.py -- versioned, signed compilation artifact
Deployment Package (JSON + schema version + git SHA)
```

Control logic is the primary artifact — expressed in a formally verifiable
Petri net formalism, compiled to spiking neural networks, executed at
hardware-compatible latencies. The physics modules provide a realistic plant
model for the controller to operate against.

| Property | How |
|----------|-----|
| Formal verification | Contract checking preserves Petri net invariants (boundedness, liveness, reachability) |
| Hardware targeting | Same Petri net compiles to NumPy, SC-NeuroCore (FPGA), or neuromorphic silicon |
| Explicit backend policy | Production solver lanes fail closed; reference backends require explicit opt-in |
| Deterministic replay | Identical inputs produce identical outputs across platforms |

## Formal Verification

SCPN Fusion Core now includes a committed Lean 4 safety-proof surface for the
native solver stack. The first machine-checkable theorem proves that invalid
Grad-Shafranov case descriptions fail closed: when `validateCase` rejects a
case, `solveGradShafranov` returns the same validation error before numerical
solver work can begin.

Evidence boundary:

| Item | Evidence |
|------|----------|
| Lean project | `scpn-fusion-lean/` |
| First theorem | `scpn-fusion-lean/SafetyProof.lean` |
| PID bounded-output proof | `scpn-fusion-lean/PIDBoundedOutput.lean` |
| Petri-to-SNN reachability proof | `scpn-fusion-lean/SNNReachabilityPreservation.lean` |
| Petri token-boundedness proof | `scpn-fusion-lean/PetriTokenBoundedness.lean` |
| Verified properties | Grad-Shafranov validation errors are propagated exactly; normalized PID magnitudes remain bounded by actuator limits and raw command magnitude, propagate actuator-limit, raw-command, and dual upper bounds, remain bounded under nested filters, preserve zero-command and at-limit boundary cases, are monotone in command and actuator limit, are idempotent under repeated saturation, take exactly the raw command or the configured limit branch, and cannot amplify a command unless saturating to the configured limit; finite Petri graph reachability paths, composed paths, direct-edge equivalence, edge-count preservation, empty-edge preservation, source/destination endpoint bounds, and full endpoint bounds are preserved and reflected by the compiled SNN edge contract; no compiled SNN direct edge or reachable path exists without a corresponding Petri edge/path or declared topology; empty Petri graphs are well-formed vacuously; well-formed Petri edges and reachable endpoints remain within the compiled SNN neuron bound; finite-capacity Petri token filters preserve per-place bounds, preserve arbitrary finite-marking length and capacity sum, do not amplify per-place tokens or aggregate token sum, keep filtered aggregate token sum below both original and filtered aggregate capacity, expose a combined original-and-capacity aggregate safety theorem, are idempotent, have stable aggregate token/capacity sums under repeated filtering, leave already bounded finite markings unchanged, and keep bounded natural-number firing updates within place capacity |
| CI surface | `lean-safety-proofs` job in `.github/workflows/ci.yml` |
| Narrative draft | `docs/blog/first_machine_checkable_safety_proof_for_tokamak_plasma_solver.md` |

This is an intentionally narrow proof boundary, not a claim that the full
plasma solver, controller stack, or nonlinear plant model is formally verified.
Next proof targets are signed PID saturation over physical coil-current units,
SNN deterministic replay over seeded stochastic traces, matrix-level incidence
preservation against the Python compiler artefact schema, and finite-capacity
token boundedness tied to executable weighted firing updates.

## Controller Stress-Test Campaign (1,000 shots)

The campaign table is a mixed-fidelity controller benchmark, not an
apples-to-apples language benchmark. `Rust-PID` uses the Rust
`PyRustFlightSim` reduced-order linearised plasma surrogate. The Python PID,
H-infinity, NMPC-JAX, and Nengo-SNN rows run through the Python flight-sim
control path; the non-surrogate PID lane performs Grad-Shafranov equilibrium
work inside the loop. Therefore the Rust/Python latency ratio measures a
physics-scope change plus implementation overhead, not "Rust is N times faster
for the same work".

| Controller | Physics/control scope | P50 Latency | P95 Latency | Disruption Rate |
|-----------|------------------------|------------|------------|-----------------|
| **Rust-PID** | Reduced-order linearised plasma surrogate in Rust | **0.52 us** | 0.67 us | **0%** |
| PID (Python) | Python flight-sim path with Grad-Shafranov equilibrium work | 3,431 us | 3,624 us | 0% |
| H-infinity | Python flight-sim research lane | diagnostic only | diagnostic only | blocked: stale scalar-plant artifact |
| NMPC-JAX | Python/JAX NMPC lane | 45,450 us | 49,773 us | 0% |
| Nengo-SNN | Python/Nengo SNN lane | 23,573 us | 24,736 us | 0% |

For same-harness controller-loop timings and the measured digital-twin
sensor-to-control path, use `validation/scpn_end_to_end_latency.py`. The
tracked 2026-06-17 local run reports Python CPU and Rust native
sensor-to-control p50/p95/p99 latency, CUDA GPU p50/p95/p99 latency on the
operator-reserved NVIDIA GeForce GTX 1060 6GB device, host-load metadata, and
degraded-mode fallback counts. It also reports actuator fanout through `256`
channels and reduced-order predictive-horizon timing at `50 ms` and `100 ms`;
see
`validation/reports/scpn_end_to_end_latency.md`.

Rust full-order equilibrium benchmarks are tracked separately from the
reduced-order control kernel. On 2026-05-24, local Criterion runs on an
Intel i5-11600K workstation measured 33x33 Grad-Shafranov SOR at 413 us,
33x33 Picard multigrid at 845 us, vacuum field 33x33 at 140 us, and vacuum
field 65x65 at 489 us. Local hardware: Linux 6.17, 31.1 GB RAM, Rust 1.95.0,
Python 3.12.3, NVIDIA GTX 1060 6GB available for JAX/CUDA tests. These numbers
are competitive for sub-millisecond low-resolution equilibrium/control-support
updates, but they are not EFIT-grade reconstruction parity evidence and should
not be compared with the reduced-order 0.16-0.52 us Rust flight-simulator
kernel.

H-infinity is a research lane (reduced-order 2x2 robust model) and is not
part of production release acceptance criteria.

---

<details>
<summary><strong>Architecture (234 modules)</strong></summary>

```
scpn-fusion-core/
+-- src/scpn_fusion/              # Python package (234 source files)
|   +-- core/           (118)    # Plasma physics: GS, transport, GK, MHD, disruptions
|   +-- control/         (54)    # Controllers: PID, H-inf, NMPC, SNN, RL, free-boundary
|   +-- phase/           (10)    # SCPN dynamics: Kuramoto, UPDE, adaptive K_nm
|   +-- scpn/            (12)    # Neuro-symbolic compiler, contracts, interlocks
|   +-- io/              (15)    # IMAS, GEQDSK, archive, logging
|   +-- diagnostics/      (5)    # Synthetic sensors, tomography
|   +-- nuclear/           (5)    # Blanket neutronics, PWI, erosion
|   +-- engineering/       (4)    # Balance of plant, thermal hydraulics
|   +-- hpc/               (2)    # HPC bridge, C library interface
|   +-- ui/                (4)    # Streamlit dashboard
+-- scpn-fusion-rs/               # Rust workspace (11 crates)
|   +-- crates/
|       +-- fusion-types/         # Shared data types
|       +-- fusion-math/          # Linear algebra, FFT
|       +-- fusion-core/          # Grad-Shafranov, transport
|       +-- fusion-physics/       # MHD, heating, turbulence
|       +-- fusion-control/       # PID, MPC, disruption
|       +-- fusion-ml/            # Inference engine
|       +-- fusion-python/        # PyO3 bindings
+-- tests/               (382)    # 3,817 test functions (Hypothesis property tests)
+-- validation/           (74)    # Benchmark pipeline + reference data
+-- examples/             (16)    # 10 Jupyter notebooks + 6 scripts
```

</details>

<details>
<summary><strong>Installation</strong></summary>

### Python (no Rust required)

```bash
pip install -e .                      # core runtime (minimal deps)
pip install "scpn-fusion[full]"       # core + UI + JAX/ML + RL + physics extras
pip install "scpn-fusion[ui,ml,rl]"   # explicit optional stacks only
```

Every module auto-detects Rust and falls back to NumPy/SciPy.

### Rust acceleration (optional)

```bash
pip install "scpn-fusion[rust]"
cd scpn-fusion-rs
cargo build --release && cargo test
cd crates/fusion-python && maturin develop --release
```

### Docker

```bash
docker compose up --build                              # dashboard
docker build --build-arg INSTALL_DEV=1 -t dev .        # with tests
docker run dev pytest tests/ -v
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v                 # Python (Hypothesis property tests)
cd scpn-fusion-rs && cargo test  # Rust (proptest)
cargo bench                      # Criterion benchmarks
```

</details>

<details>
<summary><strong>Physics Modules (118 core)</strong></summary>

### Equilibrium & Stability
| Module | Physics |
|--------|---------|
| `fusion_kernel` | Nonlinear Grad-Shafranov solves with Picard/SOR/multigrid plus external-coil vacuum/free-boundary boundary coupling |
| `jax_gs_solver` | JAX-differentiable GS solver (Picard + damped Jacobi) |
| `force_balance` | Force balance verification (J x B = grad p) |
| `stability_mhd` | 7-criterion suite: Mercier, ballooning, K-S, Troyon, NTM, RWM, peeling-ballooning |
| `ballooning_solver` | Full ideal MHD ballooning equation solver |
| `elm_model` | Peeling-ballooning boundary + Chirikov overlap + RMP suppression |

### Transport
| Module | Physics |
|--------|---------|
| `integrated_transport_solver` | 1.5D coupled (Te, Ti, ne, current diffusion) |
| `jax_solvers` | JAX Thomas + Crank-Nicolson (differentiable, batched via vmap) |
| `neural_transport` | QLKNN-10D surrogate (test rel_L2=0.094) |
| `momentum_transport` | NBI torque, ExB shearing (Waltz 1994), rotation solver |
| `neoclassical` | Chang-Hinton chi + Sauter bootstrap current |
| `impurity_transport` | Hirshman & Sigmar neoclassical pinch (multi-species) |

### Gyrokinetic Three-Path
| Module | Physics |
|--------|---------|
| `gk_eigenvalue` | Native linear GK solver (ballooning, Sugama collisions) |
| `gk_quasilinear` | Mixing-length saturation -> chi_i, chi_e, D_e |
| `gyrokinetic_transport` | TGLF-10 input, ITG/TEM/ETG mode identification |
| `gk_tglf` / `gk_gene` / `gk_gs2` / `gk_cgyro` / `gk_qualikiz` | External GK solver interfaces; GENE/GS2/CGYRO support explicit nonlinear 5D deck metadata when the external binaries are installed |
| `gk_ood_detector` + `gk_corrector` + `gk_scheduler` | Hybrid surrogate validation |

### Disruption & Edge Physics
| Module | Physics |
|--------|---------|
| `disruption_sequence` | Full chain: thermal quench -> current quench -> RE -> halo currents |
| `runaway_electrons` | Connor & Hastie primary + Rosenbluth & Putvinski avalanche + Smith hot-tail |
| `pellet_injection` | Parks & Turnbull 1978 NGS ablation + drift displacement |
| `plasma_wall_interaction` | Eckstein sputtering + 1D wall thermal + Coffin-Manson fatigue |
| `marfe` | Radiation condensation + density limit |
| `lh_transition` | Zonal flow predator-prey L-H transition |
| `locked_mode` | Error field amplification -> rotation braking -> mode locking |
| `plasma_startup` | Paschen breakdown -> Townsend avalanche -> burn-through |
| `blob_transport` | SOL filament propagation + cross-field diffusion |

### Advanced
| Module | Physics |
|--------|---------|
| `alfven_eigenmodes` | TAE/RSAE continuum + fast-particle drive |
| `tearing_mode_coupling` | Multi-mode nonlinear interaction + disruption triggering |
| `orbit_following` | Alpha particle guiding-center orbits + confinement time |
| `vmec_lite` | 3D fixed-boundary MHD equilibrium (Fourier harmonics) |
| `neural_turbulence` | QLKNN-class 10-parameter MLP surrogate |
| `vessel_model` | Vacuum vessel eddy currents (lumped circuit) |
| `kinetic_efit` | Anisotropic fast-ion pressure reconstruction |
| `integrated_scenario` | Full coupled scenario (current diffusion + transport + NTM + SOL) |

</details>

<details>
<summary><strong>Control Modules (54)</strong></summary>

| Category | Modules |
|----------|---------|
| **Classical** | PID (Rust 0.52 us), H-infinity (Riccati synthesis), gain-scheduled, sliding-mode vertical |
| **Optimal** | NMPC-JAX (SQP), MPC (gradient trajectory), optimal control |
| **Learning** | Safe RL (Lagrangian PPO), PPO 500K (beats MPC+PID), controller tuning (Bayesian) |
| **Neuro-symbolic** | SNN compiler, cybernetic controller, Nengo SNN wrapper |
| **Disruption** | Predictor (ML), SPI mitigation, checkpoint policy, disruption contracts |
| **Free-boundary** | Direct kernel tracking + supervisor rejection + EKF latency compensation |
| **Burn & Fueling** | Burn controller (alpha heating), pellet fueling, density control, detachment |
| **Stability** | RWM feedback, mu-synthesis (D-K iteration), RZIP model |
| **Infrastructure** | State estimator (EKF), volt-second manager, scenario scheduler, fault-tolerant control |
| **Simulation** | Digital twin, flight simulator (Python + Rust), Gymnasium environment |
| **Integration** | Director interface, bio-holonomic controller (SCPN L4/L5 bridge) |

</details>

<details>
<summary><strong>Tutorial Notebooks</strong></summary>

| Notebook | Description |
|----------|-------------|
| **`neuro_symbolic_control_demo_v2`** | **Golden Base v2** — formal proofs + closed-loop + replay |
| `01_compact_reactor_search` | MVR-0.96 compact reactor optimizer |
| `02_neuro_symbolic_compiler` | Petri net -> SNN compilation pipeline |
| `03_grad_shafranov_equilibrium` | Free-boundary equilibrium solver |
| `04_divertor_and_neutronics` | Divertor heat flux & TBR |
| `05_validation_against_experiments` | Cross-validation vs SPARC & ITPA |
| `06_inverse_and_transport_benchmarks` | Inverse solver & neural transport |
| `07_multi_ion_transport` | Multi-species transport evolution |
| `08_mhd_stability` | Ballooning & Mercier criteria |
| `09_coil_optimization` | Coil current optimization (Tikhonov) |
| `10_uncertainty_quantification` | Monte Carlo UQ chain |

</details>

<details>
<summary><strong>Validation Data</strong></summary>

| Dataset | Source | Contents |
|---------|--------|----------|
| **SPARC GEQDSK** | [SPARCPublic](https://github.com/cfs-energy/SPARCPublic) (MIT) | 8 public EFIT equilibria (B=12.2 T, Ip up to 8.7 MA); operator-source rows pass, while profile-source/free-boundary reconstruction rows remain open with row-level debug traces |
| **ITPA H-mode** | Verdoolaege et al., NF 61 (2021) | 53 shots from 24 machines |
| **DIII-D disruptions** | Reference profiles (16 shots) | Locked mode, VDE, tearing, density, beta |
| **Multi-machine GEQDSK** | Synthetic Solov'ev | 100 equilibria (DIII-D, JET, EAST, KSTAR, ASDEX-U) |

```bash
python validation/validate_real_shots.py        # real-shot gate
python validation/collect_results.py            # full 15-lane benchmark
python validation/benchmark_gk_linear.py        # GK eigenvalue benchmark
PYTHONPATH=src python benchmarks/gk_solver_comparison.py  # nonlinear GK NumPy/JAX comparison
python validation/benchmark_full_fidelity_acceptance.py    # fail-closed full-fidelity acceptance status
python validation/full_fidelity_end_to_end_campaign.py     # six-lane production-parity campaign status
python validation/benchmark_freegs_public_example_reconstruction.py  # FreeGS public-example comparison gate
```

</details>

<details>
<summary><strong>Benchmarks & Solver Performance</strong></summary>

All numbers are internal measurements. Reproduce with `cargo bench` and
`python validation/collect_results.py`.

| Metric | Value | Source |
|--------|-------|--------|
| SOR step @ 65x65 | us-range | `sor_bench.rs` |
| GMRES(30) @ 65x65 | ~45 iters | `gmres_bench.rs` |
| Rust multigrid V-cycle scaling | local convergence/scaling report; not a production speedup claim | `validation/reports/rust_multigrid_scaling.md` |
| Rust flight sim | 0.3 us/step | `verify_10khz_rust.py` |
| Full equilibrium (Python) | ~5 s | `profile_kernel.py` |
| Neural transport MLP | ~5 us/point | `neural_transport_bench.rs` |
| JAX GS solve (33x33) | ~50 ms | `jax_gs_solver.py` |
| Native GK eigenvalue | ~0.3 s/surface | `gk_eigenvalue.py` |
| Native nonlinear GK invariant benchmark | local report | `validation/reports/gk_nonlinear_solver_comparison.md` |
| QLKNN single-point | ~24 ns | `neural_transport.py` |

### Community context (not direct comparisons)

| Code | Category | Typical Runtime |
|------|----------|-----------------|
| GENE | 5D gyrokinetic | ~10^6 CPU-h |
| JINTRAC | Integrated modelling | ~10 min/shot |
| TORAX | Integrated (JAX) | ~30 s (GPU) |
| DREAM | Disruption / runaway | ~1 s |

</details>

<details>
<summary><strong>Code Health & Hardening</strong></summary>

263 hardening tasks across 8 waves (S2-S4, H5-H8). Every production-path
module returns structured errors.

| Metric | Value |
|--------|-------|
| Python source files | 236 |
| Python lines of code | 65,664 |
| Test functions | 3,815 |
| Validation scripts | 74 |
| Rust crates | 11 |
| CI jobs | 24 |
| Internal readiness entries | 115 (local-only governance queue) |

Validation artifacts:
- Claims evidence map and release readiness checks are generated by repository tools.
- Benchmark and acceptance reports live under `validation/reports/`.
- [Competitive analysis](docs/competitive_analysis.md)

</details>

## Community

- [Discussions](https://github.com/anulum/scpn-fusion-core/discussions) — Q&A, ideas, show and tell
- [Roadmap](ROADMAP.md) — v4.0 targets and beyond
- [Contributing](CONTRIBUTING.md) — how to get started

## Citation

```bibtex
@software{scpn_fusion_core,
  title   = {SCPN Fusion Core: Neuro-Symbolic Tokamak Control Suite},
  author  = {Sotek, Miroslav},
  year    = {2026},
  url     = {https://github.com/anulum/scpn-fusion-core},
  version = {3.9.10}
}
```

## Author

- **Miroslav Sotek** — ANULUM CH & LI — [ORCID](https://orcid.org/0009-0009-3560-0851)

## License

Dual licensing is intentional:

- Open-source use is governed by GNU Affero General Public License v3.0 or
  later. See [LICENSE](LICENSE).
- Commercial licences are available for proprietary/internal deployments,
  closed operational workflows, commercial support, and integrations that
  cannot use AGPL reciprocal terms.

Commercial licensing:
[protoscience@anulum.li](mailto:protoscience@anulum.li?subject=SCPN%20Fusion%20Core%20Commercial%20License).
