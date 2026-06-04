# Changelog

## [Unreleased]

- Added the first FRC rigid-rotor equilibrium slice: validated
  Steinhauer no-rotation analytical axial-field profile, finite pressure and
  energy diagnostics, lazy public exports, module-specific tests, and Sphinx
  API documentation while keeping rotating BVP cases fail-closed.
- Added explicit FRC radial force-balance residual diagnostics with an optional
  fail-closed validation tolerance for analytical no-rotation runs.
- Added Rust `fusion-physics::frc` support, a PyO3
  `py_solve_frc_equilibrium` bridge, parity tests, Criterion coverage, and the
  tracked `validation/reports/frc_rigid_rotor_benchmark.json` benchmark report for the
  accepted Steinhauer no-rotation FRC analytical contract.
- Hardened the FRC quality-of-equilibrium diagnostic to use the Steinhauer
  Eq. 27 local-gyroradius integral across Python, Rust, PyO3 parity tests, and
  the tracked benchmark report.
- Added finite-grid convergence evidence for the accepted no-rotation FRC
  scalar invariants across the tracked benchmark report, Python tests, Rust
  tests, and public benchmark/method documentation.
- Added explicit toroidal current-density and Ampere closure diagnostics for
  the accepted no-rotation FRC contract across Python, Rust, PyO3 parity tests,
  benchmark JSON, and public physics documentation.
- Hardened the FRC current-density contract to use the closed-form Steinhauer
  derivative while keeping Ampere residuals as independent finite-grid
  diagnostics with Python, Rust, PyO3, benchmark, and documentation coverage.
- Added an explicit FRC separatrix and field-reversal validation contract so
  the no-rotation analytical solver compares the interpolated null against the
  configured `R_s`, requires radial samples on both sides of `R_s`, carries the
  diagnostics through Python, Rust, PyO3, benchmark JSON, and public docs, and
  keeps missing non-Python/Rust FRC surfaces fail-closed.
- Replaced numerical FRC cylindrical-flux integration with the closed-form
  Steinhauer primitive and added a flux derivative closure residual across
  Python, Rust, PyO3, package exports, benchmark JSON, tests, and public docs.
- Replaced the FRC diagnostic Gaussian pressure with the local magnetic
  pressure-balance profile and added pressure-balance residual plus
  thermal-pressure consistency diagnostics across Python, Rust, PyO3,
  benchmark JSON, tests, and public docs.
- Added FRC density-closure validation so the solved density profile is derived
  from magnetic-pressure-balanced `p(r)` and configured temperatures, while
  `n0` is gated against the solved peak density across Python, Rust, PyO3,
  benchmark JSON, tests, and public docs.
- Added FRC beta and particle-line-density diagnostics derived from the same
  accepted pressure/density fields, with beta-limit validation, Python/Rust/PyO3
  parity checks, benchmark JSON coverage, tests, and public docs.
- Added FRC separatrix energy-inventory closure diagnostics so pressure-energy
  and magnetic-field-deficit integrals are independently assembled, gated,
  compared across Python/Rust/PyO3, recorded in the benchmark JSON, and
  documented without extending the claim beyond the accepted no-rotation
  analytical contract.
- Added FRC separatrix current-sheet closure diagnostics so the finite-grid
  `dB_z/dr` slope and `J_theta(R_s)` are compared against the analytical
  no-rotation identities across Python/Rust/PyO3, validation gates, benchmark
  JSON, tests, and public docs.
- Added an FRC resolved sheet-current integral closure so the radial integral
  of `J_theta` is checked against the finite-grid magnetic-field jump across
  Python/Rust/PyO3, validation gates, benchmark JSON, tests, and public docs.
- Added an analytical FRC pressure-gradient closure so finite-grid `dp/dr` is
  checked against `-(B_z / mu_0) dB_z/dr` across Python/Rust/PyO3, validation
  gates, benchmark JSON, tests, and public docs.
- Added a separatrix-normalised FRC flux coordinate `psi_N` with raw flux-span
  diagnostics, axis/separatrix endpoint gates, monotonic/bounds validation,
  Python/Rust/PyO3 parity checks, benchmark JSON coverage, and public docs for
  the accepted no-rotation analytical MIF lane.
- Expanded the accepted FRC no-rotation parity gate to a deterministic 16-case
  MIF/FRC parameter cohort across Python, Rust `fusion-physics`, and PyO3 while
  keeping rotating-BVP cases fail-closed.
- Added generated FRC no-rotation property gates for pressure monotonicity away
  from the magnetic null, beta bounds, separatrix energy closure, and
  Rust/PyO3 energy-invariant parity on accepted MIF/FRC decks.
- Added an executable FRC rigid-rotor quickstart that reproduces the accepted
  Steinhauer no-rotation field and magnetic-pressure-balance contract, writes
  reproducible JSON samples, and is covered by a module-specific example test.

## [3.9.8] - 2026-06-02

- Bumped package and release metadata to 3.9.8 for a documentation, security-hardening, and repository-polish release.
- Reworked README badges and public reader paths so the project value, applications, evidence boundary, and blocked parity lanes are easier to scan.
- Expanded onboarding, API, applications/market, notebook, quickstart, and Sphinx overview documentation with clearer contribution paths, maturity labels, and trust-boundary guidance.
- Hardened optional native C++ compilation against arbitrary code execution risks by requiring the bundled solver source, resolving `g++` from the system default path, rejecting symlinked or group/world-writable source/compiler files, and stripping ambient compiler/linker flags from the build environment.

- Hardened optional native C++ compilation against arbitrary code execution
  risks by requiring the bundled solver source, resolving `g++` from the system
  default path, rejecting symlinked or group/world-writable source/compiler
  files, stripping ambient compiler/linker flags from the build environment,
  and extending the HPC bridge tests for these fail-closed cases.

## [3.9.7] - 2026-06-02

- Reworked the public documentation surface around evidence-bounded product
  positioning, onboarding, API navigation, applications, market value, and
  reproducibility entry points.
- Added explicit public documentation for current solver maturity: validated
  local contracts and fail-closed parity gates are separated from blocked
  GENE/CGYRO/GS2, DREAM, Aurora/STRAHL, FreeGS, electromagnetic, and
  distributed-scaling acceptance evidence.
- Updated the GitHub Pages and Sphinx-facing documentation headers to remove
  stale public boilerplate and point readers to current benchmark and evidence
  reports.

- Hardened the GENE/CGYRO/GS2 nonlinear GK parity lane with a fail-closed
  evidence-package matrix requiring manifest completeness, public
  provenance/license, source checksums, converted artefact and metadata
  checksums, native same-case thresholds, grid convergence, and
  production-scaling rows before acceptance.
- Added explicit grid-convergence and production-scaling acceptance matrices to
  the nonlinear GK parity report so linked evidence rows must satisfy declared
  thresholds instead of merely existing.
- Added a GENE/CGYRO/GS2 public-output candidate matrix to the nonlinear GK
  deck inventory, keeping public decks and partial numeric snippets separate
  from accepted same-deck nonlinear parity artifacts.

- Added an explicit vertical-control replay release-gate object that accepts
  only the reduced-order RZIP replay claim and keeps full PCS production-grade
  readiness false.

- Added a Lean 4 PID bounded-output proof for the normalized saturated actuator
  magnitude.

- Added a fail-closed GPU Phase 1 readiness gate for the Rust/wgpu SOR
  implementation surfaces while blocking production scaling claims until a
  tracked GPU benchmark artifact exists.

- Added a free-boundary strict acceptance matrix that exposes same-case
  reference, native profile-source, threshold, grid-convergence, coil/vacuum
  sidecar, and machine-metadata readiness separately.

## [3.9.6] - 2026-06-01

- Added local large-grid CPU evidence to the production-decomposition benchmark, measuring `9,437,184` 5D phase cells over `24` local rank tiles while keeping MPI/multi-GPU production-scale readiness fail-closed.

- Added a public GitHub Pages financing landing page that presents the full-fidelity parity campaign, evidence links, blocked validation lanes, and staged GPU/reference-solver funding plan without promoting partial artefacts to accepted parity evidence.

- Declared the `requests` runtime dependency required by public Aurora
  acquisition and made public NPZ reference-artifact writing deterministic.

- Added JSON sidecar ingestion for future production-decomposition distributed
  runtime measurements while keeping incomplete rows blocked by the acceptance
  manifest.

- Added a fail-closed production-decomposition distributed-run acceptance
  manifest with required measurement fields, rank coverage, efficiency gates,
  hardware metadata, and checksum requirements.

- Added a fail-closed production-decomposition distributed scaling gate with
  required rank counts, efficiency thresholds, and measurement requirements
  while keeping MPI/multi-GPU acceptance blocked until real runs exist.

- Added production-decomposition reciprocal neighbour-graph evidence with
  directed halo-link symmetry, payload-shape matching, and byte-asymmetry
  diagnostics while keeping distributed MPI/multi-GPU execution blocked.

- Added FreeGS public-example geometry-containment evidence for source
  X-points, isoflux endpoints, native/external magnetic axes, and boundary
  containment metrics while keeping strict FreeGS parity blocked.

- Added production-decomposition halo-face integrity evidence with per-rank
  radial/toroidal face comparisons against the serial reference halo exchange
  while keeping distributed MPI/multi-GPU halo exchange blocked.

- Added native impurity source/sink budget evidence for conservative
  charge-state transfer matrices, ionisation/recombination source budgets,
  line-radiation power, and inventory history while keeping Aurora/STRAHL
  same-case source-budget parity fail-closed.

- Added native runaway source-term budget evidence for avalanche growth,
  synchrotron loss, partial-screening drag, and bremsstrahlung loss channels
  while keeping DREAM same-case source-budget parity fail-closed.

- Added fail-closed FreeGS public-example strict threshold evidence with
  per-case native-vs-FreeGS checks, failed-check counts, and readiness booleans
  for threshold acceptance, grid convergence, and public coil/vacuum sidecars.

- Added fail-closed native impurity transport-operator evidence that separates
  trace radial transport and charge-state artifact contracts from missing
  Aurora/STRAHL charge-state-resolved radial transport parity, external ADAS
  transport coefficients, same-case outputs, and quantitative thresholds.

- Added fail-closed native runaway kinetic-operator evidence that separates the
  finite DREAM-style multidimensional artifact export from missing coupled
  momentum-pitch-radius DREAM parity, same-case thresholds, radial transport,
  pitch-angle scattering, partial-screening, and bremsstrahlung operator gates.

- Added native electromagnetic same-case replay thresholds for `phi`,
  `A_parallel`, `B_parallel`, and total field-energy histories while keeping
  external electromagnetic same-deck parity thresholds fail-closed.

- Added local source-free spectral Maxwell evolution evidence for Faraday
  induction, Ampere-Maxwell displacement current, and inductive parallel
  electric-field diagnostics while keeping self-consistent 5D kinetic current
  coupling and external electromagnetic parity fail-closed.

- Added production-decomposition same-physics shape-convergence evidence across
  `4x2`, `8x1`, and `2x4` local radial/toroidal rank shapes, with true
  owned-tile reductions and fail-closed distributed MPI/multi-GPU scaling
  blockers.

- Added local compact-electromagnetic GK grid-convergence evidence for
  algebraic `A_parallel`/`B_parallel` field-energy histories while keeping full
  Faraday/displacement-current Vlasov-Maxwell parity fail-closed.

- Added row-level GEQDSK debug traces and aggregate first-blocker counts for
  source attribution, source-unit normalisation, reconstruction, residual,
  classification, and blocker reporting.

- Hardened the fail-closed GENE/CGYRO/GS2 nonlinear GK external-output parity
  manifest so readiness requires shared same-deck identity across all three
  solver families and per-family grid-convergence and production-scaling
  evidence before parity can pass.

- Blocked nonlinear GK external-output artifact conversion when candidate rows
  provide private provenance URLs or non-redistributable, unknown, restricted,
  or proprietary licensing.

- Required `native_output_sha256` for nonlinear GK native same-case comparison
  rows before quantitative threshold evaluation can become ready.

- Classified top-level NPZ nonlinear GK output keys by declared coordinate and
  observable contracts before writing converted artefact metadata.

- Required nonlinear GK grid-convergence and production-scaling evidence rows to
  reference the converted same-case solver-family output rows.

- Added a machine-readable electromagnetic GK Maxwell equation contract that
  distinguishes compact algebraic `A_parallel`/`B_parallel` closures from
  missing Faraday, displacement-current, and inductive parallel-field evolution.

- Added disruption contract primitives, mitigation contracts, and replay contracts to the configured global mypy strict cohort with a typed reactor-design evaluation boundary.

- Added the real-time density-profile controller to the configured global mypy strict cohort with explicit radial-profile array contracts.

- Expanded the global mypy strict cohort to burn, detachment, fuelling, sliding-mode vertical control, state-estimation, and volt-second management control modules with explicit NumPy and callback contracts.

- Expanded the global mypy strict cohort to the control docstring cluster (`runaway_electron_model`, `rust_flight_sim_wrapper`, `rwm_feedback`, `rzip_model`, `safe_rl_controller`, `scenario_scheduler`, `shape_controller`) and fixed typed callback and NumPy array contracts without new suppressions.

- Enabled global mypy strict defaults, removed the broad internal package
  `ignore_errors` override, and fixed the currently gated typed cohort so the
  strict runner passes without hidden package-level suppressions.

- Aligned public README, roadmap, benchmark, and Sphinx changelog surfaces with
  the current fail-closed full-fidelity campaign status, reproducibility
  commands, and public documentation hygiene boundaries.

- Added native same-case FreeGS public-example profile-source comparison
  documentation for `psi_N` RMSE, magnetic-axis error, boundary error,
  sampled X-point constraint error, current closure, and finite signed-q sanity
  while keeping strict free-boundary parity blocked on thresholds, grid
  convergence, and public coil/vacuum sidecars.

- Documented production-scale decomposition halo exchange and
  decomposition-invariant inventory/free-energy checks as local contracts while
  preserving the distributed MPI/multi-GPU scaling blocker.

- Added a production-decomposition rank communication contract with explicit
  neighbour ranks and halo-face payload shapes for future MPI/multi-GPU
  exchange implementation.

- Added executable local rank-tile decomposition reductions for production GK
  scheduling evidence while keeping distributed MPI/multi-GPU scaling blocked
  until external hardware runs exist.

- Added a strict fail-closed GENE/CGYRO/GS2 nonlinear GK external-output
  parity lane that converts redistribution-permitted same-deck outputs into
  tracked NPZ artefacts, validates nonlinear distribution, heat-flux,
  field-energy, zonal/saturation, convergence, and scaling evidence, and keeps
  parity blocked when external outputs or native same-case comparisons are
  missing.

- Extended the FreeGS public-example reconstruction benchmark with a Picard
  convergence sweep that records finite external `psi(R,Z)` output while
  keeping strict free-boundary parity blocked until native same-case
  reconstruction and solver-output comparison exist. Clean checkouts now
  preserve tracked FreeGS and machine-metadata evidence when gitignored public
  caches are absent.

- Added a FreeGS public-example reconstruction attempt benchmark that checks
  native-vs-FreeGS vacuum Green-function parity on public machine coils while
  keeping strict free-boundary parity blocked on nonlinear same-case
  `psi(R,Z)` output and native reconstruction comparison.

- Added a fail-closed FreeGS/FreeGSNKE public machine-metadata inventory that
  records active-coil, passive-structure, limiter/wall, magnetic-probe, and
  FreeGS example-script checksums while keeping strict free-boundary parity
  blocked until same-case reconstruction outputs and solver comparisons exist.

- Added a deterministic radial/toroidal production-decomposition contract for
  5D nonlinear GK grids, including halo-overhead and load-balance reporting,
  while keeping production-scale readiness blocked until distributed
  MPI/multi-GPU execution and scaling evidence exist.

- Added compact electromagnetic GK closure diagnostics for native
  `A_parallel`/`B_parallel` residuals in the nonlinear GK benchmark while
  explicitly keeping full Faraday/displacement-current Vlasov-Maxwell parity
  fail-closed.

- Added a fail-closed nonlinear GK public deck inventory that hashes GS2
  nonlinear decks, CGYRO nonlinear decks, CGYRO regression precision outputs,
  and GENE/GS2/CGYRO public web-source snapshots while keeping full
  GENE/CGYRO/GS2 parity blocked until runnable external outputs and native
  same-case comparisons exist.

- Added an Aurora/Open-ADAS reference execution harness that generates a
  checksummed argon charge-state fractional-abundance artifact with ADAS source
  checksums and keeps Aurora/STRAHL transport parity fail-closed until radial
  transport, radiation, source/sink conservation, and native same-case
  comparisons exist.

- Added a DREAM public reference execution harness that generates the upstream
  `examples/2kinetic` settings deck, records source/deck checksums, and reports
  the exact PETSc/`dreami` backend blocker needed before DREAM output can be
  converted into full-fidelity reference observables. Clean checkouts without
  the gitignored external cache now retain tracked settings-deck evidence
  instead of rewriting the report as missing.

- Added a fail-closed public reference-artifact converter that exports finite,
  checksummed DREAM avalanche, FreeGSNKE baseline, and FreeGSNKE MAST-U-like
  current-sidecar payloads to tracked artifacts with provenance metadata while
  keeping accepted full-fidelity artifact count at zero until required
  observables and same-case solver-output comparisons exist. The converter now
  records whether artifacts came from the external cache or tracked fallback
  evidence.

- Added a reproducible full-fidelity public source downloader and provenance
  report for GENE, CGYRO/GACODE, GS2, DREAM, Aurora, FreeGS, and FreeGSNKE,
  while keeping raw upstream snapshots gitignored and fail-closed as
  acquisition inputs rather than accepted parity artifacts.

- Added an integrated six-lane full-fidelity end-to-end campaign registry and
  report covering GENE/CGYRO/GS2 nonlinear GK parity, full Maxwell/EM fidelity,
  production-scale decomposition, DREAM runaway electrons, Aurora/STRAHL
  impurities, and strict free-boundary reconstruction in one fail-closed gate.

- Added Aurora/STRAHL-style impurity charge-state artifact contracts with
  conservative collisional-radiative source/sink matrices, deterministic
  ADAS-style coefficient ingestion-shape checks, finite line-radiation
  observables, and inventory conservation gates while keeping public
  Aurora/STRAHL parity fail-closed.

- Added Python and Rust DREAM-style runaway-electron kinetic artifact export
  contracts over the native 1D momentum Fokker-Planck kernel, including
  explicit time, radius, momentum, and pitch axes plus finite current,
  avalanche-growth, synchrotron-loss, partial-screening-drag, and
  bremsstrahlung observables while keeping public DREAM parity fail-closed.

- Added a fail-closed EFIT/GEQDSK external-coil sidecar contract for
  free-boundary reconstruction readiness, including strict unit, finite-value,
  turn-count, current-limit, unique-name, provenance, schema, test, and report
  fields while keeping current public GEQDSK-only rows blocked.

- Added EFIT/GEQDSK free-boundary metadata/blocker reporting so rows that need
  full free-boundary reconstruction distinguish available boundary/limiter/axis
  inputs from missing external coil-current data.

- Added full-domain trapezoidal EFIT/GEQDSK operator-current diagnostics with
  native Rust, Go, Julia, and Lean parity; local public GEQDSK refresh remains
  failed, confirming the open rows are source-domain/free-boundary blockers
  rather than a quadrature-only mismatch.

- Added EFIT/GEQDSK source-domain remediation contracts so each benchmark row
  states whether the next required solver mode is fixed-boundary profile-source
  repair, free-boundary coil/vacuum reconstruction, or both.

- Added aggregate EFIT/GEQDSK source-domain required-solver and next-action
  counts so benchmark reports expose the free-boundary reconstruction queue
  without row scanning.

- Added public-gate EFIT/GEQDSK source-domain required-solver and next-action
  counts to separate SPARC gate blockers from diagnostic-only proxy rows.

- Added EFIT/GEQDSK public-gate required-solver queue failure reasons so
  benchmark failures state the remaining full-order reconstruction class.

- Corrected EFIT/GEQDSK public profile-source mismatch failure semantics so
  accepted named-adapter rows are not counted as unresolved profile-source
  blockers while their free-boundary blockers remain explicit.

- Added EFIT/GEQDSK effective source-domain residual classification to separate
  plasma-source mismatch from vacuum/source-free operator residuals after
  accepted source-convention adaptation.

- Added EFIT/GEQDSK adapted-source plasma/vacuum residual diagnostics so
  accepted source-convention adapters expose masked source quality, not only a
  global residual.

- Added Go, Julia, and Lean native masked toroidal-current integration parity
  for the full-domain/plasma-domain Grad-Shafranov current-closure contract.

- Added Criterion benchmark evidence for native Rust GEQDSK full-domain and
  plasma-domain operator-current integration.

- Added native Rust and Rust-polyglot masked toroidal-current integration so
  full-domain and plasma-domain Grad-Shafranov current closure can be compared
  without Python wrapper delegation.

- Added GEQDSK operator-current domain attribution, reporting full-domain,
  plasma-domain, and best-domain current closure without relaxing the strict
  full-domain gate.

- Added current-limited EFIT/GEQDSK adapted-profile source diagnostics to preserve accepted ψ reconstruction while closing declared-current error bands where possible.

- Added effective EFIT/GEQDSK profile-current closure diagnostics that use accepted source-convention adapters while preserving raw canonical current evidence.

- Scoped EFIT/GEQDSK aggregate failure reasons to public gate rows while preserving diagnostic-only synthetic row evidence in counts and reports.

- Added EFIT/GEQDSK pressure and FFprime toroidal-current contribution diagnostics for profile-source current closure attribution.

- Added aggregate EFIT/GEQDSK profile-current failure-class counts to expose current-closure failure mode distribution without row scanning.

- Added EFIT/GEQDSK current-ratio and profile-current failure-class diagnostics to make profile-source current closure failures row-actionable.

- Added row-level EFIT/GEQDSK profile-current closure pass fields so profile-source current failures are machine-readable in benchmark reports.

- Added public formal-verification positioning for the Lean 4 solver safety proof, including README evidence boundary, formal-verification status, blog draft, and roadmap follow-up actions.

- Added aggregate EFIT/GEQDSK profile-current closure threshold counts and fail-closed row coverage.

- Added worst-row EFIT/GEQDSK operator/profile current-closure severity diagnostics to the aggregate benchmark report.

- Added aggregate EFIT/GEQDSK operator-current closure pass counts and fail-closed row coverage.

- Added an aggregate EFIT/GEQDSK source-sum identity gate over signed pressure, FFprime, and total source diagnostics.

- Added signed GEQDSK source-sum fields to EFIT/GEQDSK benchmark rows and schema validation.

- Added signed source-sum diagnostics to Python and Rust GEQDSK profile-source component assembly so pressure/FFprime sign regressions are not hidden by norms.

- Added native Rust benchmark evidence for GEQDSK second-order and current-conserving flux-profile interpolation primitives.

- Added a native Rust GEQDSK source-convention adapter-selection benchmark for explicit named transform ranking without fitted scales.

- Added a native Rust GEQDSK profile-source component benchmark for pressure/FFprime source assembly timing evidence.

- Added Rust GEQDSK profile-source component assembly with pressure, FFprime, total-source, plasma-mask, boundary-masking, and source-norm diagnostics matching the Python native solver contract.

- Added Rust `fusion-core::source` parity for GEQDSK second-order and current-conserving flux-profile interpolation, matching the Python profile-source source-construction contract.

- Added integrated `solve_free_boundary(..., optimize_shape=True)` diagnostics and a benchmark gate for shape-current optimization residuals, response rank, current recovery, and vacuum-boundary consistency.

- Added native Rust free-boundary shape-current inversion parity through `fusion-core::vacuum::reconstruct_shape_currents_from_boundary_flux`, matching the Python boundary-flux current-recovery benchmark contract without wrapper delegation.

- Added a free-boundary shape-control current-inversion gate that recovers bounded external coil currents from boundary-flux targets and reports current/flux residuals in the benchmark artifact.

- Hardened DIII-D/JET proxy GEQDSK validation to reuse the shared current-conserving profile-source contract, exposing plasma-mask fraction and source-norm diagnostics instead of maintaining a weaker linear interpolation path.

- Added schema v2 and fail-closed gate summary semantics to the native Grad-Shafranov operator/current-closure benchmark, preserving compatibility with the existing `passed` field while exposing `passes` and named gate evidence.

### Changed
- Documentation/API hardening: Completed the queued control-module docstring cluster
  (`runaway_electron_model`, `rust_flight_sim_wrapper`, `rwm_feedback`,
  `rzip_model`, `safe_rl_controller`, `scenario_scheduler`, `shape_controller`)
  and refreshed the internal consolidated task tracker status.
- Physics validation hardening: SPARC GEQDSK RMSE reporting now separates raw
  canonical profile-source metrics from explicitly requested public-SPARC
  source-convention adapter metrics and exposes a strict adapted-source
  contract gate.
- Benchmark schema hardening: SPARC GEQDSK RMSE artifacts now emit explicit
  benchmark id, reconstruction scope, reference role/class counts, and solver
  mode counts.
- Free-boundary benchmark hardening: solver computational-wall containment is
  now labelled as diagnostic-only instead of being displayed as a limiter
  containment pass condition.
- Free-boundary benchmark hardening: reports now include an aggregate `passes`
  field and machine-readable gate summary.
- Free-boundary benchmark hardening: the CLI now exits non-zero when any named
  benchmark gate is missing or failed.

## [3.9.5] - 2026-05-24

### Changed
- Physics hardening: integrated transport bootstrap current now uses the full Sauter path by default.
- Physics hardening: Miller geometry metrics now include elongation and triangularity shear in radial derivatives.
- Physics hardening: JAX gyrokinetic Bessel and equilibrium elliptic kernels now match SciPy reference contracts across operational domains.
- Physics hardening: alpha slowing-down deposition now uses exact first-order relaxation to preserve non-negative deposited power.
- Release readiness: metadata and release acceptance surfaces now target `v3.9.5`.


### Changed
- Validation governance hardening (Wave A): release preflight now enforces source P0/P1 issue-readiness drift checks and untested-module linkage guard checks.
- Governance hardening: refreshed local-only release-readiness reporting from
  the current codebase snapshot and removed stale hard-coded readiness counts
  from public docs.
- Security hardening: disruption predictor checkpoint loading now fails closed when `weights_only=True` is unavailable; insecure legacy torch deserialization is opt-in via `SCPN_ALLOW_INSECURE_TORCH_LOAD=1` for trusted checkpoints only.
- Security hardening: remaining `.npz` loaders in runtime/validation paths now use `allow_pickle=False` and context-managed reads (`tokamak_archive`, `neural_transport`, `fno_turbulence_suppressor`, `fno_jax_training`, `validate_real_shots`, `validate_fno_tglf`, `validate_transport_qlknn`).
- Runtime hardening: quantum bridge now validates required script presence and fails fast on non-zero subprocess exits instead of silently continuing.
- Runtime hardening: quantum bridge script launches now use bounded per-script timeouts with deterministic timeout errors.
- Runtime hardening: unified CLI now enforces per-mode subprocess timeout via `--mode-timeout-seconds` to prevent indefinite hangs.
- Runtime hardening: compiler git SHA probe now uses bounded subprocess timeout with deterministic fallback.
- Runtime hardening: native C++ solver compilation now uses a bounded subprocess timeout and timeout-aware failure handling.
- Runtime hardening: TGLF binary execution now validates finite positive timeouts and non-negative integer retry counts, and uses deterministic retry backoff.
- Runtime hardening: TGLF retry counts are now capped to a bounded range to prevent runaway retry loops from malformed configs.
- Runtime hardening: TGLF JSON output parsing now handles malformed/non-object files safely and coerces scalar or non-finite numeric payloads to finite defaults.
- Runtime hardening: TGLF text-output parsing now rejects non-finite coefficients, and benchmark comparison now handles empty TGLF reference sets without crashing.
- Runtime hardening: pretrained-surrogate cached manifest reuse now validates manifest schema and deterministically rebuilds artifacts when cache metadata is malformed.
- Tooling hardening: QLKNN training scripts now enforce secure `.npz` loading (`allow_pickle=False`) with required-key validation.
- Tooling hardening: claims audit git file discovery now uses a bounded subprocess timeout with safe fallback.
- CI hardening: `run_python_preflight.py` and `run_mypy_strict.py` now enforce bounded subprocess timeouts with explicit timeout exit handling.
- Tooling hardening: notebook upgrade bootstrap now bounds `pip install -e` subprocess runtime.
- Repo policy: local command-permission settings are now explicitly local-only and git-ignored.
- CI compatibility: `quantum_bridge` now uses postponed annotation evaluation to keep Python 3.9 import-safe while retaining modern union type hints.
- CI stability: Task 4 quasi-3D threshold gating now applies tiny floating-point comparison tolerance to avoid Python-version edge drift at boundary values.
- Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weights are absent.
- Packaging hardening: removed legacy `setup.py`; `pyproject.toml` is now the sole Python packaging metadata source.
- Packaging hardening: Docker build now installs from `pyproject.toml` with `README.md` + `validation/` staged before `pip install .`.
- Repo hygiene: removed stale `ci_old.yml` and `mypy.ini.bak` artifacts to reduce release-surface ambiguity.
- Onboarding hardening: README now includes a fast-start 45-second path, explicit capability bullets, direct Colab/Binder links, and a top-limitations snapshot.
- Validation hardening: TORAX cross-validation fallback now uses deterministic stable seeding (BLAKE2-based) instead of Python process-randomized `hash()`.
- Validation observability: TORAX benchmark artifacts now expose per-case backend, fallback reason, and fallback seed fields.
- Validation hardening: TORAX benchmark adds strict backend gating (`--strict-backend`) to fail when reduced-order fallback is used.
- Validation hardening: SPARC GEQDSK RMSE benchmark fallback no longer uses identity-like reconstruction; it now uses a deterministic reduced-order proxy.
- Validation observability: SPARC GEQDSK RMSE benchmark now records surrogate backend and fallback reason per case, plus strict backend requirement support (`--strict-backend`).
- CI hardening: Python coverage lane now runs with `--cov-branch`, and coverage guard supports optional global/domain/file branch-rate thresholds to prevent silent branch-coverage regressions.
- CI stability: calibrated line-coverage guard thresholds to current release-lane baseline for `cli.py`, `control`, and `integrated_transport_solver` while retaining regression gating.
- CI throughput hardening: Python 3.12 lane now avoids duplicate full-suite execution by running a single coverage-enabled release pass, reducing redundant runtime.
- CI resiliency hardening: Python 3.12 coverage run now has an explicit 50-minute timeout, writes diagnostics metadata/logs, and uploads diagnostics artifacts on failure.
- Coverage recovery: expanded unified-CLI test surface and re-tightened `cli.py` coverage thresholds to 80% after restoring measured coverage headroom.
- Governance hardening: the internal readiness generator now supports deterministic drift detection (timestamp-normalized comparison).
- Validation governance hardening: release preflight now includes an internal-register drift gate with an explicit skip flag for scoped local runs.
- Validation governance hardening: release preflight now includes split internal-readiness scope-report drift checks (`source` vs `docs_claims`) to reduce hardening triage noise.
- CI hardening: Python 3.12 release preflight now enables strict backend checks by default for release mode while explicitly skipping unstable SPARC/FreeGS strict lanes in that step.
- Benchmark hardening: TORAX and SPARC benchmark scripts now enforce standalone `src/` import-path determinism for CI/local parity.
- Runtime hardening: `traceable_runtime_parity.py` now enforces standalone `src/` import-path determinism for CI/local parity.
- Validation hardening: real-shot CI lane now runs `validate_real_shots.py --strict-coverage` and enforces a dedicated artifact guard for coverage minima + machine diversity.
- Governance hardening: internal-readiness scoring now down-weights narrative/planning docs so release-critical claim surfaces dominate P0/P1 triage.
- CI hardening: release preflight now enforces a deprecated-default-lane guard and persists guard diagnostics artifacts.
- Validation hardening: real-shot guard now enforces transport-machine diversity, disruption class-balance minima, calibration gate pass, and dataset-minima alignment checks.
- Validation hardening: FreeGS benchmark artifacts now include backend-availability metadata, and fallback-budget guard is availability-aware for strict FreeGS parity enforcement.
- Validation observability: untested-module linkage guard now supports JSON summary output for CI artifact diagnostics.
- Repo hygiene: expanded `.gitignore` for local CI logs, local benchmark scratch artifacts, and LaTeX build intermediates.
- Pre-commit: ruff v0.15.4→v0.15.6
- Formatting: eliminated black, ruff-only formatting

### Added
- Wave A scaffolding: `tools/coverage_guard.py` + `tools/coverage_guard_thresholds.json` with CI coverage regression gating and summary artifact upload.
- Wave A scaffolding: source-priority readiness generator with internal outputs.
- Wave A scaffolding: `tools/check_test_module_linkage.py` + allowlist policy file to block newly unlinked source modules.
- Validation hardening: added CI `Benchmark Provenance Smoke` lane to enforce TORAX/SPARC benchmark artifact provenance fields (`backend`, `fallback_reason`, seed metadata) and publish artifacts.
- Community hardening: added dedicated issue templates for real-data validation contributions and manuscript/claim review workflows.
- Release hardening: added local-only execution and release-readiness gates for scope-focused v3.9.3 preparation.
- Regression tests for secure deserialization defaults (object-array payload rejection and secure checkpoint load-path assertions).
- Regression test for deprecated FNO-controller missing-weight fallback path.
- Regression tests for TORAX benchmark deterministic fallback seeding and backend/fallback metadata fields.
- Regression tests for SPARC GEQDSK RMSE benchmark backend provenance and strict backend gating.
- Regression tests for internal-readiness drift-check behavior (up-to-date, missing-output, and drift detection paths).
- Validation hardening: added `tools/fallback_budget_guard.py` + thresholds config and wired CI fallback-budget contract on TORAX/SPARC/FreeGS benchmark artifacts.
- Runtime hardening: added `tools/runtime_parity_perf_guard.py` + thresholds config and new CI `Runtime Parity + Perf Gate` artifact contract.
- Validation hardening: added `tools/real_shot_validation_guard.py` + thresholds config and CI guard summary artifact for strict real-shot coverage lanes.
- Governance hardening: added internal split-readiness reporting for source and documentation-claim scopes.
- Session logging: moved detailed H12-H17 execution records to local-only coordination notes.
- CI/runtime hardening: added `tools/deprecated_default_lane_guard.py` with regression tests and release-preflight integration.
- Regression tests for expanded real-shot and FreeGS fallback-budget guard contracts.

## [3.9.3] - 2026-05-21

### Changed
- Physics hardening: runaway-electron evolution now enforces strict finite-domain contracts for step integration, replay evolution, and current-fraction reporting.
- Physics hardening: MHD q-profile and stability criteria now enforce strict structural and physical-domain invariants (normalised-radius bounds, q-profile consistency, and fail-fast malformed-profile rejection across Mercier/ballooning/KS/full checks).
- Physics hardening: extended MHD criteria (Troyon, NTM, RWM, peeling-ballooning) now enforce validated parameter domains with deterministic fail-fast rejection of malformed inputs.
- Runtime hardening: stability-analyser gradient sampling now uses boundary-safe central-difference indexing and positive-radius scan bounds, removing silent edge wrap behaviour.
- Documentation hardening: Sphinx core API index now explicitly includes `stability_mhd`, `stability_mhd_extended`, and `runaway_electrons` so API docs track current hardened physics surfaces.
- Release metadata sync: package, citation, and README version surfaces aligned to `3.9.3`.

---

## [3.9.3] - 2026-03-14

### Added
- Phase 5 physics: impurity transport, momentum transport, runaway electrons, Alfven eigenmodes, ELM model, pellet injection, plasma-wall interaction, kinetic EFIT
- Phase 5 control: free-boundary tracking, state estimator (EKF), volt-second manager, RWM feedback, mu-synthesis
- Phase 6 physics: disruption sequence, locked mode, plasma startup, L-H transition, MARFE, neural turbulence, orbit following, tearing mode coupling, VMEC-lite, blob transport
- Phase 6 control: detachment controller, density controller
- GK three-path: native linear eigenvalue solver, quasilinear flux model, 5 external GK interfaces (TGLF, GENE, GS2, CGYRO, QuaLiKiz), OOD detection, correction, scheduling, online learning, verification reporting
- JAX differentiable solvers: jax_gs_solver, jax_neural_equilibrium, jax_solvers (Thomas + Crank-Nicolson)
- Integrated scenario simulator, neoclassical transport, vessel model, tokamak config presets, IMAS adapter
- Phase dynamics subpackage: Kuramoto UPDE, adaptive K_nm, GK-to-UPDE bridge, plasma K_nm, Lyapunov guard
- 9 new validation benchmarks
- CoilSet extended with x_point_target, divertor_strike_points fields
- New minimal onboarding script `examples/minimal.py` that runs a reduced equilibrium solve and one SCPN controller step.
- Pre-commit scaffolding (`.pre-commit-config.yaml`) for Ruff, mypy, and Rust fmt/clippy parity checks.

### Fixed
- Impurity neoclassical pinch Z-factor bug (Hirshman & Sigmar 1981)
- Pellet injection core deposition index wraparound
- neural_turbulence.py allow_pickle security violation

### Changed
- Release delta baseline rebased (source_total: 15→50, p0p1: 13→43)
- Underdeveloped register: 115 entries (96→115 after GK port)
- Promoted hardening-wave governance, coverage, and validation guardrails from the release track to the published 3.9.3 baseline.
- Aligned packaging and release metadata to v3.9.3 across Python package metadata, citation, benchmark docs, and release readiness.

### Removed
- Michal Reiprich from all authorship metadata
- MIT OR Apache-2.0 license references (enforced AGPL-3.0 throughout)

---

## [3.9.2] — 2026-02-24

### Fixed
- Vacuum test: `_seed_plasma` early return + `solve_equilibrium` short-circuit for zero plasma current
- CI stress test: install `pytest-timeout`, handle exit code 5 (no tests collected)

### Changed
- Documentation: version refs, test count badge (1888 passed, 21 skipped)

---

## [3.9.1] — 2026-02-22

### Added — Rust Acceleration + Performance Optimizations

- **Rust Flight Sim in Stress Campaign**: `Rust-PID` controller at **0.52 us P50** vs Python PID 3,431 us (**6,600x faster**); all 5 controllers registered (PID, H-inf, NMPC-JAX, Nengo-SNN, Rust-PID)
- **Rust Transport Delegation**: `chang_hinton_chi_profile()` → Rust fast-path (4.7x speedup), `calculate_sauter_bootstrap_current_full()` → Rust fast-path (13.1x speedup) with transparent Python fallback
- **Benchmark Plot Generation**: `tools/generate_benchmark_plots.py` produces controller latency comparison, FNO suppression, and SNN trajectory plots in `docs/assets/`
- **Vectorized Transport Solver**: `_gyro_bohm_chi`, `_explicit_diffusion_rhs`, and `_build_cn_tridiag` replaced Python for-loops with NumPy array operations
- **Caching Quick Wins**: `_load_gyro_bohm_coefficient` singleton cache (avoids JSON file I/O every transport step), `_rho_volume_element` instance cache (avoids recomputing static geometry), `EpedPedestalModel` reuse in H-mode transport (avoids re-instantiation every step)
- **Numerical Stability**: Preemptive `max(nu_star, 0.0)` in Chang-Hinton to prevent NaN from negative `nu_star**(2/3)`

### Fixed

- `test_pyo3_transport_bridge.py`: Fixed Rust-vs-Python benchmarks to correctly isolate Python baseline by temporarily disabling `_rust_transport_available` flag
- mypy `no-redef` suppression for Rust transport import pattern

## [3.9.0] — 2026-02-21

### Added — QLKNN-10D Real-Data Training Pipeline

- **QLKNN-10D Data Acquisition**: `tools/download_qlknn10d.py` — downloads 300M QuaLiKiz flux calculations from Zenodo (DOI: 10.5281/zenodo.3497066) with SHA256 checksum verification and resume support
- **Data Pipeline**: `tools/qlknn10d_to_npz.py` — converts QLKNN-10D HDF5 to training `.npz` files with column mapping, gyro-Bohm → physical conversion, regime classification (ITG/TEM/stable), and stratified 90/5/5 splits
- **MLP Training on Real Data**: `tools/train_neural_transport_qlknn.py` — JAX trainer with Adam + cosine annealing, verification gates (refuses to save if test_relative_l2 >= 0.05), and GPU auto-detection
- **FNO Spatial Data Generation**: `tools/generate_fno_qlknn_spatial.py` — uses trained QLKNN MLP as oracle to generate (equilibrium, transport_field) spatial pairs for FNO training
- **FNO Training on Real Data**: `tools/train_fno_qlknn_spatial.py` — JAX FNO training on spatial transport data with spectral convolution, targeting relative L2 < 0.10 (down from 0.79 on synthetic data)
- **GPU Diagnostic**: `tools/check_gpu.py` — reports JAX GPU, PyTorch CUDA, and Rust wgpu availability with device details
- **Validation Suite**: `validation/validate_transport_qlknn.py` — validates trained MLP against held-out QLKNN test set with per-output and per-regime metrics, compares against published benchmarks
- **Published Benchmarks**: `validation/reference_data/qlknn10d_published_benchmarks.json` — van de Plassche et al., *Phys. Plasmas* 27, 022310 (2020) accuracy figures
- **Variable-Depth MLP Loader**: `neural_transport.py` auto-detects MLP depth from `.npz` keys (w1/b1, w2/b2, ..., wN/bN), supporting 2+ layer architectures while maintaining backward compatibility with existing 3-layer weights
- **GELU Activation**: MLP forward pass uses GELU (matching JAX/PyTorch training) instead of ReLU for hidden layers
- **GPU Optional Dependency**: `pyproject.toml` adds `gpu = ["jax[cuda12]>=0.4.20"]`

### Fixed

- Removed inflated FNO claims: "0.9997 TGLF correlation" (was based on 3 synthetic data points) and "98% suppression efficiency" (was hardcoded, never measured) from README.md, RESULTS.md, BENCHMARKS.md, and DOE pitch
- Fixed corrupted UTF-16 encoded entries in `.gitignore`
- `validation/collect_results.py` now reads actual metrics from manifest instead of hardcoding values

---

## [3.8.3] — 2026-02-21

### Added

- **Physics hardening**: Pydantic schema validation, Modified Rutherford Equation, IPB98(y,2) scaling, cylindrical neutronics, Thomas-Fermi WDM EOS, Tikhonov-regularized MIMO, Kalman state estimation, anti-windup PID
- **Telemetry**: Circular buffers, structured JSON logging across all core modules
- **D-T Reactivity Fix**: Replaced simplified Huba fit with NRL Plasma Formulary 5-coefficient Bosch-Hale parameterisation for `sigma_v_dt()` — accurate to <1% across 1-100 keV
- **C_fus Recalibration**: Fusion power constant recalibrated to `3.68e-18` matching 500 MW at n=1e20, T=15 keV
- **q_proxy Boundary Hardening**: Added `max(0.1, ...)` floor and `np.errstate` guards preventing NaN at plasma edge
- **Complete Package Exports**: All 7 Python packages (`control/`, `core/`, `diagnostics/`, `engineering/`, `hpc/`, `io/`, `ui/`) now have complete `__init__.py` with lazy `__getattr__` imports (was 2/7)
- **Circular Import Resolution**: Rewrote `control/__init__.py` and `core/__init__.py` with lazy import pattern to break `io` → `core` → `control` → `core` dependency chain
- **Maturin PyO3 Build**: `scpn_fusion_rs` wheel built and verified — all `PyRustFlightSim` tests passing
- **Version Sync**: Aligned all version references: pyproject.toml, setup.py, Rust crates (fusion-python 3.4.0→3.8.3), release readiness (v3.5.0→v3.8.3)
- **Missing Dependencies**: Added `pydantic>=2.0` and `pandas>=1.5` to pyproject.toml and setup.py
- **Python Version**: Unified `requires-python` to `>=3.9` across all manifests
- **Clippy Fix**: Removed unused import in `fusion-control/constraints.rs` (CI `-D warnings`)

### Changed

- Test suite: 1888 passed, 0 failed, 21 skipped (was 63 failures)
- CI: All 9 lanes green (5 Rust + 4 Python)
- `sync_metadata.py`: Fixed bug that corrupted all historical CHANGELOG version headers (replaced ALL `## [x.y.z]` instead of only the first)

### Also included (from 2026-02-20 pre-release)

- **Rust-native flight simulator**: `RustFlightSim` with 10 kHz+ control loops, `IsoFluxController`, actuator delay line, per-step timing
- **Neural equilibrium surrogate**: PCA+MLP `NeuralEquilibriumKernel` — ~1000x speedup over Picard iteration
- **JAX FNO**: `fno_jax_training.py` — spectral convolution for turbulence surrogate training (1000 samples, 64x64)
- **Jitter tracking**: Sub-microsecond timing instrumentation for control loop latency

## [3.8.2] — 2026-02-19

### Added — Project TOKAMAK-MASTER: Physics Hardening & NMPC

- **Nonlinear MPC (NMPC)**: Implemented a new `NonlinearMPC` controller in `src/scpn_fusion/control/fusion_nmpc_jax.py`:
  - Uses a **Neural ODE** surrogate model ($dx/dt = f_{\theta}(x, u)$) for high-fidelity nonlinear prediction.
  - Full **JAX integration** with JIT-compiled gradients, achieving **17.4x speedup** over SciPy baseline (13.18 ms vs 229 ms).
  - Robust fallback to SciPy L-BFGS-B when JAX is unavailable.
- **Resistive MHD (Rutherford)**: Upgraded magnetic island modeling in the Digital Twin (`tokamak_digital_twin.py`):
  - Implemented the **Rutherford Equation** ($dW/dt = \eta \Delta'$) for dynamic island width evolution.
  - Replaced static q-surface masking with physically evolved island growth and saturation.
- **SOC Physical Calibration**: Hardened the sandpile reactor model (`advanced_soc_fusion_learning.py`):
  - Mapped abstract "topples" to real **physical energy units (Megajoules)**.
  - Added `energy_per_topple_mj` calibration (default 0.05 MJ) for quantitative ELM energy analysis.
- **Unified State Space**: Introduced `FusionState` dataclass in `src/scpn_fusion/core/state_space.py`:
  - Standardized representation of axis position, global physics (Ip, beta), and topological danger zones.
  - Direct construction from physics kernels and easy vectorization for ML/MPC pipelines.
- **Validation**: Added `validation/validate_core_integration.py` and `validation/benchmark_nmpc.py` to confirm integration and quantify performance gains.

## [3.5.0] — 2026-02-19

### Added — v3.5.0: Peer-Review Upgrade Phase Kickoff

- Added canonical inhibitor-arc safety interlock lane in `scpn.safety_interlocks`:
  - `build_safety_net()` with 5 safety channels (`thermal`, `density`, `beta`, `current`, `vertical`)
  - deterministic runtime evaluator (`SafetyInterlockRuntime`)
  - state-to-token safety mapping and contract-proof hooks
- Added formal safety contract surface in `scpn.contracts`:
  - `SafetyContract`
  - `DEFAULT_SAFETY_CONTRACTS`
  - `verify_safety_contracts(...)`
- Integrated safety interlocks into `NeuroCyberneticController`:
  - per-step safety checks
  - control-command inhibition on safety trips
  - summary metrics for interlock trips and contract violations
- Added safety test suite `tests/test_safety_interlocks.py` covering:
  - inhibitor compile path
  - per-channel inhibition behavior
  - combined trip behavior
  - formal contract violation detection

### Changed

- Version bumped to `3.5.0` across release metadata:
  - `pyproject.toml`
  - `setup.py`
  - `src/scpn_fusion/__init__.py`
  - `CITATION.cff`
- Hardened Task-5 disruption contracts with uncertainty-aware proxies:
  - `mcnp_lite_tbr(..., return_uncertainty=True)` now returns `tbr_sigma` and p95 bounds
  - `run_disruption_episode()` now reports uncertainty-envelope metrics (`risk_p95_*`, `wall_damage_p95_*`, `tbr_p95_*`, `uncertainty_envelope`)
  - Added disruption-contract tests covering uncertainty-mode bounds and episode-envelope consistency
- Added autonomous hardening-governance tooling:
  - internal readiness queue generation
  - `tools/claims_audit.py` + `validation/claims_manifest.json`
  - `tools/run_python_preflight.py` now includes claims-evidence audit check
  - internal milestone board with prioritised v3.6 top-20 hardening tasks
- Split release vs research validation gates (A03):
  - Added gate profiles in `tools/run_python_preflight.py` (`--gate release|research|all`)
  - Added research-only pytest marker contract (`@pytest.mark.experimental`)
  - Added CI split lane `python-research-gate` and release-only pytest execution (`-m "not experimental"`)
- Added internal gate matrix documentation.
- Added claims evidence map generation (A19):
  - Added internal claims evidence map generation
  - Added preflight drift check to keep map synchronized with `validation/claims_manifest.json`
- Fixed CI claims-audit stability:
  - Replaced untracked RMSE dashboard artifact dependency in `validation/claims_manifest.json` with tracked evidence sources.
  - Hardened `tools/claims_audit.py` to require evidence files/pattern files be git-tracked.
- Added release docs version-consistency gate (A18):
  - `tests/test_version_metadata.py` now enforces current-version references in `README.md`, `RESULTS.md`, `VALIDATION.md`, and `CHANGELOG.md`.
- Added release readiness gate (A20):
  - Added internal release readiness validation tooling.
  - Wired release readiness verification into Python preflight and tag publish workflow (`.github/workflows/publish.yml`).
- Added disruption shot provenance manifest gate (A04):
  - Added `tools/generate_disruption_shot_manifest.py` and generated `validation/reference_data/diiid/disruption_shots_manifest.json`.
  - Added preflight `--check` gate to enforce hash-locked shot manifest freshness in release lanes.
- Added disruption shot split leakage gate (A05):
  - Added `validation/reference_data/diiid/disruption_shot_splits.json` and `tools/check_disruption_shot_splits.py`.
  - Added release preflight gate to fail on train/val/test overlap or missing shot coverage vs manifest.
- Hardened disruption replay ingest contracts (A06):
  - Added strict payload-schema loader `load_disruption_shot_payload()` in `validation/validate_real_shots.py`.
  - `validate_disruption()` now emits explicit per-shot schema failures for malformed payloads (signal keys, finite checks, timebase monotonicity, disruption-index bounds).
  - Added `tests/test_validate_real_shots_payload.py` and aligned `tests/test_disruption_threshold_sweep.py` to use the same payload contract.
- Added calibrated disruption-risk holdout lane (A07):
  - Added `tools/generate_disruption_risk_calibration.py` with deterministic `--check` drift mode.
  - Added calibrated artifact + report:
    - `validation/reference_data/diiid/disruption_risk_calibration.json`
    - `validation/reports/disruption_risk_holdout_report.md`
  - Added release preflight gate wiring for calibration holdout checks.
  - `validation/validate_real_shots.py` disruption lane now consumes calibrated threshold/bias settings from the calibration artifact.
- Tightened EPED-like domain validity contracts (A12):
  - Added calibrated-domain metadata + explicit domain assessment to `src/scpn_fusion/core/eped_pedestal.py`.
  - Added bounded extrapolation penalties and strict/warn/ignore domain modes for `predict(...)`.
  - Added transport telemetry surfacing (`_last_pedestal_contract`) in `src/scpn_fusion/core/integrated_transport_solver.py`.
  - Added benchmark artifact/report gate:
    - `validation/benchmark_eped_domain_contract.py`
    - `validation/reports/eped_domain_contract_benchmark.json`
    - `validation/reports/eped_domain_contract_benchmark.md`
  - Added release preflight wiring for `validation/benchmark_eped_domain_contract.py --strict`.
- Added end-to-end closed-loop latency accounting (A09):
  - Added deterministic latency benchmark lane `validation/scpn_end_to_end_latency.py`.
  - Added release artifacts:
    - `validation/reports/scpn_end_to_end_latency.json`
    - `validation/reports/scpn_end_to_end_latency.md`
  - Added release preflight wiring for `validation/scpn_end_to_end_latency.py --strict`.
  - Added tests for latency-lane determinism, thresholds, markdown output, and CLI artifact generation.
- Added default disruption replay pipeline contracts (A10):
  - Added replay pipeline module `src/scpn_fusion/control/replay_pipeline.py` with validated defaults for sensor preprocessing and actuator lag toggles.
  - Wired replay pipeline preprocessing/lag into real-shot replay and disruption validation:
    - `src/scpn_fusion/control/disruption_contracts.py`
    - `validation/validate_real_shots.py`
  - Added replay-pipeline benchmark lane + release artifacts:
    - `validation/benchmark_disruption_replay_pipeline.py`
    - `validation/reports/disruption_replay_pipeline_benchmark.json`
    - `validation/reports/disruption_replay_pipeline_benchmark.md`
  - Added release preflight wiring for `validation/benchmark_disruption_replay_pipeline.py --strict`.
  - Added contract tests for replay-pipeline config validation, deterministic behavior, and disabled-lane invariants.
- Added transport uncertainty-envelope contracts (A13):
  - `validation/validate_real_shots.py` transport lane now emits p50/p95 uncertainty-envelope metrics for absolute relative error, residual spread, sigma spread, and z-score coverage.
  - Added benchmark gate + release artifacts:
    - `validation/benchmark_transport_uncertainty_envelope.py`
    - `validation/reports/transport_uncertainty_envelope_benchmark.json`
    - `validation/reports/transport_uncertainty_envelope_benchmark.md`
  - Added release preflight wiring for `validation/benchmark_transport_uncertainty_envelope.py --strict`.
  - Added tests for envelope contract fields, benchmark strict mode, markdown output, and preflight command ordering.
- Added multi-ion transport conservation contracts (A11):
  - Added strict benchmark lane for D/T/He-ash transport integrity:
    - `validation/benchmark_multi_ion_transport_conservation.py`
    - `validation/reports/multi_ion_transport_conservation_benchmark.json`
    - `validation/reports/multi_ion_transport_conservation_benchmark.md`
  - Benchmark enforces finite/positive species state, quasineutral closure residual bounds, late-window energy-error stability, and He-ash accumulation.
  - Added release preflight wiring for `validation/benchmark_multi_ion_transport_conservation.py --strict`.
  - Added benchmark tests and updated preflight command-order/skip-flag tests.

---

## [3.4.0] — 2026-02-18

### Changed — v3.4.0: Unified CLI + Release Surface Hardening

- Replaced legacy `os.system` launcher with structured `click` CLI (`scpn-fusion`) and `subprocess` execution.
- Added single-command suite execution via `scpn-fusion all --surrogate --experimental`.
- Kept `run_fusion_suite.py` as compatibility wrapper forwarding to the new CLI.
- Added CLI regression tests in `tests/test_cli_launcher.py`.
- Aligned release metadata to `3.4.0` across `pyproject.toml`, `setup.py`, `src/scpn_fusion/__init__.py`, and `CITATION.cff`.
- Added coverage upload lane to CI and updated README badges/quickstart docs.
- Added Streamlit demo runbook: `docs/STREAMLIT_DEMO_PLAYBOOK.md`.
- Clarified type-checking lane wording from "strict" claim to explicit "mypy gate" where configured.

### Notes

- `v3.2.0` and `v3.3.0` were internal development trains and were not published as official git tags/releases.
- `v3.4.0` is the next official tagged release after `v3.1.0`.

---

## [3.3.0] — 2026-02-18 (Historical Backfill)

### Added / Changed — v3.3.0: Phase 1+2+3 Integration Wave

- Integrated multi-ion transport lane and coupled physics guardrails from the Phase 1+2 train.
- Added coil-optimization and extended physics-invariant coverage in the main suite.
- Advanced OMAS/TGLF coupling scaffolding and related interoperability paths.
- Updated publication/paper artifacts in the Phase 3 documentation wave.

### Notes

- `v3.3.0` was developed and merged on 2026-02-18, but was not tagged at the time; release/tag were backfilled later for historical continuity.
- `v3.2.0` was not published as an official release tag.

---

## [3.1.0] — 2026-02-17

### Changed — v3.1.0: Phase 0 Physics Hardening

#### P0.1 — TBR Realism (blanket_neutronics.py)
- Added `port_coverage_factor` (default 0.80), `streaming_factor` (default 0.85), and `blanket_fill_factor` (default 1.0) to `calculate_volumetric_tbr()` and `MultiGroupBlanket.solve_transport()`
- Corrected TBR now falls in Fischer/DEMO range [1.0, 1.4] instead of ideal 1.67
- New `tbr_ideal` field preserved on `VolumetricBlanketReport` and 3-group result dicts
- Validation against Fischer et al. (2015) and DEMO blanket studies

#### P0.2 — Q-Scan Greenwald & Temperature Limits (fusion_ignition_sim.py)
- Greenwald density limit enforced: `n_GW = I_p / (pi * a^2)`, scan points above 1.2x n_GW skipped
- Temperature hard cap at 25 keV with `UserWarning` emission (was 100 keV)
- Q ceiling at 15 (was unbounded, producing Q=98 artifacts from 0-D model)
- `find_q10_operating_point()` now returns `n_greenwald` in result dict

#### P0.6 — Energy Conservation Enforcement (integrated_transport_solver.py)
- Per-timestep conservation diagnostic after Crank-Nicolson solve
- `_last_conservation_error` attribute stores relative energy balance error
- `enforce_conservation=True` parameter raises `PhysicsError` if error > 1%
- New `PhysicsError(RuntimeError)` exception class

#### P0.3 — Dashboard Auto-Flagging + Plots (rmse_dashboard.py)
- `THRESHOLDS` dict with PASS/WARN/FAIL auto-flags for tau_E, beta_N, axis RMSE, FPR, TBR, Q
- Auto-flag summary table at top of rendered markdown report
- `render_plots()` generates matplotlib scatter plots (tau_E, beta_N) and bar charts (SPARC axis error)
- Plots saved to `artifacts/` and embedded as relative links in markdown

#### P0.4 — CI Gate Hardening (ci_rmse_gate.py)
- Disruption FPR promoted from soft warn to hard fail (threshold 0.15, was 0.40)
- TBR gate: fail if outside [1.0, 1.4] (new)
- Q gate: fail if Q_peak > 15 (new)

#### P0.5 — Issue Templates (.github/ISSUE_TEMPLATE/)
- `feature_request.md` and `bug_report.md` converted to YAML form format (`.yml`)
- Mandatory "Physics Reference" field (`required: true`) with DOI/arXiv + equation number
- Old markdown templates removed

#### Tests
- 24 new tests in `tests/test_phase0_physics_fixes.py` across 3 classes:
  - `TestTBRCorrection` (11 tests): correction ratios, unit factors, fill factor, invalid inputs
  - `TestQScanLimits` (7 tests): Q ceiling, T cap, Greenwald limit, extreme low current
  - `TestEnergyConservation` (6 tests): conservation attribute, zero-heating monotonic cooling, PhysicsError
- Total: 1141 tests passing (57 skipped, 1 pre-existing stochastic failure)

---

## [3.0.0] — 2026-02-17

### Added — v3.0.0: Rust SNN Bindings, Full-Chain UQ, Shot Replay

#### Rust SNN via PyO3 (Task 3.1)
- `PySnnPool` wrapping `SpikingControllerPool` with `step(error)`, `n_neurons`, `gain`, `window_size` getters
- `PySnnController` wrapping `NeuroCyberneticController` with `step(measured_r, measured_z)`, `target_r/z` getters
- Python-side wrappers in `_rust_compat.py`: `RustSnnPool`, `RustSnnController` with graceful fallback
- 19 tests in `tests/test_snn_pyo3_bridge.py` (skip when Rust extension not compiled)

#### Monte Carlo UQ Through Full Solver Chain (Task 3.2)
- `quantify_full_chain(scenario, n_samples, seed, chi_gB_sigma, pedestal_sigma, boundary_sigma)`
  propagates uncertainty through equilibrium → transport → fusion power
- `EquilibriumUncertainty`, `TransportUncertainty`, `FullChainUQResult` dataclasses
- `summarize_uq()` for JSON-serializable output with p5/p50/p95 bands
- Backward-compatible: `quantify_uncertainty()` preserved for IPB98-only UQ
- 12 tests in `tests/test_uq_full_chain.py`

#### FNO Turbulence Surrogate Deprecated (Task 3.3)
- Module docstrings updated: EXPERIMENTAL → DEPRECATED/EXPERIMENTAL
- Runtime `FutureWarning` on `FNO_Controller` initialization
- Removed from default pipeline; will be retired in v4.0 unless real gyrokinetic data available
- Trained on 60 synthetic Hasegawa-Wakatani samples only (relative L2 = 0.79)

#### Shot Replay Streamlit Tab (Task 3.5)
- 5th tab "Shot Replay" in `app.py` for measured vs simulated diagnostic overlay
- Loads DIII-D disruption NPZ files from `validation/reference_data/disruption_shots/`
- Plots Ip, ne, Te time traces with disruption predictor risk overlay
- Disruption time marker and status display

#### Paper Manuscripts Updated (Task 3.4)
- Paper A (equilibrium solver): version refs v2.1.0, added v2.1.0 improvements section
- Paper B (SNN controller): version refs v2.1.0, PyO3 bindings note, disruption recalibration note

### Changed
- Version bumped to 3.0.0
- FNO warning category changed from `UserWarning` to `FutureWarning`

---

## [2.1.0] — 2026-02-17

### Added — v2.1.0: Physics Hardening & Self-Consistent Transport

#### Disruption Predictor Recalibration
- Feature weight recalibration: instability indicators (std, slope) prioritised over
  raw amplitude (max_val) — safe high-power shots no longer trigger false alarms
- Default risk threshold shifted from 0.65 to 0.50 (Pareto-optimal on 16-shot dataset)
- Threshold/bias sweep tool (`tools/sweep_disruption_threshold.py`) with ROC curve generation
- FPR reduced from 90% to 0%, recall maintained at 100% on reference dataset
- 7 new tests in `tests/test_disruption_threshold_sweep.py`

#### GS-Transport Self-Consistency Loop
- `TransportSolver.run_self_consistent()` outer iteration: transport → `map_profiles_to_2d()` →
  `solve_equilibrium()` → psi convergence check
- Backward-compatible: `run_to_steady_state(self_consistent=True)` delegates to new loop
- Psi convergence tracked via `||psi_new - psi_old|| / ||psi_old||` with configurable tolerance
- 14 new tests in `tests/test_gs_transport_coupling.py`

#### MHD Stability Expansion (2 → 5 criteria)
- Kruskal-Shafranov: `q_edge > 1` external kink safety check
- Troyon beta limit: `beta_N < g` with configurable no-wall (g=2.8) and ideal-wall (g=3.5) coefficients
- NTM seeding: simplified Modified Rutherford equation with bootstrap-drive marginal island width
- `run_full_stability_check()` convenience function returning `StabilitySummary` dataclass
- All new types exported from `scpn_fusion.core`
- 15 new tests in `tests/test_mhd_stability.py` (25 total)

#### beta_N Estimator Calibration
- `DynamicBurnModel` replaces hardcoded `FusionBurnPhysics` in validation dashboard
- Profile peaking correction factor 1.446 (geometric mean of ITER/SPARC calibration)
- ITER beta_N error: -96% → -2.8%; SPARC: -42% → +3.0%; RMSE: 1.26 → 0.042
- CI gate tightened from 2.00 to 0.10

#### FreeGS Blind Benchmark
- `validation/benchmark_vs_freegs.py` with Solov'ev analytic fallback (no freegs dependency)
- 3 test cases: ITER-like (R0=6.2m), SPARC-like (R0=1.85m), spherical tokamak (R0=0.85m)
- PSI NRMSE threshold: 10%
- `freegs>=0.6` added to `[benchmark]` optional dependencies
- 27 new tests in `tests/test_freegs_benchmark.py` (+ 2 skipped when freegs absent)

### Changed
- Disruption predictor thermal_term coefficients: max_val 0.55→0.03, std 0.35→0.55, slope 0.25→0.50
- Disruption predictor state_term coefficients: mean 0.15→0.02, last 0.20→0.02
- RESULTS.md updated with calibrated beta_N metrics and resolved-status flags
- Version bumped to 2.1.0

---

## [2.0.0] — 2026-02-17

### Added — v2.0.0: Publication-Grade Physics Validation

#### Equilibrium Solver
- Pure-Python geometric multigrid V-cycle (full-weighting restriction, bilinear prolongation, Red-Black SOR smoother) with toroidal GS* stencil
- Python multigrid wired as default solver (replaces SOR) — 129x129 converges to <1e-6 in <500 V-cycles
- PyO3 binding for Rust multigrid `multigrid_vcycle()` with Python fallback
- Real DIII-D EFIT GEQDSK equilibria replacing synthetic Solov'ev files
- Psi/q-profile overlay validation against EFIT ground truth
- Neural surrogate retrained on expanded 18+ GEQDSK dataset

#### Transport Physics
- EPED-like pedestal model (Snyder 2009 simplified scaling) for H-mode boundary conditions
- Gyro-Bohm transport calibration against ITPA H-mode confinement database (20 shots)
- IPB98(y,2) uncertainty quantification with log-linear error propagation (Verdoolaege 2021)
- Transport validation against ITPA CSV with RMSE dashboard integration

#### Control & Disruption
- Proper H-infinity synthesis via Riccati equations (Doyle-Glover-Khargonekar) replacing fixed-gain PID
- Real DIII-D disruption shot data (5 disruptions + 5 safe controls)
- Disruption predictor retrained on mixed real+synthetic data (>70% recall target)
- Closed-loop disruption mitigation with real shot replay
- Realistic actuator model (sensor noise, coil rate limits, measurement delay)
- SNN vs MPC vs PID disturbance rejection benchmark
- Petri net formal verification documentation

#### Integration & IO
- IMAS IDS conformance audit with JSON Schema validation
- MDSplus automated download with NPZ caching
- GitHub issue templates for shot replay requests
- GitHub issue template for shot replay requests (`shot_replay_request.md`); Streamlit Shot Replay tab planned for v2.1

#### Validation
- `validate_real_shots.py` capstone validation (5 shots, equilibrium + transport + disruption)
- Rust/Python parity test suite (relative tolerance < 1e-3)
- RMSE regression gate in CI (tau_E RMSE, psi NRMSE thresholds)
- HIL demo documentation with FPGA register map

### Changed
- Default equilibrium solver changed from SOR to multigrid
- Transport model uses gyro-Bohm scaling instead of constant chi_base=0.5
- Version bumped to 2.0.0

---

## [2.0.0-pre] — 2026-02-15 (Pre-release hardening)

### Added

#### 3D MHD and Exascale Readiness
- VMEC fixed-boundary 3D equilibrium solver (Hirshman & Whitson 1983 variational approach)
  in `fusion-core/src/vmec_interface.rs`
- BOUT++ coupling interface with Newton-iteration flux surface tracing, field-aligned metric
  tensors (`g^ij`, Jacobian, `|B|`), and stability result parsing
- 2D Cartesian MPI domain decomposition with Additive Schwarz distributed GS solver
  (Rayon thread-parallel) in `fusion-core/src/mpi_domain.rs`
- Optimal process-grid factorisation (surface-to-volume minimisation)
- 4-face halo exchange (serial reference, rsmpi-ready interface)
- GS residual L2 norm for convergence monitoring

#### Physics and Transport
- Coulomb collision operator (Fokker-Planck Monte Carlo with NRL Coulomb logarithm,
  Spitzer slowing-down time, critical velocity, pitch-angle Langevin scattering,
  deterministic xorshift64 PRNG) in `fusion-core/src/particles.rs`
- GPU equilibrium solver via wgpu compute shaders (WGSL Red-Black SOR on
  DX12/Vulkan/Metal with R-singularity protection) in new `fusion-gpu` crate
- Neoclassical transport closure (Chang-Hinton 1982 ion thermal diffusivity,
  banana/plateau/PS regimes, bootstrap current) in Rust and Python

#### Validation and Experimental Data
- DIII-D shot 187070 and JET shot 92436 synthetic Solov'ev GEQDSK validation profiles
  (10 files: L-mode, H-mode, negative triangularity, snowflake, DT, hybrid, high-Ip)
  on 129x129 grids with generation script and validation runner
- DIII-D/JET validation runner with strict GS-operator and GEQDSK payload integrity contracts

#### Documentation and Community
- Sphinx documentation site (26 files): Furo-themed conf.py with autodoc, napoleon,
  mathjax, intersphinx; installation, quickstart, and 8 user guide chapters with
  physics equations; API reference for all 59+ Python modules across 8 subpackages
- Paper A LaTeX manuscript: GS equilibrium solver (840 lines, target Nuclear Fusion / CPC)
- Paper B LaTeX manuscript: SNN vertical stability controller (939 lines, target FED / IEEE TPS)
- Shared bibliography (55 BibTeX entries, 643 lines)
- DOE ARPA-E convergence pitch document (785 lines): GAMOW, BETHE, FES, ASCR program
  alignment matrices with 36-month milestone roadmap
- GitHub PR template with testing/quality/physics criteria
- Issue template config (security contact, blank issues disabled)
- CONTRIBUTING.md comprehensive upgrade (426 lines: dev setup, style guide, PR process)
- Security policy (SECURITY.md) expanded with version table, hardening summary, known limitations
- Coverage badge, Docker quickstart, pure-Python install, and data licensing in README

#### Reactor Engineering Elevation (H6 wave, 9 tasks)
- VMEC IO bridge for non-axisymmetric equilibrium interoperability (`H6-001`)
- Hybrid kinetic-fluid alpha/runaway test-particle tracker with orbit/energy statistics (`H6-002`)
- Synthetic forward diagnostics (interferometer phase, neutron count channels) (`H6-003`)
- IMAS/IDS adapter pattern for Digital Twin state interchange (`H6-004`)
- CAD geometry integration lane (STEP/STL mesh ingestion for heat/neutron load tracing) (`H6-005`)
- Explicit actuator transfer-function dynamics in flight simulation/control path (`H6-006`)
- Domain-randomization chaos monkey for deterministic sensor dropout/noise injection (`H6-007`)
- Multi-node MPI domain-decomposition scaffolding for core grid solves (`H6-008`)
- Traceable control-loop runtime lane to reduce Python interpreter overhead (`H6-009`)

### Changed

#### SCPN Controller Optimisation (H5 wave)
- Vectorized injection, action decode, and array-native stepping in SCPN controller
- Internal marking state kept in NumPy arrays with compatibility accessors
- Compiled feature-evaluation path removes per-tick dict churn
- Preallocated dense/update work buffers for tick-loop allocation reduction
- Binomial sampling replaces high-allocation mirrored-draw stochastic path
- Rust stochastic-firing kernel bridge for optional backend offload
- Default runtime profile changed to adaptive (binary probabilistic margin)
- Expanded Rust offload eligibility and chunked antithetic sampling for large nets
- Precomputed transition-delay indices remove per-tick timing mask churn

#### README Repositioning
- Repositioned project identity as a **neuro-symbolic control framework** rather than a
  physics simulation suite competing with TRANSP/GENE/JINTRAC
- Expanded Design Philosophy, Neuro-Symbolic Compiler section, simulation mode maturity
  tiers, Code Health & Hardening section, and Known Limitations & Roadmap
- Honest benchmark claims: separated validated from projected performance numbers

#### Runtime and Configuration
- Experimental modes gated behind `--experimental` flag
- Removed `sys.path` hacks from 20+ core/validation/test modules (replaced with package imports)
- Realtime simulation entrypoint resolves config relative to script root with `--config` support
- Docker image hardened with prod-by-default dependency install and non-root user
- HPC bridge restricts native solver library loading to trusted package paths with
  explicit `SCPN_SOLVER_LIB` env override
- Stabilized GAI-02 latency metric with deterministic hardware-normalized proxy
- PID/MPC and GNEU benchmark controllers pinned to adaptive non-zero binary margin
- NumPy 2.4 compatibility restored for blanket TBR integration (trapezoid fallback)
- GitHub Pages deployment gated behind explicit `DEPLOY_GH_PAGES` repo variable

#### Licensing and Legal
- AGPL-3.0-or-later added to all package manifests
- SPARC MIT license notice and ITPA provenance disclaimer added
- TestPyPI publish made best-effort in CI

### Fixed

#### Hardening Waves (266 tasks completed across 8 waves)

**S2 (8 tasks):** HPC bridge in-place solve paths, C++ SOR convergence guards,
SCPN benchmark stochastic-vs-float equivalence gate, disruption predictor fallback,
PWI angle-energy invariants and redeposition bounds, TEMHD solver edge-case regression,
path mapping normalisation, release gate queue health.

**S3 (6 tasks):** SCPN topology diagnostics for dead nodes and unseeded place cycles,
inhibitor-arc support with explicit opt-in, compact artifact serialization mode,
control simulation fallback entry points, HPC bridge edge-path validation,
S3 queue health in release-readiness report.

**S4 (4 tasks):** Compiler topology/inhibitor compile controls, deterministic compact
artifact codec smoke checks, deterministic flight-sim CI path, S4 release gate lane.

**H5 (37 tasks):** SCPN controller optimisation (vectorized injection, array-native
stepping, Rust offload, binomial sampling, compiled feature evaluation, reusable scratch
buffers); HPC library loading security (trusted-path restriction); artifact decompression
bounds (zip-bomb prevention); solver fail-fast on divergence; Docker hardening (non-root);
CI `sc-neurocore` pinning; strict py312 mypy typing; adaptive margin regression locks;
`sys.path` removal across 20+ modules.

**H6 (9 tasks):** Reactor engineering elevation -- VMEC IO, particle tracker, forward
diagnostics, IMAS/IDS adapter, CAD raytrace, actuator dynamics, chaos monkey, MPI
scaffolding, traceable control loop.

**H7 (90 tasks):** Comprehensive Python runtime hardening -- deterministic non-interactive
summary APIs for all control/nuclear/diagnostics modules; scoped RNG (elimination of
global `np.random` mutation) across disruption predictor, digital twin, flight sim,
optimal control, SOC learning, FNO turbulence, design scanner, diagnostics, and
benchmark runners; strict input guards (finite/range/type validation) across all
control, nuclear, diagnostics, validation, SCPN contract, HPC, and IO modules;
NumPy LIF fallback for neuro-cybernetic controller; director interface fallback
oversight; tomography non-SciPy solve path.

**H8 (112 tasks):** Rust runtime hardening -- finite-value guards, shape validation,
and domain bounds enforcement across all `fusion-core` crates (kernel, inverse,
particles, transport, AMR, stability, RF heating, vacuum, B-field, X-point, memory
transport, plasma source, VMEC interface, MPI domain, JIT, pedestal, ignition) and
all `fusion-control` crates (MPC, PID, IsoFlux, SNN, optimal, SPI, analytic, SOC
learning, digital twin with OU noise/chaos monkey/SimpleMLP/Plasma2D/actuator);
explicit `FusionResult` error propagation replacing silent coercions/clamps/fallbacks;
rustfmt and clippy CI fixes; Python-side IDS validation tightening, forward diagnostics
grid-axis monotonicity enforcement, psi RMSE convergence/encoding/input contracts,
solver-method bridge regression tests.

---

## [2.0.0-pre] — Pre-Hardening

### Point-Wise ψ RMSE Validation (All 8 SPARC GEQDSKs)

- Added `validation/psi_pointwise_rmse.py` — self-contained module for point-wise
  ψ(R,Z) reconstruction error metrics on real SPARC equilibrium data:
  - `gs_operator()`: finite-difference Grad-Shafranov operator Δ*ψ
  - `gs_residual()`: relative L2 and max-abs GS residual
  - `manufactured_solve_vectorised()`: Red-Black SOR with reference boundary conditions
  - `compute_psi_rmse()`: normalised ψ RMSE, plasma-region RMSE, max pointwise error
  - `validate_file()` / `validate_all_sparc()`: per-file and aggregate runners
  - `sparc_psi_rmse()`: drop-in for rmse_dashboard integration
- Added `tests/test_psi_pointwise_rmse.py` — 17 tests covering GS operator
  analytics, SPARC integration, manufactured solve, RMSE metrics
- Integrated psi RMSE into `validation/rmse_dashboard.py` report output

### Neural Equilibrium Rewrite & Mode Deprecation

- Rewrote `src/scpn_fusion/core/neural_equilibrium.py` from 157 → 530 lines:
  - Removed `sklearn` dependency → `MinimalPCA` (pure NumPy SVD)
  - Removed `pickle` persistence → `.npz` save/load (`allow_pickle=False`)
  - Added `SimpleMLP` with He initialisation, ReLU hidden layers, full backprop
  - Added `train_from_geqdsk()`: trains on real SPARC GEQDSK data with profile
    perturbations (8-dim input: I_p, B_t, R_axis, Z_axis, pprime_scale,
    ffprime_scale, simag, sibry)
  - Added `train_on_sparc()` convenience function for all 8 files
  - Added `benchmark()` timing method
- Added `tests/test_neural_equilibrium.py` — 19 tests (PCA round-trip, MLP shapes,
  save/load, SPARC training integration, benchmark timing)
- Moved "neural" mode from `PUBLIC_MODES` to `SURROGATE_MODES` in
  `run_fusion_suite.py` — now behind `--surrogate` / `SCPN_SURROGATE=1` flag

### Multi-Regime SPARC-Parameterized FNO Training

- Added multi-regime turbulence data generator to `fno_training.py`:
  - `SPARC_REGIMES` dict: ITG / TEM / ETG parameter ranges (adiabaticity α,
    gradient drive κ, viscosity ν, nonlinear damping, spectral cutoff k_c)
  - `_generate_multi_regime_pairs()`: modified Hasegawa-Wakatani with regime-
    dependent dispersion ω = α·k_y/(α+k²), growth γ = κ·k_y·k²/(α+k²)² − ν·k⁴,
    and spectral filtering
  - `train_fno_multi_regime()`: full training loop with per-regime validation
    breakdown and regime distribution logging
- Added `tests/test_fno_multi_regime.py` — 18 tests (regime sampling, spectral
  character verification, training smoke, weights round-trip)
- Weights saved to `weights/fno_turbulence_sparc.npz` (multi-regime); legacy
  single-regime weights at `weights/fno_turbulence.npz` preserved

### Documentation — README Repositioning

- Repositioned project identity in README.md as a **neuro-symbolic control framework**
  rather than a physics simulation suite competing with TRANSP/GENE/JINTRAC.
- Added Design Philosophy table (control-first, graceful degradation, explicit errors,
  real data validation, reduced-order by design).
- Expanded Neuro-Symbolic Compiler section from 7 lines to ~40 lines with pipeline
  diagram, 5-stage description, and "Why This Matters" rationale.
- Reorganised simulation modes into 4 maturity tiers (Production / Validated /
  Reduced-order / Experimental) with hardening task counts.
- Added Code Health & Hardening section documenting all 8 hardening waves (248 tasks).
- Added Known Limitations & Roadmap section with honest gap table, strengths table,
  and DOE Fusion S&T Roadmap alignment.

### Multigrid Wiring and Experimental Validation

- Wired geometric multigrid V-cycle solver into `FusionKernel` Picard loop:
  - Added `SolverMethod` enum (`PicardSor`, `PicardMultigrid`) in `fusion-core/src/kernel.rs`
  - Added `set_solver_method()` / `solver_method()` accessors on `FusionKernel`
  - Exported `multigrid` module from `fusion-math/src/lib.rs`
  - Inner Picard solve now dispatches to Red-Black SOR (default) or multigrid V-cycle
- Exposed solver method control in PyO3 bindings:
  - `PyFusionKernel.set_solver_method("sor" | "multigrid")` in `fusion-python/src/lib.rs`
  - `RustAcceleratedKernel.set_solver_method()` / `.solver_method()` in `_rust_compat.py`
- Added `validation/benchmark_solvers.py`: 3-way SOR vs Multigrid vs Python timing comparison
- Added `validation/run_experimental_validation.py`: unified runner for all 8 SPARC equilibrium files
  with topology checks (axis, q-profile, GS sign) and solver validation modes

### Cutting-Edge Release Staging

- Prepared `v2.0-cutting-edge` release contract:
  - Novum elevated - SNN + GyroSwin hybrids for resilient, 1000x fast control; MVR grounded in 2025-2026 liquid metal/HTS.

### Stability and CI

- Hardened 3D LCFS extraction for coarse smoke meshes:
  - per-ray nearest-boundary fallback when strict crossing is absent
  - conservative ellipse fallback for sparse contours
- Added geometry regression coverage for sparse/non-crossing LCFS rays.
- Fixed CI formatting regression in inverse Jacobian tests (`cargo fmt --check` pass restored).

### Solver Validation and Numerics

- Added kernel-level analytical-vs-finite-difference Jacobian consistency test for inverse reconstruction.
- Added reduced particle feedback lane in `fusion-core`:
  - new `particles.rs` with Boris pusher and toroidal current deposition
  - optional kernel hook `set_particle_current_feedback(...)` to blend particle `J_phi` into GS source updates
  - regression tests for Boris invariance, deposition non-zero support, blend renormalization, and shape guard
- Added symplectic integration baseline in `fusion-math`:
  - new `symplectic.rs` with canonical velocity-Verlet stepper and RK4 reference
  - long-horizon Hamiltonian drift checks for harmonic-oscillator stress profiles
  - regression coverage for bounded symplectic drift and coarse-step drift comparison against RK4
- Added reduced non-LTE impurity-radiation lookup in `fusion-nuclear/wall_interaction.rs`:
  - ADAS-style synthetic PEC grids with charge-state distinction (including W20+ and W40+)
  - bilinear PEC interpolation in `(log ne, log Te)` space
  - collisional-radiative power-density and volume-integrated radiative-loss helpers
- Added reduced IGA/NURBS geometry lane:
  - new `fusion-math/src/iga.rs` for NURBS curve evaluation and open-uniform knot generation
  - NURBS-based first-wall contour generator (`generate_first_wall_nurbs`) in `fusion-nuclear`
  - regression checks for endpoint consistency, contour closure, and analytic-envelope proximity
- Added latency-aware control realism extensions in `fusion-control`:
  - vector Ornstein-Uhlenbeck noise layer for multi-channel perturbations
  - actuator delay-line with configurable lag blending
  - DDE-like delayed rollout path in MPC (`plan_with_delay_dynamics`)
- Added reduced runtime regime-specialized kernel lane in `fusion-core`:
  - new `fusion-core/src/jit.rs` with deterministic regime routing and specialization specs
  - compile-cache and hot-swap manager (`RuntimeKernelJit`) for regime-triggered kernel activation
  - regression tests for cache reuse, hot-swap response divergence, and identity-safe fallback
- Upgraded AMR hierarchy and kernel integration to level-aware scaling (`2^level`) for multi-level patch colocation/prolongation.
- Added AMR multilevel tests for hierarchy generation and level-scaled interpolation.
- Added low-order toroidal mode coupling closure in `fusion-core` transport:
  - configurable `n=1..N` mode spectrum via `set_toroidal_mode_spectrum(...)`
  - edge-weighted diffusivity coupling factor for reduced `n!=0` transport effects
  - regression tests for baseline parity, edge-dominant coupling, and clamp safety
- Added reduced 3D-aware toroidal-harmonic coupling in `fusion-physics` FNO turbulence:
  - configurable toroidal harmonic spectrum via `set_toroidal_harmonics(...)`
  - non-zonal low-k spectral amplification closure with bounded coupling gain
  - regression tests for default parity, non-zonal energy increase, and clamp safety
- Added reduced 3D divertor strike-point asymmetry projection in `fusion-nuclear`:
  - toroidal/poloidal heat-flux map projection via `project_heat_flux_3d(...)`
  - low-order toroidal mode shaping through `ToroidalMode { n, amplitude, phase }`
  - map normalization preserving Eich-derived mean target load
  - regression tests for strike localization, toroidal asymmetry increase, and mean-load conservation
- Added reduced VMEC-like native 3D equilibrium interface:
  - new `VMECStyleEquilibrium3D` + `FourierMode3D` in `src/scpn_fusion/core/equilibrium_3d.py`
  - native flux-coordinate mapping `(rho, theta, phi) -> (R, Z) -> (x, y, z)`
  - integration with `Reactor3DBuilder` for direct 3D surface generation without 2D revolve-only path
  - builder utility `build_vmec_like_equilibrium(...)` to infer baseline shaping from traced 2D LCFS
  - regression coverage in `tests/test_equilibrium_3d.py` and quickstart support for VMEC-like mode
- Added reduced 3D field-line and Poincare diagnostics:
  - new `FieldLineTracer3D` in `src/scpn_fusion/core/fieldline_3d.py`
  - reduced helical field-line tracing over VMEC-like flux coordinates
  - Poincare section generation for arbitrary toroidal cut planes
  - `Reactor3DBuilder` helpers: `create_fieldline_tracer(...)` and `generate_poincare_map(...)`
  - new quickstart workflow in `examples/run_3d_fieldline_quickstart.py`
  - regression coverage in `tests/test_fieldline_3d.py`
- Added PCE checked APIs:
  - `PCEModel::try_fit(...) -> FusionResult<PCEModel>`
  - `PCEModel::try_predict(...) -> FusionResult<Array1<f64>>`
- Added PCE edge-case tests for non-finite inputs and prediction dimension mismatch.
- Extended pedestal model tests for ELM cooldown gating and minimum profile floor guarantees.

### Release and Metadata

- Added cross-file release metadata consistency test (`tests/test_version_metadata.py`) covering:
  - `pyproject.toml`
  - `setup.py`
  - `CITATION.cff`
  - `src/scpn_fusion/__init__.py`
  - `docs/sphinx/conf.py`
- Removed hardcoded Sphinx release fallback; now resolves dynamically from package source metadata.

### SCPN Controller Artifact

- Centralized artifact schema version as `ARTIFACT_SCHEMA_VERSION` in `scpn/artifact.py`.
- Updated compiler artifact export to use package version for `meta.compiler.version`.
- Added git SHA resolution for artifacts:
  - prefers `SCPN_GIT_SHA`, `GITHUB_SHA`, `CI_COMMIT_SHA`
  - falls back to `git rev-parse --short HEAD`
  - final fallback `0000000`
- Added controller/compiler tests for schema version, package version, and env-driven git SHA stamping.

### Documentation

## 1.0.2 - 2026-02-12

### Solvers and Numerics

- Added analytical Jacobian pathway for inverse reconstruction with selectable `JacobianMode`.
- Added Chebyshev-accelerated SOR and benchmark comparison against fixed SOR.
- Added phase-space memory kernel transport solver.
- Added patch-based AMR baseline:
  - `fusion-math/src/amr.rs` hierarchy and refinement criterion (`|∇²J_phi|`)
  - `fusion-core/src/amr_kernel.rs` AMR-assisted solve wrapper

### Physics and Transport

- Added EPED-like pedestal model (`fusion-core/src/pedestal.rs`):
  - pedestal width scaling: `Δ_ped ~ sqrt(beta_p,ped) * (rho_s / R)`
  - ELM trigger and crash application on edge profiles
- Integrated pedestal model into the transport solver H-mode path.

### Machine Learning and UQ

- Added multi-layer FNO training pipeline and Rust weight loading path.
- Added neural transport surrogate in Rust with `.npz` loading.
- Added neural transport training and validation scripts/docs.
- Added Polynomial Chaos Expansion UQ module in Rust:
  - multivariate Hermite basis generation
  - Latin Hypercube sampling utility
  - fit/predict and first-order Sobol indices

### Engineering and Bindings

- Implemented `fusion-engineering` modules for tritium, blanket, magnets, and layout.
- Expanded PyO3 bindings to expose inverse, transport, and plant APIs.
- Added benchmark suites for inverse Jacobians and neural transport throughput.

### Validation and Documentation

- Added ITER/SPARC regression reference data and validation tests.
- Added internal GPU acceleration implementation notes.
- Added this changelog and CFF citation metadata.

### Versioning

- Python package version bumped to `1.0.2` (`pyproject.toml`, `setup.py`, `src/scpn_fusion/__init__.py`).
- `fusion-engineering` crate version bumped to `0.2.0`.
