# API Overview


## Purpose and reading guide

This is the entry map for public interfaces before navigating generated API references. It shows where to begin for controller contracts, physics kernels, transport, and hardening surfaces.

This page maps the main public Python and native-extension surfaces. Use it as
an orientation layer before reading generated Sphinx API pages.

## Python package

Primary package: `scpn_fusion`

Important surfaces:

| Surface | Typical modules | Purpose |
|---|---|---|
| Core equilibrium and transport | `scpn_fusion.core` | Grad-Shafranov solves, GEQDSK I/O, transport, scaling laws |
| Control | `scpn_fusion.control` | PID, H-infinity, LQR, NMPC, replay, controller contracts |
| SCPN compiler | `scpn_fusion.scpn` | Petri-net structures, compiler, artifacts, deterministic replay |
| Diagnostics | `scpn_fusion.diagnostics` | Synthetic sensors and diagnostic forward models |
| Engineering | `scpn_fusion.engineering` | Balance-of-plant and engineering utilities |
| Nuclear | `scpn_fusion.nuclear` | Blanket, neutronics, and wall-interaction utilities |
| HPC | `scpn_fusion.hpc` | Optional distributed and accelerator integration surfaces |
| I/O | `scpn_fusion.io` | IMAS/OMAS and archive adapters |
| Phase | `scpn_fusion.phase` | Kuramoto, UPDE, K_nm, and phase-stream bridge surfaces |
| Studio federation | `scpn_fusion.studio` | Schema-A manifest, exactness comparator, architecture-map extension, and Studio manifest emitter |
| UI | `scpn_fusion.ui` | Streamlit dashboard launcher, dashboard generator, and security-header helpers |

Generated Sphinx API pages live under `docs/sphinx/api/`.

## Practical orientation

This map is intended as a navigation layer before opening generated API details.
Use it to confirm that every code edit has a destination and an evidence
destination:

- Validate expected behavior through tests and benchmark artifacts first.
- Confirm contract behavior through validation reports before changing public claims.
- Keep a direct note in changelog and docs whenever a public-facing contract changes.

## Public interface map

The API surface is deliberately split by responsibility. Controller-facing code
should normally enter through `scpn_fusion.control` or `scpn_fusion.scpn`.
Physics-kernel work should enter through `scpn_fusion.core`. Evidence work
should enter through `validation/` scripts and report schemas before changing
README or market-facing text.

| Work type | API surface | Evidence surface |
|---|---|---|
| Control loop or replay | `scpn_fusion.control`, `scpn_fusion.scpn` | Replay reports, latency reports, controller tests |
| Equilibrium or transport | `scpn_fusion.core` | GS, GEQDSK, transport, and benchmark reports |
| Data interchange | `scpn_fusion.io` | Provenance records, checksums, source licenses |
| Hardware or acceleration | `scpn_fusion.hpc`, `scpn-fusion-rs/` | CPU/GPU/MPI metadata and benchmark artifacts |
| Tutorial/demo | `examples/`, `docs/notebooks/` | Linked validation reports when public claims are made |

## How to choose an API surface

| Goal | Start with | Why |
|---|---|---|
| Build a controller loop | `scpn_fusion.control`, `scpn_fusion.scpn` | These modules expose controller contracts, replay metadata, and Petri-net/SNN compilation. |
| Run or inspect physics kernels | `scpn_fusion.core` | Core modules hold equilibrium, transport, gyrokinetic, electromagnetic, MIF/FRC, and validation helpers. |
| Connect external evidence | `validation/`, `scpn_fusion.io` | Validation scripts and I/O adapters carry provenance, schema, checksum, and pass/blocked status. |
| Accelerate or compare kernels | `scpn-fusion-rs/` | Rust crates implement selected native surfaces and benchmark contracts. |
| Federate with Studio/Hub | `scpn_fusion.studio`, `scpn-emit-studio-manifest` | Emits `docs/_generated/studio_manifest.json`, schema-A verbs, evidence schemas, content digest, and architecture-map metadata. |
| Build demos or onboarding material | `examples/`, `docs/notebooks/` | Notebook flows are tutorials; they must link to reports for public claims. |

## Command-line entry point

```bash
scpn-fusion --help
scpn-fusion kernel
scpn-fusion flight
scpn-fusion neuro-control
scpn-emit-studio-manifest --check
```

CLI modes are useful for demos and smoke tests. Scientific claims should point
to validation reports, not only CLI output.

## Studio federation surface

The optional `studio` extra adds the `scpn_studio_platform` SDK and activates
the `scpn_fusion.studio` package. The package publishes:

- `scpn_fusion.studio.manifest.build_manifest()` for the schema-A capability
  manifest,
- `scpn_fusion.studio.federation.build_federation_document()` for the schema-A
  plus architecture-map JSON document,
- `scpn_fusion.studio.exactness.reproduce()` for bit-exact, tolerance, and
  reduced-stochastic claim comparison,
- `scpn-emit-studio-manifest` for writing and drift-checking
  `docs/_generated/studio_manifest.json`.

The Studio manifest is a federation contract and documentation artifact. It
does not turn simulated or blocked physics rows into accepted evidence; it
describes the verbs, evidence schemas, interfaces, backend matrix, and
boundaries the Hub can ingest.

## Rust workspace

Rust crates live under `scpn-fusion-rs/`. They are used for selected native
kernels and parity surfaces. Python fallback paths remain available when the
compiled extension is absent.

Common commands:

```bash
cd scpn-fusion-rs
cargo test --all-features
cargo bench
```

## Polyglot surfaces

Go, Julia, and Lean surfaces exist where equivalent logic or proofs are
maintained. They are not wrappers for missing physics. If a Python solver
contract changes and an equivalent Rust/Go/Julia/Lean surface exists, update the
corresponding surface or explicitly document why it is not equivalent.

## External reference solvers

Reference-code adapters and benchmark requests exist for GENE, CGYRO, GS2,
DREAM, Aurora, STRAHL, FreeGS, and related data formats. These adapters do not
bundle the external solvers. Acceptance requires same-case outputs, licenses,
provenance, thresholds, checksums, and native comparisons.


## API stability model

| Surface | Stability expectation |
|---|---|
| CLI smoke commands | Stable enough for demos and CI smoke tests; scientific claims still require validation reports. |
| Python package APIs | Evolve with tests and changelog entries; public imports should keep backwards-compatible behavior when possible. |
| Rust/PyO3 kernels | Optional acceleration path; Python fallbacks remain the compatibility baseline. |
| Validation schemas | Treated as evidence contracts; changes require report and documentation updates. |
| External solver adapters | Fail closed when the solver, license, provenance, or output artifact is missing. |

## Security-sensitive surfaces

Native compilation, subprocess launchers, external solver execution, artifact
loading, and dashboard/browser entry points are security-sensitive. Changes to
these surfaces should include scoped tests, timeout handling, fixed argv where
possible, and documentation of the trust boundary.

## Documentation contract for API changes

When a public API, validation schema, benchmark report, or Rust/Python parity
surface changes, update the relevant guide and changelog in the same commit.
If the API exposes a physics claim, link it to a tracked report rather than
describing it as accepted in prose.
