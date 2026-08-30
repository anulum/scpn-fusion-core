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

## Minimal verified entry points

These examples use public imports and avoid optional services. They are API
orientation, not benchmark evidence.

```python
from scpn_fusion.core import RUST_BACKEND, read_geqdsk

equilibrium = read_geqdsk(
    "validation/reference_data/sparc/lmode_vv.geqdsk"
)
print(RUST_BACKEND, equilibrium.psirz.shape)
```

```python
from scpn_fusion.scpn import FusionCompiler, StochasticPetriNet

net = StochasticPetriNet()
compiler_type = FusionCompiler
print(net, compiler_type.__name__)
```

```python
from scpn_fusion.studio import build_manifest

manifest = build_manifest()
print(manifest.studio, manifest.studio_version)
```

The Studio import is part of the repository surface; downstream Hub integration
may additionally require the optional `studio` extra. Use the generated API
pages for signatures and the linked tests/reports for behavioral guarantees.

### Full runaway kinetic solver

`scpn_fusion.core` exports `RunawayKineticGrid`,
`RunawayKineticCoefficients`, `RunawayKineticGeometry`,
`RunawayKineticOperator`, and `RunawayKineticSolver`. The state order is
`(radius, pitch, momentum)`. `RunawayKineticSolver.solve()` returns the full
unprojected history, every named operator contribution, the independently
evolved total-density balance, and radial density/current/energy moments.
Select `backend="numpy"` or `backend="rust"` explicitly. The Rust request
fails closed if the compiled extension or full-kinetic symbol is unavailable;
it never silently substitutes NumPy. Exact-output parity and the
host-conditioned timing are recorded in
`validation/reports/runaway_kinetic_rust_benchmark.md`.

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

## Generated API reference map

| Domain | Sphinx source |
|---|---|
| Core physics and solvers | `docs/sphinx/api/core.rst` |
| Control and replay | `docs/sphinx/api/control.rst` |
| Neuro-symbolic compiler | `docs/sphinx/api/scpn.rst` |
| Studio federation | `docs/sphinx/api/studio.rst` |
| I/O and interchange | `docs/sphinx/api/io.rst` |
| HPC/native integration | `docs/sphinx/api/hpc.rst` |
| Phase, diagnostics, engineering, nuclear, and UI | `docs/sphinx/api/phase.rst`, `diagnostics.rst`, `engineering.rst`, `nuclear.rst`, `ui.rst` |

If a symbol is absent from the relevant generated page, treat the owning module
as an internal or evolving surface until an explicit public facade and tests say
otherwise.

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
scpn-fusion repro --full
scpn-emit-studio-manifest --check
```

CLI modes are useful for demos and smoke tests. Scientific claims should point
to validation reports, not only CLI output. `scpn-fusion repro --full` refreshes
the fail-closed full-fidelity campaign, public ledger, and checksummed
full-reproduction evidence report without changing blocked lane semantics.

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
kernels and parity surfaces. Some dispatchers provide observable Python
fallbacks when the compiled extension is absent. Explicit full-kinetic
`backend="rust"` requests are fail-closed and do not silently fall back.

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

### Real TORAX runtime contract

`scpn_fusion.integrations.torax` is the stable caller surface for real pinned
TORAX 1.4.3 execution. Importing it does not import TORAX or JAX. A
`ToraxRuntimeClient` launches an explicitly supplied TORAX interpreter through
the one-request CLI:

```bash
python -m scpn_fusion.integrations.torax \
  --request request.json \
  --result outcome.json \
  --output-sidecar torax-output.nc
```

`ToraxRunRequest` carries the complete JSON-compatible TORAX configuration plus
unit-bearing typed state/control bindings that must agree with it. Every run is
a fresh process with no hidden-state carry-over. V1 controls are prescribed
model sources, not actuator commands: model delay is zero and hardware
saturation, slew, and hardware-delay limits are explicitly undeclared.
`ToraxRunOutcome` distinguishes success from every TORAX 1.4.3 `SimError`,
backend/configuration/process/timeout/schema/provenance failures, and preserves
the complete output as a NetCDF DataTree with a per-variable inventory.

Sibling semantic consumers use `ToraxReviewEnvelope`, not the runtime outcome
directly. Its canonical bytes contain simulation-monotonic integer-nanosecond
samples, Ti/Te/ne/poloidal-flux profiles, source totals, state budgets, solver
completion, and numerical-refinement uncertainty. It deliberately omits
inferred q95, li3, normalized beta, stored thermal energy, regime, and phase.
The envelope includes all U0 reactor and clock facets, direct-model-output
calibration and identity-transfer declarations, and absolute RMS plus relative
L2 refinement uncertainty for every profile, source total, and state budget.
`review_envelope_to_bytes` and `review_envelope_from_bytes` provide the unique
bounded, duplicate-free, digest-verifiable canonical representation. The
envelope is review-only and non-actuating. Evidence and the exact claim boundary
are tracked in
`validation/reports/torax_runtime_contract.{json,md}`.


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
