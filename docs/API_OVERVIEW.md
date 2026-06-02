# API Overview

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

Generated Sphinx API pages live under `docs/sphinx/api/`.

## Command-line entry point

```bash
scpn-fusion --help
scpn-fusion kernel
scpn-fusion flight
scpn-fusion neuro-control
```

CLI modes are useful for demos and smoke tests. Scientific claims should point
to validation reports, not only CLI output.

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
