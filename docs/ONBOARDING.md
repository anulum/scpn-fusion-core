# Onboarding Guide

This guide gives a new contributor the shortest path from clone to useful
results without overstating solver maturity.

## Purpose

This onboarding document is for first-time contributors who need to move from
environment setup to a repeatable validation path. It separates fast local checks
from full-fidelity evidence-gated campaigns so contributors know when results are
developmental versus production-accepted.

## Positioning for new contributors

The fastest route to productive work is to start with local commands,
then move to tracked evidence surfaces. This repo treats all public claims as
evidence-backed only after corresponding reports and acceptance markers are in
place.

Contributors are expected to distinguish these two categories before edits:

- **Exploratory paths:** local-only scripts, draft references, and blocked rows.
- **Accepted paths:** published benchmark rows, reproducibility metadata, and
  documented gates in `validation/reports/`.

## 1. Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e '.[dev]'
```

Optional accelerator and distributed lanes require additional packages such as
Rust, maturin, mpi4py, CuPy, CUDA runtime libraries, or external reference
solvers. Those are optional unless you are running the corresponding benchmark.

## 2. Run the first local commands

```bash
scpn-fusion kernel
python examples/minimal.py --grid 17 --equilibrium-iters 4
python validation/full_fidelity_end_to_end_campaign.py
```

The full-fidelity campaign is expected to remain fail-closed until external
same-case reference artifacts are present. Treat blocked rows as work items, not
as failures to hide.

## 3. Understand the product in one pass

Before changing code, read the public claim boundary:

```text
README.md
docs/PROJECT_OVERVIEW.md
docs/APPLICATIONS_AND_MARKET.md
docs/BENCHMARKS.md
validation/reports/full_fidelity_end_to_end_campaign.md
```

The repository is not just a solver. It is a control-first validation stack:
Petri-net and SNN control contracts, native physics kernels, Rust acceleration,
formal proof slices, notebooks, benchmark reports, and fail-closed external
reference gates.

## 3a. Choose your reader track

Different readers should not start in the same place:

| Reader | First path | Outcome |
|---|---|---|
| New user | `README.md` -> `docs/PROJECT_OVERVIEW.md` -> `docs/sphinx/quickstart.rst` | Understand the product and run a smoke command. |
| Contributor | `docs/ONBOARDING.md` -> `docs/API_OVERVIEW.md` -> relevant tests | Know where code, docs, and evidence must change together. |
| Fusion-domain reviewer | `docs/BENCHMARKS.md` -> `validation/reports/` | Verify claim boundaries before citing numbers. |
| Investor or buyer | `docs/APPLICATIONS_AND_MARKET.md` -> GitHub Pages -> release notes | Understand current value, blockers, and funding targets. |
| Security/release reviewer | `docs/RELEASE_READINESS.md` -> workflows -> dependency locks | Verify hygiene, dependency state, and release gates. |

## 4. Learn the repository by surface

| Surface | Entry point |
|---|---|
| Equilibrium | `docs/sphinx/userguide/equilibrium.rst` |
| Transport | `docs/sphinx/userguide/transport.rst` |
| Control | `docs/sphinx/userguide/control.rst` |
| Hardware-in-the-loop | `docs/sphinx/userguide/hil.rst` |
| Validation | `docs/sphinx/userguide/validation.rst` |
| Studio federation | `docs/sphinx/userguide/studio_federation.rst` |
| Benchmarks | `docs/BENCHMARKS.md` |
| Notebooks | `docs/notebooks/README.md` |
| API map | `docs/API_OVERVIEW.md` |

## 5. Contributor workflow

- Keep public claims tied to tracked reports.
- Keep planning, scratch notes, credentials, provider instance IDs, and private
  runbooks in ignored internal paths.
- Do not promote synthetic, reduced-order, or diagnostic-only output to accepted
  parity evidence.
- Update documentation and changelog entries when behavior, benchmarks, public
  claims, or user-visible APIs change.

## 6. Before publishing benchmark numbers

A benchmark number is public only when it includes command, environment,
hardware metadata, input decks, output artifacts, checksums, thresholds, and a
report that states pass, fail, blocked, or diagnostic-only status.

## 7. Choose a first contribution path

| Path | First useful task | Evidence expected |
|---|---|---|
| Control | Add or harden a deterministic replay scenario | Trace checksum, controller inputs, outputs, and pass/fail criteria |
| Equilibrium | Improve a GEQDSK or FreeGS comparison row | Source provenance, convention metadata, RMSE/current/axis metrics |
| Gyrokinetics | Add a same-deck external output candidate | License, checksum, deck hash, converted JSON/NPZ artifact, blocked/accepted status |
| Runtime | Harden a native, Rust, MPI, CUDA, or HIL surface | Fixed command, hardware metadata, timeout, benchmark report |
| Studio federation | Add or validate a Studio evidence schema or verb | Manifest drift check, schema-A conformance, architecture-map boundary text |
| Formal methods | Extend Lean proofs toward executable contracts | Lean build, theorem boundary, linked implementation contract |
| Documentation | Improve a public guide or evidence map | Clear scope, no unsupported claims, links to reports |

## 8. Release and documentation discipline

Before a public-facing change is released, update the relevant README, guide,
API overview, benchmark report, changelog, and release note. If the change is an
internal plan or unfinished task note, keep it under ignored internal paths and do not
turn it into public capability prose.

## Evidence checkpoints for first-time contributors

After the initial setup and first local run, confirm each section below before
starting review-ready work:

1. command output and local checks are reproducible,
2. benchmark rows are interpreted as local/blocked/accepted according to evidence state,
3. all external references in docs point to an existing tracked report.
