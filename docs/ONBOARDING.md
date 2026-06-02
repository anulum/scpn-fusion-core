# Onboarding Guide

This guide gives a new contributor the shortest path from clone to useful
results without overstating solver maturity.

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

## 3. Learn the repository by surface

| Surface | Entry point |
|---|---|
| Equilibrium | `docs/sphinx/userguide/equilibrium.rst` |
| Transport | `docs/sphinx/userguide/transport.rst` |
| Control | `docs/sphinx/userguide/control.rst` |
| Hardware-in-the-loop | `docs/sphinx/userguide/hil.rst` |
| Validation | `docs/sphinx/userguide/validation.rst` |
| Benchmarks | `docs/BENCHMARKS.md` |
| Notebooks | `docs/notebooks/README.md` |
| API map | `docs/API_OVERVIEW.md` |

## 4. Contributor workflow

- Keep public claims tied to tracked reports.
- Keep planning, scratch notes, credentials, provider instance IDs, and private
  runbooks in ignored internal paths.
- Do not promote synthetic, reduced-order, or diagnostic-only output to accepted
  parity evidence.
- Update documentation and changelog entries when behavior, benchmarks, public
  claims, or user-visible APIs change.

## 5. Before publishing benchmark numbers

A benchmark number is public only when it includes command, environment,
hardware metadata, input decks, output artifacts, checksums, thresholds, and a
report that states pass, fail, blocked, or diagnostic-only status.


## 6. Choose a first contribution path

| Path | First useful task | Evidence expected |
|---|---|---|
| Control | Add or harden a deterministic replay scenario | Trace checksum, controller inputs, outputs, and pass/fail criteria |
| Equilibrium | Improve a GEQDSK or FreeGS comparison row | Source provenance, convention metadata, RMSE/current/axis metrics |
| Gyrokinetics | Add a same-deck external output candidate | License, checksum, deck hash, converted JSON/NPZ artifact, blocked/accepted status |
| Runtime | Harden a native, Rust, MPI, CUDA, or HIL surface | Fixed command, hardware metadata, timeout, benchmark report |
| Formal methods | Extend Lean proofs toward executable contracts | Lean build, theorem boundary, linked implementation contract |
| Documentation | Improve a public guide or evidence map | Clear scope, no unsupported claims, links to reports |

## 7. Release and documentation discipline

Before a public-facing change is released, update the relevant README, guide,
API overview, benchmark report, changelog, and release note. If the change is an
internal plan or unfinished TODO, keep it under ignored internal paths and do not
turn it into public capability prose.
