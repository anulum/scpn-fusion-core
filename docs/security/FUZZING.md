<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

## Fuzzing scope

This page defines the public fuzzing contract for malformed and hostile inputs. It records the bounded targets, tools, and failure-handling paths used in hardening lanes.

## Hardening role

This document is the public contract for robustness against malformed input.
Successful fuzzing builds and reviewed findings support acceptance gates where
input-path stability and bounded resource behaviour are required.

<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Fuzzing Guide -->

# Fuzzing guide

The repository ships Atheris-compatible fuzz targets for malformed user-controlled file inputs:

- `fuzz/fuzz_geqdsk.py`: G-EQDSK parser input, including absurd grid dimensions and malformed Fortran tokens.
- `fuzz/fuzz_imas_ids.py`: IMAS IDS JSON input, including malformed equilibrium schema and bounded-depth/list validation.
- `fuzz/fuzz_fusion_config.py`: `FusionKernel` JSON configuration loading and schema validation.
- `fuzz/fuzz_disruption_npz.py`: disruption-shot NumPy archive loading with pickle disabled.
- `fuzz/fuzz_snn_artifact.py`: compiled SNN controller JSON and bounded packed-weight decoding.

Install the optional fuzzing dependencies before running targets:

```bash
python -m pip install '.[fuzz]'
```

Run one target at a time with a dedicated corpus and findings directory:

```bash
python fuzz/fuzz_geqdsk.py corpus/geqdsk findings/geqdsk
python fuzz/fuzz_imas_ids.py corpus/imas_ids findings/imas_ids
python fuzz/fuzz_fusion_config.py corpus/fusion_config findings/fusion_config
python fuzz/fuzz_disruption_npz.py corpus/disruption_npz findings/disruption_npz
python fuzz/fuzz_snn_artifact.py corpus/snn_artifact findings/snn_artifact
```

All targets truncate single generated inputs to bounded sizes before writing temporary files and instrument the exact parser plus harness entrypoint before starting Atheris. Production loaders enforce 10 MiB gates for JSON, GEQDSK, and NumPy archive paths before parsing. GEQDSK parsing also enforces bounded numeric-token, grid, contour, finite-value, and shape invariants. IMAS JSON loading enforces bounded nesting, bounded list lengths, minimal IDS properties, and equilibrium-grid schema checks before conversion.

The dedicated `Python Fuzz` GitHub Actions workflow runs all five targets every
night and on explicit dispatch. Each isolated matrix job performs 512 Atheris
iterations with bounded input length, per-input timeout, runner timeout, and
resident-memory limit. Dependencies come from the normal hashed minimal lock
plus the hashed Python 3.12 Atheris wheel in `requirements/fuzz.txt`. A failing
campaign uploads any crash, timeout, out-of-memory, or leak reproducer for 14
days. Use `-atheris_runs=<count>` for bounded local campaigns; unlike
libFuzzer's `-runs`, it terminates Atheris gracefully.
The job sets `SCPN_DISABLE_JULIA=1` because none of these parser boundaries
uses the optional Julia backend; this prevents unrelated backend discovery from
starting package resolution during a fuzz campaign.

## CodeQL language coverage

The `CodeQL` workflow analyzes Python and Rust in no-build extraction mode and
manually builds the nested Go module for Go extraction. The three language
jobs run independently so one extractor failure does not hide the other
language results. Rust cargo-fuzz remains a separate bounded CI smoke job and
is not duplicated by the Python nightly workflow.

## Rust cargo-fuzz target

The Rust solver workspace also ships a `cargo-fuzz` harness for domain-decomposition boundary handling:

- `scpn-fusion-rs/fuzz/fuzz_targets/mpi_domain.rs`: exercises bounded 2D decomposition, tile extraction, and serial halo exchange over finite generated arrays.

Install the Rust fuzzing toolchain before running the target locally:

```bash
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz --locked
```

Run the same bounded smoke command used by CI:

```bash
cd scpn-fusion-rs
cargo +nightly fuzz run mpi_domain -- -runs=256 -max_len=128 -timeout=10 -rss_limit_mb=2048
```

Generated Rust fuzz corpora, crash artefacts, and build outputs live under `scpn-fusion-rs/fuzz/{corpus,artifacts,target}` and are intentionally ignored. Preserve only reduced, reviewed regression inputs when a fuzz finding is promoted into a dedicated test.

## Related deployment hardening

Container runtime confinement is documented in `docs/security/CONTAINER_HARDENING.md`.
Use the hardened Compose profile when exposing the dashboard or phase-stream
support services outside a local development shell.
