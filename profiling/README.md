<!--
SCPN Fusion Core — Profiling Quickstart
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Profiling Quickstart

## Python cProfile

From repo root:

```bash
python profiling/profile_kernel.py --top 50
python profiling/profile_geometry_3d.py --toroidal 48 --poloidal 48 --top 50
```

Outputs are written to `artifacts/profiling/`.

## Rust flamegraph (Linux/macOS)

```bash
cargo install flamegraph
cd scpn-fusion-rs
cargo flamegraph -p fusion-core --bench inverse_bench -- --bench
```

If your platform requires elevated `perf` privileges, configure system `perf_event_paranoid` accordingly before running flamegraphs.
