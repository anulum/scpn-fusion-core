# Social Media Draft — SCPN Fusion Core Benchmark Results

> **Status:** DRAFT — review before posting.
> **Date:** 2026-02-16

---

## Thread (LinkedIn / X / Mastodon)

**Post 1 — Hook**

We just published full benchmark results for SCPN Fusion Core, an open-source
tokamak plasma physics simulation and neuro-symbolic control suite written in
Python + Rust.

Unlike most fusion codes that only show convergence plots in papers, we run
every benchmark on CI and publish the raw numbers in RESULTS.md — anyone can
reproduce them on their own hardware with a single command.

**Post 2 — Key Numbers**

Highlights from the latest run:

- ITER-like Q ≥ 10 operating point identified (density + heating scan)
- Tritium Breeding Ratio > 1 from 3-group blanket neutronics (80 cm, 90% ⁶Li)
- Sub-millisecond hardware-in-the-loop control latency (P99 < 1 ms)
- 50-run disruption mitigation ensemble with halo current + runaway electron tracking
- 3 pretrained neural surrogates shipped out of the box (MLP, FNO, PCA+MLP equilibrium)
- 3D force-balance solver with spectral variational method

**Post 3 — What It Is**

SCPN Fusion Core covers the full tokamak simulation stack:

- Grad-Shafranov equilibrium (2D + reduced 3D)
- 1.5D transport with IPB98(y,2) scaling
- Disruption prediction + SPI mitigation
- Neuro-symbolic compiler: Petri net → spiking neural network
- Digital twin with reinforcement learning
- Full Rust acceleration workspace (10 crates)

All validated against SPARC GEQDSKs and ITPA H-mode confinement data.

**Post 4 — Call to Action**

The code is open source under AGPL v3:
https://github.com/anulum/scpn-fusion-core

Documentation + tutorials:
https://anulum.github.io/scpn-fusion-core/

We'd love feedback from the fusion community — especially on:
- Validation methodology (are we testing the right things?)
- Surrogate model architecture choices
- Integration with existing workflow tools (IMAS, OMAS, etc.)

Open issues welcome. PRs even more welcome.

---

## Short Version (single post, < 280 chars)

Open-sourced full benchmark results for our tokamak simulation suite: Q≥10, TBR>1, sub-ms control loop, 50-run disruption ensemble. All reproducible. Feedback welcome.

https://github.com/anulum/scpn-fusion-core

---

*Review checklist before posting:*
- [ ] Verify RESULTS.md numbers match latest run
- [ ] Confirm GitHub repo is public
- [ ] Check docs site is live
- [ ] Remove any internal references
