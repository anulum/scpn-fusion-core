# Honest Scope & Limitations

SCPN Fusion Core is a **control-algorithm development framework** with enough
physics fidelity to validate reactor control strategies against real equilibrium
data. It is **not** a replacement for TRANSP, JINTRAC, GENE, or any
first-principles transport/gyrokinetic code.

## What it does

| Capability | Evidence |
|-----------|---------|
| Petri net → SNN compilation with formal verification | 37 hardening tasks, deterministic replay |
| Sub-microsecond Rust control loop (0.52 µs P50) | `validation/verify_10khz_rust.py`, Criterion benches |
| QLKNN-10D real-gyrokinetic transport surrogate | test_rel_L2 = 0.0943, Zenodo DOI 10.5281/zenodo.3497066 |
| IPB98(y,2) confinement scaling on 53 shots / 24 machines | `validation/reference_data/itpa/hmode_confinement.csv` |
| 8 SPARC EFIT GEQDSK equilibrium validation | `validation/reference_data/sparc/` (MIT, CFS) |
| 0% disruption rate across 1,000-shot stress campaigns | `validation/stress_test_campaign.py` |
| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) | Every module has a pure-Python fallback |

## What it does not do

| Gap | Why | Alternative |
|-----|-----|-------------|
| 5D gyrokinetic turbulence | Deliberately reduced-order for real-time control | Use GENE/GS2; couple via surrogate training |
| Full 3D MHD | Not planned for real-time loop | Use NIMROD/M3D-C1 externally |
| Publication-quality equilibrium reconstruction | Default 65x65 grid, Picard+SOR | Use EFIT/CHEASE for production equilibria |
| Complete impurity transport | Simple diffusion only | Use JINTRAC/STRAHL |
| Free-boundary equilibrium | Coil currents are fixed inputs | Use FreeGS/CREATE-NL |

## Physics model fidelity

| Module | Actual fidelity | Known limitations |
|--------|----------------|-------------------|
| Equilibrium | Picard + SOR/multigrid, converges on SPARC GEQDSKs | 65x65 default grid; not EFIT-quality inverse reconstruction |
| Transport | 1.5D Bohm/gyro-Bohm + Chang-Hinton neoclassical | No ITG/TEM/ETG channels; no NBI slowing-down |
| Neural equilibrium | PCA+MLP on 78 samples (SPARC L-mode family) | Useful only for the specific equilibrium family it was trained on |
| FNO turbulence | Synthetic-data trained; **not validated against gyrokinetics** | Proxy mapping only; DEPRECATED in v3.9 |
| Neural transport MLP | 53-row ITPA illustrative dataset | Cannot capture full H-mode parameter space |
| Stability | Vertical n-index + ballooning/Mercier criteria | No kink, peeling-ballooning, or RWM analysis |

## Pretrained surrogate status

4 of 7 surrogate lanes ship pretrained weights:

| Surrogate | Status | Evidence |
|-----------|--------|---------|
| MLP ITPA confinement | Shipped | 13.5% RMSE on training set |
| FNO EUROfusion-proxy | Shipped | rel_L2 = 0.79 (synthetic only) |
| Neural equilibrium (SPARC) | Shipped | PCA+MLP, 78 samples |
| QLKNN-10D transport | Shipped | test_rel_L2 = 0.0943 |
| Heat ML shadow | Requires user training | No pretrained weights |
| Gyro-Swin | Requires user training | No pretrained weights |
| Turbulence oracle | Requires user training | No pretrained weights |

## Validation is mostly synthetic

The validation pipeline uses real SPARC GEQDSK files (8 shots, MIT license from
CFS) and a 53-entry ITPA confinement subset, but the bulk of testing uses
synthetic Solov'ev equilibria and template-generated profiles. The DIII-D
disruption shots are reference profiles reconstructed from published parameters,
not raw MDSplus data.

Full claims-to-evidence audit: [`docs/CLAIMS_EVIDENCE_MAP.md`](CLAIMS_EVIDENCE_MAP.md)

## Underdeveloped flags

The auto-generated [`UNDERDEVELOPED_REGISTER.md`](../UNDERDEVELOPED_REGISTER.md)
tracks 312 flags across the codebase. Of these, 114 are P0/P1 severity. The
register is regenerated on each release and CI-gated.
