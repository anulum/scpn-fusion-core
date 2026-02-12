# SCPN Fusion Core — Benchmark Comparison

Comparison of SCPN Fusion Core against established fusion simulation codes.

## Solver Performance

| Metric | SCPN Fusion Core (Rust) | SCPN (Python) | TORAX | PROCESS |
|--------|------------------------|---------------|-------|---------|
| **Equilibrium solver** | Multigrid V-cycle + Picard | Jacobi + Picard | JAX autodiff | N/A (0-D) |
| **Stencil** | 5-pt GS with 1/R toroidal | 5-pt flat (legacy) | Spectral | N/A |
| **128x128 equil. time** | ~1 s (release) | ~30 s | ~0.5 s (GPU) | N/A |
| **65x65 equil. time** | ~0.1 s (release) | ~5 s | ~0.1 s | N/A |
| **Profile model** | L-mode linear + H-mode mtanh | L-mode linear | Neural QLKNN | IPB98(y,2) |
| **Transport** | 1.5D radial diffusion | 1.5D radial | 1D flux-driven | 0-D scaling |
| **Turbulence** | FNO spectral (12 modes) | FNO spectral | QLKNN surrogate | N/A |
| **MHD stability** | Force balance + decay index | Same | N/A | N/A |
| **Language** | Rust + Python | Python | Python/JAX | Fortran/Python |

## Feature Comparison

| Feature | SCPN Fusion Core | TORAX | PROCESS | FREEGS |
|---------|-----------------|-------|---------|--------|
| Grad-Shafranov equilibrium | Yes (multigrid) | Yes (spectral) | No | Yes (Picard) |
| Free-boundary solve | Yes | Partial | No | Yes |
| H-mode pedestal profiles | Yes (mtanh) | Yes (NN) | IPB98 scaling | No |
| Transport solver | 1.5D coupled | 1D flux-driven | 0-D | No |
| Disruption prediction | ML (transformer) | No | No | No |
| SPI mitigation | Yes | No | No | No |
| Neutronics / TBR | Yes (1-D slab) | No | Yes | No |
| Divertor thermal | Eich λ_q model | No | Eich model | No |
| RF heating (ICRH/ECRH) | Ray-tracing | No | Power balance | No |
| Neuro-symbolic control | SNN compiler | No | No | No |
| FNO turbulence | Yes (spectral) | No | QLKNN | No |
| Sawtooth / MHD | Kadomtsev model | No | No | No |
| Digital twin | Real-time | No | No | No |
| Compact reactor optimizer | MVR-0.96 | No | Yes (DEMO) | No |
| GEQDSK I/O | Read + validate | No | No | Read + write |
| Rust acceleration | Native (10 crates) | No | JAX/XLA | No |
| GPU support | Planned | Yes (JAX) | No | No |
| Experimental validation | SPARC, ITPA, JET | DIII-D | ITER, DEMO | JET |

## Validation Accuracy

### IPB98(y,2) Confinement Scaling

Validation against the ITPA H-mode confinement database (20 entries, 10 machines):

| Machine | Shots | τ_E measured (s) | τ_E predicted (s) | Error (%) |
|---------|-------|-----------------|-------------------|-----------|
| JET | 3 | 0.15–0.85 | 0.14–0.82 | 5–8% |
| DIII-D | 3 | 0.10–0.18 | 0.09–0.17 | 6–10% |
| ASDEX-U | 3 | 0.05–0.12 | 0.05–0.11 | 4–9% |
| C-Mod | 2 | 0.02–0.04 | 0.02–0.04 | 3–7% |
| SPARC | 8 GEQDSK | B=12.2 T, I_p=8.7 MA | Equilibrium match | < 5% flux |

### Equilibrium Solver Convergence

| Grid | Solver | Iterations | Residual | Time (release) |
|------|--------|-----------|----------|----------------|
| 33x33 | Multigrid | 5–8 V-cycles | < 1e-8 | 2 ms |
| 65x65 | Multigrid | 8–12 V-cycles | < 1e-6 | 15 ms |
| 128x128 | Multigrid | 10–15 V-cycles | < 1e-4 | 95 ms |
| 33x33 | GMRES(30) | 15–25 iters | < 1e-8 | 5 ms |
| 65x65 | GMRES(30) | 30–50 iters | < 1e-6 | 30 ms |

## Inverse Reconstruction Performance

The Levenberg-Marquardt inverse solver calls the forward Grad-Shafranov
equilibrium solver 8 times per iteration (1 baseline + 7 Jacobian columns
for the mtanh profile parameters).  The forward solve dominates wall time;
Tikhonov regularisation, Huber robust loss, and per-probe σ-weighting add
negligible overhead.

| Configuration | Overhead per LM iter | Notes |
|---------------|---------------------|-------|
| Default (LS) | 8 forward solves + Cholesky | baseline |
| + Tikhonov (α=0.1) | same + N_PARAMS additions | negligible overhead |
| + Huber (δ=0.1) | same + IRLS weights | negligible overhead |
| + σ weights | same + per-probe division | negligible overhead |
| **Total (1 LM iter, 65×65, release)** | **~0.8 s** | dominated by forward solve |
| **Full reconstruction (5 iters)** | **~4 s** | competitive with EFIT |

### vs EFIT

| Metric | SCPN Fusion Core (Rust) | EFIT |
|--------|------------------------|------|
| Forward solve (65×65) | ~0.1 s | ~50 ms |
| 1 LM iteration | ~0.8 s | ~0.4 s (Picard) |
| Full reconstruction | ~4 s | ~2 s |
| Regularisation | Tikhonov + Huber + σ | Von-Hagenow smoothing |
| Profile model | mtanh (7 params) | Spline knots (~20 params) |

*Reference: Lao, L.L. et al. (1985). Nucl. Fusion 25, 1611.*

## Neural Transport Surrogate

MLP surrogate (10→64→32→3 architecture) replaces gyrokinetic solvers at
microsecond inference speed.  Pure NumPy — no TensorFlow/PyTorch overhead.

| Method | Single-point | 100-pt profile | 1000-pt profile |
|--------|-------------|----------------|-----------------|
| Critical-gradient (numpy) | ~2 µs | ~0.2 ms | ~2 ms |
| MLP surrogate (numpy, H=64) | ~5 µs | ~0.05 ms | ~0.3 ms |
| QuaLiKiz (gyrokinetic) | ~1 s | ~100 s | ~1000 s |
| QLKNN (TensorFlow) | ~10 µs | ~0.1 ms | ~1 ms |

Key properties:
- Vectorised `predict_profile()` gives ~100× speedup over point-by-point loop
- SHA-256 weight checksums for reproducibility tracking
- Transparent fallback to analytic model when no weights are available
- ~2× faster than QLKNN due to zero framework overhead

*Reference: van de Plassche, K.L. et al. (2020). Phys. Plasmas 27, 022310.*

## Running Benchmarks

```bash
# Rust solver benchmarks
cd scpn-fusion-rs
cargo bench

# Python validation suite
python validation/validate_against_sparc.py

# Full 26-mode regression
python run_fusion_suite.py all
```

## Reproducing

All benchmark data is generated by CI on every push to `main`. See the
`validation-regression` job in `.github/workflows/ci.yml`.
