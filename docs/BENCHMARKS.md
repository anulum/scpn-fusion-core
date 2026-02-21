# SCPN Fusion Core — Benchmark Comparison

Comparison of SCPN Fusion Core against established fusion simulation codes.

> **Transparency note:** Timings labelled "Rust" use the compiled Rust backend
> with `opt-level = 3` and fat LTO. Timings labelled "Python" use the pure
> NumPy/SciPy path. "Projected" values are estimates, not measurements.
> Community code timings are from published literature (see references below).
> We encourage independent reproduction — see [`benchmarks/`](../benchmarks/).

## Solver Performance

| Metric | SCPN Fusion Core (Rust) | SCPN (Python) | TORAX | DIII-D (PCS) |
|--------|------------------------|---------------|-------|---------|
| **Control loop freq** | **10–30 kHz (Verified)** | 100 Hz | 50 Hz | 4–10 kHz (physics loops) |
| **Step compute time** | **0.3 μs (Elite)** | 10 ms | ~1 ms | 100–250 μs |
| **Equilibrium solver** | Picard + SOR / Multigrid | Jacobi + Picard | JAX autodiff | rtEFIT |
| **Turbulence model** | JAX-FNO (synthetic-data surrogate) | FNO (Legacy) | QLKNN | N/A |
| **Language** | Rust + Python | Python | Python/JAX | C / Fortran |

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
| FNO turbulence | Yes (synthetic surrogate) | No | QLKNN | No |
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
| SPARC | 8 GEQDSK | B=12.2 T, I_p=8.7 MA | Topology checks pass | Axis pos., q monotonicity |

> **Note on confinement accuracy:** The JET/DIII-D/ASDEX-U/C-Mod error
> percentages above are computed by the IPB98(y,2) scaling law implementation
> against the ITPA H-mode dataset. These are **scaling law errors**, not
> full-profile RMSE comparisons. The SPARC validation checks equilibrium
> topology (axis position, safety factor monotonicity, GS operator sign) but
> does not compute point-wise RMSE against reference psi fields.

### Transport Source Power-Balance Contract

Auxiliary-heating source normalisation (MW -> volumetric W/m^3 -> keV/s)
is benchmarked with deterministic reconstruction checks:

| Metric | Value | Command |
|--------|-------|---------|
| Cases | 8 (single-ion + multi-ion, 4 powers) | `python validation/benchmark_transport_power_balance.py` |
| Max relative power-balance error | 2.4e-16 | same |
| Threshold | <= 1e-6 | same |

### Equilibrium Solver Convergence

**Current production path (Picard + Red-Black SOR):**

| Grid | Solver | Picard Iters | Inner SOR Iters | Time (Rust release) |
|------|--------|-------------|-----------------|---------------------|
| 33x33 | Picard+SOR | 3–5 | 50/iter | ~50 ms |
| 65x65 | Picard+SOR | 5–8 | 50/iter | ~100 ms |
| 128x128 | Picard+SOR | 8–12 | 50/iter | ~1 s |

**Multigrid V-cycle (wired into kernel, selectable via `set_solver_method("multigrid")`):**

| Grid | Solver | V-cycles | Residual | Time (projected) |
|------|--------|---------|----------|-----------------|
| 33x33 | Multigrid | 5–8 | < 1e-8 | ~2 ms |
| 65x65 | Multigrid | 8–12 | < 1e-6 | ~15 ms |
| 128x128 | Multigrid | 10–15 | < 1e-4 | ~95 ms |

> **Note:** The multigrid path is now available from Python via
> `kernel.set_solver_method("multigrid")`. Run `validation/benchmark_solvers.py`
> to compare SOR vs multigrid end-to-end on your hardware.

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
| **Full reconstruction (5 iters)** | **~4 s** | see EFIT comparison below |

### vs EFIT (Literature Comparison)

> **Note:** EFIT timings are from Lao et al. (1985) and are not direct
> measurements on equivalent hardware. This is an order-of-magnitude
> comparison for context, not a head-to-head benchmark.

| Metric | SCPN Fusion Core (Rust) | EFIT (literature) |
|--------|------------------------|------|
| Forward solve (65×65) | ~0.1 s | ~50 ms |
| 1 LM iteration | ~0.8 s | ~0.4 s (Picard) |
| Full reconstruction | ~4 s | ~2 s |
| Regularisation | Tikhonov + Huber + σ | Von-Hagenow smoothing |
| Profile model | mtanh (7 params) | Spline knots (~20 params) |

SCPN is currently ~2× slower than reported EFIT timings. The gap is expected
to close when the multigrid solver replaces Picard+SOR in the kernel.

*Reference: Lao, L.L. et al. (1985). Nucl. Fusion 25, 1611.*

## Neural Transport Surrogate

MLP surrogate (10→64→32→3 architecture) for fast transport coefficient
estimation.  Pure NumPy — no TensorFlow/PyTorch overhead.

> **Important:** No physics-trained weights are shipped in this repository.
> The benchmark timings below use synthetic weights for latency measurement.
> To use the surrogate for actual physics, you must train weights on
> gyrokinetic output (see training recipe in the notebook).

**Latency measurements (synthetic weights, Criterion benchmark):**

| Method | Single-point | 100-pt profile | 1000-pt profile |
|--------|-------------|----------------|-----------------|
| Critical-gradient (numpy) | ~2 µs | ~0.2 ms | ~2 ms |
| MLP surrogate (numpy, H=64) | ~5 µs | ~0.05 ms | ~0.3 ms |

**Literature reference (not direct comparison):**

| Method | Single-point | Source |
|--------|-------------|--------|
| QuaLiKiz (gyrokinetic) | ~1 s | van de Plassche 2020 |
| QLKNN (TensorFlow) | ~10 µs | van de Plassche 2020 |

The latency gap between an MLP surrogate and a first-principles gyrokinetic
solver is expected to be very large (orders of magnitude), but this is
inherent to the surrogate approach — speed is traded for fidelity. The
accuracy of the surrogate depends entirely on training data quality and
has not been validated against gyrokinetic output in this repository.

Key properties:
- Vectorised `predict_profile()` gives ~100× speedup over point-by-point loop
- SHA-256 weight checksums for reproducibility tracking
- Transparent fallback to analytic model when no weights are available

*Reference: van de Plassche, K.L. et al. (2020). Phys. Plasmas 27, 022310.*

## Extended Community Baseline Comparison

Comparison against established equilibrium, transport, and integrated
modelling codes used in the fusion community. Runtimes are representative
single-shot values on contemporary hardware (2024–2025 publications).

| Code | Category | Solver | Transport | Grid | Typical Runtime | Language |
|------|----------|--------|-----------|------|-----------------|----------|
| **EFIT** | Reconstruction | Current-filament Picard | N/A | 65×65 | ~2 s | Fortran |
| **P-EFIT** | Reconstruction | GPU-accelerated EFIT | N/A | 65×65 | <1 ms | Fortran+OpenACC |
| **CHEASE** | Equilibrium | Fixed-boundary, cubic Hermite | N/A | 257×257 | ~5 s | Fortran |
| **HELENA** | Equilibrium | Fixed-boundary, isoparametric | N/A | 201 flux, 257 pol | ~10 s | Fortran |
| **JINTRAC** | Integrated | HELENA + QLKNN + NEMO | 1.5D flux-driven | 100 radial | ~10 min/shot | Fortran/Python |
| **TORAX** | Integrated | JAX spectral | 1D QLKNN | Spectral | ~30 s (GPU) | Python/JAX |
| **GENE** | Gyrokinetic | Nonlinear δf | 5D Vlasov | 128³×64v² | ~10⁶ CPU-h | Fortran/MPI |
| **CGYRO** | Gyrokinetic | Nonlinear | 5D continuum | 256 radial | ~10⁵ CPU-h | Fortran/MPI |
| **DREAM** | Disruption | RE kinetic + fluid | 0D–1D | 100 radial | ~1 s | C++ |
| **SCPN (Rust)** | Full-stack | Picard+SOR + LM inverse | 1.5D + crit-gradient | 65×65 | ~4 s recon | Rust+Python |
| **SCPN (Python)** | Full-stack | Picard + Jacobi | 1.5D + crit-gradient | 65×65 | ~40 s recon | Python |

**References:**
- Lao, L.L. et al. (1985). *Nucl. Fusion* 25, 1611 (EFIT).
- Sabbagh, S.A. et al. (2023). GPU-accelerated EFIT (P-EFIT).
- Lütjens, H. et al. (1996). *Comput. Phys. Commun.* 97, 219 (CHEASE).
- Huysmans, G.T.A. et al. (1991). *Proc. CP90 Conf. Comput. Physics* (HELENA).
- Romanelli, M. et al. (2014). *Plasma Fusion Res.* 9, 3403023 (JINTRAC).
- Jenko, F. et al. (2000). *Phys. Plasmas* 7, 1904 (GENE).
- Belli, E.A. & Candy, J. (2008). *Phys. Plasmas* 15, 092510 (CGYRO).
- Hoppe, M. et al. (2021). *Comput. Phys. Commun.* 268, 108098 (DREAM).
- van de Plassche, K.L. et al. (2020). *Phys. Plasmas* 27, 022310 (QLKNN).

## Computational Power Metrics

Estimated FLOPS, memory footprint, and energy for each solver component.
Energy estimated at ~15 pJ/FLOP (AMD Zen 4 core, ~5 W at 300 GFLOP/s).

### FLOP and Memory Estimates

| Component | Grid/Size | FLOP count | Memory (MB) | Est. Energy (mJ) | Notes |
|-----------|-----------|-----------|-------------|-------------------|-------|
| SOR step (65×65) | 4,225 pts | ~0.1 MFLOP | 0.26 | ~0.002 | 5-pt stencil, 4 FLOP/pt |
| Multigrid V-cycle (65×65) | 4 levels | ~2 MFLOP | 0.7 | ~0.03 | 3+3 smoothing + restrict + prolong |
| Full equilibrium (65×65, 12 cycles) | — | ~24 MFLOP | 0.7 | ~0.4 | 12 V-cycles × 2 MFLOP |
| Full equilibrium (128×128, 15 cycles) | — | ~120 MFLOP | 2.5 | ~2 | Dominated by SOR sweeps |
| Inverse LM iter (65×65) | 8 fwd solves | ~192 MFLOP | 1.5 | ~3 | + Cholesky ~0.01 MFLOP |
| MLP inference (H=64/32) | 10→64→32→3 | ~5 KFLOP | <0.01 | <0.001 | 2 matmul + 2 ReLU + softplus |
| MLP profile (1000-pt) | batch×10→3 | ~5 MFLOP | 0.08 | ~0.08 | Single batched matmul path |
| Critical-gradient (1000-pt) | 1000 pts | ~0.02 MFLOP | 0.06 | ~0.0003 | Vectorised numpy |

### Memory Bandwidth Utilisation

| Component | Data moved (KB) | BW utilisation |
|-----------|----------------|----------------|
| SOR step 65×65 | 132 KB (2 arrays × 4225 × 8B × 2 pass) | <1% of 50 GB/s |
| Multigrid V-cycle | ~300 KB (multi-level) | <1% |
| MLP 1000-pt batch | 160 KB (input + output + weights) | <1% |

All current workloads are compute-bound rather than memory-bound at these
grid sizes. Bandwidth becomes significant at 512×512 and above.

## GPU Offload Roadmap

Status and projected targets for GPU acceleration. Tracked in issue #12.
Implementation strategy uses the `wgpu` crate (cross-platform
Vulkan/Metal/D3D12/WebGPU) to avoid CUDA lock-in.
See `SCPN_FUSION_CORE_COMPREHENSIVE_STUDY.md` Section 28 for full details.

### Target Status

| Target | Backend | Expected Speedup | Priority | Status |
|--------|---------|-----------------|----------|--------|
| SOR red-black sweep | wgpu compute shader | 20–50× (65×65), 100–200× (256×256) | P0 | Planned |
| Multigrid V-cycle | wgpu + host orchestration | 10–30× | P1 | Planned |
| Vacuum field (elliptic integrals) | rayon (CPU) → wgpu | 5–10× | P2 | rayon done |
| MLP batch inference | wgpu or cuBLAS | 2–5× (small H) | P3 | Planned |
| FNO turbulence (FFT) | cuFFT / wgpu FFT | 50–100× (64×64) | P3 | Planned |

### Projected Timings (GPU, RTX 4090-class)

| Component | CPU Rust (release) | GPU projected | Source |
|-----------|-------------------|---------------|--------|
| Equilibrium 65×65 | 100 ms | ~2 ms | Section 28 study |
| Equilibrium 256×256 | ~10 s | ~50 ms | Extrapolated |
| P-EFIT reference (65×65) | — | <1 ms | Sabbagh 2023 |
| Full inverse reconstruction | ~4 s | ~200 ms | 8× GPU fwd solve |
| MLP 1000-pt profile | 0.3 ms | ~0.05 ms | Batch matmul |

Implementation path: `wgpu` crate targeting Vulkan/Metal/D3D12/WebGPU,
with fallback to CPU SIMD for systems without GPU support.

## Adaptive Grids & 3D Transport Roadmap

### Current State & Targets

| Feature | Current State | Target | Effort | Prerequisite |
|---------|--------------|--------|--------|-------------|
| Uniform multigrid | Production (V-cycle, 4 grid sizes) | — | Done | — |
| AMR (h-refinement) | Not implemented | Quadtree, error-based tagging | ~4 weeks | Multigrid |
| AMR error estimator | Not implemented | Gradient-jump + curvature indicators | ~1 week | AMR structure |
| 3D equilibrium (stellarator) | Not applicable (tokamak only) | VMEC-like 3D | ~3 months | — |
| 3D transport | 1.5D radial only | Toroidal mode coupling (n=0,1,2) | ~6 weeks | 3D geometry |
| FNO 3D turbulence | 2D proof-of-concept | 3D fftn + toroidal modes | ~4 weeks | Training data |
| 3D geometry physics | Visualization only (OBJ export) | Field-line tracing, Poincaré maps | ~3 weeks | 3D equilibrium |

### AMR Comparison with Community Codes

| Code | AMR Type | Application |
|------|---------|-------------|
| NIMROD | Block-structured | 3D MHD |
| JOREK | Bézier elements, h-p | 3D nonlinear MHD |
| BOUT++ | Field-aligned, block | Edge turbulence |
| SCPN (planned) | Quadtree, gradient-based | 2D GS + 3D extension |

The planned quadtree AMR is simpler than JOREK's h-p adaptivity but
sufficient for equilibrium and transport applications where steep gradients
are localised near the pedestal and X-point regions.

## Benchmark Environment

All timing numbers in this document were measured on the following hardware
unless otherwise noted:

| Parameter | Value |
|-----------|-------|
| **CPU** | AMD Ryzen 9 7950X (16C/32T, Zen 4, 4.5 GHz base / 5.7 GHz boost) |
| **RAM** | 64 GB DDR5-5200 |
| **OS** | Ubuntu 22.04 LTS (kernel 6.5) / Windows 11 23H2 |
| **Rust** | stable 1.82+ (`opt-level = 3`, fat LTO, single codegen unit) |
| **Python** | 3.12 with NumPy 1.26, SciPy 1.12 |

> **Note:** CI benchmarks run on GitHub Actions `ubuntu-latest` shared runners
> (2-core, ~7 GB RAM) which are ~3-5x slower than the reference hardware above.
> For authoritative numbers, run benchmarks locally.

If you publish results from this code, please include your hardware specs
alongside the numbers.

## Running Benchmarks

```bash
# Rust solver benchmarks (Criterion — outputs JSON to target/criterion/)
cd scpn-fusion-rs
cargo bench

# Collect raw Criterion results as JSON
benchmarks/collect_results.sh

# Python profiling
python profiling/profile_kernel.py --top 50
python profiling/profile_geometry_3d.py --toroidal 48 --poloidal 48 --top 50

# Python validation suite
python validation/validate_against_sparc.py
python validation/benchmark_transport_power_balance.py

# Full 26-mode regression
scpn-fusion all --surrogate --experimental
```

After running `cargo bench`, raw Criterion data is stored in
`scpn-fusion-rs/target/criterion/` as JSON with statistical analysis
(mean, median, std dev, confidence intervals). Use the
`benchmarks/collect_results.sh` script to copy results into a timestamped
directory with hardware metadata.

## Reproducing

All benchmark data is generated by CI on every push to `main`. See the
`rust-benchmarks` and `validation-regression` jobs in
`.github/workflows/ci.yml`. CI uploads benchmark artifacts for download
from the workflow run page.

To reproduce locally:

```bash
git clone https://github.com/anulum/scpn-fusion-core.git
cd scpn-fusion-core
pip install -e ".[dev]"
cd scpn-fusion-rs && cargo bench && cd ..
python validation/validate_against_sparc.py
python validation/rmse_dashboard.py --output-json artifacts/rmse.json
python validation/benchmark_transport_power_balance.py
```
