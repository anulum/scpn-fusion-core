# r/programming Post Draft

**Title:** SCPN Fusion Core: a Rust+Python tokamak plasma physics simulator with neuro-symbolic control

---

**Body:**

We've open-sourced SCPN Fusion Core, a full-stack tokamak fusion reactor simulator. The engineering side might interest this community:

**Architecture:**
- 46-module Python package (physics, AI, diagnostics) + 10-crate Rust workspace (native solvers via PyO3)
- Transparent fallback: `try: import scpn_fusion_rs` -- if the Rust extension isn't built, NumPy kicks in
- Property-based testing on both sides: Hypothesis (Python) and proptest (Rust) covering numerical invariants and solver convergence

**Rust solver stack:**
- Multigrid V-cycle Grad-Shafranov equilibrium solver (15 ms @ 65x65, 50x faster than Python)
- Levenberg-Marquardt inverse reconstruction with Tikhonov regularisation (~4 s, competitive with EFIT in Fortran)
- Fat LTO + single codegen unit for maximum inlining
- No C/Fortran dependencies -- pure Rust with ndarray, nalgebra, rayon, rustfft

**MLP transport surrogate:**
- Pure NumPy inference (10->64->32->3 architecture), no TensorFlow/PyTorch
- 5 us/point, 200,000x faster than gyrokinetic solvers (GENE, CGYRO)
- ~2x faster than QLKNN due to zero framework overhead
- SHA-256 weight checksums for reproducibility

**Benchmarks vs community codes:**

| Code | Category | Runtime | Language |
|------|----------|---------|----------|
| GENE | 5D gyrokinetic | ~10^6 CPU-h | Fortran/MPI |
| JINTRAC | Integrated | ~10 min | Fortran/Python |
| EFIT | Reconstruction | ~2 s | Fortran |
| **SCPN (Rust)** | **Full-stack** | **~4 s recon** | **Rust+Python** |
| P-EFIT | GPU recon | <1 ms | Fortran+OpenACC |

**GPU roadmap:** wgpu (Vulkan/Metal/D3D12/WebGPU) targeting ~2 ms equilibrium on RTX 4090-class hardware.

**Links:**
- Repository: https://github.com/anulum/scpn-fusion-core
- Tutorial notebooks (HTML): https://anulum.github.io/scpn-fusion-core/notebooks/
- Benchmark comparison: https://github.com/anulum/scpn-fusion-core/blob/main/docs/BENCHMARKS.md
- Rust API docs: https://anulum.github.io/scpn-fusion-core/rust/fusion_core/

Licensed AGPL-3.0. Feedback welcome.
