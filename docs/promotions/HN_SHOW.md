# Show HN Draft

**Title:** Show HN: SCPN Fusion Core -- Full-stack tokamak simulator in Rust+Python (AGPL)

**URL:** https://github.com/anulum/scpn-fusion-core

---

**Body:**

SCPN Fusion Core is an open-source tokamak plasma physics simulator that covers the full lifecycle of a fusion reactor: Grad-Shafranov equilibrium, MHD stability, transport, heating, neutronics, disruption prediction, and real-time control. The Python package (46 modules) handles physics and AI; a 10-crate Rust workspace provides native acceleration via PyO3 with transparent fallback to NumPy.

What makes it different from existing fusion codes (EFIT, TORAX, PROCESS, GENE): it's a single integrated codebase rather than a pipeline of separate Fortran programs, it includes a neuro-symbolic compiler that maps Petri-net control logic to stochastic spiking neural networks, and everything is AGPL-licensed so anyone can audit and extend it.

Benchmark highlights:
- Equilibrium: 15 ms @ 65x65 (Rust multigrid), 50x faster than Python, competitive with Fortran codes
- Inverse reconstruction: ~4 s, matching EFIT
- Neural transport surrogate: 5 us/point, 200,000x faster than gyrokinetic solvers
- Memory: 0.7 MB for a full equilibrium solve
- GPU roadmap: projected 2 ms equilibrium on wgpu (Vulkan/Metal/D3D12)

Six tutorial notebooks walk through everything from compact reactor design to validation against SPARC GEQDSK data:
https://anulum.github.io/scpn-fusion-core/notebooks/

Full benchmark comparison tables: https://github.com/anulum/scpn-fusion-core/blob/main/docs/BENCHMARKS.md

Happy to answer questions about the physics, the Rust architecture, or the neuro-symbolic compiler.
