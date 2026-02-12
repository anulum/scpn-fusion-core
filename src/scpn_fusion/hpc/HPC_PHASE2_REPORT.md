# Phase 2: Domain Decomposition & Parallelism

**Status**: IMPLEMENTED (Initial OpenMP Support)
**Date**: 2026-02-01

## Updates
- Integrated **Red-Black SOR** ordering in `solver.cpp`.
- Added OpenMP `#pragma` directives for shared-memory parallelism.
- Updated `hpc_bridge.py` to auto-detect platform (Windows .dll vs Linux .so) and search path.
- Integrated `HPCBridge` into `FusionKernel` (Python).

## Next Steps
1. **Compile**: Run `g++ -shared -o scpn_solver.dll src/scpn_fusion/hpc/solver.cpp -O3 -mavx2 -fopenmp` (Windows) or similar on Linux.
2. **MPI Integration**: For multi-node scaling, replace OpenMP loops with domain decomposition blocks exchanging ghost cells.
