# SCPN Migration Plan: From Workstation to Exascale

This document outlines the roadmap for transitioning the SCPN Fusion Framework from a Python prototype to a C++/MPI High-Performance Computing (HPC) application.

## Phase 1: The Hybrid Era (Current State)
We use Python for orchestration and C++ for compute-heavy kernels.
- **Data:** Numpy arrays (Python) -> Pointers (C++).
- **Kernels:** `solver.cpp` implements SOR solver.
- **Benefit:** 10-50x speedup on single node.

## Phase 2: Domain Decomposition (MPI)
To scale beyond one CPU, we must split the grid.
1.  **Decomposition:** Divide the $(R, Z)$ grid into $N$ sub-domains.
2.  **Ghost Cells:** Each sub-domain stores a layer of neighbors' data.
3.  **Communication:** Use `MPI_Send` / `MPI_Recv` to sync Ghost Cells after every iteration.

### C++ MPI Architecture Sketch
```cpp
MPI_Init(&argc, &argv);
int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Each rank solves only its chunk
Solver local_solver(global_config, rank);

while(error > tol) {
    local_solver.step();
    local_solver.sync_boundaries(); // MPI Communication
    
    // Global reduction for convergence check
    double local_err = local_solver.get_error();
    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}
```

## Phase 3: GPU Acceleration (CUDA/HIP)
The loop in `solver.cpp` is perfectly parallel.
1.  **Direct Port:** Replace `for` loops with CUDA Kernels.
2.  **Memory:** Copy `psi` and `j_phi` to GPU VRAM at start. Keep them there.
3.  **Libraries:** Use `cuBLAS` or `AmgX` for linear algebra.

## Phase 4: In-Situ Visualization
Don't save all data (too big). Use **ParaView Catalyst** to render images *during* simulation on the cluster nodes.

---
**Status:** Phase 1 Prepared.
**Next Step:** Deploy `solver.cpp` to a Linux environment with GCC/Clang.
