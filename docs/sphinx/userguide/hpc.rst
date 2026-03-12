============================================
HPC and GPU Acceleration
============================================

SCPN-Fusion-Core supports high-performance computing through a Rust
native backend, C++ FFI bridge, and a planned GPU acceleration path.

Rust Workspace
---------------

The ``scpn-fusion-rs/`` directory contains a 10-crate Rust workspace
that mirrors the Python package structure:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Crate
     - Purpose
   * - ``fusion-types``
     - Shared data types, configuration structs, error types
   * - ``fusion-math``
     - Linear algebra (SOR, GMRES, multigrid), FFT, interpolation,
       Chebyshev polynomials, elliptic integrals, tridiagonal solver
   * - ``fusion-core``
     - Grad-Shafranov kernel, transport, inverse reconstruction,
       stability, pedestal model, AMR
   * - ``fusion-physics``
     - MHD sawtooth, Hall-MHD, turbulence, FNO, heating, compact
       reactor optimiser, design scanner, sandpile
   * - ``fusion-nuclear``
     - Neutronics, divertor, wall interaction, PWI erosion, TEMHD,
       balance of plant
   * - ``fusion-engineering``
     - Blanket engineering, magnet design, tritium systems, plant layout
   * - ``fusion-control``
     - PID, MPC, SNN controller, SPI mitigation, disruption predictor,
       digital twin, analytic solver, SOC learning
   * - ``fusion-diagnostics``
     - Sensor models, tomography
   * - ``fusion-ml``
     - Neural equilibrium, neural transport, disruption classifier,
       polynomial chaos expansion (PCE) UQ
   * - ``fusion-python``
     - PyO3 bindings producing ``scpn_fusion_rs.pyd`` / ``.so``

Build Configuration
^^^^^^^^^^^^^^^^^^^^

The workspace is optimised for maximum performance::

    # Cargo.toml [profile.release]
    opt-level = 3
    lto = "fat"
    codegen-units = 1

Key dependencies: ``ndarray``, ``nalgebra``, ``rayon`` (parallelism),
``rustfft``, ``serde``, ``pyo3`` (Python bindings).

The Rust workspace has no external C or Fortran dependencies -- it is
pure Rust.

Python-Rust FFI
^^^^^^^^^^^^^^^^

The ``fusion-python`` crate provides PyO3 bindings that expose the Rust
solvers as a native Python extension module.  The Python package
auto-detects the extension at import time::

    try:
        from ._rust_compat import FusionKernel, RUST_BACKEND
    except ImportError:
        from .fusion_kernel import FusionKernel
        RUST_BACKEND = False

All API signatures are identical between the Python and Rust paths,
ensuring zero code changes when switching backends.

C++ FFI Bridge
---------------

The ``hpc_bridge`` module (``hpc/hpc_bridge.py``) provides a C++ FFI
bridge for interfacing with external HPC solvers:

- ``solver.cpp`` -- C++ solver implementation using ``types.h`` shared
  data structures
- ``ctypes``-based Python bindings for calling compiled C++ from Python
- Shared-memory data exchange to avoid serialisation overhead

This bridge is primarily used for prototyping custom solver kernels
before porting them to Rust.

GPU Acceleration Roadmap
--------------------------

GPU support is planned in three phases (tracked in
``docs/GPU_ACCELERATION_ROADMAP.md``):

**Phase 1: wgpu SOR kernel**
   Red-Black SOR stencil implemented as a ``wgpu`` compute shader,
   providing cross-platform GPU acceleration (Vulkan, Metal, D3D12,
   WebGPU) with deterministic CPU fallback.

   Performance targets:

   - 65x65 grid: 2x--4x speedup
   - 257x257 grid: 5x--12x speedup

**Phase 2: GPU-backed GMRES preconditioning**
   CUDA/ROCm adapters for the GMRES linear solver with CPU fallback
   for environments without GPU drivers.

   Performance target: 2x--6x speedup on inverse solves.

**Phase 3: Full multigrid on device**
   Smooth, restrict, prolong, and coupled nonlinear multigrid path
   running entirely on GPU.

   Performance targets:

   - < 1 ms for control-loop grids
   - 10x--30x speedup for 257x257+ workloads

**Acceptance gates** for each phase:

- Correctness: residual behaviour matches CPU reference within
  configured tolerance
- Performance: measured speedups meet declared minimum floors
- Operations: runtime capability detection + automatic CPU fallback

The ``gpu_runtime`` module (``core/gpu_runtime.py``) provides the
``GPURuntimeBridge`` class for managing GPU device detection, memory
allocation, and kernel dispatch.

Benchmarking
--------------

Criterion micro-benchmarks are included in the Rust workspace::

    cd scpn-fusion-rs
    cargo bench

Available benchmarks:

- ``sor_bench.rs`` -- Red-Black SOR stencil at 65x65 and 128x128
- ``inverse_bench.rs`` -- Levenberg-Marquardt inverse reconstruction
  (FD vs analytical Jacobian comparison)
- ``neural_transport_bench.rs`` -- Neural transport MLP inference

Python-side profiling is available via::

    python profiling/profile_kernel.py --top 50
    python profiling/profile_geometry_3d.py --toroidal 48 --poloidal 48 --top 50

Results are written to ``artifacts/profiling/``.

Performance Summary
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Metric
     - Rust (release)
     - Python (NumPy)
     - Speedup
   * - 65x65 equilibrium
     - ~100 ms
     - ~5 s
     - ~50x
   * - 128x128 equilibrium
     - ~1 s
     - ~30 s
     - ~30x
   * - SOR step (65x65)
     - microseconds
     - milliseconds
     - ~100x
   * - Neural transport MLP
     - ~5 microseconds/point
     - ~500 microseconds/point
     - ~100x
   * - Inverse reconstruction
     - ~4 s (5 LM iters)
     - ~60 s
     - ~15x

.. note::

   These are internal measurements on specific hardware.  We encourage
   independent reproduction using ``cargo bench`` and
   ``benchmarks/collect_results.sh``.

Related Modules
-----------------

- :mod:`scpn_fusion.hpc.hpc_bridge` -- C++/Rust FFI bridge
- :mod:`scpn_fusion.core.gpu_runtime` -- GPU runtime management
- :mod:`scpn_fusion.core._rust_compat` -- Rust backend auto-detection
