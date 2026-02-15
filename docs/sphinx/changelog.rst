=========
Changelog
=========

The full changelog is maintained in the project root at
``CHANGELOG.md``.  Key releases are summarised below.

v1.0.2 (Current)
------------------

- GPU acceleration roadmap and ``gpu_runtime`` module
- Full analytical Jacobian mode for inverse reconstruction
- AMR hierarchy and EPED-like pedestal model in Rust kernel
- Polynomial chaos UQ module (``fusion-ml/pce``)
- Point-wise :math:`\psi` RMSE validation on all 8 SPARC GEQDSKs
- Neural equilibrium rewrite (removed sklearn/pickle dependencies)
- Multi-regime SPARC-parameterised FNO training
- Multigrid V-cycle wired into FusionKernel Picard loop
- Solver method selection API (``set_solver_method("sor"|"multigrid")``)
- 3D flux-surface mesh generation (OBJ export + PNG preview)
- GEQDSK reader/writer with 8-file SPARC validation
- RMSE dashboard report generation
- Criterion benchmarks for SOR, inverse, and neural transport

v1.0.1
-------

- Hardening waves H7 and H8 (184 tasks total)
- Every ``unwrap()`` in Rust workspace replaced with ``FusionResult<T>``
- Input validation guards on all public API boundaries
- Scoped RNG isolation for deterministic replay
- Sensor model guards in diagnostics
- MPC input validation

v1.0.0
-------

- Initial public release
- Grad-Shafranov equilibrium solver (Picard + Red-Black SOR)
- 1.5D radial transport with IPB98(y,2) scaling
- Neuro-symbolic compiler (Petri net to SNN pipeline)
- FNO turbulence suppressor
- Digital twin with RL-trained MLP policy
- ML disruption predictor
- Shattered pellet injection mitigation
- Blanket neutronics and PWI erosion
- Synthetic diagnostics and tomography
- Compact reactor optimiser (MVR-0.96)
- 10-crate Rust workspace with PyO3 bindings
- Validation against 8 SPARC GEQDSK files and ITPA H-mode database
- 6 tutorial Jupyter notebooks
- Docker support
- CI/CD pipeline with GitHub Actions
