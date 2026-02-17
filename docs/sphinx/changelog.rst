=========
Changelog
=========

The full changelog is maintained in the project root at
``CHANGELOG.md``.  Key releases are summarised below.

v3.1.0 (Current)
------------------

- **TBR realism**: Port-coverage (0.80), streaming (0.85), blanket-fill correction factors
- **Greenwald density limit**: ``n_GW = I_p / (pi * a^2)``; scan skips unphysical points
- **Temperature cap**: Hard 25 keV cap with warning emission
- **Q ceiling**: Capped at 15 (was unbounded at Q=98)
- **Energy conservation**: Per-timestep ``W_before``/``W_after`` diagnostic in transport solver
- **Dashboard auto-flagging**: PASS/WARN/FAIL flags + matplotlib plots
- **CI gate hardening**: Disruption FPR hard fail (15%), TBR [1.0, 1.4], Q <= 15
- **Issue templates**: YAML forms with mandatory physics reference field
- 24 new tests, 1141 total passing

v3.0.0
-------

- Rust SNN via PyO3 (``PySnnPool``, ``PySnnController``)
- Full-chain Monte Carlo UQ: equilibrium -> transport -> fusion power
- Shot Replay Streamlit tab with NPZ disruption data overlay
- FNO turbulence surrogate deprecated (runtime ``FutureWarning``)

v2.1.0
-------

- GEQDSK expansion (100+ multi-machine equilibria)
- Sauter bootstrap current model
- Spitzer resistivity with neoclassical corrections
- Free-boundary Grad-Shafranov extension

v2.0.0
-------

- Multigrid V-cycle solver (3-5x faster convergence)
- Gyro-Bohm + EPED pedestal transport model
- H-infinity controller (Riccati ARE synthesis)
- Disruption predictor on 10,000 synthetic + 10 reference shots
- Real-shot validation gate in CI

v1.0.2
-------

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
