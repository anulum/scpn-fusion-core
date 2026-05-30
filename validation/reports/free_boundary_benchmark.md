# Free-Boundary Validation Benchmark

- Benchmark ID: `free_boundary_coil_vacuum_reconstruction`
- Benchmark scope: `free_boundary_reconstruction`
- Contract: External coil Green-function vacuum flux on boundary, limiter, axis, and X-point metadata; not a fixed-boundary Dirichlet replay or reduced-order surrogate.

- Gate result: `True` (5/5 gates passed)

| Test | Metric | Result | Pass |
|------|--------|--------|------|
| Single Coil | Rel Error | 0.00e+00 | True |
| Boundary Green reconstruction | RMSE | 0.00e+00 | True |
| Boundary Green reconstruction | Response rank | 1/1 coils, 5 points | N/A |
| Boundary Green reconstruction | Limiter/topology metadata | 4 limiter, 2 X-points, axis flux 2.589381e-01 | True |
| Boundary Green reconstruction | Min limiter clearance | 0.380789 m | True |
| Boundary Green reconstruction | Limiter containment | 1.000 | True |
| Boundary Green reconstruction | X-point pair flux residual | 0.00e+00 | True |
| Shape-control current inversion | Current relative L2 error | 2.96e-15 | True |
| Shape-control current inversion | Flux relative RMSE | 8.44e-17 | True |
| Shape-control current inversion | Response rank | 3/3 coils, 5 points | N/A |
| Solver free-boundary contract | Vacuum boundary abs error | 0.00e+00 | True |
| Solver free-boundary contract | Boundary points | 256 over 1 outer iter | N/A |
| Solver free-boundary contract | Topology metadata | 4 limiter, 2 X-points, axis flux 2.589381e-01 | True |
| Solver free-boundary contract | Limiter containment | 0.000 (diagnostic_computational_wall_outside_limiter) | N/A |
| Solver free-boundary contract | X-point pair flux residual | 0.00e+00 | True |
| Helmholtz | B_z Axis Ref | 0.8992 T | N/A |
| X-point diagnostic | Detected Z | -1.5000 | N/A |
| JAX free-boundary wall flux | Vacuum boundary abs error | 7.11e-15 | True |
| JAX free-boundary wall flux | 33x33 solve wall time | 0.625122 s | N/A |
| JAX free-boundary axis | Boundary distance | 2 cells | True |
| JAX free-boundary source | Vacuum boundary flux level | 1.655107e+01 | N/A |
| JAX free-boundary source | Interior axis flux | 6.604981e+01 | N/A |
