# Disturbance Rejection Benchmark

ITER-like parameters: Ip=15 MA, BT=5.3 T, R=6.2 m, a=2.0 m

## Results

| Controller | Scenario | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Stable |
|------------|----------|----:|-------------------:|---------------:|---------------:|--------|
| SNN (H-inf) | VDE | 4.844e-06 | 0.0545 | 1.670e-02 | 6.887e+00 | Yes |
| PID | VDE | 1.358e+01 | 0.1999 | 1.004e+01 | 2.789e+02 | No |
| MPC | VDE | 1.368e+01 | 0.1999 | 1.006e+01 | 2.155e+01 | No |
| SNN (H-inf) | Density ramp | 4.461e-05 | 2.9990 | 7.904e-03 | 1.620e+02 | Yes |
| PID | Density ramp | 2.637e+02 | 2.9990 | 1.041e+01 | 5.064e+03 | No |
| MPC | Density ramp | 2.483e+02 | 2.9990 | 1.011e+01 | 4.324e+02 | No |
| SNN (H-inf) | ELM pacing | 2.071e-07 | 0.4597 | 1.487e-03 | 3.311e+00 | Yes |
| PID | ELM pacing | 4.069e+01 | 0.4999 | 1.007e+01 | 8.124e+02 | No |
| MPC | ELM pacing | 4.094e+01 | 0.4999 | 1.009e+01 | 6.271e+01 | No |

## Scenario Descriptions

- **VDE**: Vertical Displacement Event: gamma=100/s instability, impulsive kick.  Must stabilise within 50 ms.
- **Density ramp**: Linear density ramp 0.5 to 1.2 Greenwald fraction over 2 s.  Overshoot must be < 20%.
- **ELM pacing**: ELM pacing at 10 Hz, 5% beta_N drop per burst.  Must recover within 30 ms per burst.

## Metrics Definitions

- **ISE**: Integral of Squared Error (lower is better)
- **Settling Time**: Time to reach and stay within 5% band (lower is better)
- **Peak Overshoot**: Maximum absolute deviation from target (lower is better)
- **Control Effort**: Integral of |u| over time (lower = more efficient)
- **Stable**: Whether the plant state remained bounded during the scenario

*Generated: 2026-02-17 09:22:04 UTC*