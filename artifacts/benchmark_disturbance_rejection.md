# Disturbance Rejection Benchmark

ITER-like parameters: Ip=15 MA, BT=5.3 T, R=6.2 m, a=2.0 m, gamma_growth=100/s

## Results

### VDE

_Vertical Displacement Event: gamma=100/s exponential instability, impulsive kick. Duration: 2 s._
Duration: 2.0 s | dt: 1e-04 s | Steps: 20,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 1.077e-05 | 0.0843 | 1.996e-02 | 1.581e+01 | 0.3118 | Yes |
| H-infinity | 4.858e-05 | 0.0894 | 3.974e-02 | 2.001e+01 | 0.6953 | Yes |
| MPC | 5.274e-06 | 0.0553 | 1.705e-02 | 7.109e+00 | 0.4135 | Yes |
| SNN | 7.742e-02 | 1.9999 | 3.100e-01 | 3.910e+03 | 1.3101 | Yes |

### Density ramp

_Linear density ramp 0.5 to 1.2 Greenwald fraction over 2 s, then 2 s settling. Duration: 4 s._
Duration: 4.0 s | dt: 1e-04 s | Steps: 40,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 5.955e-05 | 3.9999 | 9.067e-03 | 2.130e+02 | 0.6699 | Yes |
| H-infinity | 9.081e-04 | 3.9999 | 3.165e-02 | 6.219e+02 | 1.4714 | Yes |
| MPC | 2.058e-05 | 3.9999 | 4.807e-03 | 1.628e+02 | 0.8471 | Yes |
| SNN | 1.548e-01 | 3.9999 | 3.143e-01 | 7.715e+03 | 2.9256 | Yes |

### ELM pacing

_ELM pacing at 10 Hz, 5 % beta_N drop per burst, recovery tracking. Duration: 3 s._
Duration: 3.0 s | dt: 1e-04 s | Steps: 30,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 7.181e-07 | 2.9999 | 1.304e-03 | 1.892e+01 | 0.5322 | Yes |
| H-infinity | 1.367e-05 | 2.9999 | 4.525e-03 | 5.772e+01 | 1.0992 | Yes |
| MPC | 4.855e-07 | 2.9593 | 1.115e-03 | 1.503e+01 | 0.7214 | Yes |
| SNN | 1.152e-01 | 2.9999 | 3.104e-01 | 5.824e+03 | 2.0510 | Yes |

## Metrics Definitions

- **ISE**: Integral of Squared Error (lower is better)
- **Settling Time**: Time to reach and stay within 5 % band (lower is better)
- **Peak Overshoot**: Maximum absolute deviation from setpoint (lower is better)
- **Control Effort**: Integral of |u| over time (lower = more efficient)
- **Wall-Clock**: Real execution time in seconds (lower = faster)
- **Stable**: Whether the plant state remained bounded during the scenario

## Verdict

- **VDE**: Best ISE = MPC (5.274e-06)
- **Density ramp**: Best ISE = MPC (2.058e-05)
- **ELM pacing**: Best ISE = MPC (4.855e-07)

*Generated: 2026-03-09 02:12:47 UTC*