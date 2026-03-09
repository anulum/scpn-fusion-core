# Disturbance Rejection Benchmark

ITER-like parameters: Ip=15 MA, BT=5.3 T, R=6.2 m, a=2.0 m, gamma_growth=100/s

## Results

### VDE

_Vertical Displacement Event: gamma=100/s exponential instability, impulsive kick. Duration: 2 s._
Duration: 2.0 s | dt: 1e-04 s | Steps: 20,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 1.077e-05 | 0.0843 | 1.996e-02 | 1.581e+01 | 0.2468 | Yes |
| H-infinity | 1.931e+02 | 1.9999 | 1.008e+01 | 2.449e+05 | 0.0127 | No |
| MPC | 1.904e+02 | 1.9999 | 1.001e+01 | 5.752e-03 | 0.7487 | No |
| SNN | 1.921e+02 | 1.9999 | 1.005e+01 | 9.772e+03 | 0.0275 | No |

### Density ramp

_Linear density ramp 0.5 to 1.2 Greenwald fraction over 2 s, then 2 s settling. Duration: 4 s._
Duration: 4.0 s | dt: 1e-04 s | Steps: 40,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 5.955e-05 | 3.9999 | 9.067e-03 | 2.130e+02 | 0.5649 | Yes |
| H-infinity | 3.889e+02 | 3.9999 | 1.008e+01 | 4.927e+05 | 0.0196 | No |
| MPC | 3.864e+02 | 3.9999 | 1.003e+01 | 1.163e-02 | 0.9381 | No |
| SNN | 3.853e+02 | 3.9999 | 1.001e+01 | 1.967e+04 | 0.0512 | No |

### ELM pacing

_ELM pacing at 10 Hz, 5 % beta_N drop per burst, recovery tracking. Duration: 3 s._
Duration: 3.0 s | dt: 1e-04 s | Steps: 30,000

| Controller | ISE | Settling Time (s) | Peak Overshoot | Control Effort | Wall-Clock (s) | Stable |
|------------|----:|-------------------:|---------------:|---------------:|---------------:|--------|
| PID | 7.181e-07 | 2.9999 | 1.304e-03 | 1.892e+01 | 0.3713 | Yes |
| H-infinity | 2.919e+02 | 2.9999 | 1.011e+01 | 3.687e+05 | 0.0182 | No |
| MPC | 2.889e+02 | 2.9999 | 1.006e+01 | 8.674e-03 | 1.2140 | No |
| SNN | 2.863e+02 | 2.9999 | 1.001e+01 | 1.463e+04 | 0.0490 | No |

## Metrics Definitions

- **ISE**: Integral of Squared Error (lower is better)
- **Settling Time**: Time to reach and stay within 5 % band (lower is better)
- **Peak Overshoot**: Maximum absolute deviation from setpoint (lower is better)
- **Control Effort**: Integral of |u| over time (lower = more efficient)
- **Wall-Clock**: Real execution time in seconds (lower = faster)
- **Stable**: Whether the plant state remained bounded during the scenario

## Verdict

- **VDE**: Best ISE = PID (1.077e-05)
- **Density ramp**: Best ISE = PID (5.955e-05)
- **ELM pacing**: Best ISE = PID (7.181e-07)

*Generated: 2026-03-09 01:16:42 UTC*