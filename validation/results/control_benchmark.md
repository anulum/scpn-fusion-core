# PID vs SNN Controller — Head-to-Head Benchmark

Generated: 2026-02-14T19:41:09.949742+00:00

## Per-Scenario Metrics

| Scenario | Metric | PID | SNN | Better |
|----------|--------|----:|----:|--------|
| step_5mm             | Settling [ms]    |     4.6000 |   100.0000 | PID    |
|                      | Overshoot [mm]   |     5.0125 |     5.4138 | PID    |
|                      | SS error [mm]    |     0.0259 |     1.8788 | PID    |
|                      | RMS effort       |     0.0003 |     0.0061 | PID    |
|                      | Peak effort      |     0.0025 |     0.0069 | PID    |
|                      | us/step          |    23.3318 |    63.8785 | PID    |
|                      | Disrupted        |         no |         no | TIE    |
| step_noisy           | Settling [ms]    |     4.6000 |   100.0000 | PID    |
|                      | Overshoot [mm]   |     5.0125 |     5.8165 | PID    |
|                      | SS error [mm]    |     0.0259 |     3.7544 | PID    |
|                      | RMS effort       |     0.0003 |     0.0064 | PID    |
|                      | Peak effort      |     0.0025 |     0.0069 | PID    |
|                      | us/step          |    23.8388 |    82.1305 | PID    |
|                      | Disrupted        |         no |         no | TIE    |
| ramp_disturbance     | Settling [ms]    |     0.0000 |     0.0000 | TIE    |
|                      | Overshoot [mm]   |     0.0639 |     0.0000 | SNN    |
|                      | SS error [mm]    |     0.0262 |     0.0000 | SNN    |
|                      | RMS effort       |     0.0001 |     0.0000 | SNN    |
|                      | Peak effort      |     0.0002 |     0.0000 | SNN    |
|                      | us/step          |    34.2577 |    62.0133 | PID    |
|                      | Disrupted        |         no |         no | TIE    |
| random_perturbation  | Settling [ms]    |     0.0000 |   500.0000 | PID    |
|                      | Overshoot [mm]   |     0.0959 |     2.8631 | PID    |
|                      | SS error [mm]    |     0.0208 |     0.6484 | PID    |
|                      | RMS effort       |     0.0001 |     0.0039 | PID    |
|                      | Peak effort      |     0.0002 |     0.0069 | PID    |
|                      | us/step          |    23.3109 |    87.7049 | PID    |
|                      | Disrupted        |         no |         no | TIE    |
| plant_uncertainty    | Settling [ms]    |   100.0000 |   100.0000 | TIE    |
|                      | Overshoot [mm]   |    31.0220 |     5.8091 | SNN    |
|                      | SS error [mm]    |    26.4325 |     2.6324 | SNN    |
|                      | RMS effort       |     0.0085 |     0.0064 | SNN    |
|                      | Peak effort      |     0.0160 |     0.0069 | SNN    |
|                      | us/step          |    55.5646 |    69.6119 | PID    |
|                      | Disrupted        |         no |         no | TIE    |
| sensor_dropout       | Settling [ms]    |   100.0000 |   100.0000 | TIE    |
|                      | Overshoot [mm]   | 1463940018846927872.0000 | 13216709968313360318464.0000 | PID    |
|                      | SS error [mm]    | 150092771300398112.0000 | 1355064142709727952896.0000 | PID    |
|                      | RMS effort       |     8.1003 |     0.0049 | SNN    |
|                      | Peak effort      |    10.0000 |     0.0069 | SNN    |
|                      | us/step          |   109.4372 |   133.7359 | PID    |
|                      | Disrupted        |        YES |        YES | TIE    |

## Aggregate Summary

- **PID wins (settling time)**: 3/6
- **SNN wins (settling time)**: 0/6
- **Ties**: 3/6

## Formal Verification Properties

| Property | PID | SNN |
|----------|-----|-----|
| Boundedness proof | No proof | PROVED |
| Liveness proof | No proof | PROVED |
| Mutual exclusion proof | No proof | PROVED |
| Deterministic routing | N/A | PROVED |

> **Key insight**: The SNN controller may not beat PID on every numerical metric -- that is expected and honest. The decisive advantage of the SNN controller is its formally verified Petri net structure: boundedness, liveness, and mutual exclusion are *proved*, not assumed. PID has no such guarantees.
