# Task 10 Free-Boundary State Estimation And Disturbance Rejection

- Generated: `2026-03-11T21:53:05.585291+00:00`
- Overall pass: `YES`

## Nominal Baseline

- Nominal P95 axis error: `0.0336 m`
- Nominal P95 X-point error: `0.0364 m`
- Nominal stabilization rate: `1.000`

## Observer Performance

- Faulted mean state-estimation error: `0.0302 m` (threshold `<= 0.040`)
- Faulted mean actuator-bias estimation error: `0.0126` (threshold `<= 0.020`)
- Faulted final actuator-bias estimation error: `0.0018` (threshold `<= 0.010`)
- Faulted max uncertainty norm: `0.0642` (threshold `<= 0.080`)

## Disturbance Rejection

- Sensor-bias recovery steps: `9` (threshold `<= 12`)
- Actuator-bias recovery steps: `1` (threshold `<= 4`)
- Axis tracking degradation ratio: `1.531` (threshold `<= 1.70`)
- X-point tracking degradation ratio: `1.053` (threshold `<= 1.20`)
- Faulted stabilization rate: `1.000` (threshold `>= 0.95`)
- Faulted P95 axis error: `0.0515 m` (threshold `<= 0.055`)
- Faulted P95 X-point error: `0.0383 m` (threshold `<= 0.045`)
