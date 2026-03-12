# Task 8 Free-Boundary Supervisory Control

- Generated: `2026-03-11T21:53:07.208521+00:00`
- Overall pass: `YES`

## Scenario

- Shot length: `72`
- Control dt: `0.050 s`
- Current ramp start: `18`
- Coil kick step: `26`
- Sensor bias step: `32`

## Closed-Loop Acceptance

- P95 axis error: `0.0336 m` (threshold `<= 0.08 m`)
- P95 X-point error: `0.0336 m` (threshold `<= 0.11 m`)
- Stabilization rate: `1.000` (threshold `>= 0.88`)
- Mean estimation error: `0.01973 m` (threshold `<= 0.035 m`)
- Max |action|: `0.350` (threshold `<= 0.35`)
- Max |coil current|: `1.191` (threshold `<= 1.50`)

## Safety Supervisor

- Supervisor interventions: `13` (threshold `>= 1`)
- Saturation events: `13`
- Max bias norm: `0.0095 m`
