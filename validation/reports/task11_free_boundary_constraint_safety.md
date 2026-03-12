# Task 11 Free-Boundary Constraint And Safety Gate

- Generated: `2026-03-11T21:53:05.536562+00:00`
- Overall pass: `YES`

## Constraint Envelope

- Axis R bounds: `5.96 .. 6.04 m`
- Axis Z bounds: `-0.07 .. 0.07 m`
- X-point R bounds: `4.99 .. 5.05 m`
- X-point Z bounds: `-3.51 .. -3.45 m`
- Max action L1: `0.76`

## Constraint-Aware Control

- P95 axis error: `0.0586 m` (threshold `<= 0.065`)
- P95 X-point error: `0.0560 m` (threshold `<= 0.060`)
- Stabilization rate: `1.000` (threshold `>= 0.95`)
- Max |action|: `0.310` (threshold `<= 0.35`)
- Max action L1: `0.760` (threshold `<= 0.76`)
- Max |coil current|: `1.460` (threshold `<= 1.50`)

## Supervisory Safety

- Fallback mode count: `18` (threshold `>= 5`)
- Invariant violation count: `6` (threshold `<= 6`)
- Failsafe trip count: `0` (threshold `<= 0`)
- Max risk score: `1.000` (threshold `<= 1.05`)
