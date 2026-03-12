# Task 14 Free-Boundary Fail-Safe Dropout Replay

- Generated: `2026-03-11T21:53:34.527395+00:00`
- Overall pass: `YES`

## Fault Envelope

- Replay deterministic: `YES`
- Diagnostic dropout count: `16` (threshold `>= 16`)
- Actuator dropout count: `18` (threshold `>= 18`)
- Degraded mode count: `24` (threshold `>= 24`)
- Fallback mode count: `21` (threshold `>= 12`)

## Fail-Safe Degradation

- P95 axis error: `0.0548 m` (threshold `<= 0.085`)
- P95 X-point error: `0.0495 m` (threshold `<= 0.090`)
- Stabilization rate: `0.984` (threshold `>= 0.95`)
- Failsafe trip count: `0` (threshold `<= 0`)
- Recovery transitions: `3` (threshold `>= 2`)

## Recovery Window

- Final alert level: `0` (threshold `<= 1`)
- Late P95 axis error: `0.0233 m` (threshold `<= 0.055`)
- Late P95 X-point error: `0.0048 m` (threshold `<= 0.060`)
- Late max alert level: `0` (threshold `<= 1`)
- Late degraded mode count: `0` (threshold `<= 0`)
