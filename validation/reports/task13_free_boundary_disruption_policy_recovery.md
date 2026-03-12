# Task 13 Free-Boundary Disruption Policy Recovery

- Generated: `2026-03-11T21:53:34.355125+00:00`
- Overall pass: `YES`

## Alert Regimes

- q95 floor: `4.08`
- beta_N ceiling: `2.18`
- disruption-risk ceiling: `0.23`
- warning mode count: `42` (threshold `>= 6`)
- guarded mode count: `5` (threshold `>= 5`)
- fallback mode count: `41` (threshold `>= 8`)
- peak alert level: `3` (required `3`)
- final alert level: `1` (threshold `<= 1`)
- alert transitions: `5` (threshold `>= 4`)
- recovery transitions: `2` (threshold `>= 2`)

## Recovery Window

- Late mean disruption risk: `0.2066` (threshold `<= 0.21`)
- Late P95 axis error: `0.0185 m` (threshold `<= 0.055`)
- Late P95 X-point error: `0.0058 m` (threshold `<= 0.055`)
- Late max alert level: `1` (threshold `<= 1`)

## Closed-Loop Outcome

- P95 axis error: `0.0548 m` (threshold `<= 0.064`)
- P95 X-point error: `0.0558 m` (threshold `<= 0.058`)
- Stabilization rate: `1.000` (threshold `>= 0.97`)
- Max disruption risk: `0.2360` (threshold `>= 0.23`)
- Invariant violation count: `0` (threshold `<= 0`)
- Failsafe trip count: `0` (threshold `<= 0`)
