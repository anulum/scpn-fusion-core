# Task 9 Free-Boundary Replay And HIL Gate

- Generated: `2026-03-11T21:53:05.732033+00:00`
- Overall pass: `YES`

## Replay Determinism

- Replay deterministic: `True`
- Replay signature: `04f3fc95e88f6289`
- Nominal P95 axis error: `0.0336 m`
- Nominal P95 X-point error: `0.0336 m`

## Watchdog And Fail-Safe

- Watchdog trip observed: `True`
- Faulted supervisor interventions: `52`
- Faulted failsafe trip count: `5` (threshold `>= 1`)
- Faulted max risk score: `1.510` (threshold `>= 1.35`)
- Faulted final target Ip: `7.000 MA`
- Faulted max |action|: `0.350` (threshold `<= 0.35`)
- Faulted max |coil current|: `1.380` (threshold `<= 1.50`)

## HIL Compatibility

- HIL P95 latency: `49.83 us` (threshold `<= 1000 us`)
- State-estimation P95: `32.50 us`
- Controller-step P95: `3.00 us`
- Actuator-command P95: `11.88 us`
