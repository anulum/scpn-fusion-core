# Task 5 Disruption + Mitigation Integration

- Generated: `2026-02-19T17:38:21.914590+00:00`
- Runtime: `11.690 s`
- Ensemble runs: `50`
- Overall pass: `YES`

## SPI / Impurity + Post-Disruption Physics

- Disruption prevention rate: `1.000` (threshold `>= 0.90`)
- Mean halo current: `1.184 MA` (threshold `<= 2.60 MA`)
- P95 halo peak current: `1.791 MA` (threshold `<= 3.40 MA`)
- P95 runaway beam: `0.000 MA` (threshold `<= 1.00 MA`)
- P95 runaway peak beam: `0.000 MA` (threshold `<= 1.20 MA`)
- Mean impurity decay tau: `7.212 ms`
- Mean wall-damage index: `0.276`
- P95 risk upper bound: `0.327`
- P95 wall-damage upper bound: `0.727`
- Mean uncertainty envelope: `0.443`

## MPC ELM Disturbance Rejection

- ELM rejection rate: `1.000` (threshold `>= 0.90`)
- TORAX parity: `99.87%`
- P95 loop latency: `0.2835 ms`

## RL Multi-Objective Optimization

- Success rate (`Q>10`, `TBR>1`, no wall damage): `0.980` (threshold `>= 0.75`)
- Q>=10 rate: `0.980` (threshold `>= 0.90`)
- TBR>=1 rate: `1.000` (threshold `>= 0.90`)
- Robust prevention rate (p95 bounds): `1.000`
- Mean Q proxy: `11.804`
- Mean TBR proxy: `1.087`
