# Task 5 Disruption + Mitigation Integration

- Generated: `2026-02-15T22:58:49.493967+00:00`
- Runtime: `2.454 s`
- Ensemble runs: `24`
- Overall pass: `YES`

## SPI / Impurity + Post-Disruption Physics

- Disruption prevention rate: `1.000` (threshold `>= 0.90`)
- Mean halo current: `1.120 MA` (threshold `<= 2.60 MA`)
- P95 halo peak current: `1.627 MA` (threshold `<= 3.40 MA`)
- P95 runaway beam: `0.000 MA` (threshold `<= 1.00 MA`)
- P95 runaway peak beam: `0.000 MA` (threshold `<= 1.20 MA`)
- Mean impurity decay tau: `7.319 ms`
- Mean wall-damage index: `0.260`

## MPC ELM Disturbance Rejection

- ELM rejection rate: `1.000` (threshold `>= 0.90`)
- TORAX parity: `99.46%`
- P95 loop latency: `0.2842 ms`

## RL Multi-Objective Optimization

- Success rate (`Q>10`, `TBR>1`, no wall damage): `0.958` (threshold `>= 0.75`)
- Q>=10 rate: `0.958` (threshold `>= 0.90`)
- TBR>=1 rate: `1.000` (threshold `>= 0.90`)
- Mean Q proxy: `11.865`
- Mean TBR proxy: `1.099`
