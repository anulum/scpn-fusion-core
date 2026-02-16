# Task 6 Heating + Neutronics Realism

- Generated: `2026-02-15T22:58:58.911871+00:00`
- Runtime: `1.966 s`
- Overall pass: `YES`

## GENRAY-Like Heating Proxies

- Mean RF absorption efficiency: `0.598` (threshold `>= 0.55`)
- Mean NBI absorption efficiency: `0.591` (threshold `>= 0.45`)
- Mean RF reflection rate: `0.341` (threshold `<= 0.55`)
- Mean NBI reflection rate: `0.216` (threshold `<= 0.55`)

## MCNP-Lite Neutronics Optimization (MVR-0.96 Lane)

- Optimized configs meeting Q/TBR gate: `10` (threshold `>= 10`)
- Min Q in optimized set: `10.470` (threshold `>= 10.0`)
- Min TBR in optimized set: `1.067` (threshold `>= 1.05`)
- Mean MC TBR (history transport): `1.136`
- Mean neutron leakage rate: `0.000` (threshold `<= 0.50`)
- Mean Q: `11.762`
- Mean TBR: `1.141`

## ARIES-AT Scaling Parity

- Q parity score: `76.72%` (threshold `>= 75.0%`)
