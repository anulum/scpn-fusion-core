# SCPN Fusion Core — Real-Shot Validation Report

- **Generated**: `2026-02-28T21:09:21.088944+00:00`
- **Runtime**: `5.61s`
- **Overall**: PASS

## 1. Equilibrium Validation
- Files tested: 18
- Psi NRMSE pass: 12/18 (67%)
- q95 pass: 18/18 (100%)
- **Status**: PASS

| File | Machine | q95 | Psi NRMSE | GS Residual |
|------|---------|-----|-----------|-------------|
| lmode_hv.geqdsk | SPARC | 3.15 | 1.2464 | 16.6572 |
| lmode_vh.geqdsk | SPARC | 3.17 | 1.2553 | 16.2726 |
| lmode_vv.geqdsk | SPARC | 3.08 | 1.2498 | 16.5921 |
| sparc_1300.eqdsk | SPARC | -36.67 | 7.8434 | 4.5633 |
| sparc_1305.eqdsk | SPARC | -4.03 | 1.4530 | 5.7733 |
| sparc_1310.eqdsk | SPARC | -3.49 | 1.3102 | 8.6751 |
| sparc_1315.eqdsk | SPARC | -3.52 | 1.3135 | 8.4640 |
| sparc_1349.eqdsk | SPARC | -3.45 | 1.2943 | 7.9500 |
| diiid_hmode_1p5MA.geqdsk | DIII-D | 3.95 | 6.9805 | 6.7866 |
| diiid_hmode_2MA.geqdsk | DIII-D | 4.18 | 6.8136 | 9.8192 |
| diiid_lmode_1MA.geqdsk | DIII-D | 3.73 | 7.3666 | 3.8347 |
| diiid_negdelta.geqdsk | DIII-D | 3.82 | 8.8238 | 4.7497 |
| diiid_snowflake.geqdsk | DIII-D | 3.73 | 6.7517 | 4.0713 |
| jet_dt_3p5MA.geqdsk | JET | 4.86 | 2.2689 | 6.9877 |
| jet_high_ip_4p8MA.geqdsk | JET | 5.45 | 2.3842 | 9.4983 |
| jet_hmode_3MA.geqdsk | JET | 4.63 | 2.2689 | 5.9894 |
| jet_hybrid_2p5MA.geqdsk | JET | 4.41 | 2.2901 | 4.9208 |
| jet_lmode_2MA.geqdsk | JET | 4.18 | 2.3171 | 3.8509 |

## 2. Transport Validation (ITPA)
- Shots: 53
- RMSE: 0.0969 s (37.4% relative)
- Within 2-sigma: 74%
- Uncertainty envelope: |rel error| p95=2.16, z-score p95=3.78, coverage 1sigma=38%
- **Status**: PASS

## 3. Disruption Prediction
- Shots: 16 (6 disruptions, 10 safe)
- Recall: 100%
- FPR: 0%
- **Status**: PASS
- Calibration: `diiid-disruption-risk-calibration-v1` (threshold=0.50, bias_delta=-0.00)
- Replay pipeline: sensor_preprocess=ON, actuator_lag=ON
  (mean |sensor delta|=3.368383, mean |actuator lag|=0.006291)

## 4. Dataset Coverage Gates
| Dataset | Observed | Required Min | Status |
|---------|----------|--------------|--------|
| Equilibrium files | 18 | 12 | PASS |
| Transport shots | 53 | 52 | PASS |
| Disruption shots | 16 | 12 | PASS |
- **Status**: PASS

## Summary

| Lane | Status | Key Metric |
|------|--------|------------|
| Equilibrium | PASS | Psi NRMSE pass 67% |
| Transport | PASS | 2-sigma 74% |
| Disruption | PASS | Recall 100%, FPR 0% |
| Dataset Coverage | PASS | Eq=18, Tr=53, Dis=16 |
