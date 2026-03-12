# Validation Against ITER and Reference Machines

**Date**: 2026-02-12  
**Version**: v1.0.2 preparation (WP-E1)  
**Scope**: Regression-grade confinement validation using published reference scenarios

---

## 1. Purpose

This document defines a reproducible validation baseline for confinement performance
in SCPN-Fusion-Core using three reference machines:

1. ITER 15 MA baseline (Q = 10 target)
2. SPARC V2C compact high-field scenario
3. DIII-D representative L-mode sanity case

The objective is not to force exact equality with all published values in a single
0D scaling expression. The objective is to detect regressions and preserve physically
reasonable scaling behavior over time.

---

## 2. Validation Data Pack

Reference scenario files are stored in:

- `validation/reference_data/iter_reference.json`
- `validation/reference_data/sparc_reference.json`

Each file contains geometry, operating point, and confinement targets used by
`tests/test_validation_regression.py`.

### 2.1 ITER Baseline Dataset

Primary scenario:

- `I_p = 15.0 MA`
- `B_t = 5.3 T`
- `R = 6.2 m`
- `a = 2.0 m`
- `kappa = 1.7`
- `n_e = 10.1e19 m^-3`
- `P_aux = 50 MW`
- `P_loss = 85 MW` (used for IPB98(y,2) cross-check)
- `tau_E = 3.7 s`
- `P_fusion = 500 MW`
- `Q = 10`

### 2.2 SPARC V2C Dataset

Compact high-field scenario:

- `I_p = 8.7 MA`
- `B_t = 12.2 T`
- `R = 1.85 m`
- `a = 0.57 m`
- `kappa = 1.97`
- `n_e = 37.0e19 m^-3`
- `P_aux = 25 MW`
- `P_loss = 35 MW` (used for IPB98(y,2) cross-check)
- `tau_E = 0.77 s`
- `P_fusion = 140 MW`
- `Q = 11`

---

## 3. Governing Confinement Scaling

The regression suite uses IPB98(y,2):

```
tau_E = 0.0562 * I_p^0.93 * B_t^0.15 * n_e19^0.41 * P_loss^-0.69
        * R^1.97 * kappa^0.78 * epsilon^0.58 * M^0.19
```

Where:

- `I_p` in MA
- `B_t` in T
- `n_e19` in 1e19 m^-3
- `P_loss` in MW
- `R` in m
- `kappa` elongation
- `epsilon = a / R`
- `M` effective ion mass in amu

In tests, `M = 2.5` is used for a D-T mixture.

---

## 4. Case Study A: ITER 15 MA Baseline

### 4.1 Inputs

| Quantity | Value |
| :-- | --: |
| Plasma current `I_p` | 15.0 MA |
| Toroidal field `B_t` | 5.3 T |
| Major radius `R` | 6.2 m |
| Minor radius `a` | 2.0 m |
| Elongation `kappa` | 1.7 |
| Density `n_e` | 10.1e19 m^-3 |
| Loss power `P_loss` | 85 MW |
| Target confinement `tau_E` | 3.7 s |

### 4.2 Regression expectation

The IPB98(y,2)-estimated confinement from the reference input should be in:

- `[2.9 s, 4.4 s]` absolute range
- `<= 20%` relative error from the stored reference `tau_E`

This is implemented in `test_iter_tau_e_within_20pct`.

### 4.3 Why this tolerance?

The formula is empirical and global. Differences in scenario assumptions,
`P_loss` partitioning, and composition model can shift `tau_E` materially.
A 20% guardrail is tight enough to catch coding regressions while avoiding
false failures from minor model-choice differences.

---

## 5. Case Study B: SPARC V2C High-Field Compact Regime

### 5.1 Inputs

| Quantity | Value |
| :-- | --: |
| Plasma current `I_p` | 8.7 MA |
| Toroidal field `B_t` | 12.2 T |
| Major radius `R` | 1.85 m |
| Minor radius `a` | 0.57 m |
| Elongation `kappa` | 1.97 |
| Density `n_e` | 37.0e19 m^-3 |
| Loss power `P_loss` | 35 MW |
| Target confinement `tau_E` | 0.77 s |

### 5.2 Compact-machine comparison metric

To compare machines of very different physical size, the regression test uses
confinement density:

```
confinement_density = tau_E / V
V = 2 * pi^2 * R * a^2 * kappa
```

The expected behavior is:

- SPARC has higher `tau_E / V` than ITER (high-field compact advantage)

This is enforced in `test_sparc_high_field_advantage`.

---

## 6. Case Study C: DIII-D L-Mode Sanity Check

The regression includes a representative DIII-D L-mode point:

- `I_p = 1.5 MA`
- `B_t = 2.1 T`
- `R = 1.67 m`
- `a = 0.67 m`
- `kappa = 1.8`
- `n_e = 5.0e19 m^-3`
- `P_loss = 5 MW`

Expected confinement range:

- `0.12 s <= tau_E <= 0.25 s`

This check is implemented in `test_diiid_lmode_baseline`.

---

## 7. Regression Test Inventory

`tests/test_validation_regression.py` provides:

1. `test_iter_tau_e_within_20pct`  
   Confirms ITER confinement remains in expected range.
2. `test_sparc_high_field_advantage`  
   Confirms high-field compact advantage in confinement density.
3. `test_ipb98_scaling_sanity`  
   Confirms deterministic reference output for a fixed ITER-like point.
4. `test_diiid_lmode_baseline`  
   Confirms physically plausible L-mode confinement scale.

---

## 8. How To Run

From repo root:

```bash
python -m pytest tests/test_validation_regression.py -v
```

Optional: run full Python suite

```bash
python -m pytest tests/ -v
```

---

## 9. Pass/Fail Criteria

Validation passes when:

- all four regression tests pass
- the reference JSON files are present and parse cleanly

Validation fails when:

- confinement leaves expected tolerance band
- compact high-field advantage regression disappears
- deterministic IPB98 reference point changes unexpectedly

---

## 10. Known Limits

1. IPB98(y,2) is an empirical global law, not a transport PDE.
2. `P_loss` definitions vary by source and scenario assumptions.
3. Auxiliary physics (impurity radiation, pedestal, alpha partition) can shift
   operating point interpretation even when scaling math is unchanged.

These tests are intentionally narrow: they check consistency and regression
detection, not full-fidelity discharge reconstruction.

---

## 11. Next Validation Extensions

Planned follow-up after WP-E1:

1. Add machine-specific uncertainty bands in reference JSON.
2. Include JET and DIII-D published points as explicit files.
3. Add automated trend plots in CI artifacts for `tau_E`, `H98`, and `Q`.

This keeps validation transparent, reproducible, and actionable for future solver
and transport updates.
