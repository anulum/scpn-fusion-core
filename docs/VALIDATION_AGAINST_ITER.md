# Validation Against ITER: Benchmarking the SCPN-Fusion-Core

**Date**: 2026-01-20
**Benchmark**: ITER baseline (Q=10, P=500MW)
**Status**: CALIBRATION PHASE

---

## 1. Objective

To establish the scientific credibility of the SCPN-Fusion-Core, we benchmark our simulation results against the verified parameters of the **International Thermonuclear Experimental Reactor (ITER)**. The goal is to reproduce the ITER Q=10 operating point (50 MW input $\rightarrow$ 500 MW output) within a $\pm 10\%$ tolerance.

## 2. Validation Methodology

We use the `validate_iter.py` suite, which performs the following steps:
1.  **Configuration Loading**: Importing the standard ITER geometry (R=6.2m, a=2.0m, B0=5.3T).
2.  **MHD Equilibrium Solver**: Solving the Grad-Shafranov equation to determine the plasma flux surfaces and X-point location.
3.  **Thermodynamic Analysis**: Calculating the fusion power density, alpha heating, and total $Q$-factor using standard D-T reaction cross-sections.

---

## 3. Current Benchmark Results (Jan 2026)

| Parameter | ITER Nominal | SCPN-Fusion-Core | Status |
| :--- | :--- | :--- | :--- |
| **Major Radius ($R$)** | 6.2 m | 4.62 m | **DISCREPANCY** |
| **Minor Radius ($a$)** | 2.0 m | 1.85 m | **PASS** |
| **Fusion Power ($P_{fus}$)**| 500 MW | 833.0 MW | **OVERESTIMATE** |
| **Q-Factor** | 10.0 | 16.66 | **OVERESTIMATE** |

### Discrepancy Analysis
*   **Geometry ($R$)**: The current vacuum field solver in the `FusionKernel` tends to shift the magnetic axis inward. This is likely due to the simplified toroidal current representation.
*   **Power ($P_{fus}$)**: The overestimation of fusion power (833MW vs 500MW) suggests that our **Integrated Transport Solver** is currently too optimistic about the L-to-H mode transition or the suppression of turbulence by the FNO.

---

## 4. Planned Calibration Steps

To bridge the gap between our model and the ITER baseline, we are implementing:
1.  **H-mode Pedestal Model**: Incorporating a more rigorous transport barrier model to accurately reflect energy confinement times ($\tau_E$).
2.  **Impurity Radiation**: Adding tungsten ($W$) and beryllium ($Be$) impurity models, which will provide a more realistic radiative cooling effect.
3.  **Refined Toroidal Field**: Improving the discretization of the toroidal field coils in `geometry_3d.py` to fix the radial shift.

---

## 5. Conclusion

While the SCPN-Fusion-Core captures the correct physics scaling, the current numerical discrepancy indicates that the system is tuned for **compact, high-performance designs** (like the MVR-0.96) rather than large-scale, lower-power-density machines like ITER. Validation remains an ongoing priority to ensure the framework's predictive reliability.
