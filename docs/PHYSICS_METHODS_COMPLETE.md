# Physics Methods: The SCPN-Fusion-Core Mathematical Foundation

**Version**: 1.0.0
**Scope**: Advanced Magnetohydrodynamics (MHD) and Plasma Physics

---

## 1. Equilibrium: The Grad-Shafranov Solver

The heart of the simulation is the **Grad-Shafranov Equation** [1][2], which describes the magnetostatic equilibrium of an axisymmetric plasma.

$$ \Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi)F'(\psi) $$

Where:
*   $\psi$ is the magnetic flux.
*   $p(\psi)$ is the plasma pressure.
*   $F(\psi) = R B_\phi$ is the poloidal current function.

**Implementation**: We use a **Non-Linear Finite Difference Solver** with an iterative Picard scheme enhanced by multigrid preconditioning [9] and GMRES [10] to achieve force balance. For real-time applications, we employ a **Neural Equilibrium** model (`neural_equilibrium.py`) that predicts the flux surface geometry in milliseconds.

---

## 2. Stability: Hall-MHD and Sawtooth Oscillations

Beyond ideal MHD, we incorporate **Hall-MHD** effects [6], which are critical for understanding reconnection and high-frequency stability in compact tokamaks.

*   **Sawtooth Oscillations**: Modeled using a modified Kadomtsev reconnection model [7] (`mhd_sawtooth.py`).
*   **Vertical Stability**: Active feedback control of the $n=0$ mode using proportional-integral-derivative (PID) and Model Predictive Control (MPC) algorithms.

---

## 3. Heating and Transport: RF and WDM

### RF Heating (`rf_heating.py`)
Simulates Ion Cyclotron Resonance Heating (ICRH) and Lower Hybrid Current Drive (LHCD). We use a ray-tracing algorithm to calculate the power deposition profile $P(r)$.

### Integrated Transport Solver (`integrated_transport_solver.py`)
Couples the heating profiles with a 1D transport model (WDM - Whole Device Modeling). Confinement scaling follows the IPB98(y,2) law [3] with Bayesian uncertainty quantification via the Verdoolaege regression framework [4].
$$ \frac{\partial n}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \rho} [V' (D \frac{\partial n}{\partial \rho})] + S $$
Where $D$ is the anomalous diffusion coefficient derived from the **Turbulence Oracle** (a Fourier Neural Operator [5] trained on gyrokinetic data).

---

## 4. Turbulence Suppression: FNO-Based Control

The most innovative feature of the Fusion-Core is the **FNO Turbulence Suppressor** (`fno_turbulence_suppressor.py`).
*   **Concept**: Uses a **Fourier Neural Operator (FNO)** [5] to simulate the Kolmogorov scale turbulence in real-time.
*   **Action**: Predicts the onset of Edge Localized Modes (ELMs) and adjusts the magnetic shear to suppress them before they trigger a disruption. When suppression fails, Shattered Pellet Injection (SPI) [8] is deployed as a mitigation strategy.

---

## 5. Summary of Core Modules

| Module | Description |
| :--- | :--- |
| `force_balance.py` | Iterative Grad-Shafranov solver. |
| `hall_mhd_discovery.py` | Non-ideal MHD effect discovery. |
| `stability_analyzer.py` | Nyquist and Lyapunov stability checks. |
| `fusion_ignition_sim.py` | Lawson criterion and ignition margin calculation using IPB98(y,2) scaling [3]. |
| `divertor_thermal_sim.py` | Heat flux mapping on plasma-facing components following the Eich model [11]. |

---

## References

[1] H. Grad and H. Rubin, "Hydromagnetic Equilibria and Force-Free Fields," in *Proceedings of the 2nd United Nations International Conference on the Peaceful Uses of Atomic Energy*, vol. 31, pp. 190-197, Geneva, 1958.

[2] V. D. Shafranov, "Plasma Equilibrium in a Magnetic Field," in *Reviews of Plasma Physics*, vol. 2, pp. 103-151, Consultants Bureau, New York, 1966.

[3] ITER Physics Basis Editors et al., "Chapter 2: Plasma confinement and transport," *Nuclear Fusion*, vol. 39, no. 12, pp. 2175-2249, 1999. DOI: [10.1088/0029-5515/39/12/302](https://doi.org/10.1088/0029-5515/39/12/302)

[4] G. Verdoolaege, G. Kaye, F. Scheibl, G. Tartarini *et al.*, "The updated ITPA global H-mode confinement database: description and analysis," *Nuclear Fusion*, vol. 61, no. 7, p. 076006, 2021. DOI: [10.1088/1741-4326/abdb91](https://doi.org/10.1088/1741-4326/abdb91)

[5] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhatt, A. Stuart, and A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021. arXiv: [2010.08895](https://arxiv.org/abs/2010.08895)

[6] J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory, Washington, DC, 2019. Report No. NRL/PU/6790--19-640.

[7] B. B. Kadomtsev, "Disruptive instability in tokamaks," *Soviet Journal of Plasma Physics*, vol. 1, no. 5, pp. 389-391, 1975.

[8] N. Commaux, L. R. Baylor, T. C. Jernigan, E. M. Hollmann, P. B. Parks *et al.*, "Demonstration of rapid shutdown using large shattered deuterium pellet injection in DIII-D," *Nuclear Fusion*, vol. 56, no. 4, p. 046007, 2016. DOI: [10.1088/0029-5515/56/4/046007](https://doi.org/10.1088/0029-5515/56/4/046007)

[9] W. L. Briggs, V. E. Henson, and S. F. McCormick, *A Multigrid Tutorial*, 2nd ed., Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2000. DOI: [10.1137/1.9780898719505](https://doi.org/10.1137/1.9780898719505)

[10] Y. Saad and M. H. Schultz, "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems," *SIAM Journal on Scientific and Statistical Computing*, vol. 7, no. 3, pp. 856-869, 1986. DOI: [10.1137/0907058](https://doi.org/10.1137/0907058)

[11] T. Eich, A. W. Leonard, R. A. Pitts, W. Fundamenski, R. J. Goldston *et al.*, "Scaling of the tokamak near the scrape-off layer H-mode power width and implications for ITER," *Nuclear Fusion*, vol. 53, no. 9, p. 093031, 2013. DOI: [10.1088/0029-5515/53/9/093031](https://doi.org/10.1088/0029-5515/53/9/093031)
