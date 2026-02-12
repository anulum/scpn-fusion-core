# Physics Methods: The SCPN-Fusion-Core Mathematical Foundation

**Version**: 1.0.0
**Scope**: Advanced Magnetohydrodynamics (MHD) and Plasma Physics

---

## 1. Equilibrium: The Grad-Shafranov Solver

The heart of the simulation is the **Grad-Shafranov Equation**, which describes the magnetostatic equilibrium of an axisymmetric plasma.

$$ \Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi)F'(\psi) $$

Where:
*   $\psi$ is the magnetic flux.
*   $p(\psi)$ is the plasma pressure.
*   $F(\psi) = R B_\phi$ is the poloidal current function.

**Implementation**: We use a **Non-Linear Finite Difference Solver** with an iterative Picard scheme to achieve force balance. For real-time applications, we employ a **Neural Equilibrium** model (`neural_equilibrium.py`) that predicts the flux surface geometry in milliseconds.

---

## 2. Stability: Hall-MHD and Sawtooth Oscillations

Beyond ideal MHD, we incorporate **Hall-MHD** effects, which are critical for understanding reconnection and high-frequency stability in compact tokamaks.

*   **Sawtooth Oscillations**: Modeled using a modified Kadomtsev reconnection model (`mhd_sawtooth.py`).
*   **Vertical Stability**: Active feedback control of the $n=0$ mode using proportional-integral-derivative (PID) and Model Predictive Control (MPC) algorithms.

---

## 3. Heating and Transport: RF and WDM

### RF Heating (`rf_heating.py`)
Simulates Ion Cyclotron Resonance Heating (ICRH) and Lower Hybrid Current Drive (LHCD). We use a ray-tracing algorithm to calculate the power deposition profile $P(r)$.

### Integrated Transport Solver (`integrated_transport_solver.py`)
Couples the heating profiles with a 1D transport model (WDM - Whole Device Modeling).
$$ \frac{\partial n}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \rho} [V' (D \frac{\partial n}{\partial \rho})] + S $$
Where $D$ is the anomalous diffusion coefficient derived from the **Turbulence Oracle** (a Fourier Neural Operator trained on gyrokinetic data).

---

## 4. Turbulence Suppression: FNO-Based Control

The most innovative feature of the Fusion-Core is the **FNO Turbulence Suppressor** (`fno_turbulence_suppressor.py`).
*   **Concept**: Uses a **Fourier Neural Operator (FNO)** to simulate the Kolmogorov scale turbulence in real-time.
*   **Action**: Predicts the onset of Edge Localized Modes (ELMs) and adjusts the magnetic shear to suppress them before they trigger a disruption.

---

## 5. Summary of Core Modules

| Module | Description |
| :--- | :--- |
| `force_balance.py` | Iterative Grad-Shafranov solver. |
| `hall_mhd_discovery.py` | Non-ideal MHD effect discovery. |
| `stability_analyzer.py` | Nyquist and Lyapunov stability checks. |
| `fusion_ignition_sim.py` | Lawson criterion and ignition margin calculation. |
| `divertor_thermal_sim.py` | Heat flux mapping on plasma-facing components. |
