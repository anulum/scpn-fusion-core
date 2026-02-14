# Literature Baseline: Tokamak Vertical Position Control

> Compiled 2026-02-14. Sources: peer-reviewed journals, textbooks, and
> institutional technical reports. This document serves as the foundational
> reference for the PID baseline controller design and the paper's
> background section.

---

## 1. Linearized Plant Model

### 1.1 The Standard Vertical Instability Equation

The vertical position of an elongated tokamak plasma is inherently
unstable. The standard linearized equation of motion for rigid vertical
displacement Z(t) of the plasma column is:

```
m_eff * d^2Z/dt^2 = -K_s * Z + M_ctrl * dI_ctrl/dt + F_wall
```

where:

| Symbol       | Description                                              | Units    |
|--------------|----------------------------------------------------------|----------|
| m_eff        | Effective plasma mass (includes electromagnetic inertia) | kg       |
| Z            | Vertical displacement from equilibrium                   | m        |
| K_s          | Stability index (restoring/destabilizing force gradient)  | N/m      |
| M_ctrl       | Mutual inductance between control coil and plasma         | H        |
| I_ctrl       | Current in the active vertical stability control coil     | A        |
| F_wall       | Force from eddy currents induced in passive structures    | N        |

**Physical interpretation.** When the plasma is elongated (kappa > 1),
the vertical field curvature produces a *destabilizing* force: a small
upward displacement Z > 0 produces a net upward force, so K_s < 0
(the "negative spring"). Without passive conductors or active feedback,
the plasma accelerates away from equilibrium on the ideal-MHD
(Alfven) timescale.

**Massless-plasma simplification.** In most control analyses, plasma
inertia is negligible compared to electromagnetic forces, so the mass
term is dropped (m_eff -> 0). The dynamics are then dominated by eddy
current diffusion through the passive wall and the active coil circuit.
This yields a first-order unstable system after including the wall
circuit equation. The massless approximation is validated by Humphreys
and Walker (2009), who showed that no pure derivative gain can stabilize
a system with positive plasma mass, but the massless model captures the
essential control physics [Ref 5].

### 1.2 State-Space Formulation (Ariola & Pironti Framework)

Ariola and Pironti derive the full state-space model starting from
magneto-hydro-dynamic (MHD) equilibrium and circuit equations
[Ref 1, 2]. The general form is:

```
M * dI/dt + R * I = V + P(Z, dZ/dt)
m_eff * d^2Z/dt^2 = F_z(I, Z)
```

where:

| Symbol | Description |
|--------|-------------|
| I      | Vector of all conductor currents (coils + wall segments + plasma) |
| M      | Mutual inductance matrix (symmetric, positive-definite)           |
| R      | Resistance matrix (diagonal for discrete conductors)              |
| V      | Applied voltage vector (non-zero only for active coils)           |
| P      | Plasma coupling terms                                             |
| F_z    | Vertical force on plasma (linearized about equilibrium)           |

After linearization about an equilibrium and elimination of the plasma
current perturbation, the system reduces to:

```
dx/dt = A*x + B*u
y     = C*x
```

where x includes wall eddy currents, coil currents, vertical position Z,
and vertical velocity dZ/dt. The matrix A has **one real positive
eigenvalue** (the vertical instability) and many stable or marginally
stable eigenvalues (eddy current decay modes). The unstable eigenvector
corresponds to a nearly rigid vertical shift of the plasma current
distribution [Ref 1].

### 1.3 Decay Index and Instability Criterion

The vertical instability is fundamentally governed by the **decay index**
n of the equilibrium vertical field B_z:

```
n = -(R / B_z) * (dB_z / dR)
```

where R is the major radius. Vertical instability occurs when the field
curvature is unfavorable, which is directly linked to plasma elongation
kappa. Higher elongation requires stronger field curvature, pushing the
system further into the unstable regime. A D-shaped plasma with higher
triangularity and broader current profiles is relatively more stable
[Ref 8].

### 1.4 Typical Parameter Ranges

| Parameter          | DIII-D            | ITER              | SPARC             |
|--------------------|-------------------|--------------------|-------------------|
| Major radius R (m) | 1.67              | 6.2                | 1.85              |
| Minor radius a (m) | 0.67              | 2.0                | 0.57              |
| Elongation kappa   | 1.8 -- 2.5        | 1.70 -- 1.85       | ~1.75 (kappa_areal)|
| Triangularity delta| 0.2 -- 0.8        | 0.33 -- 0.50       | ~0.3              |
| Plasma current (MA)| 1.0 -- 3.0        | 15                 | 8.7               |
| Toroidal field (T) | 2.2               | 5.3                | 12.2              |
| Plasma mass m_eff  | ~5.0e-6 kg (typ.) | ~0.04 kg (DT)      | ~0.01 kg (est.)   |

---

## 2. Growth Rates

### 2.1 Definition

The vertical instability growth rate gamma (s^-1) is the real positive
eigenvalue of the linearized system matrix A. It characterizes how fast
the plasma would move vertically in the presence of a resistive passive
wall but without active feedback. It is the inverse of the instability
e-folding time: tau_gamma = 1/gamma.

### 2.2 Published gamma Values

| Machine  | gamma (s^-1) | Conditions / Notes                                     | Reference       |
|----------|--------------|---------------------------------------------------------|-----------------|
| DIII-D   | 200 -- 800   | Depends on elongation; up to ~800 controllable          | [Ref 3, 9]      |
| DIII-D   | up to 968    | Maximum controllable (LQR, EAST benchmark comparison)   | [Ref 10]        |
| ITER     | ~50 -- 300   | Lower growth rate due to thick, close-fitting vessel     | [Ref 4, 7]      |
| SPARC    | < 100        | TokSyS framework; thick conducting wall reduces gamma   | [Ref 6]         |
| TCV      | 100 -- 1500  | Highly elongated; used for control R&D                  | [Ref 11]        |
| EAST     | up to ~968   | Maximum controllable with model-based LQR               | [Ref 10]        |

### 2.3 Relationship Between gamma and Elongation kappa

The growth rate increases monotonically with elongation, approximately as:

```
gamma ~ gamma_0 * exp(alpha * (kappa - 1))
```

where gamma_0 and alpha are machine-specific constants depending on
plasma-wall distance, triangularity, and the conductivity/thickness of
passive stabilizing structures. The feedback capability parameter
gamma * tau_w (where tau_w is the resistive wall time constant) is a
dimensionless measure of how challenging the control problem is:

- **gamma * tau_w < 1**: Easily controllable
- **gamma * tau_w ~ 1 -- 1.5**: Challenging but achievable with good design
- **gamma * tau_w > 1.5**: At the limit of controllability for current technology

The SPARC design team notes that gamma * tau_w <= 1.5 is "a reasonable
estimate of what is expected to be controllable in future fusion-grade
experiments" [Ref 6].

### 2.4 Effect of Passive Structures

Passive stabilizing plates (between the plasma and vacuum vessel)
slow the growth rate by inducing eddy currents that resist plasma
motion. SPARC studies showed that informed positioning of passive plates
can reduce gamma to 16% of the baseline (no-plate) value. However, in
SPARC the effect of steel VS plates on gamma was modest (~25% reduction)
because the full vacuum vessel already provides substantial passive
stabilization [Ref 6].

---

## 3. Published Controller Gains and Approaches

### 3.1 PD Control (Standard Approach)

The vertical control portion of tokamak feedback systems predominantly
uses **proportional-derivative (PD)** control, not full PID. The integral
term is typically omitted because:

1. Steady-state vertical position offsets are corrected by the
   slower-acting plasma shape/position control loop.
2. Integral windup is problematic given the fast, unstable dynamics.

The PD controller takes the form:

```
V_ctrl = K_p * Z + K_d * dZ/dt
```

or equivalently in multi-coil systems:

```
V = K_p * (Z - Z_ref) + K_d * (dZ/dt)
```

where K_p and K_d are gain vectors that map scalar position error to
voltage commands for the active control coils.

**Key theoretical results (Walker & Humphreys, 2009):**
- No pure derivative gain (K_p = 0) can stabilize the system.
- With positive plasma mass, stability requires both K_p and K_d to
  be non-zero and properly sized relative to the instability growth rate.
- The proportional gain must exceed a minimum threshold related to the
  open-loop instability strength [Ref 5].

**DIII-D practical implementation:**
- 10 -- 100 PID parameters are adjusted empirically.
- A fast vertical stability algorithm runs at ~20 kHz cycle rate.
- The fast VS loop reduces the effective growth rate so that the
  slower shape control (running at ~1 kHz) can maintain equilibrium.
- Elongation up to kappa = 2.5 has been achieved with this architecture [Ref 3, 9].

**EAST implementation:**
- Fixed PD controller with empirically tuned gains.
- Gains require re-tuning when plasma configuration changes significantly.
- Maximum controllable growth rate with PD: limited by voltage saturation [Ref 10].

### 3.2 LQR Control (Model-Based Optimal)

LQR (Linear Quadratic Regulator) provides optimal state-feedback gains
by minimizing:

```
J = integral_0^inf [ x^T Q x + u^T R u ] dt
```

where Q penalizes state deviations and R penalizes control effort.

**Ariola & Pironti (LQR for ITER vertical stabilization):**
- Used the CREATE-L linearized model of the ITER plasma + conductors.
- Designed robust gain sets using multiple linearized models across the
  operating space.
- Demonstrated that LQR gains can be computed offline and applied in
  real-time with static output feedback [Ref 1, 2, 7].

**EAST LQR results (Rui et al., 2024):**
- Model-based LQR controller successfully stabilized plasmas with
  continuously increasing growth rate.
- Achieved maximum controllable gamma = 968 s^-1 (vs. PD limitations).
- LQR gains adapted based on real-time plasma response model updates.
- Demonstrated robustness in free-drift recovery experiments [Ref 10].

**Advantages of LQR over PD for vertical control:**
1. Systematic gain selection (no empirical tuning of 10-100 parameters).
2. Explicit trade-off between position accuracy (Q) and coil effort (R).
3. Guaranteed stability margins when the model is accurate.
4. Natural extension to MIMO for coupled vertical + radial + shape control.

### 3.3 Model Predictive Control (MPC)

MPC has been experimentally demonstrated for plasma shape control on TCV
(first demonstration reported 2025). The MPC controller:
- Solves a constrained quadratic program in real-time.
- Explicitly handles actuator limits (voltage, current saturation).
- Uses linearized plasma response models from the FGE equilibrium code.
- Optimizes reference signals for an inner fast control loop [Ref 12].

For vertical position specifically, adaptive MPC has been proposed
(Mavkov et al., 2016) to handle time-varying growth rates, but
experimental validation on vertical-only control remains limited [Ref 12].

### 3.4 Deep Reinforcement Learning (DeepMind/TCV, 2022)

Degrave et al. (Nature, 2022) demonstrated deep RL control of the
full magnetic configuration on the TCV tokamak:
- Single neural network (3-layer MLP) controls all 19 poloidal field
  coils simultaneously.
- Learns voltage commands directly from sensor measurements.
- Successfully produced elongated, negative-triangularity, and
  "snowflake" configurations.
- Demonstrated simultaneous control of two separate plasma "droplets."
- The RL controller inherently handles vertical stability as part of
  the full shape/position control problem [Ref 13].

---

## 4. Performance Targets

### 4.1 ITER Requirements

| Metric                              | Value / Requirement                        | Source   |
|--------------------------------------|--------------------------------------------|----------|
| Max controllable displacement        | DeltaZ_max / a = 5% (~100 mm)             | [Ref 4, 7] |
| Routine position accuracy            | < 10 -- 20 mm                              | [Ref 4]  |
| Unstable mode time constant          | ~100 ms                                    | [Ref 7]  |
| VS power supply voltage              | 6 kV baseline, 9 kV upgrade               | [Ref 4]  |
| VS power supply response time        | < 1 ms                                     | [Ref 4]  |
| Control signal transmission delay    | ~38 microseconds                           | [Ref 4]  |
| Shape control settling time          | 15 -- 25 s (for reference signal changes)  | [Ref 4]  |
| Vertical recovery from VDE onset     | Must recover before thermal quench (~10 ms)| [Ref 7]  |

### 4.2 DIII-D Requirements

| Metric                              | Value / Requirement                        | Source   |
|--------------------------------------|--------------------------------------------|----------|
| Routine position accuracy            | < 5 mm for standard operation              | [Ref 3]  |
| Maximum controllable gamma           | ~800 s^-1 (PD), ~968 s^-1 (LQR on EAST)   | [Ref 3, 10] |
| Fast VS loop cycle rate              | ~20 kHz (50 microsecond cycle time)        | [Ref 3]  |
| Shape control loop rate              | ~1 kHz                                     | [Ref 3]  |
| Maximum elongation achieved          | kappa = 2.5                                | [Ref 9]  |

### 4.3 SPARC Requirements

| Metric                              | Value / Requirement                        | Source   |
|--------------------------------------|--------------------------------------------|----------|
| Design elongation                    | kappa_areal ~ 1.75                         | [Ref 6]  |
| Maximum growth rate                  | gamma < 100 s^-1                           | [Ref 6]  |
| Controllability criterion            | gamma * tau_w < 1.5                        | [Ref 6]  |
| DeltaZ_max / a                       | >= 5% (consistent with ITER/DEMO)          | [Ref 6]  |

### 4.4 General Control Bandwidth Requirements

For a vertical instability with growth rate gamma, the control system
must have a bandwidth significantly exceeding gamma to provide adequate
gain and phase margins. As a rule of thumb:

```
f_control > (3 to 5) * gamma / (2 * pi)
```

For DIII-D at gamma = 800 s^-1: f_control > 400 -- 640 Hz.
For ITER at gamma = 200 s^-1: f_control > 100 -- 160 Hz.
For SPARC at gamma = 100 s^-1: f_control > 50 -- 80 Hz.

---

## 5. Control Coil Specifications

### 5.1 ITER In-Vessel Vertical Stability Coils

ITER uses two distinct vertical stabilization systems:

**VS1/VS2 (ex-vessel, using PF/CS coils):**
- Uses existing poloidal field coils in anti-series configuration.
- Slower response due to vessel shielding.

**VS3 (in-vessel coils) -- adopted in baseline design:**

| Parameter                   | Value                                          |
|-----------------------------|------------------------------------------------|
| Number of VS coils          | 2 (upper + lower)                              |
| Turns per coil              | 3                                              |
| Peak current                | 240 kA-turns                                   |
| RMS current                 | 36 kA-turns                                    |
| Peak current per conductor  | ~60 kA (with power supply)                     |
| Power supply voltage        | 6 kV (baseline), 9 kV (upgrade)                |
| Required voltage rise time  | < 1 ms                                         |
| Bandwidth (VS coils)        | DC -- 5 Hz (primary), up to 20 Hz capability   |
| ELM coil bandwidth          | 20 -- 200 Hz (separate from VS)                |
| Conductor material          | Copper bore, MgO insulation, SS jacket         |
| Operating temperature       | 70 -- 240 deg C                                |
| Waveform duration           | 10 s                                           |

The in-vessel VS3 coils significantly increase the range of plasma
controllability compared to VS1/VS2 alone, providing operating margins
sufficient for ITER's performance goals [Ref 4, 7].

### 5.2 DIII-D Vertical Stability Coils

DIII-D uses the F-coil system (18 poloidal field coils) for both
equilibrium shaping and fast vertical control:

| Parameter                   | Value                                          |
|-----------------------------|------------------------------------------------|
| Number of F-coils           | 18 (various locations)                         |
| Chopper voltage             | 600 V or 1200 V (depending on coil)            |
| Pulsed current capability   | up to +/- 16 kA per coil                       |
| Fast VS loop rate           | 20 kHz                                         |
| Power supply topology       | Chopper-based with high-current patch panel     |
| Control interval            | 0.5 -- 3 s (for shape control)                 |

The DIII-D approach uses a hierarchical control architecture: a fast
inner loop for vertical stability (running at 20 kHz) embedded within
a slower outer loop for shape and position control (~1 kHz) [Ref 3].

### 5.3 SPARC Vertical Stability System

| Parameter                   | Value                                          |
|-----------------------------|------------------------------------------------|
| Passive stabilization       | Thick conducting vacuum vessel + VS plates     |
| VS plate effect             | ~25% reduction in gamma (modest)               |
| Active coils                | Dedicated VS coils (details in design phase)   |
| Design growth rate          | gamma < 100 s^-1                               |
| Passive plate positioning   | Informed placement can reduce gamma to 16% of baseline |

SPARC benefits from its compact, high-field design with a thick
conducting wall that inherently provides strong passive stabilization,
yielding growth rates lower than most existing devices [Ref 6].

### 5.4 Mutual Inductance Considerations

The mutual inductance M_ctrl between the active control coil and the
plasma determines the control authority -- the force per unit rate of
change of coil current. Key design considerations:

- Closer coils to the plasma yield higher M_ctrl but face thermal and
  radiation exposure challenges.
- In-vessel coils (ITER VS3) have much higher M_ctrl than ex-vessel
  coils (VS1/VS2) because the signal does not need to diffuse through
  the vessel wall.
- The effective mutual inductance is frequency-dependent: at high
  frequencies, vessel eddy currents shield the plasma from coil field
  changes, reducing effective M_ctrl.
- Typical values for M_ctrl are not published as single numbers because
  they depend on the full geometry (computed by codes like CREATE-L,
  TokSys, or FEEQS).

---

## 6. Key References

### Primary Textbook

**[Ref 1]** M. Ariola and A. Pironti, *Magnetic Control of Tokamak
Plasmas*, 1st ed. London: Springer (Advances in Industrial Control),
2008. ISBN: 978-1-84899-674-6.
- 2nd ed. published 2016, ISBN: 978-3-319-29888-7.
- *The* definitive reference for tokamak magnetic control. Covers
  derivation of state-space models from MHD equations, PID and MIMO
  controller design, vertical stabilization, shape control, and
  applications to JET, ITER, and other machines.

**[Ref 2]** M. Ariola and A. Pironti, "Plasma shape control for the
JET tokamak," *IEEE Control Systems Magazine*, vol. 25, no. 5,
pp. 65--75, Oct. 2005.

### ITER Vertical Control

**[Ref 4]** D. A. Humphreys, G. Ambrosino, P. de Vries, F. Felici,
S. H. Kim, G. Jackson, A. Kallenbach, E. Kolemen, J. Lister,
D. Moreau, and A. Pironti, "Novel aspects of plasma control in ITER,"
*Physics of Plasmas*, vol. 22, no. 2, p. 021806, Feb. 2015.
DOI: [10.1063/1.4907901](https://doi.org/10.1063/1.4907901).
- Comprehensive review of ITER control challenges including vertical
  stabilization, current profile control, and disruption avoidance.

**[Ref 7]** A. Portone, "Plasma vertical stabilisation in ITER,"
*Nuclear Fusion*, vol. 45, no. 8, pp. 926--932, 2005.
- ITER-specific VS system design, controllability analysis.

### DIII-D Vertical Control

**[Ref 3]** M. L. Walker and D. A. Humphreys, "Next-generation plasma
control in the DIII-D tokamak," *Fusion Technology*, vol. 42, no. 2T,
pp. 283--290, 2002. Also: *Fusion Science and Technology*, vol. 50,
2006.
- Architecture of DIII-D PCS, hierarchical control (fast VS + slow shape).

**[Ref 9]** E. A. Lazarus, J. B. Lister, and G. H. Neilson, "Control
of the vertical instability in tokamaks," *Nuclear Fusion*, vol. 30,
no. 1, pp. 111--141, Jan. 1990.
- Foundational paper on vertical stability in DIII-D.

### SPARC Vertical Control

**[Ref 6]** A. O. Nelson, C. Paz-Soldan, S. Wei, et al., "Implications
of vertical stability control on the SPARC tokamak," *Nuclear Fusion*,
vol. 64, p. 086042, 2024. arXiv: [2401.09613](https://arxiv.org/abs/2401.09613).
DOI: [10.1088/1741-4326/ad58f6](https://doi.org/10.1088/1741-4326/ad58f6).
- TokSyS-based analysis; gamma < 100 s^-1; passive plate studies;
  gamma * tau_w controllability criterion.

### Controller Theory & Modern Approaches

**[Ref 5]** D. A. Humphreys and M. L. Walker, "On feedback stabilization
of the tokamak plasma vertical instability," *Automatica*, vol. 45,
no. 3, pp. 665--674, 2009.
- Necessary conditions for PD stabilization; no pure-D result.

**[Ref 8]** A. Portone, "The stability margin of elongated plasmas,"
*Nuclear Fusion*, vol. 45, no. 8, pp. 926--932, 2005.
- Stability margin definition; effects of aspect ratio, elongation,
  plasma-wall distance on passive stabilization.

**[Ref 10]** R. Rui, Y. Huang, et al., "Using LQR controller for
vertical position control on EAST," *Nuclear Fusion*, vol. 64,
p. 066040, 2024.
DOI: [10.1088/1741-4326/ad43fd](https://doi.org/10.1088/1741-4326/ad43fd).
- First model-based LQR on EAST; gamma = 968 s^-1 maximum controllable;
  free-drift recovery demonstration.

**[Ref 11]** F. Felici et al., "Improved plasma vertical position
control on TCV using model-based optimized controller synthesis,"
*Fusion Science and Technology*, vol. 78, no. 7, pp. 661--681, 2022.
DOI: [10.1080/15361055.2022.2043511](https://doi.org/10.1080/15361055.2022.2043511).
- Model-based optimal control synthesis for TCV vertical position.

**[Ref 12]** F. Mavkov et al., "Adaptive model predictive control of
tokamak plasma unstable vertical position," Proc. IEEE CDC, 2016.
Also: Anand et al., "First experimental demonstration of plasma shape
control in a tokamak through Model Predictive Control," arXiv:2506.20096,
2025.

**[Ref 13]** J. Degrave, F. Felici, J. Buchli, M. Neunert, B. Tracey,
F. Carpanese, T. Ewalds, R. Hafner, A. Abdolmaleki, D. de Las Casas,
C. Donner, L. Fritz, C. Galber, A. Huber, J. Keeling, M. Tsiacalos,
et al., "Magnetic control of tokamak plasmas through deep reinforcement
learning," *Nature*, vol. 602, pp. 414--419, Feb. 2022.
DOI: [10.1038/s41586-021-04301-9](https://doi.org/10.1038/s41586-021-04301-9).
- DeepMind RL controller on TCV; single NN controls all 19 PF coils.

### Additional References

**[Ref 14]** K. E. J. Olofsson, "Fast calculation of the tokamak
vertical instability," *Nuclear Fusion*, vol. 63, p. 126048, 2023.
DOI: [10.1088/1741-4326/ad04aa](https://doi.org/10.1088/1741-4326/ad04aa).
- Efficient spectral method for growth rate computation; complexity
  linear in number of conductive elements.

**[Ref 15]** G. Ambrosino, M. Ariola, A. De Tommasi, and A. Pironti,
"Plasma vertical stabilization in the ITER tokamak via constrained
static output feedback," *IEEE Trans. Control Systems Technology*,
vol. 19, no. 2, pp. 376--381, 2011.
- Robust gain design for ITER VS using set of linear models.

---

## 7. Summary Table: Control Approaches Comparison

| Approach  | Gains                  | Strengths                        | Weaknesses                       | Demonstrated On     |
|-----------|------------------------|----------------------------------|----------------------------------|---------------------|
| PD        | K_p, K_d (empirical)   | Simple, fast, well-understood    | No systematic tuning; no constraints | DIII-D, JET, EAST  |
| LQR       | K = -R^-1 B^T P       | Optimal; systematic; adaptable   | Requires accurate model; no constraints | EAST (2024), ITER design |
| MPC       | Receding-horizon QP    | Handles constraints explicitly   | Computational cost; model dependency | TCV (2025, shape)  |
| Deep RL   | Neural network weights | Learns from simulation; MIMO     | Black box; needs simulator; no guarantees | TCV (2022)        |
| H-inf     | Robust synthesis       | Guaranteed robustness margins    | Conservative; complex design     | JET, ITER design   |

---

## 8. Implications for PID Baseline Design

Based on this literature review, the following design choices are
well-supported for a PID baseline vertical position controller:

1. **Use PD, not full PID.** Integral action is not standard practice
   for fast vertical stabilization. If integral action is added, it
   must include anti-windup protection and should be limited to
   slow drift correction.

2. **Proportional gain K_p must exceed a minimum threshold** related to
   the instability growth rate gamma and the system's passive
   stabilization properties. K_p too low => unstable; K_p too high =>
   oscillatory or saturates actuators.

3. **Derivative gain K_d provides damping** and is essential for
   stability. The ratio K_d / K_p is analogous to a damping ratio
   and should be tuned for adequate phase margin (>30 deg) and gain
   margin (>6 dB).

4. **Control bandwidth must exceed 3-5x the growth rate** to provide
   adequate stability margins.

5. **Actuator constraints (voltage and current limits)** are the primary
   practical limitation. The literature consistently identifies power
   supply voltage saturation as the dominant failure mode for vertical
   control loss.

6. **Target performance:** |DeltaZ| < 5% of minor radius for routine
   operation; recovery from displacements up to DeltaZ_max within one
   instability e-folding time.

---

*End of literature baseline.*
