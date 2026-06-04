<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Physics Methods Reference -->

# SCPN Fusion Core — Physics Methods Reference

**Version**: 4.0.0 &nbsp;|&nbsp; **Scope**: All implemented physics closures with equations, numerical methods, references, and validation scripts.

---

## 1. Grad-Shafranov Equilibrium

The axisymmetric magnetostatic equilibrium satisfies the Grad-Shafranov equation:

$$\Delta^* \psi \;=\; -\mu_0\, R^2\, p'(\psi) \;-\; F(\psi)\,F'(\psi)$$

where the Grad-Shafranov operator is $\Delta^* \equiv R \frac{\partial}{\partial R}\!\left(\frac{1}{R}\frac{\partial}{\partial R}\right) + \frac{\partial^2}{\partial Z^2}$, $\psi$ is the poloidal magnetic flux, $p(\psi)$ is the plasma pressure, and $F(\psi) = R\,B_\phi$ is the poloidal current function.

**Source terms.** The code parameterises the source profiles as:

- $p'(\psi) = p_0\,\alpha\,(1 - \hat\psi)^{\alpha-1}$ — peaked pressure gradient
- $FF'(\psi) = \mu_0\,R_0\,\lambda\,(1 - \hat\psi)^{\lambda-1}$ — peaked current

where $\hat\psi = (\psi - \psi_{\rm axis})/(\psi_{\rm bdry} - \psi_{\rm axis})$.

**Numerical method.** Nonlinear Picard iteration on a rectangular $(R,Z)$ finite-difference grid with Newton-Raphson acceleration for the coil-current equilibrium (`ForceBalanceSolver`). The Jacobian $\partial F_R / \partial I_{\rm PF}$ is evaluated by numerical perturbation, and the Newton step is clamped to $|\Delta I| \le 5$ MA for robustness. Convergence tolerance: $|F_R| < 10$ kN.

**Boundary conditions.** Free-boundary: vacuum Green's-function contributions from external PF coils (CS, PF1–PF6). The vacuum field $\psi_{\rm vac}$ is computed via the Biot-Savart kernel in `FusionKernel.calculate_vacuum_field()`.

**Key files:**

| File | Role |
|:---|:---|
| `core/fusion_kernel.py` | 2D GS finite-difference solver, vacuum field, X-point finder |
| `core/force_balance.py` | Newton-Raphson coil-current solver for radial force balance |
| `core/analytic_solver.py` | Solov'ev analytic equilibrium (test and initialisation) |
| `core/neural_equilibrium.py` | MLP surrogate for real-time flux-surface prediction |

**References:**

1. Grad, H. & Rubin, H. "Hydromagnetic Equilibria and Force-Free Fields." *Proc. 2nd UN Conf. Peaceful Uses Atomic Energy* 31, 190–197 (1958).
2. Shafranov, V.D. "Plasma Equilibrium in a Magnetic Field." *Rev. Plasma Phys.* 2, 103–151 (1966).
3. Briggs, W.L. et al. *A Multigrid Tutorial*, 2nd ed. SIAM (2000). doi:[10.1137/1.9780898719505](https://doi.org/10.1137/1.9780898719505).
4. Saad, Y. & Schultz, M.H. "GMRES: A Generalized Minimal Residual Algorithm." *SIAM J. Sci. Stat. Comput.* 7, 856–869 (1986). doi:[10.1137/0907058](https://doi.org/10.1137/0907058).

**Validation:** `tests/test_force_balance.py`, `tests/test_gs_convergence.py`, `tests/test_gs_residual_gate.py`, `validation/validate_against_sparc.py`, `validation/benchmark_vs_freegs.py`.

---

## 1A. Field-Reversed Configuration Rigid-Rotor Analytical Limit

The FRC workstream now exposes the Steinhauer no-rotation analytical limit as
the first accepted `FUS-C.1` contract.  It is not yet the full rotating
rigid-rotor BVP.  The accepted no-rotation contract is implemented in Python
and Rust, with optional PyO3 exposure when the native extension is built.

The same accepted no-rotation equations are exposed through
`frc_no_rotation_jax_observables` when JAX is installed. That helper evaluates
the Steinhauer field, pressure-balance profile, cylindrical flux primitive,
energy inventory, and Eq. 27 `s` integral on a fixed normalised grid
`x = r / R_s`, making `B_ext` and `R_s` gradients well-defined for optimisation
workflows. It is not a substitute for the unresolved rotating rigid-rotor BVP
or nonlinear MIF parity gates.

The axial field follows Steinhauer (2011), Eq. 7:

$$B_z(r) = -B_{\rm ext}\tanh\left(\frac{r^2 - R_s^2}{2R_s\delta}\right)$$

The cylindrical flux primitive is evaluated analytically from the same
accepted field:

$$\psi(r) = -B_{\rm ext}R_s\delta\left[\log\cosh(a(r)) -
\log\cosh(a(0))\right],\qquad
a(r)=\frac{r^2-R_s^2}{2R_s\delta}.$$

The accepted MIF-facing coordinate is the separatrix-normalised flux:

$$\psi_N(r)=\frac{\psi(r)-\psi(0)}{\psi(R_s)-\psi(0)}.$$

Validation gates the raw flux span as nonzero, requires
$\psi_N(0)=0$ and $\psi_N(R_s)=1$, and checks that $\psi_N$ is monotone and
bounded on $0\leq r\leq R_s$. These gates are carried through Python, Rust,
PyO3 parity, and the tracked benchmark report.

The solver validates a one-dimensional, strictly increasing radial grid,
requires the grid to start at the magnetic axis and extend outside the
separatrix radius, derives `delta = rho_i` from the deuterium thermal ion
gyroradius when no layer thickness is supplied, and rejects rotating cases
until the BVP implementation is added. The validation report now compares the
interpolated zero-crossing against the configured `R_s` and requires the axial
field sign to reverse between the last inner sample and the first outer sample.

The reported quality-of-equilibrium parameter follows Steinhauer Eq. 27:

$$s = \frac{1}{R_s}\int_0^{R_s}\frac{r}{\rho_i(r)}\,dr,$$

where $\rho_i(r) = \sqrt{2m_iT_i}/(e|B_z(r)|)$.  Numerically the code
integrates the finite form
$r e |B_z(r)|/\sqrt{2m_iT_i}$ over the clipped interval $[0,R_s]$.

The solver derives the toroidal diamagnetic current density from the
closed-form Steinhauer derivative:

$$J_\theta =
\frac{B_{\rm ext}}{\mu_0R_s\delta}
\left[1-\tanh^2\left(\frac{r^2-R_s^2}{2R_s\delta}\right)\right]r.$$

The Ampere residual then compares this analytical current against an
independent second-order finite-difference derivative on the active grid:

$$\mathcal{A}_r = \mu_0J_\theta + \frac{dB_z}{dr}.$$

The flux derivative residual compares the analytical primitive against an
independent second-order finite-difference derivative:

$$\mathcal{F}_r = \frac{d\psi}{dr} - rB_z.$$

The accepted no-rotation pressure profile is the local magnetic-pressure
balance profile:

$$p(r)=\frac{B_{\rm ext}^2-B_z(r)^2}{2\mu_0},\qquad
\mathcal{P}_r=p+\frac{B_z^2}{2\mu_0}-\frac{B_{\rm ext}^2}{2\mu_0}.$$

It also carries the analytical pressure-gradient identity implied by the same
magnetic-pressure-balance field:

$$\frac{dp}{dr}=-\frac{B_z}{\mu_0}\frac{dB_z}{dr},\qquad
\mathcal{G}_r=\left(\frac{dp}{dr}\right)_{\rm finite-grid}-
\left(\frac{dp}{dr}\right)_{\rm analytical}.$$

The solved density profile is derived from the accepted pressure profile and
temperature contract:

$$n(r)=\frac{p(r)}{(T_i+T_e)e}.$$

The scalar input pressure $n_0(T_i+T_e)e$ is reported as a consistency
diagnostic relative to the magnetic-pressure-balance peak. Validation now
gates the configured central density $n_0$ against the solved peak density,
so thermally inconsistent input decks fail closed instead of being accepted as
valid equilibria. The scalar input pressure is not used to replace the solved
local pressure profile.

The same accepted pressure and density profiles define beta and line-density
invariants on the separatrix domain:

$$\beta(r)=\frac{p(r)}{B_{\rm ext}^2/(2\mu_0)},\qquad
\langle\beta\rangle_s=\frac{1}{\pi R_s^2}\int_0^{R_s}\beta(r)\,2\pi r\,dr,$$

$$N_{\rm line}=\int_0^{R_s} n(r)\,2\pi r\,dr.$$

The separatrix pressure-energy inventory is checked against the independently
assembled magnetic-field deficit:

$$E_{p,s}=\int_0^{R_s}p(r)\,2\pi r\,dr,\qquad
E_{{\rm def},s}=\int_0^{R_s}\frac{B_{\rm ext}^2-B_z(r)^2}{2\mu_0}\,2\pi r\,dr.$$

The validation gate keeps $\beta_{\rm peak}\leq 1$ within finite-grid
tolerance for the accepted no-rotation pressure-balance contract.  The line
density is reported in particles per metre of axial length and is included in
cross-surface parity reports. The accepted pressure-balance contract also
requires $E_{p,s}=E_{{\rm def},s}$, so the validation report carries the
relative closure error as a fail-closed implementation gate. The current-sheet
closure also carries the resolved sheet-current integral above, which fails
closed when the implemented current profile no longer conserves the finite-grid
field jump.

The validation report carries both the Ampere closure residual
$\mathcal{A}_r$ and the normalised radial force-balance diagnostic

$$\mathcal{R}_r = \frac{d p}{d r} - (\mathbf{J}\times\mathbf{B})_r,$$

with $\mathbf{J}_\theta = -\mu_0^{-1} dB_z/dr$ for the no-rotation axial-field
slice.  The Ampere closure gate is active by default with a finite-grid
tolerance, and its report row must converge under grid refinement. The
force-balance diagnostic is visible by default and becomes a fail-closed gate
only when an explicit `force_balance_tolerance` is supplied.

The separatrix current sheet is also checked directly from the analytical
no-rotation field:

$$\left.\frac{dB_z}{dr}\right|_{R_s}=-\frac{B_{\rm ext}}{\delta},\qquad
J_\theta(R_s)=\frac{B_{\rm ext}}{\mu_0\delta}.$$

It also integrates the resolved current-density profile over the radial domain
and compares it with the magnetic-field jump on that same domain:

$$K_{\theta,{\rm grid}}=\int_0^{r_{\rm out}}J_\theta\,dr,\qquad
K_{\theta,{\rm expected}}=\frac{B_z(0)-B_z(r_{\rm out})}{\mu_0}.$$

The solver interpolates the finite-grid derivative and current-density profile
to $R_s$ and reports relative closure errors for both identities. These
current-sheet diagnostics are active validation gates for the accepted
analytical field; they are not a substitute for the unresolved rotating FRC BVP.

The tracked benchmark report now includes finite-grid convergence evidence for
the accepted no-rotation scalar invariants: null radius, separatrix radius
error, Eq. 27 `s`, energy per metre, pressure-balance ratio, analytical
pressure-gradient residual, central-density relative error, beta peak,
separatrix-averaged beta, particle line density,
separatrix pressure energy, separatrix magnetic-deficit energy, energy-closure
relative error, separatrix field-gradient/current-density closure,
`psi_N` axis/separatrix closure, `psi_N` monotonic/bounds diagnostics, and the
independent pressure, flux, and Ampere residuals. This is local
convergence evidence for the implemented analytical contract, not validation
of the unresolved rotating FRC BVP or kinetic/transport evolution.

The same report now carries a deterministic 16-case MIF/FRC no-rotation
parameter cohort across Python, Rust `fusion-physics`, and PyO3. The cohort
spans accepted separatrix radii, layer thicknesses, external axial fields, and
radial grid sizes. Go, Julia, and Lean remain explicit
`not_applicable_no_frc_surface` rows until equivalent native FRC solver logic
exists; wrappers are not treated as parity.

The no-rotation test surface also includes generated MIF/FRC property gates:
pressure must decrease outward from the magnetic null, `beta_peak` must remain
within the local pressure-balance bound, separatrix pressure energy must close
against magnetic-field deficit, and Rust/PyO3 must preserve energy and
separatrix-inventory invariants for generated accepted decks.

The executable quickstart `examples/03_frc_rigid_rotor_quickstart.py` runs the
same accepted solver path, emits the validated radial profiles and scalar
diagnostics, and is covered by a module-specific test that checks the
Steinhauer analytical field, magnetic-pressure-balance profile, validation
report, and JSON output. This keeps onboarding examples aligned with the
accepted no-rotation contract and does not extend the claim to the unresolved
rotating BVP.

**Key files:** `core/frc_rigid_rotor.py`, `scpn-fusion-rs/crates/fusion-physics/src/frc/`.

**Validation:** `tests/test_frc_rigid_rotor.py`, `tests/test_frc_rigid_rotor_property.py`, `tests/test_frc_rigid_rotor_rust_parity.py`,
`examples/03_frc_rigid_rotor_quickstart.py`,
`benchmarks/bench_frc_rigid_rotor.py`.

---

## 1B. MIF/FRC Pulsed Hall-MHD Flux Carrier

The MIF lane exposes FUS-C.2 as an axisymmetric pulsed Hall-MHD flux carrier in
Python and Rust. The accepted public contract is the flattened Ono Eq. 8 form:

$$
\frac{\partial\psi}{\partial t}
=-\frac{\psi}{\tau_\psi}+R_{\rm null}E_\theta-\eta_{\rm Spitzer}J_\theta.
$$

When no explicit azimuthal electric-field profile is supplied, the external
field ramp supplies the circular-loop Faraday drive:

$$E_\theta(r,t)=-\frac{r}{2}\frac{dB_{\rm ext}}{dt}.$$

Spitzer resistivity uses the NRL-style scaling:

$$\eta_{\rm Spitzer}=1.65\times10^{-9}
\frac{Z_{\rm eff}\ln\Lambda}{T_e[\mathrm{eV}]^{3/2}}\;\Omega\mathrm{m}.$$

The default step treats damping implicitly and the Ono source explicitly:

$$\psi_{n+1} =
\frac{\psi_n+\Delta t\left(R_{\rm null}E_{\theta,n+1}
-\eta J_{\theta,n+1}\right)}
{1+\Delta t/\tau_\psi}.$$

The axial field diagnostic is reconstructed as
$B_z=(1/r)\,d\psi/dr$ with finite-axis handling. The report carries Hall-drive,
resistive-sink, damping-sink, source-residual, and magnetic-energy proxy
diagnostics.

Full 2D two-fluid Hall-MHD, Gkeyll/BOUT++ same-case parity, WGPU execution,
and Ono figure reproduction remain explicit blocked rows until redistributable
reference artefacts are available.

**Key files:** `core/hall_mhd_pulsed.py`,
`scpn-fusion-rs/crates/fusion-physics/src/hall_mhd_pulsed.rs`,
`docs/physics/hall_mhd_pulsed.md`.

**Validation:** `tests/test_hall_mhd_pulsed.py`, Rust
`fusion_physics::hall_mhd_pulsed` unit tests,
`benchmarks/bench_hall_mhd_pulsed.py`,
`validation/reports/hall_mhd_pulsed_benchmark.json`.

---

## 1C. MIF/FRC MRTI Growth Spectrum

The MIF lane now exposes an analytical Magneto-Rayleigh-Taylor instability
growth-spectrum contract in Python and Rust. It evaluates the resolved
linear-mode growth rate with magnetic-tension stabilisation:

$$\gamma^2(k) = k\,a_{\rm eff} -
\frac{k^2B_\perp^2}{\mu_0\rho}.$$

Modes with negative radicand are reported as zero-growth stabilised modes. The
hydrodynamic limit is recovered as $\gamma=\sqrt{k\,a_{\rm eff}}$ when
$B_\perp=0$. `MRTISpectrumTracker` advances perturbation amplitudes through
frozen-coefficient exponential growth, records the fastest growing mode, and
captures the first saturation-threshold breach.

The coupling helper `effective_acceleration_from_radius_rate()` estimates
$d^2R_s/dt^2$ from a supplied separatrix radial-speed history using finite
differences and optional edge-padded smoothing. This is only an adapter for
future pulsed-compression trajectories; it is not the unresolved FUS-C.6
Hall-MHD compression solver and is not accepted as nonlinear saturation
evidence.

**Key files:** `core/mrti.py`, `scpn-fusion-rs/crates/fusion-physics/src/mrti.rs`,
`docs/physics/mrti.md`.

**Validation:** `tests/test_mrti.py`, Rust `fusion_physics::mrti` unit tests,
`benchmarks/bench_mrti.py`, `validation/reports/mrti_benchmark.json`.

---

## 1D. FRC Faraday Recovery

The MIF lane now exposes the FUS-C.7 classical recovery-coil induction contract
in Python and Rust. It operates on a supplied separatrix-radius and external
field trajectory:

$$\Phi(t) = B_{\rm ext}(t)\,\pi R_s(t)^2,$$

$$\mathrm{EMF}(t) = -N_{\rm turns}\,\pi
\left(R_s^2\frac{dB_{\rm ext}}{dt}
+ 2B_{\rm ext}R_s\frac{dR_s}{dt}\right).$$

For a resistive recovery load, the integrated recovered energy is evaluated as

$$E_{\rm recovered} = \int \frac{\mathrm{EMF}(t)^2}{R_{\rm load}}\,dt.$$

The implementation does not generate the unresolved FUS-C.6 Hall-MHD
pulsed-compression trajectory. If a caller does not supply a self-consistent
compression-work sidecar, `FaradayRecoveryReport.budget_claim_status` is
`blocked_missing_compression_work` instead of a fabricated pass/fail result.

**Key files:** `core/faraday_recovery.py`,
`scpn-fusion-rs/crates/fusion-physics/src/faraday_recovery.rs`,
`docs/physics/faraday_recovery.md`.

**Validation:** `tests/test_faraday_recovery.py`, Rust
`fusion_physics::faraday_recovery` unit tests,
`benchmarks/bench_faraday_recovery.py`,
`validation/reports/faraday_recovery_benchmark.json`.

---

## 1E. FRC Pulsed Compression

The MIF lane now exposes the FUS-C.6 pulsed-compression contract in Python and
Rust. The model evolves a separatrix-radius trajectory under internal thermal
pressure, external coil magnetic pressure, adiabatic compression heating, and a
one-step non-adiabatic flux-diffusion coupling. It is a full-state trajectory
contract for the implemented FRC/MIF lane, not a reduced-order surrogate or a
claim of same-case Slough Fig. 5 reproduction.

The external coil field is computed from the declared solenoid geometry:

$$B_{\rm ext}(t)=\mu_0\,N_{\rm coil}\,I_{\rm coil}(t)/L_{\rm coil}.$$

The radial shell update uses the TODO-specified momentum balance with explicit
geometry and mass:

$$m_{\rm shell}\,\frac{d^2R_s}{dt^2}
= \left(p_{\rm int}-p_{\rm mag,ext}\right)\,A,$$

where

$$p_{\rm int}=n\,(T_i+T_e)e,\qquad
p_{\rm mag,ext}=B_{\rm ext}^2/(2\mu_0).$$

The code reports the instantaneous beta diagnostic

$$\beta = \frac{2\mu_0 p_{\rm int}}{B_{\rm ext}^2},$$

and advances adiabatic ion/electron heating with cylindrical volume
compression:

$$T_{\rm next}=T_{\rm prev}\left(V_{\rm prev}/V_{\rm next}\right)^{\gamma-1}.$$

Flux diffusion is coupled through the accepted non-adiabatic current-diffusion
kernel using the instantaneous Spitzer resistivity and external-field-induced
flux target. The trajectory therefore carries both mechanical compression
state and magnetic-flux state at every step.

The public acceptance boundary is fail-closed: the helper
`slough_fig5_acceptance_status()` reports `blocked_missing_public_reference`
until a redistributable, digitised reference trajectory is available. The
tracked benchmark report includes that blocked row instead of fabricating
external parity.

**Key files:** `core/pulsed_compression.py`,
`scpn-fusion-rs/crates/fusion-physics/src/compression/pulsed.rs`,
`scpn-fusion-rs/crates/fusion-physics/src/compression/coil_geometry.rs`,
`docs/physics/pulsed_compression.md`.

**Validation:** `tests/test_pulsed_compression.py`, Rust
`fusion_physics::compression` unit tests,
`benchmarks/bench_pulsed_compression.py`,
`validation/reports/pulsed_compression_benchmark.json`.

---

## 1F. FRC n=1 Tilt-Mode Diagnostics

The MIF lane exposes FUS-C.5 as a conservative FRC n=1 tilt diagnostic in
Python and Rust. The accepted surface uses the Belova-normalised MHD
Alfvén-time growth scaling and the Steinhauer `s` parameter already carried by
the validated FRC equilibrium. It does not claim full Belova hybrid eigenvalue
parity.

For a supplied FRC equilibrium, the reference Alfvén speed is

$$V_A = \frac{B_{\rm ref}}{\sqrt{\mu_0 n_{\rm peak}m_i}},$$

and the axial half-length is

$$Z_s = E R_s,$$

where `E` is the prolate elongation. The MHD tilt diagnostic is

$$\gamma_{\rm tilt} = C V_A/Z_s,$$

with default `C = 1.2`. The rigid-body FLR diagnostic reports `s / E` against
the thresholds `1.7`, `2.2`, and `2.8`. These thresholds are labelled as
diagnostics only; the public stability boolean remains fail-closed while the
Belova Table I and hybrid-eigenvalue parity row is blocked.

**Key files:** `core/tilt_mode_frc.py`,
`scpn-fusion-rs/crates/fusion-physics/src/tilt_mode_frc.rs`,
`docs/physics/tilt_mode_frc.md`.

**Validation:** `tests/test_tilt_mode_frc.py`, Rust
`fusion_physics::tilt_mode_frc` unit tests,
`benchmarks/bench_tilt_mode_frc.py`,
`validation/reports/tilt_mode_frc_benchmark.json`.

---

## 2. JAX Differentiable GS Transport

A fully differentiable 1.5D transport kernel in JAX, enabling `jax.grad` through the transport evolution for optimal-control and inverse problems.

**Equation solved (explicit step):**

$$\frac{\partial T}{\partial t} = \frac{1}{n_e}\,\nabla\!\cdot\!\left(n_e\,\chi\,\nabla T\right) + S$$

**Numerical method.** Explicit finite-difference on a uniform $\rho$ grid. Spatial gradients use `jnp.gradient`. The domain is cylindrical-like in the normalised coordinate. Boundary conditions: $T(\rho=0) = T(\rho=\delta\rho)$ (axis symmetry), $T(\rho=1) = 0.1$ keV (fixed edge). Positivity enforced by `jnp.maximum(T, 0.01)`.

For multi-step rollout, `jax.lax.scan` is used, making the entire pulse trajectory JIT-compilable and differentiable end-to-end.

**Key file:** `core/jax_transport_solver.py`

**Validation:** `tests/test_jax_transport_solver.py`, `tests/test_gpu_jax_backend.py`.

---

## 3. Integrated 1.5D Transport

The production transport solver evolves four channels — $T_e$, $T_i$, $n_e$, $J_\phi$ — on 50 flux surfaces, coupled self-consistently with the 2D GS equilibrium and EPED pedestal.

### 3.1 Crank-Nicolson scheme

The implicit system for each profile channel is:

$$\frac{T^{n+1} - T^n}{\Delta t} = \frac{1}{2}\left[\mathcal{L}_h(T^{n+1}) + \mathcal{L}_h(T^n)\right] + S - \Lambda$$

where $\mathcal{L}_h(T) = \frac{1}{\rho}\frac{\partial}{\partial\rho}\!\left(\rho\,\chi\,\frac{\partial T}{\partial\rho}\right)$ is the radial heat diffusion operator, $S$ is the heating source, and $\Lambda$ is the radiation sink. The scheme is unconditionally stable (no CFL restriction).

The tridiagonal system $(I - \tfrac{1}{2}\Delta t\,L_h)\,T^{n+1} = \text{RHS}$ is solved by Thomas algorithm. Boundary conditions: axis symmetry ($\partial T/\partial\rho|_0 = 0$), prescribed edge from EPED pedestal or fixed $T_{\rm edge}$.

### 3.2 Anomalous transport

The anomalous diffusivity is provided by the gyrokinetic three-path closure (Section 4) or the neural surrogates (Section 5). A calibrated gyro-Bohm coefficient $c_{\rm gB}$ is loaded from `validation/reference_data/itpa/gyro_bohm_coefficients.json`.

### 3.3 Neoclassical ion transport — Chang-Hinton

The neoclassical ion thermal diffusivity follows Chang & Hinton, *Phys. Fluids* 25, 1493 (1982):

$$\chi_i^{\rm nc} = 0.66\,(1 + 1.54\,\epsilon)\,\frac{q^2\,\rho_i^2\,\nu_{ii}}{\epsilon^{3/2}\,(1 + 0.74\,\nu_*^{2/3})}$$

where $\epsilon = r/R_0$ is the inverse aspect ratio, $\rho_i$ is the ion Larmor radius, $\nu_{ii}$ is the ion-ion collision frequency from the NRL Plasma Formulary, and $\nu_* = \nu_{ii}\,q\,R_0 / (\epsilon^{3/2}\,v_{ti})$.

Both a Python vectorised reference implementation and a Rust fast-path (via PyO3 `PyTransportSolver`) are available; production backend selection is explicit rather than silent.

### 3.4 Bootstrap current — Sauter model

The bootstrap current density follows Sauter et al., *Phys. Plasmas* 6, 2834 (1999):

$$j_{\rm bs} = -\frac{p_e}{B_\theta}\left[L_{31}\,\frac{1}{n_e}\frac{dn_e}{dr} + L_{32}\,\frac{1}{T_e}\frac{dT_e}{dr} + L_{34}\,\frac{1}{T_i}\frac{dT_i}{dr}\right]$$

The Sauter coefficients $L_{31}$, $L_{32}$, $L_{34}$ depend on the trapped-particle fraction $f_t$, collisionality $\nu_*^e$, and $Z_{\rm eff}$. Interior gradients use central finite differences; axis and edge are set to zero.

### 3.5 Multi-ion transport

When `multi_ion=True`, the solver evolves separate D/T fuel densities, He-ash with configurable pumping time $\tau_{\rm He}$, independent $T_e$, coronal-equilibrium tungsten radiation (Pütterich et al. 2010), and per-cell Bremsstrahlung.

**Key files:** `core/integrated_transport_solver.py`, `core/integrated_transport_solver_runtime.py`, `core/integrated_transport_solver_model.py`, `core/current_diffusion.py`.

**Validation:** `tests/test_integrated_transport_solver.py`, `tests/test_cn_transport.py`, `tests/test_multi_ion_transport.py`, `validation/benchmark_transport_power_balance.py`.

### 3.6 Non-adiabatic MIF/FRC flux carrier

For pulsed MIF/FRC workflows the current-diffusion surface now includes the
flattened Ono-style non-adiabatic carrier:

$$\frac{\partial\psi}{\partial t} =
-\frac{\psi}{\tau_\psi} + R_{\rm null} E_\theta - \eta J_\theta.$$

The public Python helper `solve_flux_evolution_nonadiabatic` returns a
`FluxEvolutionTrajectory` with `psi`, `R_null E_theta`, `eta J_theta`, net
source, and damping histories on the supplied radial grid. Each step advances
the local linear damping analytically with endpoint-averaged source sampling,
so constant damping recovers the exact exponential solution and zero drive,
zero damping preserves the input flux exactly. Rust parity math lives in
`fusion-core::current_diffusion`, and the benchmark report is local regression
evidence only unless rerun under the repository benchmark-isolation policy.

**Validation:** `tests/test_current_diffusion_nonadiabatic.py`, Rust unit tests
in `fusion-core::current_diffusion`, and
`validation/reports/current_diffusion_nonadiabatic_benchmark.json`.

---

## 4. Gyrokinetic Three-Path Closure

The transport closure separates three operational lanes with strict provenance tracking.

### Path A: External gyrokinetic codes (5)

Input-deck generation and output parsing for TGLF, GENE, GS2, GKW, and CGYRO. The `TGLFInputDeck` builder in `core/tglf_interface.py` constructs a complete TGLF input from the `TransportSolver` state. Reference data is stored in `validation/tglf_reference/` for ITG-dominated, TEM-dominated, and ETG-dominated regimes.

### Path B: Native gyrokinetic solvers

The native path has two distinct surfaces. `core/gk_eigenvalue.py` and
`core/gk_quasilinear.py` provide the linear eigenvalue and quasilinear
transport lane. `core/gk_nonlinear.py` provides a bounded 5D delta-f nonlinear
operator research solver with NumPy and JAX execution paths. It now reports an
explicit E x B invariant contract: the dealiased nonlinear bracket must not
inject free energy in an undriven collisionless step, and all bracket energy
outside the 2/3 spectral mask must remain zero. This is not a replacement for
GENE or CGYRO nonlinear turbulence campaigns.

The reduced multichannel model in `core/neural_transport.py::critical_gradient_model()` resolves three branches:

- **ITG**: $\chi_i^{\rm ITG} = \chi_{\rm gB}\,\left(\max(R/L_{T_i} - R/L_{T_i}^{\rm crit},\,0)\right)^\alpha\,f_s\,f_\beta$
- **TEM**: $\chi_e^{\rm TEM} = \chi_{\rm gB}\,\left(\max(R/L_{T_e} - R/L_{T_e}^{\rm crit},\,0)\right)^\alpha\,f_t\,f_{\nu_*}\,f_\beta\,f_n$
- **ETG**: $\chi_e^{\rm ETG} = 0.85\,\chi_{\rm gB}\,\left(\max(R/L_{T_e} - R/L_{T_e,{\rm ETG}}^{\rm crit},\,0)\right)^{0.9\alpha}\,f_{\nu_*}\,f_s\,(T_e/T_i)$

where $\alpha = 2.0$ is the stiffness exponent (Dimits PoP 2000; physical range 1.5–4.0), $f_s = (1 + 0.35\,\hat{s}^2)^{-1}$ is shear suppression, $f_\beta = (1 + \beta_e/0.03)^{-1}$ is electromagnetic stabilisation, and $f_t = 1.46\sqrt{\epsilon}$ is the trapped-particle fraction.

Critical gradients incorporate magnetic shear and electromagnetic corrections: $R/L_{T_i}^{\rm crit} = 4.0 + 0.4\hat{s} + 8\beta_e$.

The instability domain classification uses `_dominant_channel()` based on relative transport magnitudes.

### Path C: Hybrid OOD detection + correction + online learning

The neural surrogate (Section 5) provides per-point out-of-distribution (OOD) detection via z-scores against training statistics. Points with $\max|z| > 5\sigma$ are flagged; the TGLF interface can be called for targeted correction at up to `tglf_max_points` OOD locations.

**Mixing-length saturation.** All three paths use a gyro-Bohm reference: $\chi_{\rm gB} = \rho_s^2\,c_s / R$, where $\rho_s = \sqrt{m_i T_e}/(eB)$ and $c_s = \sqrt{T_e/m_i}$.

**Key files:** `core/neural_transport.py`, `core/tglf_interface.py`, `core/tglf_surrogate_bridge.py`, `core/tglf_validation_runtime.py`.

**Validation:** `tests/test_neural_transport.py`, `tests/test_tglf_interface.py`, `validation/validate_fno_tglf.py`, `validation/validate_transport_qlknn.py`.

---

## 5. Neural Transport Surrogates

### 5.1 QLKNN-10D

A variable-depth feedforward MLP maps 10 local plasma parameters $(ρ, T_e, T_i, n_e, R/L_{T_e}, R/L_{T_i}, R/L_{n_e}, q, \hat{s}, \beta_e)$ to turbulent fluxes $(\chi_e, \chi_i, D_e)$.

**Architecture:** Auto-detected from `.npz` weight keys: `input(D) → hidden1 → [hidden2 → ...] → output(3)`, with GELU activation on hidden layers and softplus on output (ensures $\chi > 0$). Supports optional log-transform, gyro-Bohm skip-connection, and gated output modes.

**Training data:** QLKNN-10D public dataset (doi:[10.5281/zenodo.3700755](https://doi.org/10.5281/zenodo.3700755)). Training recipe in `docs/NEURAL_TRANSPORT_TRAINING.md`.

**Accuracy:** Relative $L_2$ error = 0.094 against QLKNN-10D test set (see `weights/neural_transport_qlknn.metrics.json`).

**OOD detection:** Per-point z-score monitoring. Points with $\max|z| > 3\sigma$ or $> 5\sigma$ are tracked in the surrogate contract.

**Key file:** `core/neural_transport.py` (`NeuralTransportModel`).

**References:** van de Plassche et al., "Fast modeling of turbulent transport in fusion plasmas using neural networks." *Phys. Plasmas* 27, 022310 (2020). doi:[10.1063/1.5134126](https://doi.org/10.1063/1.5134126).

### 5.2 FNO Turbulence Suppressor

A Fourier Neural Operator predicts turbulence suppression from a 2D $(64 \times 64)$ drift-wave turbulence field. Default execution uses a deterministic reduced-order compatibility backend; the JAX-FNO backend requires explicit opt-in.

The `SpectralTurbulenceGenerator` drives an ITG-like spectral drift-wave simulation with a predator-prey zonal-flow coupling: $d(\text{ZF})/dt = 5 \langle \phi^2 \rangle - 0.5\,\text{ZF}$.

**Accuracy:** Relative $L_2$ error = 0.055 on synthetic proxy data. This lane is research-grade, not gyrokinetic-validated.

**Key file:** `core/fno_turbulence_suppressor.py` (`FNO_Controller`).

**References:** Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR* (2021). arXiv:[2010.08895](https://arxiv.org/abs/2010.08895).

### 5.3 GyroSwin-Like Surrogate (GAI-01)

A deterministic NumPy-only surrogate for CI benchmarking of core-turbulence transport. Uses a synthetic reference law with drive/shear/shape factors calibrated to JET/ITPA-like plasma regimes. Provides speed benchmarks against a "GENE-like proxy" baseline.

**Key file:** `core/gyro_swin_surrogate.py`.

**Validation:** `tests/test_gai_01_turbulence_surrogate.py`, `validation/gai_01_turbulence_surrogate.py`.

---

## 6. MHD Stability Suite

Seven stability criteria are evaluated in `run_full_stability_check()`:

### 6.1 Mercier interchange stability

$$D_M = \frac{s^2}{4} - \alpha_{\rm MHD} \;\ge\; 0$$

where $s = (r/q)(dq/dr)$ is the magnetic shear and $\alpha_{\rm MHD} = -2\mu_0 R_0 q^2 (dp/dr) / B_0^2$ is the normalised pressure gradient. Freidberg, *Ideal MHD*, Ch. 12 (2014).

### 6.2 Ideal ballooning — Connor-Hastie-Taylor

Critical pressure gradient in the first-stability window:

$$\alpha_{\rm crit}(s) = \begin{cases} s(1 - s/2) & s < 1 \\ 0.6\,s & s \ge 1 \end{cases}$$

Stable where $\alpha \le \alpha_{\rm crit}$. Connor, Hastie & Taylor, *Phys. Rev. Lett.* 40, 396 (1978).

### 6.3 Kruskal-Shafranov external kink

$$q_{\rm edge} > 1$$

Stable against the $m=1/n=1$ external kink. Kruskal & Schwarzschild, *Proc. R. Soc. Lond. A* 223, 348 (1954).

### 6.4 Troyon normalised beta limit

$$\beta_N = \frac{100\,\beta_t}{I_p[\text{MA}]/(a[\text{m}]\,B_0[\text{T}])} \;<\; g$$

where $g = 2.8$ (no-wall) or $g = 3.5$ (ideal-wall). Troyon et al., *Plasma Phys. Control. Fusion* 26, 209 (1984).

### 6.5 Neoclassical tearing mode (NTM)

Reduced modified Rutherford equation for island width $w$:

$$\tau_R\,\frac{dw}{dt} = r_s\,\Delta' + r_s\,a_{\rm bs}\,\frac{j_{\rm bs}}{j_{\rm total}}\,\frac{w}{w^2 + w_d^2}$$

The NTM is unstable when bootstrap drive exceeds classical stabilisation ($\Delta' < 0$). La Haye, *Phys. Plasmas* 13, 055501 (2006).

### 6.6 Resistive wall mode (RWM)

Between the no-wall and ideal-wall beta limits, the RWM growth rate scales as:

$$\gamma\,\tau_w \;\sim\; \frac{\beta_N - \beta_N^{\rm no\text{-}wall}}{\beta_N^{\rm wall} - \beta_N}$$

Stable when $\beta_N < \beta_N^{\rm no\text{-}wall}$, or when active feedback suppresses the mode.

### 6.7 Peeling-ballooning (ELM boundary)

Coupled pedestal stability per Snyder et al., *Phys. Plasmas* 9, 2037 (2002) and *Nucl. Fusion* 51, 103016 (2011). The stability boundary is parameterised by normalised edge current $j_{\rm edge}/j_{\rm crit}$ (peeling drive) and normalised $\alpha / \alpha_{\rm crit}$ (ballooning drive). ELM type classification: Type-I (high $\alpha$, low $j$), Type-III (low $\alpha$, high $j$).

**Key files:** `core/stability_mhd.py`, `core/stability_mhd_extended.py`.

**Validation:** `tests/test_mhd_stability.py`, `tests/test_phase1_hardening.py`.

---

## 7. Disruption Physics

### 7.1 Thermal quench

SPI-induced radiative collapse. The thermal energy decays via impurity radiation:

$$\frac{dW_{\rm th}}{dt} = -P_{\rm rad}, \qquad P_{\rm rad} \propto Z_{\rm eff}^{1/2}\,T_e^{1/2}\,n_e$$

Temperature tracks energy: $T_e \propto W_{\rm th}$. The thermal quench completes when $T_e < 5$ keV.

### 7.2 Current quench

L/R decay after thermal quench:

$$\frac{dI_p}{dt} = -\frac{I_p}{\tau_{\rm CQ}}, \qquad \tau_{\rm CQ} = 0.02\,\frac{2}{Z_{\rm eff}}\,\left(\frac{T_e}{0.1}\right)^{0.25}\;\text{s}$$

clamped to $[2, 50]$ ms. The ITER design limit is $\tau_{\rm CQ} \ge 50$ ms for thermal loads and $\le 150$ ms for eddy-current forces.

### 7.3 Halo currents — Fitzpatrick L/R circuit

During vertical displacement events (VDEs), the halo current $I_h$ is driven by an L/R circuit:

$$L_h\,\frac{dI_h}{dt} + R_h\,I_h = M\,\left|\frac{dI_p}{dt}\right|$$

The ITER design limit: $\text{TPF} \times I_h / I_{p0} \le 0.75$. Wall force: $F \sim \mu_0 I_h I_p / (2\pi a)$.

Ref: Fitzpatrick, R. *Phys. Plasmas* 9, 3459 (2002).

**Key files:** `control/spi_mitigation.py` (`ShatteredPelletInjection`), `control/halo_re_physics.py` (`HaloCurrentModel`, `run_disruption_ensemble`), `control/disruption_predictor.py` (`DisruptionTransformer`).

**Validation:** `tests/test_spi_mitigation.py`, `tests/test_halo_re_physics.py`, `validation/task5_disruption_mitigation_integration.py`, `validation/benchmark_disruption_replay_pipeline.py`.

---

## 8. ELM & Edge Physics

### 8.1 EPED pedestal model

Reduced-order EPED-like scaling per Snyder et al., *Phys. Plasmas* 16, 056118 (2009):

$$\Delta_{\rm ped} = 0.076\,\beta_{p,{\rm ped}}^{0.5}\,\nu_*^{-0.2}$$

with high-density width broadening: $\Delta \times (1 + 0.25\,n_{\rm ped}/10)$ (Frassinetti et al., *Nucl. Fusion* 61, 016001, 2021). The pedestal temperature is iterated to satisfy the ballooning pressure-gradient limit $\alpha_{\rm crit}$.

Domain-validity bounds enforce calibrated ranges for all 10 input parameters (R0, a, B0, Ip, $\kappa$, etc.) with continuous extrapolation penalty.

### 8.2 Sawtooth oscillations — Kadomtsev reconnection

Reduced MHD in cylindrical geometry. The $m=1, n=1$ internal kink mode is evolved with:

$$\frac{\partial \psi_{11}}{\partial t} = \eta\,\nabla_\perp^2 \psi_{11} + \{...\}$$

Crash trigger at $|\psi_{11}| > 0.1$; post-crash amplitude reduction by 99% inside $q < 1$. Safety-factor relaxation rate: $0.05\,\text{s}^{-1}$.

Ref: Kadomtsev, B.B. *Sov. J. Plasma Phys.* 1, 389 (1975).

### 8.3 Predator-prey zonal flow (L-H transition)

The turbulence suppressor includes a zonal-flow predator-prey model:

$$\frac{d(\text{ZF})}{dt} = 5\,\langle \phi^2 \rangle - 0.5\,\text{ZF}$$

The zonal flow shearing rate contributes to total turbulence damping: $\gamma_{\rm tot} = \gamma_{\rm ext} + 0.2\,\text{ZF}$.

**Key files:** `core/eped_pedestal.py`, `core/mhd_sawtooth.py`, `core/fno_turbulence_suppressor.py`.

**Validation:** `tests/test_eped_pedestal.py`, `tests/test_eped_domain_contract_benchmark.py`, `tests/test_mhd_sawtooth.py`.

---

## 9. Pellet Injection

### 9.1 NGS ablation — Parks & Turnbull

The ablation rate for a spherical pellet of radius $r_p$ in a plasma with density $n_e$ and temperature $T_e$:

$$\frac{dm}{dt} = C_P\,n_e^{1/3}\,T_e^{1.64}\,r_p^{4/3}$$

where $C_P$ is the Parks coefficient, calibrated to mixed units (Parks et al., *Phys. Plasmas* 5, 1024, 1998; Parks, *Nucl. Fusion* 57, 2017, Eq. 8).

### 9.2 Multi-fragment SPI ablation

The `SpiAblationSolver` tracks $N$ Lagrangian fragments (position, velocity, mass, radius). Each fragment ablates according to the Parks scaling at its local plasma parameters, with mass deposited into the radial density profile weighted by the local volume element $dV = 4\pi^2 R\,r\,(a\,dr)$.

Fragment generation: uniform mass, solid-neon density ($\rho_{\rm Ne} = 1444\;\text{kg/m}^3$, CRC Handbook). Velocity dispersion from Gaussian perturbation of the injector direction.

### 9.3 Grad-B drift displacement

Not yet implemented as an explicit Pegourie (2005) drift model. Pellet deposition currently assumes radial penetration only. Toroidal drift effects enter through the SPI mitigation campaign's empirical $Z_{\rm eff}$ cocktail model.

**Key files:** `control/spi_ablation.py` (`SpiAblationSolver`), `control/spi_mitigation.py` (`ShatteredPelletInjection`).

**Validation:** `tests/test_spi_ablation.py`, `tests/test_spi_mitigation.py`, `tests/test_spi_mitigation_runtime.py`.

---

## 10. Impurity Transport

### 10.1 Tungsten radiation — Pütterich cooling curve

When `multi_ion=True`, the transport solver applies coronal-equilibrium tungsten radiation per Pütterich et al., *Nucl. Fusion* 50, 025012 (2010). The cooling-rate function $L_z(T_e)$ is interpolated from tabulated values; contributions from C, Ar, and Ne are also available via ADAS atomic data.

### 10.2 Bremsstrahlung

Per-cell Bremsstrahlung: $P_{\rm brem} = 5.35 \times 10^{-37}\,Z_{\rm eff}\,n_e^2\,T_e^{1/2}\;\text{W/m}^3$.

### 10.3 SPI cocktail model

The impurity cocktail estimator (`ShatteredPelletInjection.estimate_mitigation_cocktail`) computes Ne/Ar/Xe mole fractions as a function of disruption risk and disturbance amplitude, with empirical $Z_{\rm eff}$ weighting: $Z_{\rm eff} = (1 - f_{\rm imp}) + f_{\rm imp}\,\langle Z^2 \rangle$ where $\langle Z^2 \rangle$ is the mass-weighted average over impurity species.

### 10.4 Trace radial impurity transport

The trace impurity solver advances species densities with implicit radial
diffusion, volume-normalised edge sources, and a Hirshman-Sigmar-style
neoclassical pinch using inverse scale lengths:

$$V_Z = -D_{\rm neo}\left[\frac{Z}{L_n} + \left(\frac{Z}{2} - H_Z\right)\frac{1}{L_{T_i}}\right],\quad \frac{1}{L_x}=-\frac{d\ln x}{dr}$$

The native benchmark validates source conservation, positivity, inward pinch
for peaked profiles, and monotonic radiated power. This is a trace transport
contract, adds a charge-state artifact/source-sink contract, and does not claim
Aurora/STRAHL/JINTRAC collisional-operator parity.

**Key files:** `core/impurity_transport.py`, `core/integrated_transport_solver.py` (multi-ion mode), `control/spi_mitigation.py`.

**Validation:** `tests/test_impurity_transport.py`, `tests/test_multi_ion_transport.py`, `validation/benchmark_impurity_transport_contract.py`, `validation/benchmark_multi_ion_transport_conservation.py`.

---

## 11. Momentum Transport

### 11.1 RF Heating — ICRH and LHCD

Ray-tracing for ion cyclotron resonance heating (ICRH) at 50 MHz. The wave propagation follows the local dispersion relation with $B_{\rm tor}(R) = B_0 R_0 / R$. Absorption is computed at the fundamental cyclotron resonance layer where $\omega = \Omega_{\rm ci} = eB/m_i$.

Lower Hybrid Current Drive (LHCD) power deposition profiles are computed from the parallel refractive index $n_\parallel$ and the Landau-damping condition.

### 11.2 ExB shearing

The turbulence suppressor implements ExB shearing through the zonal-flow predator-prey model. The total damping includes a shearing contribution $0.2 \times \text{ZF}$ applied in Fourier space, following the mechanism described by Waltz, *Phys. Plasmas* 1, 2229 (1994).

**Key file:** `core/rf_heating.py` (`RFHeatingSystem`).

**Validation:** `tests/test_phase0_physics_fixes.py`.

---

## 12. Plasma-Wall Interaction

### 12.1 Eckstein sputtering yield

The sputtering yield $Y(E, \theta)$ uses the Eckstein empirical formula:

$$Y = Q\,s_n(\epsilon)\,\left(1 - \left(\frac{E_{\rm th}}{E}\right)^{2/3}\right)\left(1 - \frac{E_{\rm th}}{E}\right)^2\,f(\theta)$$

where $s_n(\epsilon) = \ln(1 + 1.2288\,\epsilon)/(1 + \sqrt{\epsilon})$ is the nuclear stopping cross-section, $\epsilon = E/E_{\rm th}$ is the reduced energy, and $f(\theta) = 1/\cos\theta$ is the angular enhancement factor.

Material parameters:
- **Tungsten**: $E_{\rm th} = 200$ eV, $Q = 0.03$, $A = 183.84$, $\rho = 19.25\;\text{g/cm}^3$.
- **Carbon**: $E_{\rm th} = 30$ eV, $Q = 0.1$, $A = 12.0$, $\rho = 2.2\;\text{g/cm}^3$.

### 12.2 Net erosion rate

$$\dot{d} = \frac{\Gamma_{\rm ion}\,Y\,(1 - f_{\rm redep})\,A\,m_u}{\rho_{\rm wall}}$$

where $f_{\rm redep}$ is the redeposition fraction (typically 0.95 for W divertor), and the impact energy is $E_{\rm impact} = 5\,T_i$.

### 12.3 Divertor heat flux — Eich model

The scrape-off layer heat-flux width follows Eich et al., *Nucl. Fusion* 53, 093031 (2013), mapped to divertor tile surfaces in `core/divertor_thermal_sim.py`.

**Key files:** `nuclear/pwi_erosion.py` (`SputteringPhysics`), `core/divertor_thermal_sim.py`.

**Validation:** `tests/test_pwi_erosion.py`, `tests/test_nuclear_wall_interaction.py`.

---

## 13. Runaway Electrons

### 13.1 Critical and Dreicer fields

The Connor-Hastie critical field for runaway sustainment:

$$E_c = \frac{n_e\,e^3\,\ln\Lambda}{4\pi\,\epsilon_0^2\,m_e\,c^2}$$

The Dreicer field: $E_D = E_c\,m_e c^2 / T_e$.

Ref: Connor, J.W. & Hastie, R.J. *Nucl. Fusion* 15, 415 (1975).

### 13.2 Primary generation (Dreicer)

$$\dot{n}_{\rm RE}^{\rm primary} = \frac{n_e}{\tau_{\rm coll}}\,C_D\,\left(\frac{E_D}{E}\right)^{h(Z_{\rm eff})}\,\exp\!\left(-\frac{E_D}{4E} - \sqrt{\nu_{\rm eff}}\right)$$

In the Fokker-Planck solver, this is modelled as a source injection at low momentum ($p < 5\,p_{\rm th}$).

### 13.3 Avalanche generation — Rosenbluth-Putvinski

$$\dot{n}_{\rm RE}^{\rm aval} = n_{\rm RE}\,\frac{(E/E_c - 1)}{\tau_{\rm av}\,\ln\Lambda}\,\sqrt{\frac{\pi(Z_{\rm eff} + 1)}{2}}$$

Ref: Rosenbluth, M.N. & Putvinski, S.V. *Nucl. Fusion* 37, 1355 (1997), Eq. 19 and 66.

### 13.4 DREAM-style fluid density balance

$$\frac{dn_{\rm RE}}{dt} = S_{\rm Dreicer} + \gamma_{\rm aval} n_{\rm RE} - \frac{n_{\rm RE}}{\tau_{\rm loss}},\quad 0 \le n_{\rm RE} \le f_{\rm cap} n_e$$

The native contract validates the scalar density balance used by DREAM fluid
runs, including subcritical avalanche suppression, mitigation loss accounting,
and density-cap enforcement. It is a fluid benchmark contract and does not
claim parity with DREAM's kinetic momentum-space distribution solver.

### 13.5 Hot-tail seed

Gaussian tail from rapid thermal quench: $f_{\rm hottail}(p) \propto \exp(-p^2/p_{\rm th,initial}^2)$, per Aleynikov & Breizman, *Phys. Rev. Lett.* 114, 155001 (2015). The seed amplitude is fixed at $10^{10}\;\text{m}^{-3}$ (quench-time dependence ignored; stated assumption).

### 13.6 Fokker-Planck solver

The 1D-in-momentum kinetic equation for $f(p, t)$:

$$\frac{\partial f}{\partial t} + \frac{\partial}{\partial p}\!\left[(F_{\rm acc} - F_{\rm drag} - F_{\rm synch})\,f - D\,\frac{\partial f}{\partial p}\right] = S_{\rm av} + S_{\rm Dr} + S_{\rm ko}$$

**Numerical method:** MUSCL-Hancock 2nd-order advection with minmod limiter (Toro, *Riemann Solvers*, Ch. 13), operator-split central-difference diffusion half-step. Log-spaced momentum grid with 200 points up to $p = 100\,m_e c$.

**Synchrotron radiation force:** $F_{\rm synch} = p\gamma\sqrt{1 + Z_{\rm eff}} / \tau_{\rm rad}$, where $\tau_{\rm rad} = 6\pi\epsilon_0 (m_e c)^3 / (e^4 B^2)$.

Ref: Hesslow et al., *J. Plasma Phys.* 85, 475850601 (2019).

**Key files:** `core/runaway_electrons.py` (fluid source terms and density balance), `control/fokker_planck_re.py` (`FokkerPlanckSolver`), `control/halo_re_physics.py` (`RunawayElectronModel`), `control/runaway_electron_model.py`.

**Validation:** `tests/test_runaway_electrons.py`, `tests/test_fokker_planck.py`, `tests/test_halo_re_physics.py`, `validation/benchmark_runaway_dream_contract.py`.

---

## 14. Alfven Eigenmodes

### TAE/RSAE continuum

The toroidicity-induced Alfven eigenmode (TAE) frequency gap is located at:

$$\omega_{\rm TAE} = \frac{v_A}{2qR_0}$$

where $v_A = B/\sqrt{\mu_0 n_i m_i}$ is the Alfven speed. Reversed-shear Alfven eigenmodes (RSAEs) appear near the minimum of the safety-factor profile with $\omega_{\rm RSAE} \approx v_A\,|m - nq_{\rm min}|/(qR_0)$.

### Fast-particle drive

The energetic-particle drive from NBI or alpha particles is estimated via the critical beta for fast-ion destabilisation. This analysis is integrated into the stability suite as a diagnostic quantity alongside the Troyon $\beta_N$.

**Status:** Reduced-order; continuum frequency estimates only. Full eigenmode stability requires external codes (NOVA, CASTOR).

**Key file:** `core/stability_analyzer.py`.

---

## 15. Neutronics

### 15.1 Single-group blanket

1D cylindrical diffusion-reaction for neutron flux $\Phi(r)$:

$$-D\,\frac{1}{r}\frac{d}{dr}\!\left(r\,\frac{d\Phi}{dr}\right) + \Sigma_{\rm rem}\,\Phi = 0$$

Solved by finite-difference tridiag on a radial grid from $r_{\rm inner}$ to $r_{\rm outer}$. Boundary conditions: incident flux at first wall, rear albedo at shield. Cross sections: $\Sigma_{\rm Li6}^{\rm capture} = 0.15\,f_{\rm enrich}$, $\Sigma_{\rm scatter} = 0.2$, $\Sigma_{\rm mult} = 0.08$ (Be (n,2n), gain = 1.8).

TBR from cylindrical volume integration: $\text{TBR} = \int \Sigma_{\rm Li6}\,\Phi\,2\pi r\,dr \,/\, J_{\rm in}\,2\pi r_{\rm inner}$.

### 15.2 Three-group blanket

Energy-dependent transport with down-scatter chain:

| Group | Energy | Dominant process |
|:---|:---|:---|
| 1 (fast) | $> 1$ MeV | Be(n,2n) multiplication, inelastic scatter |
| 2 (epithermal) | 1 eV – 1 MeV | Li-6 resonance capture |
| 3 (thermal) | $< 1$ eV | Li-6(n,t) at 940 b, dominant T production |

Each group solved independently as 1D cylindrical diffusion with inter-group source terms $\Sigma_{\rm down}\,\Phi_g$. Correction factors for port coverage (0.80), neutron streaming (0.85), and blanket fill (1.0) per Fischer et al., DEMO blanket studies (2015).

### 15.3 Reduced volumetric 3D surrogate

The 1D radial profile is extended over a toroidal mesh $(N_r \times N_\theta \times N_\phi)$ with elongation-weighted poloidal and toroidal angle modulation. This captures first-order 3D effects (inboard/outboard asymmetry, port coverage gaps).

### 15.4 Alpha heating — Bosch-Hale reactivity

The D-T reactivity $\langle\sigma v\rangle_{\rm DT}$ uses the parameterisation from Bosch & Hale, *Nucl. Fusion* 32, 611 (1992), implemented in `core/uncertainty.py::_dt_reactivity()`. The fusion power and Q-factor are computed in `FusionBurnPhysics.calculate_thermodynamics()`.

**Key files:** `nuclear/blanket_neutronics.py` (`BreedingBlanket`, `MultiGroupBlanket`), `core/fusion_ignition_sim.py` (`FusionBurnPhysics`).

**Validation:** `tests/test_blanket_neutronics.py`, `tests/test_heating_neutronics_q10.py`, `validation/gneu_01_benchmark.py`, `validation/task6_heating_neutronics_realism.py`.

---

## 16. 3D Equilibrium

### 16.1 VMEC-lite Fourier representation

The boundary shape is parameterised by VMEC-like Fourier harmonics in $(\theta, \phi)$:

$$R(\rho,\theta,\phi) = R_{\rm axis} + \rho\,a\left[\cos\theta - \delta\cos 2\theta + \sum_{m,n} R_{mn}^c\cos(m\theta - n\,N_{\rm fp}\,\phi)\right]$$

$$Z(\rho,\theta,\phi) = Z_{\rm axis} + \rho\,a\,\kappa\,\sin\theta + \sum_{m,n} Z_{mn}^s\sin(m\theta - n\,N_{\rm fp}\,\phi)$$

where $N_{\rm fp}$ is the number of field periods, and $\delta$ is triangularity. Each `FourierMode3D(m, n, r_cos, r_sin, z_cos, z_sin)` adds a non-axisymmetric perturbation.

The `VMECStyleEquilibrium3D` class supports:

- Fixed-boundary equilibrium in native flux coordinates $(\rho, \theta, \phi)$
- Cartesian geometry output for 3D visualisation
- `from_axisymmetric_lcfs()` constructor for automatic shaping inference

### 16.2 3D field-line tracing

The `FieldLineTracer3D` integrates the field-line ODE in flux coordinates using RK4. Poincare sections are generated at specified toroidal angles $\phi_0$.

### 16.3 Toroidal asymmetry observables

Fourier analysis of the traced field-line positions yields $n=1,2,3$ toroidal amplitudes and an asymmetry index used by the disruption predictor for RMP-aware control.

**Key files:** `core/equilibrium_3d.py` (`VMECStyleEquilibrium3D`, `FourierMode3D`), `core/fieldline_3d.py` (`FieldLineTracer3D`, `PoincareSection3D`).

**Validation:** `tests/test_equilibrium_3d.py`, `tests/test_force_balance_3d.py`.

---

## Confinement Scaling

The IPB98(y,2) empirical scaling law for H-mode energy confinement time:

$$\tau_E = C\,I_p^{\alpha_I}\,B_T^{\alpha_B}\,\bar{n}_{e19}^{\alpha_n}\,P_{\rm loss}^{\alpha_P}\,R^{\alpha_R}\,\kappa^{\alpha_\kappa}\,\epsilon^{\alpha_\epsilon}\,M^{\alpha_M}$$

Coefficients loaded from `validation/reference_data/itpa/ipb98y2_coefficients.json`, fitted to 5920 H-mode data points from 18 tokamaks.

Updated Bayesian uncertainty quantification via Verdoolaege et al., *Nucl. Fusion* 61, 076006 (2021).

**Key file:** `core/scaling_laws.py`.

**Validation:** `tests/test_ipb98y2_benchmark.py`.

---

## Summary of Core Physics Modules

| Module | Physics | Section |
|:---|:---|:---|
| `core/fusion_kernel.py` | 2D GS equilibrium | §1 |
| `core/force_balance.py` | Newton-Raphson coil balance | §1 |
| `core/jax_transport_solver.py` | JAX differentiable transport | §2 |
| `core/integrated_transport_solver.py` | 1.5D Crank-Nicolson transport | §3 |
| `core/neural_transport.py` | QLKNN-10D + critical-gradient | §4, §5 |
| `core/fno_turbulence_suppressor.py` | FNO turbulence controller | §5 |
| `core/stability_mhd.py` | 7-criterion MHD stability | §6 |
| `control/spi_mitigation.py` | Thermal/current quench, SPI | §7 |
| `control/halo_re_physics.py` | Halo currents, RE ensemble | §7, §13 |
| `control/fokker_planck_re.py` | Kinetic RE Fokker-Planck | §13 |
| `core/eped_pedestal.py` | EPED pedestal model | §8 |
| `core/mhd_sawtooth.py` | Kadomtsev sawtooth | §8 |
| `control/spi_ablation.py` | Multi-fragment SPI ablation | §9 |
| `nuclear/pwi_erosion.py` | Eckstein sputtering | §12 |
| `nuclear/blanket_neutronics.py` | 1D/3-group neutron TBR | §15 |
| `core/fusion_ignition_sim.py` | Bosch-Hale, Q-factor | §15 |
| `core/equilibrium_3d.py` | VMEC-lite 3D equilibrium | §16 |
| `core/fieldline_3d.py` | 3D field-line tracing | §16 |
| `core/scaling_laws.py` | IPB98(y,2) confinement | Scaling |
| `core/rf_heating.py` | ICRH ray-tracing | §11 |
| `core/divertor_thermal_sim.py` | Eich SOL heat width | §12 |
| `core/stability_analyzer.py` | Nyquist, Alfven continuum | §14 |
| `core/hall_mhd_discovery.py` | Non-ideal Hall MHD | §8 |

---

## References

[1] H. Grad and H. Rubin, "Hydromagnetic Equilibria and Force-Free Fields," *Proc. 2nd UN Conf. Peaceful Uses Atomic Energy*, vol. 31, pp. 190–197 (1958).

[2] V.D. Shafranov, "Plasma Equilibrium in a Magnetic Field," *Rev. Plasma Phys.* 2, 103–151 (1966).

[3] ITER Physics Basis Editors et al., "Chapter 2: Plasma confinement and transport," *Nucl. Fusion* 39, 2175–2249 (1999). doi:[10.1088/0029-5515/39/12/302](https://doi.org/10.1088/0029-5515/39/12/302).

[4] G. Verdoolaege et al., "The updated ITPA global H-mode confinement database," *Nucl. Fusion* 61, 076006 (2021). doi:[10.1088/1741-4326/abdb91](https://doi.org/10.1088/1741-4326/abdb91).

[5] Z. Li et al., "Fourier Neural Operator for Parametric PDEs," *ICLR* (2021). arXiv:[2010.08895](https://arxiv.org/abs/2010.08895).

[6] J.D. Huba, *NRL Plasma Formulary*, NRL/PU/6790–19-640 (2019).

[7] B.B. Kadomtsev, "Disruptive instability in tokamaks," *Sov. J. Plasma Phys.* 1, 389–391 (1975).

[8] P.B. Parks and R.J. Turnbull, "Effect of transonic flow in the ablation cloud on the lifetime of a solid hydrogen pellet in a plasma," *Phys. Fluids* 21, 1735 (1978). doi:[10.1063/1.862088](https://doi.org/10.1063/1.862088).

[9] J.W. Connor and R.J. Hastie, "Relativistic limitations on runaway electrons," *Nucl. Fusion* 15, 415–424 (1975). doi:[10.1088/0029-5515/15/3/007](https://doi.org/10.1088/0029-5515/15/3/007).

[10] M.N. Rosenbluth and S.V. Putvinski, "Theory for avalanche of runaway electrons in tokamaks," *Nucl. Fusion* 37, 1355–1362 (1997). doi:[10.1088/0029-5515/37/10/I03](https://doi.org/10.1088/0029-5515/37/10/I03).

[11] H.M. Smith et al., "Hot tail runaway electron generation in tokamak disruptions," *Phys. Plasmas* 15, 072502 (2008). doi:[10.1063/1.2949692](https://doi.org/10.1063/1.2949692).

[12] P. Aleynikov and B.N. Breizman, "Generation of Runaway Electrons during the Thermal Quench," *Phys. Rev. Lett.* 114, 155001 (2015). doi:[10.1103/PhysRevLett.114.155001](https://doi.org/10.1103/PhysRevLett.114.155001).

[13] L. Hesslow et al., "Generalized collision operator for fast electrons interacting with partially ionized impurities," *J. Plasma Phys.* 85, 475850601 (2019). doi:[10.1017/S0022377819000874](https://doi.org/10.1017/S0022377819000874).

[14] O. Sauter et al., "Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime," *Phys. Plasmas* 6, 2834 (1999). doi:[10.1063/1.873240](https://doi.org/10.1063/1.873240).

[15] C.S. Chang and F.L. Hinton, "Effect of finite aspect ratio on the neoclassical ion thermal conductivity in the banana regime," *Phys. Fluids* 25, 1493 (1982). doi:[10.1063/1.863934](https://doi.org/10.1063/1.863934).

[16] P.B. Snyder et al., "A first-principles predictive model of the pedestal height and width," *Phys. Plasmas* 16, 056118 (2009). doi:[10.1063/1.3122146](https://doi.org/10.1063/1.3122146).

[17] P.B. Snyder et al., "Pedestal stability comparison and ITER pedestal prediction," *Nucl. Fusion* 51, 103016 (2011). doi:[10.1088/0029-5515/51/10/103016](https://doi.org/10.1088/0029-5515/51/10/103016).

[18] F. Troyon et al., "MHD-limits to plasma confinement," *Plasma Phys. Control. Fusion* 26, 209 (1984). doi:[10.1088/0741-3335/26/1A/319](https://doi.org/10.1088/0741-3335/26/1A/319).

[19] R.J. La Haye, "Neoclassical tearing modes and their control," *Phys. Plasmas* 13, 055501 (2006). doi:[10.1063/1.2180747](https://doi.org/10.1063/1.2180747).

[20] J.W. Connor, R.J. Hastie, and J.B. Taylor, "Shear, Periodicity, and Plasma Ballooning Modes," *Phys. Rev. Lett.* 40, 396 (1978). doi:[10.1103/PhysRevLett.40.396](https://doi.org/10.1103/PhysRevLett.40.396).

[21] T. Eich et al., "Scaling of the tokamak near SOL H-mode power width," *Nucl. Fusion* 53, 093031 (2013). doi:[10.1088/0029-5515/53/9/093031](https://doi.org/10.1088/0029-5515/53/9/093031).

[22] H.P. Bosch and G.M. Hale, "Improved formulas for fusion cross-sections and thermal reactivities," *Nucl. Fusion* 32, 611 (1992). doi:[10.1088/0029-5515/32/4/I07](https://doi.org/10.1088/0029-5515/32/4/I07).

[23] R. Fitzpatrick, "Halo current and error field interaction," *Phys. Plasmas* 9, 3459 (2002). doi:[10.1063/1.1491955](https://doi.org/10.1063/1.1491955).

[24] K.L. van de Plassche et al., "Fast modeling of turbulent transport using neural networks," *Phys. Plasmas* 27, 022310 (2020). doi:[10.1063/1.5134126](https://doi.org/10.1063/1.5134126).

[25] R.E. Waltz, "Toroidal gyro-Landau fluid model turbulence simulations," *Phys. Plasmas* 1, 2229 (1994). doi:[10.1063/1.870934](https://doi.org/10.1063/1.870934).

[26] T. Pütterich et al., "Calculation and experimental test of the cooling factor of tungsten," *Nucl. Fusion* 50, 025012 (2010). doi:[10.1088/0029-5515/50/2/025012](https://doi.org/10.1088/0029-5515/50/2/025012).

[27] J.P. Freidberg, *Ideal MHD*, Cambridge University Press (2014).

[28] E. Frassinetti et al., "The EUROfusion JET-ILW pedestal database," *Nucl. Fusion* 61, 016001 (2021). doi:[10.1088/1741-4326/abb79e](https://doi.org/10.1088/1741-4326/abb79e).

[29] A.M. Dimits et al., "Comparisons and physics basis of tokamak transport models," *Phys. Plasmas* 7, 969 (2000). doi:[10.1063/1.873896](https://doi.org/10.1063/1.873896).

[30] J. Citrin et al., "Real-time capable first-principles based modelling of tokamak turbulent transport," *Nucl. Fusion* 55, 092001 (2015). doi:[10.1088/0029-5515/55/9/092001](https://doi.org/10.1088/0029-5515/55/9/092001).
