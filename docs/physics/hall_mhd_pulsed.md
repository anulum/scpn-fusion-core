<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Pulsed Hall-MHD -->

# Axisymmetric Pulsed Hall-MHD


## Context

This page records pulsed Hall-MHD use assumptions and model coverage so readers can distinguish supported use cases from conceptual extensions before interpreting results as production evidence.

FUS-C.2 now exposes an accepted axisymmetric Hall-MHD flux carrier across
Python and Rust. The implemented contract is the flattened Ono Eq. 8 form:

$$
\frac{\partial \psi}{\partial t} =
-\frac{\psi}{\tau_\psi} + R_{\rm null}E_\theta - \eta_{\rm Spitzer}J_\theta.
$$

It is a production surface for MIF/FRC trigger and replay workflows that need a
strict flux-state contract. It is not a claim of full 2D two-fluid Hall-MHD,
Gkeyll/BOUT++ same-case parity, or Ono figure reproduction.

## Field drive

When no explicit `E_theta_t` is supplied, the module derives the circular-loop
Faraday drive from the external axial-field ramp:

$$E_\theta(r,t) = -\frac{r}{2}\frac{dB_{\rm ext}}{dt}.$$

The axial field diagnostic is reconstructed from the cylindrical flux:

$$B_z(r)=\frac{1}{r}\frac{d\psi}{dr},$$

with finite-axis handling at `r = 0`.

## Resistivity

The Spitzer term uses the NRL-style temperature scaling:

$$\eta_{\rm Spitzer}=1.65\times 10^{-9}
\frac{Z_{\rm eff}\ln\Lambda}{T_e[\mathrm{eV}]^{3/2}}\;\Omega\,\mathrm{m}.
$$

## Integrator

The default integrator treats damping implicitly and the Ono source explicitly:

$$\psi_{n+1} =
\frac{\psi_n + \Delta t\left(R_{\rm null}E_{\theta,n+1}
- \eta J_{\theta,n+1}\right)}
{1+\Delta t/\tau_\psi}.
$$

An IMEX RK2 source midpoint path is also available for smooth prescribed
sources. Both paths report source-residual diagnostics.

## Public API

```python
from scpn_fusion.core.hall_mhd_pulsed import (
    HallMHDPulsedConfig,
    initial_hall_mhd_pulsed_state,
    run_hall_mhd_pulsed,
)

config = HallMHDPulsedConfig(
    equilibrium=eq,
    B_ext_t=lambda t: 5.0,
    tau_psi_s=5.0e-6,
    electron_temperature_eV=5_000.0,
)
state = initial_hall_mhd_pulsed_state(config)
trajectory = run_hall_mhd_pulsed(state, config, dt_s=1.0e-8, n_steps=256)
```

## Validation and benchmarks

Validation surfaces:

- `tests/test_hall_mhd_pulsed.py`
- Rust `fusion_physics::hall_mhd_pulsed` unit tests
- `benchmarks/bench_hall_mhd_pulsed.py`
- `validation/reports/hall_mhd_pulsed_benchmark.json`

The tracked benchmark report includes Python rows, Rust Criterion rows, source
checksums, and blocked external rows:

- `gkeyll_axisymmetric_small_hall`:
  `blocked_missing_public_same_case_reference`
- `ono_1997_fig4_flux_decay`:
  `blocked_missing_public_digitised_reference`

References:

- Ono et al., *Magnetic-mode-based modeling for field-reversed
  configurations*, Physics of Plasmas 4, 1953 (1997).
- NRL Plasma Formulary, Spitzer resistivity scaling.
