<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — FRC Tilt-Mode Diagnostics -->

# FRC n=1 Tilt-Mode Diagnostics

The FUS-C.5 MIF lane exposes a conservative n=1 FRC tilt diagnostic across
Python and Rust. The accepted public contract is intentionally bounded:

- accepted: MHD Alfvén-time growth scaling for a supplied FRC equilibrium;
- accepted: Steinhauer `s` parameter reuse from the validated FRC equilibrium;
- accepted: rigid-body `s / E` threshold diagnostics;
- blocked: full Belova hybrid eigenvalue parity and Belova Table I
  reproduction until a redistributable digitised reference exists.

## Equations

The diagnostic computes the peak-density Alfvén speed

$$V_A = \frac{B_{\rm ref}}{\sqrt{\mu_0 \rho_m}},$$

with `B_ref = max(abs(B_z))` and
`rho_m = n_peak * m_i`. For a prolate FRC with elongation `E`, the axial
half-length contract is

$$Z_s = E R_s.$$

The MHD tilt growth estimate follows the Belova-normalised Alfvén-time form

$$\gamma_{\rm tilt} = C\frac{V_A}{Z_s},$$

where the default coefficient is `C = 1.2`. This is a diagnostic scaling, not
the unresolved hybrid eigenvalue solver.

The rigid-body FLR diagnostic reports

$$s/E,$$

using the Steinhauer `s` parameter already carried by the accepted FRC
equilibrium. The default threshold labels are:

| Regime | Threshold |
|---|---:|
| diamagnetic FLR diagnostic | `s / E <= 1.7` |
| gyroviscous FLR diagnostic | `s / E <= 2.2` |
| combined FLR diagnostic | `s / E <= 2.8` |
| MHD susceptible diagnostic | `s / E > 2.8` |

These thresholds are reported as diagnostics only. The public stability
boolean remains fail-closed until Belova same-case parity evidence is present.

## Pulsed-compression trajectory adapter

The FUS-C.5 diagnostic is now wired into the accepted FUS-C.6 supplied-current
compression trajectory. The adapter consumes ordered `PulsedCompressionState`
samples and recomputes the Alfvén-time growth rate from instantaneous
`R_s`, `B_ext`, and density. Because FUS-C.6 states do not carry a full radial
equilibrium profile at every time step, the Steinhauer `s` number is projected
with the self-similar ion-gyroradius scaling

$$s(t)=s_0\frac{R_s(t)}{R_s(0)}
\frac{B_{\rm ext}(t)}{B_{\rm ext}(0)}
\sqrt{\frac{T_i(0)}{T_i(t)}}.$$

In code this is a product of the three factors. The trajectory adapter remains
a diagnostic coupling contract, not a Belova hybrid eigenvalue replacement.

The adapter also integrates the diagnostic growth exposure along the supplied
trajectory:

```text
G(t) = integral gamma_tilt dt
amplification = exp(G)
```

`FRCTiltModeTrajectoryPoint` reports `cumulative_growth_integral`,
`perturbation_amplification`, and `amplification_overflow_limited`. If the
diagnostic exposure exceeds finite `float64` amplification range, the
amplification is limited to a finite value while the cumulative integral keeps
the actual e-folding count.

## Public API

```python
from scpn_fusion.core.tilt_mode_frc import (
    frc_tilt_growth_rate,
    rigid_body_flr_regime,
    tilt_mode_report,
    tilt_mode_stable,
    tilt_mode_trajectory_from_pulsed_compression,
)

growth = frc_tilt_growth_rate(eq, elongation=4.0)
report = tilt_mode_report(eq, elongation=4.0)
stable, growth = tilt_mode_stable(eq, elongation=4.0)
trajectory = tilt_mode_trajectory_from_pulsed_compression(states, eq, elongation=4.0)
```

`tilt_mode_stable()` returns `False` while the external parity row is blocked.
Downstream safety logic should consume the reported growth rate and regime as
preemption diagnostics, not as an accepted stability proof.

## Validation and benchmarks

Validation surfaces:

- `tests/test_tilt_mode_frc.py`
- Rust `fusion_physics::tilt_mode_frc` unit tests
- `benchmarks/bench_tilt_mode_frc.py`
- `validation/reports/tilt_mode_frc_benchmark.json`

The benchmark report is local non-isolated regression evidence. It records
Python rows, Rust Criterion rows, FUS-C.6 coupled trajectory rows, source
checksums, and a blocked external row for
`belova_2001_table1_tilt_stability`.
Schema `scpn-fusion-core.tilt_mode_frc_benchmark.v4` records the final
trajectory growth integral and finite amplification diagnostic for coupled
rows.

## Evidence boundary

The blocked external row requires a digitised Belova Table I or equivalent
hybrid-eigenvalue reference with provenance, checksum, and matching equilibrium
metadata. Until that artifact exists, the code must not report accepted Belova
same-case parity.

References:

- Belova et al., *Numerical study of tilt stability of prolate field-reversed
  configurations*, Physics of Plasmas 8, 1267 (2001),
  PPPL-3456: <https://bp-pub.pppl.gov/pub_report/2000/PPPL-3456.pdf>
- Steinhauer, *Review of field-reversed configurations*, Physics of Plasmas 18,
  070501 (2011).
