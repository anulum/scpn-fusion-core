# MRTI growth spectrum

This page documents the public MRTI physics surface added for the MIF/FRC work
lane. The implementation is analytical and deterministic. It is not a learned
surrogate. It now consumes the accepted FUS-C.6 supplied-current pulsed
compression state history, while external nonlinear MRTI image/data parity
remains blocked until redistributable same-case references exist.

## Role in integrated workflows

MRTI is used as a physics-derived instability signal within the MIF/FRC lane.
Its output is consumed for trending and diagnostics, with higher-priority control
actions still gated on explicit interlocks and validated acceptance rows.

When MRTI artifacts are exported, coupling assumptions and source states must be
retained with the same-run manifest so downstream reproducibility checks can be
re-run from the same time grid and initial conditions.

## Scope

The accepted contract evaluates the linear magneto-Rayleigh-Taylor instability
growth rate over a resolved wavenumber spectrum:

```text
gamma^2 = k a_eff - k^2 B_perp^2 / (mu0 rho)
```

where `k` is the perturbation wavenumber in `m^-1`, `a_eff` is the effective
interface acceleration in `m s^-2`, `B_perp` is the stabilising perpendicular
field in tesla, `rho` is the mass density in `kg m^-3`, and `gamma` is the
growth rate in `s^-1`.

Negative radicands are clipped to zero. That represents magnetic-tension
stabilisation for the supplied mode, not nonlinear mode removal.

Perturbation amplitudes are integrated in log space. Each interval adds the
cumulative linear growth exponent, evaluated with the trapezoidal rule from the
interval-endpoint growth rates:

```text
log(A_i) <- log(A_i) + max(0.5 * (gamma_i(start) + gamma_i(end)) * dt, 0)
```

The trapezoidal endpoint integration is second-order accurate in `dt` for a
smoothly varying acceleration or perpendicular field, against a first-order
endpoint-frozen exponent. When the start and end drivers are equal it reduces
exactly to the frozen-coefficient form `log(A_i) <- log(A_i) + max(gamma_i * dt, 0)`,
so the constant-driver `step()` path is unchanged. The Velikovich eq. (18)
dispersion relation is evaluated unchanged at each endpoint; only the temporal
amplitude integration is hardened.

The public state still exposes physical amplitudes in metres, but also reports
`log_amplitudes`, `max_log_amplitude`, and `amplitude_overflow_limited`. If an
extreme-growth trajectory exceeds finite `float64` amplitude range, the
physical amplitude is limited to the largest representable finite value while
the log-amplitude diagnostic preserves the actual integrated exponent.

The spectrum state reports two complementary dominant-mode diagnostics:
`fastest_growing_k_m_inv` is the instantaneous fastest-growing wavenumber at the
current time (argmax of the end-of-interval growth spectrum), while
`most_amplified_k_m_inv` is the cumulatively most-amplified wavenumber (argmax of
the integrated log-amplitude). For constant drivers the two coincide; under a
time-varying compression they can differ, and the cumulative diagnostic is the
relevant signal for pre-empting a separatrix breach.

## Public API

Python:

```python
from scpn_fusion.core import (
    MRTISpectrumTracker,
    mrti_growth_rate,
    track_mrti_from_pulsed_compression,
)

gamma = mrti_growth_rate([10.0, 40.0], a_eff=1.0e8, B_perp=8.0e-4, rho_kg_m3=1.0e-3)

tracker = MRTISpectrumTracker(k_max_m_inv=1.0e4, n_modes=256)
# Constant-driver step (frozen coefficient, first-order):
state = tracker.step(dt_s=2.0e-7, a_eff_m_s2=6.5e6, B_perp_t=8.0e-4)
# Time-varying interval (trapezoidal, second-order) when the acceleration or
# field differs across the interval endpoints:
state = tracker.step_interval(
    dt_s=2.0e-7,
    a_eff_start_m_s2=6.5e6,
    a_eff_end_m_s2=7.1e6,
    B_perp_start_t=8.0e-4,
    B_perp_end_t=7.6e-4,
)
coupled_states = track_mrti_from_pulsed_compression(compression_states, tracker)
```

Rust:

```rust
use fusion_physics::mrti::{
    mrti_growth_rate, track_mrti_from_pulsed_compression, MrtiSpectrumTracker
};

let gamma = mrti_growth_rate(40.0, 1.0e8, 8.0e-4, 1.0e-3)?;
let mut tracker = MrtiSpectrumTracker::new(1.0e4, 256, 1.0e-9, 1.0e-3, 1.0e-3)?;
let state = tracker.step(2.0e-7, 6.5e6, 8.0e-4)?;
let state = tracker.step_interval(2.0e-7, 6.5e6, 7.1e6, 8.0e-4, 7.6e-4)?;
let coupled_states = track_mrti_from_pulsed_compression(
    &compression_states, &mut tracker, 1, -1.0, 1.0
)?;
```

## Trajectory coupling boundary

`effective_acceleration_from_radius_rate()` remains available for external
separatrix radial-speed histories that do not carry a solver acceleration
sidecar. `effective_acceleration_from_pulsed_compression()` consumes the FUS-C.6
`PulsedCompressionState.radial_acceleration_m_s2` field directly, validates
strictly increasing time, positive radius, finite speed, finite acceleration,
and finite field, and applies an explicit radial projection sign. The default
`-1` maps inward compression in the outward `R_s` coordinate to positive MRTI
effective acceleration.

`track_mrti_from_pulsed_compression()` advances one MRTI spectrum interval for
each supplied trajectory step, using the endpoint acceleration and endpoint
external field for magnetic-tension stabilisation. This is internal
same-solver coupling evidence. It is not external nonlinear MRTI saturation
parity and does not substitute for pulsed-power image/diagnostic references.

## Validation evidence

Tracked tests cover:

- Hydrodynamic `B_perp = 0` limit: `gamma = sqrt(k a_eff)`.
- Magnetic-tension stabilisation for short wavelengths.
- Constant-acceleration exponential amplitude growth.
- Exact equivalence of `step_interval` to the frozen-coefficient `step` for
  equal start and end drivers (Python and Rust).
- Second-order accuracy of the trapezoidal interval integration on an analytic
  acceleration ramp, verified against the closed-form cumulative growth exponent
  and against the first-order endpoint-frozen step (Python and Rust).
- Cumulative `most_amplified_k_m_inv` diagnostic tracking the integrated dominant
  mode rather than the instantaneous fastest-growing mode (Python and Rust).
- Trapezoidal interval integration of the FUS-C.6 pulsed-compression coupling.
- Log-amplitude accounting for long or extreme growth trajectories.
- First saturation-threshold breach detection.
- Smoothed acceleration recovery from a synthetic radial-speed ramp.
- FUS-C.6 supplied-current pulsed-compression force-balance acceleration driving
  MRTI spectra.
- Python and Rust fail-closed validation for invalid inputs.

Tracked benchmark report:

- `validation/reports/mrti_benchmark.json`

The benchmark report is local, non-isolated regression evidence. It includes
analytical fixed-acceleration rows and internal FUS-C.6 compression-coupled
rows, including log-amplitude and overflow-limiting diagnostics. Rust rows are
included after running the Criterion benchmark:

```bash
cargo bench -p fusion-physics --bench mrti_bench
PYTHONPATH=src python benchmarks/bench_mrti.py
```

## Explicitly omitted physics

The current MRTI surface does not include:

- Nonlinear MRTI saturation and bubble/spike morphology.
- Coupled liner/plasma equation-of-state feedback.
- Experimental validation against pulsed-power MRTI image sequences.

Those items remain roadmap work and must be gated separately.
