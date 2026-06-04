# MRTI growth spectrum

This page documents the public MRTI physics surface added for the MIF/FRC work
lane. The implementation is analytical and deterministic. It is not a learned
surrogate and it does not claim full pulsed-compression Hall-MHD coupling.

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

## Public API

Python:

```python
from scpn_fusion.core import MRTISpectrumTracker, mrti_growth_rate

gamma = mrti_growth_rate([10.0, 40.0], a_eff=1.0e8, B_perp=8.0e-4, rho_kg_m3=1.0e-3)

tracker = MRTISpectrumTracker(k_max_m_inv=1.0e4, n_modes=256)
state = tracker.step(dt_s=2.0e-7, a_eff_m_s2=6.5e6, B_perp_t=8.0e-4)
```

Rust:

```rust
use fusion_physics::mrti::{mrti_growth_rate, MrtiSpectrumTracker};

let gamma = mrti_growth_rate(40.0, 1.0e8, 8.0e-4, 1.0e-3)?;
let mut tracker = MrtiSpectrumTracker::new(1.0e4, 256, 1.0e-9, 1.0e-3, 1.0e-3)?;
let state = tracker.step(2.0e-7, 6.5e6, 8.0e-4)?;
```

## Trajectory coupling boundary

`effective_acceleration_from_radius_rate()` estimates `d²R_s/dt²` from a
separatrix radial-speed history using finite differences and optional
edge-padded smoothing. This is a coupling adapter for a future pulsed
compression trajectory. It does not synthesize the missing Hall-MHD trajectory
and must not be treated as FUS-C.6 acceptance evidence.

## Validation evidence

Tracked tests cover:

- Hydrodynamic `B_perp = 0` limit: `gamma = sqrt(k a_eff)`.
- Magnetic-tension stabilisation for short wavelengths.
- Constant-acceleration exponential amplitude growth.
- First saturation-threshold breach detection.
- Smoothed acceleration recovery from a synthetic radial-speed ramp.
- Python and Rust fail-closed validation for invalid inputs.

Tracked benchmark report:

- `validation/reports/mrti_benchmark.json`

The benchmark report is local, non-isolated regression evidence. Rust rows are
included after running the Criterion benchmark:

```bash
cargo bench -p fusion-physics --bench mrti_bench
PYTHONPATH=src python benchmarks/bench_mrti.py
```

## Explicitly omitted physics

The current MRTI surface does not include:

- Nonlinear MRTI saturation and bubble/spike morphology.
- Full Hall-MHD pulsed-compression trajectory generation.
- Coupled liner/plasma equation-of-state feedback.
- Experimental validation against pulsed-power MRTI image sequences.

Those items remain roadmap work and must be gated separately.

