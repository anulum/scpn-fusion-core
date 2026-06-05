# MIF/FRC pulsed compression

This page documents the FUS-C.6 supplied-current pulsed-compression surface.
The implementation evolves a pressure-driven FRC separatrix radius, adiabatic
temperature and density, an energy-accounting residual, and the existing Ono
non-adiabatic flux carrier.

## Governing contract

The external compression field uses the uniform-solenoid approximation:

```text
B_ext(t) = mu0 * N_turns * I_coil(t) / L_coil
```

The radial force is assembled from the internal scalar pressure and external
magnetic pressure:

```text
p_internal = n * (T_i + T_e) * e
p_external = B_ext^2 / (2 * mu0)
F_R = (p_internal - p_external) * 2*pi*R_s*L_plasma
a_R = F_R / m_plasma
```

The compression thermodynamics use the ideal adiabatic invariant:

```text
T * V^(gamma - 1) = constant
V = pi * R_s^2 * L_plasma
```

The state also advances the poloidal-flux profile through the already accepted
non-adiabatic carrier:

```text
dpsi/dt = -psi/tau_psi + R_null*E_theta - eta_spitzer*J_theta
```

The trajectory carries the compression-work sidecar consumed by FUS-C.7
Faraday recovery:

```text
W_compression(t_n) = sum_k (E_thermal,adiabatic,k - E_thermal,k-1)
```

FUS-C.7 maps each state to `(t, R_s, B_ext, dR_s/dt)` and compares recovered
load energy against the final `compression_work_J` when that sidecar is
provided.

## Public API

Python:

```python
from scpn_fusion.core import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
)

cfg = PulsedCompressionConfig(
    equilibrium=frc_state,
    coil=CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    ),
    coil_current_t=lambda t: 5.0e5,
    plasma_mass_kg=2.0e-5,
    ion_temperature_eV=10_000.0,
    electron_temperature_eV=5_000.0,
)

initial = initial_pulsed_compression_state(cfg)
trajectory = run_pulsed_compression(initial, cfg, dt_s=1.0e-9, n_steps=256)
```

Rust:

```rust
use fusion_physics::compression::{
    run_pulsed_compression,
    CoilGeometry,
    PulsedCompressionConfig,
    PulsedCompressionState,
};
```

## Evidence boundary

The implementation is complete for the supplied-current pressure-balance and
adiabatic-compression contract in the internal MIF lane. It is wired to the
existing Ono flux carrier.

It does not claim Slough 2011 Fig. 5 parity yet. That acceptance row is
blocked until a public digitised reference trajectory exists with provenance
and checksums. The tracked benchmark report records this explicitly.

## Validation evidence

Tracked tests cover:

- uniform-solenoid coil-field mapping,
- adiabatic invariant preservation,
- radial acceleration under pressure imbalance,
- external compression heating and shrinkage,
- flux-history propagation,
- FUS-C.7 compression-work sidecar consumption through Faraday recovery,
- Spitzer resistivity scaling,
- fail-closed invalid-input paths.

Tracked report:

- `validation/reports/pulsed_compression_benchmark.json`

Benchmark commands:

```bash
cargo bench -p fusion-physics --bench pulsed_compression_bench
PYTHONPATH=src python benchmarks/bench_pulsed_compression.py
```

The committed timings are local non-isolated regression evidence only.
