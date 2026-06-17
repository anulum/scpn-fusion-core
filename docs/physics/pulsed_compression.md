# MIF/FRC pulsed compression


## Context and scope

This note captures the current pulsed-compression modeling contract in the project, including active assumptions, module coverage, and links to the validation commands that keep results reproducible.

This page documents the FUS-C.6 supplied-current and voltage-driven
pulsed-compression surface. The implementation evolves a pressure-driven FRC
separatrix radius, adiabatic temperature and density, an energy-accounting
residual, an exact lumped R-L coil-current path for declared bank voltage, and
the existing Ono non-adiabatic flux carrier with explicit source, damping, and
update-residual diagnostics.

## Governing contract

The external compression field uses the uniform-solenoid approximation:

```text
B_ext(t) = mu0 * N_turns * I_coil(t) / L_coil
```

When a caller supplies a voltage drive instead of a current function, the coil
current is advanced by the exact constant-voltage solution of the declared
lumped circuit over each interval:

```text
L_coil * dI_coil/dt + R_coil * I_coil = V_bank(t)
|V_bank(t)| <= bank_voltage_max
```

The circuit state records magnetic energy, source work, Ohmic loss, and a
normalised energy residual. This is a bank-limited lumped coil-circuit
contract, not a 3D coil electromagnetic or liner-circuit field solve.

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

Each compression step records the discrete carrier closure:

```text
psi[n+1] = psi[n] - damping_decrement[n] + source_increment[n]
update_residual[n] = psi[n+1] - expected_psi[n+1]
```

The public state exposes the flux checksum, source-increment checksum,
damping-decrement checksum, maximum absolute update residual, and a
`flux_budget_claim_status` gate. This prevents downstream Faraday, MRTI, and
tilt diagnostics from consuming an opaque flux checksum without knowing whether
the underlying carrier update actually closed.

The public state also exposes `radial_acceleration_m_s2` from the same
force-balance evaluation that advances `dR_s/dt`. This gives downstream MRTI,
Faraday, and tilt adapters a finite acceleration diagnostic tied to the
implemented pressure-balance step instead of forcing each lane to reconstruct
acceleration from differenced velocities.

The trajectory carries the compression-work sidecar consumed by FUS-C.7
Faraday recovery:

```text
W_compression(t_n) = sum_k (E_thermal,adiabatic,k - E_thermal,k-1)
```

FUS-C.7 maps each state to `(t, R_s, B_ext, dR_s/dt)` and compares recovered
load energy against the final `compression_work_J` when that sidecar is
provided.

The public trajectory diagnostics contract aggregates the same state history
without changing the step physics:

```text
min_radius = min(R_s[n])
compression_ratio = R_s[0] / min_radius
max_abs_acceleration = max(|a_R[n]|)
```

It also reports whether time is strictly increasing, how many samples touched a
declared radius floor, how many radial turning points occurred, and whether every
post-initial flux-budget row passed. Non-finite states, non-monotonic time, and
invalid radius floors raise instead of emitting ambiguous summary rows.

## Public API

Python:

```python
from scpn_fusion.core import (
    CoilGeometry,
    PulsedCompressionConfig,
    pulsed_compression_trajectory_diagnostics,
    initial_pulsed_compression_state,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
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
diagnostics = pulsed_compression_trajectory_diagnostics(
    trajectory,
    radius_floor_m=cfg.min_radius_m,
)
voltage_driven = run_voltage_driven_pulsed_compression(
    cfg,
    lambda t: 20_000.0,
    dt_s=1.0e-9,
    n_steps=256,
    initial_current_A=5.0e5,
)
```

Rust:

```rust
use fusion_physics::compression::{
	    run_pulsed_compression,
	    run_voltage_driven_pulsed_compression,
	    pulsed_compression_trajectory_diagnostics,
    CoilGeometry,
    PulsedCompressionConfig,
    PulsedCompressionState,
};
```

## Evidence boundary

The implementation is complete for the supplied-current pressure-balance,
exact lumped R-L voltage-drive, and adiabatic-compression contract in the
internal MIF lane. It is wired to the existing Ono flux carrier and records
the discrete flux-source/damping budget at every step in Python and Rust.

It does not claim Slough 2011 Fig. 5 parity yet. That acceptance row is
blocked until a public digitised reference trajectory exists with provenance,
checksums, and same-case pulsed-compression parity. Reconstructed operational
sidecars and digitised-looking sidecars without checksum/parity evidence remain
explicit blocked states. The tracked benchmark report records this boundary.

The public C-2U positive-net-heating table from Baltz et al., Scientific
Reports 7, 6425 (2017), is tracked separately under
`validation/reference_data/frc_public/`. It supports bounded FRC performance
context only. It is not Slough 2011 Fig. 5 trajectory data and does not satisfy
the pulsed-compression trajectory parity gate.

## Validation evidence

Tracked tests cover:

- uniform-solenoid coil-field mapping,
- adiabatic invariant preservation,
- radial acceleration under pressure imbalance,
- finite force-balance acceleration exposure,
- trajectory-level min-radius, acceleration, floor-contact, turning-point,
  compression-ratio, and flux-budget diagnostics,
- exact lumped R-L coil-current trajectory and circuit-energy residual,
- external compression heating and shrinkage,
- voltage-driven coil-current coupling into pulsed compression,
- flux-history propagation and flux-budget closure diagnostics,
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
