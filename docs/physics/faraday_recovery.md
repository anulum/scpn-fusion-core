# FRC Faraday recovery

This page documents the FUS-C.7 classical Faraday recovery surface for MIF/FRC
trajectories. The implementation is a closed-form electromagnetic calculation
over supplied trajectory samples. It now includes an explicit adapter from the
implemented FUS-C.6 supplied-current and voltage-driven pulsed-compression
trajectories into Faraday trajectory samples, plus compression-work and
coil-source-work sidecars. It also consumes the FUS-C.6 flux-budget sidecar so
Faraday rows can fail closed when the upstream non-adiabatic flux carrier did
not close. It is not a substitute for external Slough same-case acceptance
data.

## Governing equations

The linked flux per turn is:

```text
Phi = B_ext(t) * pi * R_s(t)^2
```

The recovery-coil back-EMF is:

```text
EMF = -N_turns * pi * (R_s^2 * dB_ext/dt + 2 * B_ext * R_s * dR_s/dt)
```

The report now independently checks the sampled Faraday-law closure:

```text
finite_difference(Phi) + EMF / N_turns = 0
```

This closure is evaluated from the flux samples and reported as
`flux_derivative_residual_linf`, `flux_derivative_residual_l2`, and
`flux_derivative_closure_passed`. It is intentionally separate from the
compression-work and source-work budget gates because it checks trajectory and
derivative-sidecar consistency, not external facility acceptance.
FUS-C.6 coupled rows therefore publish the residual honestly even when the
upstream compression derivative sidecars do not meet the stricter sampled-flux
closure tolerance.

For a resistive recovery load:

```text
I_load = EMF / R_load
P_load = EMF^2 / R_load
E_recovered = integral P_load dt
```

## Public API

Python:

```python
from scpn_fusion.core import (
    compression_flux_budget_from_pulsed_compression,
    compression_flux_budget_from_voltage_driven_compression,
    FaradayRecoveryTrajectoryPoint,
    coil_source_work_from_voltage_driven_compression,
    compression_work_from_pulsed_compression,
    compression_work_from_voltage_driven_compression,
    faraday_back_emf,
    faraday_trajectory_from_pulsed_compression,
    faraday_trajectory_from_voltage_driven_compression,
    integrated_recovery_energy,
)

emf_v = faraday_back_emf(
    lambda t: 0.20 + 4.0e3 * t,
    lambda t: 20.0,
    6,
    1.0e-6,
)

report = integrated_recovery_energy(
    [
        FaradayRecoveryTrajectoryPoint(t_s=0.0, separatrix_radius_m=0.20, b_ext_t=20.0),
        FaradayRecoveryTrajectoryPoint(t_s=1.0e-6, separatrix_radius_m=0.204, b_ext_t=20.0),
    ],
    N_turns=6,
    coil_resistance_ohm=0.08,
)
```

When FUS-C.6 compression states are available:

```python
trajectory = faraday_trajectory_from_pulsed_compression(compression_states)
compression_work_j = compression_work_from_pulsed_compression(compression_states)
compression_flux_budget = compression_flux_budget_from_pulsed_compression(compression_states)
report = integrated_recovery_energy(
    trajectory,
    N_turns=recovery_turns,
    coil_resistance_ohm=recovery_resistance_ohm,
    compression_work_j=compression_work_j,
    compression_flux_budget=compression_flux_budget,
)
```

When a voltage-driven FUS-C.6 result is available:

```python
trajectory = faraday_trajectory_from_voltage_driven_compression(result)
compression_work_j = compression_work_from_voltage_driven_compression(result)
coil_source_work_j = coil_source_work_from_voltage_driven_compression(result)
compression_flux_budget = compression_flux_budget_from_voltage_driven_compression(result)
report = integrated_recovery_energy(
    trajectory,
    N_turns=recovery_turns,
    coil_resistance_ohm=recovery_resistance_ohm,
    compression_work_j=compression_work_j,
    coil_source_work_j=coil_source_work_j,
    compression_flux_budget=compression_flux_budget,
)
```

Rust:

```rust
use fusion_physics::faraday_recovery::{
    faraday_back_emf_from_values,
    integrated_recovery_energy,
    FaradayRecoveryTrajectoryPoint,
};

let emf = faraday_back_emf_from_values(0.20, 20.0, 4.0e3, 0.0, 6)?;
let trajectory = vec![
    FaradayRecoveryTrajectoryPoint {
        t_s: 0.0,
        separatrix_radius_m: 0.20,
        b_ext_t: 20.0,
        d_radius_dt_m_s: None,
        d_b_ext_dt_t_s: None,
    },
    FaradayRecoveryTrajectoryPoint {
        t_s: 1.0e-6,
        separatrix_radius_m: 0.204,
        b_ext_t: 20.0,
        d_radius_dt_m_s: None,
        d_b_ext_dt_t_s: None,
    },
];
let report = integrated_recovery_energy(&trajectory, 6, 0.08, None, None, None, 0.01)?;
```

## Evidence boundary

The implementation accepts supplied trajectory samples from FUS-C.6
pulsed-compression states, voltage-driven FUS-C.6 results, or an external
validated trajectory. If the caller does not provide a self-consistent
compression-work value, the energy-budget status is
`blocked_missing_compression_work`. If the caller does not provide a
coil-source-work sidecar, the source-budget status is
`blocked_missing_coil_source_work`. If the caller does not provide the FUS-C.6
flux-budget sidecar, the compression-flux budget status is
`blocked_missing_compression_flux_budget`. When sidecars are supplied, each
budget is evaluated as `passed` or `failed`; failure is a real
load/trajectory/source/flux mismatch, not a placeholder blocked row.

This avoids fabricating Slough-style trajectory acceptance from a synthetic
radius trace.

## Validation evidence

Tracked tests cover:

- Zero EMF for constant field and radius.
- Closed-form constant-field radial expansion.
- Callable finite-difference agreement for linear histories.
- Integrated recovery energy against an analytical linear-radius integral.
- Explicit flux-derivative closure for the Faraday identity
  `dPhi/dt + EMF/N_turns = 0`.
- Fail-closed detection of inconsistent supplied derivative sidecars.
- Explicit blocked budget status when compression work is absent.
- FUS-C.6 supplied-current trajectory conversion into Faraday samples and
  evaluated compression-work and compression-flux sidecar status.
- FUS-C.6 voltage-driven trajectory conversion into Faraday samples and
  evaluated coil-source-work and compression-flux sidecar status.
- Python and Rust fail-closed invalid-input paths.

Tracked report:

- `validation/reports/faraday_recovery_benchmark.json`

Benchmark commands:

```bash
cargo bench -p fusion-physics --bench faraday_recovery_bench
PYTHONPATH=src python benchmarks/bench_faraday_recovery.py
```

The committed benchmark report is local non-isolated regression evidence, not a
production throughput claim.

## Not yet accepted

Full external FUS-C.7 acceptance still requires a public Slough-style or
facility trajectory with compression-work and source-work sidecars, provenance,
checksums, and compatible upstream flux-budget evidence. The internal FUS-C.6
supplied-current and voltage-driven trajectory paths are now evaluated directly
and no longer marked missing when those states are supplied.
