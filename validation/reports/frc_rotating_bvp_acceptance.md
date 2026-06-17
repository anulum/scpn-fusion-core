# FRC Rotating BVP Acceptance

- Generated: `2026-06-16T23:59:55+00:00`
- Status: `blocked_rotating_bvp_reference_missing_fail_closed_contract_passed`
- Accepted full-fidelity rotating BVP: `False`
- Fail-closed contract passed: `True`
- Python: `3.12.3`
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`

## Contract Checks

| Check | Result | Evidence |
|---|:---:|---|
| Python status | `blocked_missing_verified_steinhauer_rotating_closure` | `raise_not_implemented_for_nonzero_theta_dot` |
| Python no-rotation solve | `True` | residual `0.000000e+00`, s `2.107790e+01` |
| Python nonzero rotation | `True` | `NotImplementedError` |
| Rust status | `True` | `blocked_missing_verified_steinhauer_rotating_closure` |
| Steinhauer reference gate | `True` | `blocked_by_publisher_http_403` |

## Missing Requirements

- Steinhauer 2011 Section II.B plus Figure 3 closure
- machine-readable Steinhauer rotating-BVP reference profile
- Python/Rust rotating-BVP parity after verified closure lands
- same-case rotating-BVP benchmark evidence before any acceleration claim

The accepted production contract remains the Steinhauer no-rotation analytical FRC equilibrium. Nonzero `theta_dot` remains fail-closed until the missing reference requirements are satisfied.
