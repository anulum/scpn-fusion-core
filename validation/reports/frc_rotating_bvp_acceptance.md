# FRC Rotating Rigid-Rotor Acceptance

- Generated: `2026-07-01T13:33:13+00:00`
- Status: `implemented_rostoker_qerushi_rotating_closure_accepted`
- Accepted rotating closure: `True`
- Python: `3.12.3`
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`

## Contract Checks

| Check | Result | Evidence |
|---|:---:|---|
| Python status | `implemented_rostoker_qerushi_1d_rotating_closure` | implemented=`True` |
| No-rotation contract | `True` | residual `0.000e+00`, s `2.108e+01` |
| Rotating equilibrium | `True` | Mach `0.071`, rot-FB `2.157e-03` |
| Reduces to contract (omega^2) | `True` | ratios `[100.0, 100.0]` |
| Rust parity | `True` | `implemented_rostoker_qerushi_1d_rotating_closure` |
| Steinhauer Fig. 3 boundary | `True` | parity-claimed=`False` |

## Claim Boundary

The rotating rigid-rotor equilibrium solves the source-verified Rostoker & Qerushi (2002) one-dimensional one-ion centrifugal force balance d/dr[p + B_z^2/(2 mu_0)] = rho omega^2 r with the rigid-rotor density closure, reducing bit-exactly to the accepted Steinhauer no-rotation contract at theta_dot == 0. Verbatim Steinhauer 2011 Figure 3 digitised parity is NOT claimed and remains a separate external-parity gate; C-2U performance/topology references are context only, not a figure-parity certification.

