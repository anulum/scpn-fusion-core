# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 1.0.2   | :white_check_mark: | Current stable — PyPI + GitHub release |
| 1.0.1   | :x:                | Superseded (missing license metadata) |
| 1.0.0   | :x:                | Superseded (initial release) |
| < 1.0   | :x:                | Pre-release / unreleased |

Only the latest `1.0.x` patch receives security fixes. Upgrade with:

```bash
pip install --upgrade scpn-fusion
```

## Reporting a Vulnerability

If you discover a security vulnerability in SCPN Fusion Core, please report it
responsibly:

1. **Email:** protoscience@anulum.li
2. **Subject:** `[SECURITY] SCPN Fusion Core — <brief description>`
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

SCPN Fusion Core is a simulation library. It does not handle user
authentication, financial data, or network services in its default
configuration. Security concerns are primarily:

- Malicious input files (JSON configs, GEQDSK equilibria, NumPy `.npz`)
- Unsafe deserialization (serde, pickle, NumPy load)
- Numerical overflow / denial of service via pathological inputs
- Native code memory safety (Rust crates via PyO3)
- Supply chain integrity (dependency audit)

## Hardening Measures in Place

### Input Validation (v1.0.2)
Over 30 hardening commits add runtime guards across all physics and control
modules: array shape/dtype checks, non-finite rejection, range clamping,
and constructor parameter validation. See the git log for commits prefixed
with `Harden`.

### Dependency Auditing
- **Rust:** `cargo audit` runs in CI on every push (added in commit `a582ef13`).
  Known advisory RUSTSEC-2025-0020 (pyo3 <0.24) was patched by upgrading to
  pyo3 0.24.
- **Python:** Dependencies are minimal (`numpy`, `scipy`, `matplotlib`,
  `streamlit`). No `pickle.load` of untrusted data in any module.

### RNG Isolation
Global NumPy RNG state is never mutated by library code. All stochastic
modules use scoped `numpy.random.Generator` instances seeded explicitly,
preventing cross-module interference. See commits prefixed with `Scope`
and `Stop global RNG mutation`.

## Known Limitations

- **No fuzzing harness yet.** Property-based testing via Hypothesis covers
  many input paths, but dedicated fuzzing (e.g., `cargo-fuzz` for Rust,
  `atheris` for Python) has not been set up.
- **No third-party security audit.** The codebase has not been reviewed by
  an external security firm.
- **No CVE history.** No vulnerabilities have been reported to date.

Contributions to improve security coverage (fuzzing harnesses, static
analysis integration, audit reports) are welcome.

## Disclosure Timeline

| Date | Event |
|------|-------|
| 2026-02-12 | v1.0.0 initial release |
| 2026-02-12 | pyo3 0.23 → 0.24 security upgrade (RUSTSEC-2025-0020) |
| 2026-02-12 | `cargo audit` added to CI |
| 2026-02-13–14 | Input hardening sprint (30+ commits) |
| 2026-02-14 | v1.0.2 released with full license metadata |
