# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 3.1.0   | :white_check_mark: | Current stable — Phase 0 physics hardening |
| 3.0.0   | :white_check_mark: | Previous stable — Rust SNN, full-chain UQ, shot replay |
| 2.0.0   | :x:                | Superseded (multigrid, gyro-Bohm, H-inf controller) |
| 2.1.0   | :x:                | Superseded (GEQDSK, Sauter bootstrap, Spitzer) |
| 1.0.2   | :x:                | Superseded (initial public release) |
| < 1.0   | :x:                | Pre-release / unreleased |

Only the latest `3.x` release receives security fixes. Upgrade with:

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

### Input Validation (v1.0.2 — v3.1.0)
Over 30 hardening commits add runtime guards across all physics and control
modules: array shape/dtype checks, non-finite rejection, range clamping,
and constructor parameter validation. See the git log for commits prefixed
with `Harden`.

### Physics Constraint Enforcement (v3.1.0)
- Greenwald density limit rejects unphysical density points in Q-scan
- Temperature capped at 25 keV with warning emission
- Q factor capped at 15 to prevent 0-D model artifacts
- Energy conservation diagnostic in transport solver with optional `PhysicsError`
- TBR correction factors enforce realistic [1.0, 1.4] range
- CI gates: hard fail on FPR > 15%, TBR outside range, Q > 15

### Dependency Auditing
- **Rust:** `cargo audit` runs in CI on every push (added in commit `a582ef13`).
  Known advisory RUSTSEC-2025-0020 (pyo3 <0.24) was patched by upgrading to
  pyo3 0.24.
- **Python:** Dependencies are minimal (`numpy`, `scipy`, `matplotlib`,
  `streamlit`). No `pickle.load` of untrusted data in any module.
- **Checkpoint hygiene:** disruption-model checkpoint loading requires
  `torch.load(..., weights_only=True)` by default; legacy torch fallback
  deserialization is disabled unless
  `SCPN_ALLOW_INSECURE_TORCH_LOAD=1` is set for trusted checkpoints.
- **Secure NumPy loading:** runtime, validation, and QLKNN training tooling
  paths use `np.load(..., allow_pickle=False)` with required-key checks.
- **Bounded subprocesses:** CLI mode launches, compiler git-SHA probe,
  quantum bridge script orchestration, native C++ compile calls, and claims
  audit git file discovery use
  explicit subprocess timeouts to avoid indefinite process hangs.
- **CI pipeline resilience:** Python preflight and strict-mypy tool runners
  now treat hung subprocesses as deterministic timeout failures.
- **Local command policy:** `.claude/settings.local.json` is treated as
  machine-local command-permission policy and is excluded from version
  control.

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
| 2026-02-15 | v2.0.0 released — multigrid solver, H-infinity controller |
| 2026-02-16 | v2.1.0 released — GEQDSK expansion, Sauter bootstrap |
| 2026-02-17 | v3.0.0 released — Rust SNN PyO3, full-chain UQ, shot replay |
| 2026-02-17 | v3.1.0 released — Phase 0 physics hardening (Greenwald, TBR, conservation) |
