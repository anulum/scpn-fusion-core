# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |

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

- Malicious input files (JSON configs, data files)
- Unsafe deserialization (serde, pickle)
- Numerical overflow / denial of service via pathological inputs
- Supply chain integrity (dependency audit)
