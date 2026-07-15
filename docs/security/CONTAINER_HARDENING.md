<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

## Security scope

This page defines concrete container hardening configuration used in deployment. It is the runtime security companion to benchmark, proof, and solver reproducibility docs.

## Operational context

This hardening profile applies to deployment readiness. Pair this with transport
security controls and CI gating for privileged-capability surfaces.

<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — Container Hardening -->

# Container hardening

The Docker image runs as UID/GID `10001` and the Compose profile adds runtime
confinement:

- `cap_drop: [ALL]`
- `no-new-privileges:true`
- read-only root filesystem
- bounded tmpfs mounts for `/tmp` and Streamlit state
- default-deny seccomp allowlist at `docker/seccomp-scpn-fusion.json`
- default Docker AppArmor confinement

For hosts that support custom AppArmor profiles, load the stricter project
profile before deployment:

```bash
sudo apparmor_parser -r docker/apparmor-scpn-fusion.profile
```

Then replace this Compose option:

```yaml
security_opt:
  - apparmor:docker-default
```

with:

```yaml
security_opt:
  - apparmor:scpn-fusion
```

The seccomp profile is a default-deny allowlist (`defaultAction:
SCMP_ACT_ERRNO`) derived from the upstream moby v27.5.1 default profile, so any
syscall outside the curated allowlist is rejected rather than permitted. High-risk
kernel and namespace operations — module loading, mounting, keyring and kexec
access, BPF, performance counters, reboot, and namespace creation — are allowed
only behind an explicit capability gate, and because the Compose profile applies
`cap_drop: [ALL]` none of those capabilities are present, so those syscalls are
denied at runtime. `ptrace` is removed from the allowlist entirely, so it is
denied unconditionally. Ordinary Python, Rust, and Streamlit runtime syscalls
remain available. Regenerate or update the profile from the pinned upstream
source and re-run `tests/test_seccomp_profile.py`, which pins these invariants.
