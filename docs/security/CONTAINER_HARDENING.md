# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Container Hardening

# Container hardening

The Docker image runs as UID/GID `10001` and the Compose profile adds runtime
confinement:

- `cap_drop: [ALL]`
- `no-new-privileges:true`
- read-only root filesystem
- bounded tmpfs mounts for `/tmp` and Streamlit state
- custom seccomp deny profile at `docker/seccomp-scpn-fusion.json`
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

The custom seccomp profile explicitly denies high-risk kernel and namespace
operations such as module loading, mounting, keyring access, `ptrace`, BPF,
performance counters, reboot, and namespace creation while leaving ordinary
Python, Rust, and Streamlit runtime syscalls available.
