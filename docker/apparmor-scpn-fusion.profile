# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Docker AppArmor Profile

# Optional host profile for hardened Docker deployments.
# Load with:
#   sudo apparmor_parser -r docker/apparmor-scpn-fusion.profile
# Then replace `apparmor:docker-default` in docker-compose.yml with
# `apparmor:scpn-fusion`.

#include <tunables/global>

profile scpn-fusion flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  network inet stream,
  network inet6 stream,
  network inet tcp,
  network inet6 tcp,

  deny mount,
  deny remount,
  deny umount,
  deny ptrace,
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_ptrace,
  deny capability net_admin,
  deny capability mknod,
  deny /proc/sys/** wklx,
  deny /sys/** wklx,
  deny /dev/mem rwklx,
  deny /dev/kmem rwklx,

  /app/** r,
  /app/artifacts/** rw,
  /tmp/** rw,
  /home/appuser/.streamlit/** rw,
  /usr/local/bin/** ix,
  /usr/local/lib/** r,
  /usr/lib/** r,
  /lib/** r,
  /bin/** ix,
  /proc/** r,
  /etc/** r,
}
