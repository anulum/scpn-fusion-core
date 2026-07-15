# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Seccomp Profile Invariants
"""Structural guards on the container seccomp profile.

The profile is a default-deny allowlist derived from the moby v27.5.1 default
profile (``defaultAction: SCMP_ACT_ERRNO``), with ``ptrace`` removed so it is
denied even where a capability would otherwise permit it.  These tests pin the
security-relevant invariants so a future edit cannot silently re-open the
default-allow posture the audit flagged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PROFILE_PATH = Path(__file__).resolve().parents[1] / "docker" / "seccomp-scpn-fusion.json"

#: Syscalls that must never be *unconditionally* allowed (only behind a capability).
CAP_GATED_ONLY = frozenset({"mount", "umount2", "bpf", "setns", "unshare", "init_module", "reboot"})

#: Syscalls that must not appear in any allow group at all.
FULLY_DENIED = frozenset(
    {"ptrace", "add_key", "keyctl", "request_key", "kexec_load", "swapon", "swapoff"}
)


@pytest.fixture(scope="module")
def profile() -> dict[str, Any]:
    """Return the parsed seccomp profile JSON document."""
    with PROFILE_PATH.open("r", encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def test_profile_is_default_deny(profile: dict[str, Any]) -> None:
    """The profile denies every syscall by default (allowlist model)."""
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert profile["syscalls"], "profile must declare an explicit allowlist"


def test_dangerous_syscalls_are_only_capability_gated(profile: dict[str, Any]) -> None:
    """Privileged syscalls are allowed only under an explicit capability gate."""
    for group in profile["syscalls"]:
        if group["action"] != "SCMP_ACT_ALLOW":
            continue
        gated = bool(group.get("includes", {}).get("caps"))
        for name in group["names"]:
            if name in CAP_GATED_ONLY:
                assert gated, f"{name} is allowed without a capability gate"


def test_fully_denied_syscalls_never_allowed(profile: dict[str, Any]) -> None:
    """ptrace and key/kexec/swap syscalls appear in no allow group."""
    allowed = {
        name
        for group in profile["syscalls"]
        if group["action"] == "SCMP_ACT_ALLOW"
        for name in group["names"]
    }
    leaked = FULLY_DENIED & allowed
    assert not leaked, f"denied syscalls leaked into allowlist: {sorted(leaked)}"


def test_common_runtime_syscalls_are_allowed(profile: dict[str, Any]) -> None:
    """The allowlist keeps the syscalls a Python runtime needs to start."""
    allowed = {
        name
        for group in profile["syscalls"]
        if group["action"] == "SCMP_ACT_ALLOW"
        for name in group["names"]
    }
    required = {"read", "write", "mmap", "futex", "clone", "openat", "epoll_wait"}
    missing = required - allowed
    assert not missing, f"runtime-critical syscalls missing from allowlist: {sorted(missing)}"
