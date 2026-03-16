# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Lazy Import Integrity Tests
# ──────────────────────────────────────────────────────────────────────
"""Verify every lazy-import entry in core/ and control/ __init__.py resolves."""

from __future__ import annotations

import importlib

import pytest


def _collect_lazy_entries(package_path: str) -> list[tuple[str, str, str]]:
    mod = importlib.import_module(package_path)
    lazy = getattr(mod, "_LAZY_IMPORTS", {})
    return [(package_path, name, lazy[name]) for name in lazy if lazy[name] is not None]


CORE_ENTRIES = _collect_lazy_entries("scpn_fusion.core")
CONTROL_ENTRIES = _collect_lazy_entries("scpn_fusion.control")


@pytest.mark.parametrize(
    "pkg,name,target",
    CORE_ENTRIES,
    ids=[e[1] for e in CORE_ENTRIES],
)
def test_core_lazy_import(pkg, name, target):
    mod = importlib.import_module(pkg)
    obj = getattr(mod, name)
    assert obj is not None, f"{pkg}.{name} resolved to None"


@pytest.mark.parametrize(
    "pkg,name,target",
    CONTROL_ENTRIES,
    ids=[e[1] for e in CONTROL_ENTRIES],
)
def test_control_lazy_import(pkg, name, target):
    mod = importlib.import_module(pkg)
    obj = getattr(mod, name)
    assert obj is not None, f"{pkg}.{name} resolved to None"
