# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — preflight wrapper tests
"""Tests for the compatibility preflight entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "preflight.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("preflight", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_delegates_to_canonical_runner() -> None:
    module = _load_module()
    calls: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(argv)
        return 17

    module._run_python_preflight_main = fake_main

    assert module.main(["--gate", "research"]) == 17
    assert calls == [["--gate", "research"]]


def test_preflight_defaults_to_process_arguments() -> None:
    module = _load_module()
    calls: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(argv)
        return 0

    module._run_python_preflight_main = fake_main

    assert module.main() == 0
    assert calls == [None]


def test_repo_root_is_importable_for_script_execution() -> None:
    module = _load_module()

    assert str(ROOT) in sys.path
    assert module.REPO_ROOT == ROOT
    assert module.TOOLS_DIR == ROOT / "tools"
