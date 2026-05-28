# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fusion Kernel Tests
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scpn_fusion.core.fusion_kernel import MAX_CONFIG_BYTES, FusionKernel


def test_load_config_rejects_oversized_json_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "oversized.json"
    config_path.write_text("{}", encoding="utf-8")

    real_stat = Path.stat

    def oversized_stat(path: Path, follow_symlinks: bool = True) -> os.stat_result:
        stat_result = real_stat(path, follow_symlinks=follow_symlinks)
        if path == config_path:
            values = list(stat_result)
            values[6] = MAX_CONFIG_BYTES + 1
            return os.stat_result(values)
        return stat_result

    monkeypatch.setattr(Path, "stat", oversized_stat)
    kernel = FusionKernel.__new__(FusionKernel)

    with pytest.raises(ValueError, match="configuration file exceeds"):
        kernel.load_config(config_path)
