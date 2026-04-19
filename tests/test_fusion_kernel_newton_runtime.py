# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Unit tests for Newton runtime helper contracts."""

from __future__ import annotations

import pytest

from scpn_fusion.core.fusion_kernel_newton_runtime import (
    GmresTelemetry,
    parse_newton_dispatch_config,
)


def _base_cfg() -> dict[str, dict[str, object]]:
    return {
        "solver": {
            "max_iterations": 40,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.1,
            "gmres_preconditioner": "none",
        },
        "physics": {"vacuum_permeability": 1.0},
    }


def test_parse_newton_dispatch_config_maps_legacy_preconditioner_flags() -> None:
    cfg = _base_cfg()
    cfg["solver"]["gmres_preconditioner"] = "legacy"
    cfg["solver"]["gmres_diagonal_preconditioner"] = True
    out = parse_newton_dispatch_config(cfg)
    assert out.gmres_preconditioner_mode == "diagonal"
    assert out.max_iter == 40
    assert out.warmup_steps == 15


def test_parse_newton_dispatch_config_rejects_invalid_budget_bool() -> None:
    cfg = _base_cfg()
    cfg["solver"]["gmres_nonconverged_budget"] = True
    with pytest.raises(ValueError, match="gmres_nonconverged_budget"):
        parse_newton_dispatch_config(cfg)


def test_gmres_telemetry_budget_enforced() -> None:
    telem = GmresTelemetry()
    telem.record(
        info=1,
        iter_idx=3,
        fail_on_breakdown=True,
        nonconverged_budget=1,
    )
    assert telem.nonconverged_count == 1
    with pytest.raises(RuntimeError, match="GMRES non-convergence budget exceeded"):
        telem.record(
            info=2,
            iter_idx=4,
            fail_on_breakdown=True,
            nonconverged_budget=1,
        )


def test_gmres_telemetry_breakdown_can_be_nonfatal() -> None:
    telem = GmresTelemetry()
    telem.record(
        info=-1,
        iter_idx=2,
        fail_on_breakdown=False,
        nonconverged_budget=None,
    )
    assert telem.breakdown_count == 1
    assert telem.last_info == -1
