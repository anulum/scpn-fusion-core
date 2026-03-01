from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from scpn_fusion.core.config_schema import ReactorConfig, validate_config


def _base_config() -> dict:
    return {
        "reactor_name": "UnitTest-Reactor",
        "grid_resolution": [65, 65],
        "dimensions": {
            "R_min": 1.0,
            "R_max": 8.0,
            "Z_min": -4.0,
            "Z_max": 4.0,
        },
        "coils": [
            {"name": "PF1", "r": 6.5, "z": 1.5, "current": 2.0},
        ],
    }


def test_validate_config_returns_model() -> None:
    model = validate_config(_base_config())
    assert isinstance(model, ReactorConfig)
    assert model.reactor_name == "UnitTest-Reactor"
    assert model.grid_resolution == (65, 65)
    assert len(model.coils) == 1


def test_validate_config_rejects_low_resolution() -> None:
    cfg = _base_config()
    cfg["grid_resolution"] = [3, 65]
    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_dimensions() -> None:
    cfg = _base_config()
    cfg["dimensions"]["R_max"] = cfg["dimensions"]["R_min"]
    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_reactor_config_coils_default_is_not_shared() -> None:
    cfg = _base_config()
    cfg.pop("coils")
    cfg1 = validate_config(copy.deepcopy(cfg))
    cfg2 = validate_config(copy.deepcopy(cfg))
    cfg1.coils.append({"name": "PFX", "r": 2.0, "z": 0.0, "current": 0.1})  # type: ignore[arg-type]
    assert len(cfg1.coils) == 1
    assert len(cfg2.coils) == 0
