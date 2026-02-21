# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Configuration Schema
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Strict schema validation for reactor configurations using Pydantic.
Prevents late-stage simulation failures by catching malformed configs early.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict

# All sub-models use extra='allow' so that extension fields (solver_method,
# fail_on_diverge, anderson_depth, profiles, etc.) pass through validation
# without being silently dropped.  The schema validates what it knows;
# the runtime code uses .get() for optional extension keys.

class Dimensions(BaseModel):
    model_config = ConfigDict(extra='allow')
    R_min: float = Field(..., gt=0)
    R_max: float = Field(..., gt=0)
    Z_min: float
    Z_max: float

    @field_validator("R_max")
    @classmethod
    def r_max_greater_than_min(cls, v: float, info):
        if "R_min" in info.data and v <= info.data["R_min"]:
            raise ValueError("R_max must be greater than R_min")
        return v

class Coil(BaseModel):
    model_config = ConfigDict(extra='allow')
    name: str = "unnamed"
    r: float = Field(..., gt=0)
    z: float
    current: float = 0.0

class PhysicsParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    plasma_current_target: float = Field(default=5.0)
    vacuum_permeability: float = Field(default=1.25663706e-6, ge=0)
    beta_scale: float = Field(default=1.0, ge=0)
    pedestal_mode: Optional[str] = "analytic"
    confinement_scaling: str = "IPB98y2"

class SolverParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    max_iterations: int = Field(default=1000, gt=0)
    convergence_threshold: float = Field(default=1e-4, gt=0)
    relaxation_factor: float = Field(default=0.1, gt=0, le=1.0)

class ReactorConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    reactor_name: str = "Unnamed-Reactor"
    grid_resolution: Tuple[int, int] = Field(default=(65, 65))
    dimensions: Dimensions
    coils: List[Coil] = []
    physics: PhysicsParams = Field(default_factory=PhysicsParams)
    solver: SolverParams = Field(default_factory=SolverParams)

    @field_validator("grid_resolution")
    @classmethod
    def check_resolution(cls, v: Tuple[int, int]):
        if v[0] < 4 or v[1] < 4:
            raise ValueError("Grid resolution must be at least 4x4")
        return v

def validate_config(config_dict: dict) -> ReactorConfig:
    """Validate a raw configuration dictionary and return a validated ReactorConfig."""
    return ReactorConfig.model_validate(config_dict)
