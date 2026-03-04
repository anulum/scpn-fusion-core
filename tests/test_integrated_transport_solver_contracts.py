from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver_contracts import (
    _RUST_CHANG_HINTON_DEFAULTS,
    coerce_matching_1d_profiles,
    require_positive_finite_scalar,
    rust_chang_hinton_params_match_defaults,
)


def test_require_positive_finite_scalar_accepts_valid_value() -> None:
    assert require_positive_finite_scalar("x", 2.5) == pytest.approx(2.5)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf, "abc"])
def test_require_positive_finite_scalar_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        require_positive_finite_scalar("x", value)


def test_coerce_matching_1d_profiles_coerces_and_validates_shape() -> None:
    out = coerce_matching_1d_profiles(a=[1, 2, 3], b=np.array([4.0, 5.0, 6.0]))
    assert set(out) == {"a", "b"}
    assert out["a"].dtype == np.float64
    assert out["a"].shape == out["b"].shape == (3,)

    with pytest.raises(ValueError):
        coerce_matching_1d_profiles(a=[1, 2, 3], b=[1, 2])


def test_rust_chang_hinton_params_match_defaults_contract() -> None:
    assert rust_chang_hinton_params_match_defaults(
        R0=_RUST_CHANG_HINTON_DEFAULTS["R0"],
        a=_RUST_CHANG_HINTON_DEFAULTS["a"],
        B0=_RUST_CHANG_HINTON_DEFAULTS["B0"],
        A_ion=_RUST_CHANG_HINTON_DEFAULTS["A_ion"],
        Z_eff=_RUST_CHANG_HINTON_DEFAULTS["Z_eff"],
    )
    assert not rust_chang_hinton_params_match_defaults(
        R0=_RUST_CHANG_HINTON_DEFAULTS["R0"] + 1e-3,
        a=_RUST_CHANG_HINTON_DEFAULTS["a"],
        B0=_RUST_CHANG_HINTON_DEFAULTS["B0"],
        A_ion=_RUST_CHANG_HINTON_DEFAULTS["A_ion"],
        Z_eff=_RUST_CHANG_HINTON_DEFAULTS["Z_eff"],
    )
