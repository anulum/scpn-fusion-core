# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — EPED Pedestal Input Hardening Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import pytest

from scpn_fusion.core.eped_pedestal import EpedPedestalModel


def _valid_model_kwargs() -> dict[str, float]:
    return {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "Ip_MA": 15.0,
        "kappa": 1.7,
        "A_ion": 2.5,
        "Z_eff": 1.6,
    }


def test_predict_returns_physical_finite_outputs() -> None:
    model = EpedPedestalModel(**_valid_model_kwargs())
    res = model.predict(n_ped_1e19=8.0, T_ped_guess_keV=3.0)

    assert np.isfinite(res.p_ped_kPa) and res.p_ped_kPa > 0.0
    assert np.isfinite(res.T_ped_keV) and res.T_ped_keV > 0.0
    assert np.isfinite(res.n_ped_1e19) and res.n_ped_1e19 > 0.0
    assert np.isfinite(res.beta_p_ped) and res.beta_p_ped > 0.0
    assert np.isfinite(res.nu_star_ped) and res.nu_star_ped > 0.0
    # Model clips pedestal width to this physical range by design.
    assert 0.01 <= res.Delta_ped <= 0.15


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("R0", 0.0),
        ("a", -1.0),
        ("B0", np.nan),
        ("Ip_MA", np.inf),
        ("kappa", 0.0),
        ("A_ion", -0.1),
        ("Z_eff", 0.0),
    ],
)
def test_constructor_rejects_non_physical_inputs(field, value):
    kwargs = _valid_model_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        EpedPedestalModel(**kwargs)


@pytest.mark.parametrize(
    ("n_ped_1e19", "T_ped_guess_keV", "field"),
    [
        (0.0, 3.0, "n_ped_1e19"),
        (np.nan, 3.0, "n_ped_1e19"),
        (8.0, 0.0, "T_ped_guess_keV"),
        (8.0, -1.0, "T_ped_guess_keV"),
        (8.0, np.inf, "T_ped_guess_keV"),
    ],
)
def test_predict_rejects_invalid_runtime_inputs(n_ped_1e19, T_ped_guess_keV, field):
    model = EpedPedestalModel(**_valid_model_kwargs())
    with pytest.raises(ValueError, match=field):
        model.predict(n_ped_1e19=n_ped_1e19, T_ped_guess_keV=T_ped_guess_keV)
