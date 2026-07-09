# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full-Chain UQ Defensive-Branch Contract Tests
"""Contract tests for the defensive branches of the full-chain UQ propagator.

These branches are not reachable with valid inputs and a well-conditioned draw,
so they are exercised by fault injection rather than left uncovered: a sigma that
cannot be coerced to float, a ``LinAlgError`` from the correlated IPB98 draw that
must be recovered by a jittered retry, and a non-finite fusion power that must
clamp the gain sample to zero. The happy path and the finite-but-out-of-range
sigma guards are covered by ``tests/test_uq_full_chain.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import uncertainty_full_chain as frc_module
from scpn_fusion.core.uncertainty import PlasmaScenario

_SCENARIO = PlasmaScenario(I_p=15.0, B_t=5.3, P_heat=50.0, n_e=10.1, R=6.2, A=3.1, kappa=1.7, M=2.5)


class _FirstCallLinAlgErrorRng:
    """Generator wrapper that raises ``LinAlgError`` on the first draw only."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._calls = 0

    def multivariate_normal(self, *args: Any, **kwargs: Any) -> Any:
        """Raise once to force the covariance-jitter retry, then delegate."""
        self._calls += 1
        if self._calls == 1:
            raise np.linalg.LinAlgError("forced non-PSD covariance")
        return self._delegate.multivariate_normal(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate every other Generator method to the wrapped rng."""
        return getattr(self._delegate, name)


def _infinite_power(*_args: Any, **_kwargs: Any) -> float:
    """Return a non-finite fusion power to exercise the gain-clamp guard."""
    return float(np.inf)


@pytest.mark.parametrize("bad_sigma", ["not-a-number", None])
def test_non_floatable_sigma_rejected(bad_sigma: object) -> None:
    """A sigma that cannot be coerced to float fails closed with a located error."""
    with pytest.raises(ValueError, match="chi_gB_sigma must be finite"):
        frc_module.quantify_full_chain(
            _SCENARIO,
            n_samples=8,
            seed=1,
            chi_gB_sigma=bad_sigma,  # type: ignore[arg-type]  # deliberate invalid-type probe
        )


def test_non_psd_covariance_triggers_jitter_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LinAlgError from the correlated draw is recovered by a jittered retry."""
    real_default_rng = np.random.default_rng

    def _flaky(seed: int | None = None) -> Any:
        return _FirstCallLinAlgErrorRng(real_default_rng(seed))

    monkeypatch.setattr(np.random, "default_rng", _flaky)
    result = frc_module.quantify_full_chain(_SCENARIO, n_samples=8, seed=1)

    assert result.n_samples == 8
    assert np.isfinite(result.tau_E)


def test_non_finite_gain_is_clamped_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite fusion power collapses the gain sample to zero, not NaN or inf."""
    monkeypatch.setattr(frc_module, "fusion_power_from_tau", _infinite_power)
    with np.errstate(invalid="ignore"):
        result = frc_module.quantify_full_chain(_SCENARIO, n_samples=8, seed=1)

    assert result.Q == 0.0
