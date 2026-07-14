# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Blanket Neutronics Validator Tests
"""Contract tests for the shared blanket-neutronics input validators."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear._blanket_validators import _require_finite_float, _require_int


def test_require_finite_float_accepts_and_bounds() -> None:
    assert _require_finite_float("x", 2.5, min_value=0.0, max_value=5.0) == 2.5
    with pytest.raises(ValueError, match="x must be finite"):
        _require_finite_float("x", float("nan"))
    with pytest.raises(ValueError, match="x must be >="):
        _require_finite_float("x", -1.0, min_value=0.0)
    with pytest.raises(ValueError, match="x must be <="):
        _require_finite_float("x", 9.0, max_value=5.0)


def test_require_int_accepts_and_rejects() -> None:
    assert _require_int("n", 5, 3) == 5
    # np.integer is accepted at runtime via the isinstance guard (static type is int).
    assert _require_int("n", np.int64(4), 3) == 4  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be an integer"):
        _require_int("n", True, 1)  # bool is rejected
    with pytest.raises(ValueError, match="n must be an integer"):
        _require_int("n", 2.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be an integer"):
        _require_int("n", 1, 3)  # below minimum
