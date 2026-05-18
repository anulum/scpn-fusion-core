# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FNO QLKNN Spatial Generator Tests
from __future__ import annotations

import pytest

from tools.generate_fno_qlknn_spatial import _require_neural_transport_oracle


class _NonNeuralModel:
    is_neural = False
    _weights = None

    def __init__(self, _weights_path: object) -> None:
        pass


class _NeuralModel:
    is_neural = True
    _weights = {"layer": "weights"}

    def __init__(self, _weights_path: object) -> None:
        pass


def test_require_neural_transport_oracle_rejects_non_neural_model() -> None:
    with pytest.raises(RuntimeError, match="requires trained QLKNN neural weights"):
        _require_neural_transport_oracle("missing.npz", model_cls=_NonNeuralModel)


def test_require_neural_transport_oracle_accepts_loaded_neural_model() -> None:
    model = _require_neural_transport_oracle("weights.npz", model_cls=_NeuralModel)
    assert model.is_neural is True
    assert model._weights is not None
