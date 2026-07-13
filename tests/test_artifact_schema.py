# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: raw controller-artifact schema guard

from __future__ import annotations

from typing import Any

import pytest

from scpn_fusion.scpn.artifact_schema import (
    _require_array,
    _require_object,
    _validate_object_keys,
    _validate_packed_weight_schema,
    _validate_weight_matrix_schema,
)


class SchemaError(ValueError):
    """Distinct error type injected into the schema guard for assertion."""


class TestRequireObject:
    """``_require_object`` returns dicts unchanged and rejects everything else."""

    def test_returns_dict_unchanged(self) -> None:
        payload = {"a": 1}
        assert _require_object(payload, "root", error_type=SchemaError) is payload

    @pytest.mark.parametrize("value", [[], "text", 3, None, 1.5])
    def test_rejects_non_object(self, value: Any) -> None:
        with pytest.raises(SchemaError, match=r"root must be an object"):
            _require_object(value, "root", error_type=SchemaError)


class TestRequireArray:
    """``_require_array`` returns lists unchanged and rejects everything else."""

    def test_returns_list_unchanged(self) -> None:
        payload = [1, 2, 3]
        assert _require_array(payload, "root.items", error_type=SchemaError) is payload

    @pytest.mark.parametrize("value", [{}, "text", 3, None, 1.5])
    def test_rejects_non_array(self, value: Any) -> None:
        with pytest.raises(SchemaError, match=r"root\.items must be an array"):
            _require_array(value, "root.items", error_type=SchemaError)


class TestValidateObjectKeys:
    """``_validate_object_keys`` enforces the required/allowed key contract."""

    def test_accepts_exact_key_set(self) -> None:
        _validate_object_keys(
            {"a": 1, "b": 2},
            path="node",
            required={"a", "b"},
            allowed={"a", "b"},
            error_type=SchemaError,
        )

    def test_rejects_missing_required_key(self) -> None:
        with pytest.raises(SchemaError, match=r"missing required property node\.b"):
            _validate_object_keys(
                {"a": 1},
                path="node",
                required={"a", "b"},
                allowed={"a", "b"},
                error_type=SchemaError,
            )

    def test_rejects_unexpected_key(self) -> None:
        with pytest.raises(SchemaError, match=r"unexpected property node\.c"):
            _validate_object_keys(
                {"a": 1, "c": 3},
                path="node",
                required={"a"},
                allowed={"a"},
                error_type=SchemaError,
            )


class TestValidateWeightMatrixSchema:
    """A weight matrix must be an object carrying array ``shape`` and ``data``."""

    def test_accepts_valid_matrix(self) -> None:
        _validate_weight_matrix_schema(
            {"shape": [2, 2], "data": [1, 2, 3, 4]}, "weights.w_in", error_type=SchemaError
        )

    def test_rejects_non_array_shape(self) -> None:
        with pytest.raises(SchemaError, match=r"weights\.w_in\.shape must be an array"):
            _validate_weight_matrix_schema(
                {"shape": 2, "data": [1, 2]}, "weights.w_in", error_type=SchemaError
            )


class TestValidatePackedWeightSchema:
    """The packed weight guard accepts the expanded and compact encodings only."""

    def test_accepts_expanded_data_u64_payload(self) -> None:
        _validate_packed_weight_schema(
            {"shape": [2, 2], "data_u64": [1, 2, 3]},
            "weights.packed.w_in_packed",
            error_type=SchemaError,
        )

    def test_accepts_compact_zlib_payload(self) -> None:
        _validate_packed_weight_schema(
            {
                "shape": [2, 2],
                "encoding": "u64_b64_zlib",
                "count": 3,
                "data_u64_b64_zlib": "eJx=",
            },
            "weights.packed.w_in_packed",
            error_type=SchemaError,
        )

    def test_expanded_payload_requires_array_data_u64(self) -> None:
        with pytest.raises(
            SchemaError, match=r"weights\.packed\.w_in_packed\.data_u64 must be an array"
        ):
            _validate_packed_weight_schema(
                {"shape": [2, 2], "data_u64": "notarray"},
                "weights.packed.w_in_packed",
                error_type=SchemaError,
            )

    def test_rejects_payload_without_recognised_encoding(self) -> None:
        with pytest.raises(
            SchemaError,
            match=r"must contain either data_u64 or compact data_u64_b64_zlib payload",
        ):
            _validate_packed_weight_schema(
                {"shape": [2, 2]}, "weights.packed.w_in_packed", error_type=SchemaError
            )
