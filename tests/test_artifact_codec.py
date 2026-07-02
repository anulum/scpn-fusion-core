# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any
import zlib

import pytest

from scpn_fusion.scpn.artifact_codec import (
    decode_u64_compact_payload,
    encode_u64_compact_payload,
)


def _decode(encoded: Mapping[str, Any], *, max_decompressed_bytes: int = 1_000_000) -> list[int]:
    """Decode a compact payload through the production decoder limits."""
    return decode_u64_compact_payload(
        dict(encoded),
        error_type=ValueError,
        max_packed_words=1_000,
        max_compressed_bytes=1_000_000,
        max_decompressed_bytes=max_decompressed_bytes,
    )


def test_encode_decode_u64_compact_roundtrip() -> None:
    """Compact little-endian payloads preserve full unsigned 64-bit words."""
    values = [0, 1, 2, (1 << 63) + 7, (1 << 64) - 1]
    encoded = encode_u64_compact_payload(values)

    decoded = _decode(encoded)

    assert decoded == values


def test_decode_u64_compact_rejects_invalid_count_type() -> None:
    """The decoder rejects non-integer compact count metadata."""
    encoded = encode_u64_compact_payload([1, 2, 3])
    encoded["count"] = "3"

    with pytest.raises(ValueError, match="count type"):
        _decode(encoded)


def test_decode_u64_compact_rejects_wrong_encoding_marker() -> None:
    """Unsupported packed encoding labels are rejected before payload decoding."""
    encoded = encode_u64_compact_payload([1, 2, 3])
    encoded["encoding"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported packed encoding"):
        _decode(encoded)


def test_decode_u64_compact_accepts_missing_count_as_available_words() -> None:
    """Missing count metadata falls back to all decompressed words."""
    encoded = encode_u64_compact_payload([11, 22, 33])
    del encoded["count"]

    assert _decode(encoded) == [11, 22, 33]


def test_decode_u64_compact_rejects_missing_payload_string() -> None:
    """The compact payload must contain a base64 zlib string."""
    encoded = encode_u64_compact_payload([1])
    encoded["data_u64_b64_zlib"] = 123

    with pytest.raises(ValueError, match="payload string"):
        _decode(encoded)


def test_decode_u64_compact_rejects_invalid_base64_payload() -> None:
    """Invalid base64 is reported as compact payload corruption."""
    encoded = encode_u64_compact_payload([1])
    encoded["data_u64_b64_zlib"] = "$$$not-base64$$$"

    with pytest.raises(ValueError, match="Invalid base64 payload"):
        _decode(encoded)


def test_decode_u64_compact_rejects_compressed_payload_size() -> None:
    """Compressed payload byte limits are enforced before decompression."""
    encoded = encode_u64_compact_payload([1])

    with pytest.raises(ValueError, match="Compressed payload too large"):
        decode_u64_compact_payload(
            encoded,
            error_type=ValueError,
            max_packed_words=1_000,
            max_compressed_bytes=1,
            max_decompressed_bytes=1_000_000,
        )


def test_decode_u64_compact_rejects_decompression_limit_tail() -> None:
    """The streaming decompressor rejects payloads beyond the configured limit."""
    encoded = encode_u64_compact_payload([1, 2, 3])

    with pytest.raises(ValueError, match="exceeds configured limit"):
        _decode(encoded, max_decompressed_bytes=7)


def test_decode_u64_compact_rejects_decompressed_payload_size() -> None:
    """The decoded byte-size guard catches complete streams that exceed limits."""
    encoded = encode_u64_compact_payload([1])

    with pytest.raises(ValueError, match="exceeds configured limit"):
        _decode(encoded, max_decompressed_bytes=7)


def test_decode_u64_compact_rejects_invalid_zlib_stream() -> None:
    """Valid base64 wrapping a non-zlib stream fails closed."""
    encoded = encode_u64_compact_payload([1])
    encoded["data_u64_b64_zlib"] = base64.b64encode(b"not-zlib").decode("ascii")

    with pytest.raises(ValueError, match="Invalid compact packed payload"):
        _decode(encoded)


def test_decode_u64_compact_rejects_misaligned_decompressed_bytes() -> None:
    """Decompressed byte streams must contain whole uint64 words."""
    encoded = {
        "encoding": "u64-le-zlib-base64",
        "count": None,
        "data_u64_b64_zlib": base64.b64encode(zlib.compress(b"abc", level=9)).decode("ascii"),
    }

    with pytest.raises(ValueError, match="not divisible by 8"):
        _decode(encoded)


def test_decode_u64_compact_rejects_count_above_limits_and_available_words() -> None:
    """Count metadata must stay within configured and decompressed word limits."""
    encoded = encode_u64_compact_payload([1, 2, 3])
    encoded["count"] = 1_001

    with pytest.raises(ValueError, match="exceeds limit"):
        _decode(encoded)

    encoded["count"] = 4
    with pytest.raises(ValueError, match="available words=3"):
        _decode(encoded)
