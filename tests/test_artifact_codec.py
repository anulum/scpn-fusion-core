# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import pytest

from scpn_fusion.scpn.artifact_codec import (
    decode_u64_compact_payload,
    encode_u64_compact_payload,
)


def test_encode_decode_u64_compact_roundtrip() -> None:
    values = [0, 1, 2, (1 << 63) + 7, (1 << 64) - 1]
    encoded = encode_u64_compact_payload(values)
    decoded = decode_u64_compact_payload(
        encoded,
        error_type=ValueError,
        max_packed_words=1_000,
        max_compressed_bytes=1_000_000,
        max_decompressed_bytes=1_000_000,
    )
    assert decoded == values


def test_decode_u64_compact_rejects_invalid_count_type() -> None:
    encoded = encode_u64_compact_payload([1, 2, 3])
    encoded["count"] = "3"
    with pytest.raises(ValueError, match="count type"):
        decode_u64_compact_payload(
            encoded,
            error_type=ValueError,
            max_packed_words=1_000,
            max_compressed_bytes=1_000_000,
            max_decompressed_bytes=1_000_000,
        )


def test_decode_u64_compact_rejects_wrong_encoding_marker() -> None:
    encoded = encode_u64_compact_payload([1, 2, 3])
    encoded["encoding"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported packed encoding"):
        decode_u64_compact_payload(
            encoded,
            error_type=ValueError,
            max_packed_words=1_000,
            max_compressed_bytes=1_000_000,
            max_decompressed_bytes=1_000_000,
        )
