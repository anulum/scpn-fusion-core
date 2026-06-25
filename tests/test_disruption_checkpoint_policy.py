# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Checkpoint Policy Tests
"""Security-policy tests for the disruption checkpoint loading helpers.

Covers the path/seq-len/fallback helpers, the SHA256 allowlist parser, the
fail-closed torch checkpoint loader (size/suffix/digest/weights-only guards) and
the loaded-state-dict validator. Torch is injected as a controlled fake so the
guard branches are exercised deterministically regardless of the environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.control import disruption_checkpoint_policy as policy


class _FakeTensor:
    """Minimal stand-in for a torch tensor with a parameter count."""

    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class _FakeTorch:
    """Controllable torch replacement for the checkpoint loader paths."""

    Tensor = _FakeTensor

    def __init__(self, load_result: Any = None, load_error: Exception | None = None) -> None:
        self._load_result = load_result
        self._load_error = load_error
        self.load_calls: list[dict[str, Any]] = []

    def load(self, path: Any, *, map_location: str, weights_only: bool) -> Any:
        self.load_calls.append(
            {"path": path, "map_location": map_location, "weights_only": weights_only}
        )
        if self._load_error is not None:
            raise self._load_error
        return self._load_result


def test_default_model_path_points_into_repo_artifacts() -> None:
    path = policy.default_model_path("disruptor.pth")
    assert path.name == "disruptor.pth"
    assert path.parent.name == "artifacts"


def test_normalize_seq_len_enforces_minimum() -> None:
    assert policy._normalize_seq_len(8) == 8
    assert policy._normalize_seq_len(64) == 64
    with pytest.raises(ValueError, match="seq_len"):
        policy._normalize_seq_len(4)


@pytest.mark.parametrize(
    ("allow", "env", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (True, "0", True),
        (True, "1", False),
        (True, "yes", False),
        (True, "ON", False),
    ],
)
def test_resolve_allow_fallback_respects_argument_and_strict_env(
    allow: bool, env: str | None, expected: bool, monkeypatch
) -> None:
    if env is None:
        monkeypatch.delenv(policy._DISRUPTION_STRICT_NO_FALLBACK_ENV, raising=False)
    else:
        monkeypatch.setenv(policy._DISRUPTION_STRICT_NO_FALLBACK_ENV, env)
    assert policy._resolve_allow_fallback(allow) is expected


def test_augment_with_fallback_telemetry_adds_snapshot_without_mutating_input() -> None:
    meta = {"model": "x"}
    out = policy._augment_with_fallback_telemetry(meta)
    assert out["model"] == "x"
    assert "fallback_telemetry" in out
    assert "fallback_telemetry" not in meta


def test_record_recovery_event_returns_a_telemetry_record() -> None:
    record = policy._record_recovery_event("unit_test_reason", context={"k": "v"})
    assert isinstance(record, dict)


def test_prepare_signal_window_truncates_then_pads_with_edge() -> None:
    truncated = policy._prepare_signal_window(np.arange(20.0), 8)
    np.testing.assert_array_equal(truncated, np.arange(8.0))

    padded = policy._prepare_signal_window([1.0, 2.0, 3.0], 8)
    assert padded.size == 8
    np.testing.assert_array_equal(padded[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(padded[3:], np.full(5, 3.0))  # edge padding


def test_parse_sha256_allowlist_accepts_mixed_separators(monkeypatch) -> None:
    a, b = "a" * 64, "b" * 64
    monkeypatch.setenv(policy._CHECKPOINT_SHA256_ALLOWLIST_ENV, f"{a}, ;{b.upper()};")
    assert policy._parse_checkpoint_sha256_allowlist() == {a, b}


def test_parse_sha256_allowlist_empty_returns_empty_set(monkeypatch) -> None:
    monkeypatch.delenv(policy._CHECKPOINT_SHA256_ALLOWLIST_ENV, raising=False)
    assert policy._parse_checkpoint_sha256_allowlist() == set()


@pytest.mark.parametrize("bad", ["abc", "g" * 64, "a" * 63])
def test_parse_sha256_allowlist_rejects_malformed_digests(bad: str, monkeypatch) -> None:
    monkeypatch.setenv(policy._CHECKPOINT_SHA256_ALLOWLIST_ENV, bad)
    with pytest.raises(ValueError, match="invalid SHA256"):
        policy._parse_checkpoint_sha256_allowlist()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    payload = b"scpn-fusion-core" * 4096
    f.write_bytes(payload)
    assert policy._sha256_file(f) == hashlib.sha256(payload).hexdigest()


def _write_checkpoint(tmp_path: Path, name: str = "model.pth", data: bytes = b"weights") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_safe_load_requires_torch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", None)
    with pytest.raises(RuntimeError, match="Torch is required"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path))


def test_safe_load_rejects_missing_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(FileNotFoundError):
        policy._safe_torch_checkpoint_load(tmp_path / "absent.pth")


def test_safe_load_rejects_disallowed_suffix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(RuntimeError, match="suffix is not allowed"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path, "model.bin"))


def test_safe_load_rejects_empty_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(RuntimeError, match="empty"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path, data=b""))


def test_safe_load_blocks_oversize_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    monkeypatch.setattr(policy, "_MAX_CHECKPOINT_BYTES", 4)
    with pytest.raises(RuntimeError, match="exceeds safety size budget"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path, data=b"too-large"))


def test_safe_load_blocks_non_allowlisted_digest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    monkeypatch.setenv(policy._CHECKPOINT_SHA256_ALLOWLIST_ENV, "c" * 64)
    with pytest.raises(RuntimeError, match="not allowlisted"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path))


def test_safe_load_accepts_allowlisted_digest_and_uses_weights_only(
    monkeypatch, tmp_path: Path
) -> None:
    path = _write_checkpoint(tmp_path)
    digest = policy._sha256_file(path)
    expected = {"layer.weight": _FakeTensor(4)}
    fake = _FakeTorch(load_result=expected)
    monkeypatch.setattr(policy, "torch", fake)
    monkeypatch.setenv(policy._CHECKPOINT_SHA256_ALLOWLIST_ENV, digest)

    result = policy._safe_torch_checkpoint_load(path)
    assert result is expected
    assert fake.load_calls[0]["weights_only"] is True
    assert fake.load_calls[0]["map_location"] == "cpu"


def test_safe_load_blocks_legacy_weights_only_typeerror(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeTorch(load_error=TypeError("unexpected keyword argument 'weights_only'"))
    monkeypatch.setattr(policy, "torch", fake)
    with pytest.raises(RuntimeError, match="Legacy torch checkpoint loading is disabled"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path))


def test_safe_load_reraises_unrelated_typeerror(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeTorch(load_error=TypeError("some other failure"))
    monkeypatch.setattr(policy, "torch", fake)
    with pytest.raises(TypeError, match="some other failure"):
        policy._safe_torch_checkpoint_load(_write_checkpoint(tmp_path))


def test_validated_state_dict_accepts_tensor_and_ndarray(monkeypatch) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    state = {"a.weight": _FakeTensor(3), "b.bias": np.zeros(2, dtype=np.float64)}
    assert policy._validated_checkpoint_state_dict(state) is state


def test_validated_state_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        policy._validated_checkpoint_state_dict([1, 2, 3])


def test_validated_state_dict_rejects_non_string_keys(monkeypatch) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(ValueError, match="keys must be strings"):
        policy._validated_checkpoint_state_dict({3: np.zeros(2)})


def test_validated_state_dict_rejects_unsupported_value_type(monkeypatch) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(ValueError, match="must be tensor/ndarray"):
        policy._validated_checkpoint_state_dict({"a": "not-a-tensor"})


def test_validated_state_dict_rejects_empty_parameter_set(monkeypatch) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    with pytest.raises(ValueError, match="at least one parameter"):
        policy._validated_checkpoint_state_dict({"a": np.zeros(0, dtype=np.float64)})


def test_validated_state_dict_enforces_parameter_budget(monkeypatch) -> None:
    monkeypatch.setattr(policy, "torch", _FakeTorch())
    monkeypatch.setattr(policy, "_MAX_CHECKPOINT_PARAMETER_COUNT", 3)
    with pytest.raises(ValueError, match="exceeds safety budget"):
        policy._validated_checkpoint_state_dict({"a": np.zeros(4, dtype=np.float64)})
