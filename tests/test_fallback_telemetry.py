# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fallback Telemetry Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from scpn_fusion.fallback_telemetry import (
    FallbackBudgetExceeded,
    record_fallback_event,
    reset_fallback_telemetry,
    snapshot_fallback_telemetry,
)


def test_record_fallback_event_updates_snapshot() -> None:
    reset_fallback_telemetry()
    record_fallback_event("tokamak_archive", "poll_reference_fallback")
    snap = snapshot_fallback_telemetry()
    assert snap["total_count"] == 1
    assert snap["domain_counts"]["tokamak_archive"] == 1
    assert snap["recent_events"]


def test_total_fallback_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_fallback_telemetry()
    monkeypatch.setenv("SCPN_MAX_FALLBACK_EVENTS_TOTAL", "1")

    record_fallback_event("disruption_predictor", "checkpoint_missing_fallback")
    with pytest.raises(FallbackBudgetExceeded):
        record_fallback_event("disruption_predictor", "checkpoint_missing_fallback")


def test_domain_fallback_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_fallback_telemetry()
    monkeypatch.setenv("SCPN_MAX_FALLBACK_EVENTS_SCPN_CONTROLLER", "0")

    with pytest.raises(FallbackBudgetExceeded):
        record_fallback_event("scpn_controller", "rust_backend_unavailable")

