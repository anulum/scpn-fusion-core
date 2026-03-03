# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fallback Telemetry and Budget Gates
# ──────────────────────────────────────────────────────────────────────
"""Runtime fallback telemetry with optional fail-closed budget enforcement."""

from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


_TOTAL_BUDGET_ENV = "SCPN_MAX_FALLBACK_EVENTS_TOTAL"
_DOMAIN_BUDGET_PREFIX = "SCPN_MAX_FALLBACK_EVENTS_"
_RECENT_EVENT_LIMIT = 256

_LOCK = threading.Lock()
_TOTAL_COUNT = 0
_DOMAIN_COUNTS: dict[str, int] = {}
_RECENT_EVENTS: deque[dict[str, Any]] = deque(maxlen=_RECENT_EVENT_LIMIT)


class FallbackBudgetExceeded(RuntimeError):
    """Raised when fallback-event budgets are exceeded."""


def _normalize_domain(domain: str) -> str:
    text = str(domain).strip()
    if not text:
        raise ValueError("fallback domain must be non-empty")
    return text.lower()


def _domain_env_key(domain: str) -> str:
    chars: list[str] = []
    for ch in domain.upper():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    return f"{_DOMAIN_BUDGET_PREFIX}{''.join(chars)}"


def _parse_limit(env_name: str) -> int | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"{env_name} must be an integer >= 0, got {raw!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{env_name} must be an integer >= 0, got {raw!r}.")
    return parsed


def reset_fallback_telemetry() -> None:
    """Reset in-process fallback telemetry state."""
    global _TOTAL_COUNT  # noqa: PLW0603
    with _LOCK:
        _TOTAL_COUNT = 0
        _DOMAIN_COUNTS.clear()
        _RECENT_EVENTS.clear()


def snapshot_fallback_telemetry() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of fallback telemetry."""
    with _LOCK:
        return {
            "total_count": int(_TOTAL_COUNT),
            "domain_counts": {k: int(v) for k, v in sorted(_DOMAIN_COUNTS.items())},
            "recent_events": [dict(evt) for evt in _RECENT_EVENTS],
        }


def record_fallback_event(
    domain: str,
    reason: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a runtime fallback event and enforce optional environment budgets.

    Budget environment variables:
    - ``SCPN_MAX_FALLBACK_EVENTS_TOTAL``
    - ``SCPN_MAX_FALLBACK_EVENTS_<DOMAIN>`` where ``<DOMAIN>`` is upper-cased
      and non-alphanumeric characters replaced with ``_``.
    """
    normalized_domain = _normalize_domain(domain)
    reason_text = str(reason).strip() or "unspecified"
    total_limit = _parse_limit(_TOTAL_BUDGET_ENV)
    domain_limit = _parse_limit(_domain_env_key(normalized_domain))
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain": normalized_domain,
        "reason": reason_text,
        "context": dict(context or {}),
    }

    with _LOCK:
        global _TOTAL_COUNT  # noqa: PLW0603
        _TOTAL_COUNT += 1
        _DOMAIN_COUNTS[normalized_domain] = _DOMAIN_COUNTS.get(normalized_domain, 0) + 1
        _RECENT_EVENTS.append(event)

        total_count = int(_TOTAL_COUNT)
        domain_count = int(_DOMAIN_COUNTS[normalized_domain])
        over_total = total_limit is not None and total_count > total_limit
        over_domain = domain_limit is not None and domain_count > domain_limit
        if over_total or over_domain:
            raise FallbackBudgetExceeded(
                "Fallback budget exceeded: "
                f"domain={normalized_domain} reason={reason_text} "
                f"domain_count={domain_count} domain_limit={domain_limit} "
                f"total_count={total_count} total_limit={total_limit}"
            )

        event["domain_count"] = domain_count
        event["total_count"] = total_count
        event["domain_limit"] = domain_limit
        event["total_limit"] = total_limit
        return event

