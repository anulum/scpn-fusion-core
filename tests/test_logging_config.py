# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_fusion.io.logging_config import FusionJSONFormatter, setup_fusion_logging


def _restore_logger(
    logger: logging.Logger,
    handlers: list[logging.Handler],
    level: int,
) -> None:
    """Restore a logger after a logging-configuration test."""
    logger.handlers[:] = handlers
    logger.setLevel(level)


def _as_object(payload: str) -> dict[str, Any]:
    """Decode a JSON log payload into a mapping."""
    decoded = json.loads(payload)
    assert isinstance(decoded, dict)
    return decoded


def test_fusion_json_formatter_includes_context_fields() -> None:
    """JSON log formatting includes the physics context payload."""
    record = logging.LogRecord(
        name="scpn_fusion",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="unit test message",
        args=(),
        exc_info=None,
    )
    record.physics_context = {"ip_ma": 8.7}
    payload = _as_object(FusionJSONFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "unit test message"
    assert payload["physics_context"]["ip_ma"] == 8.7


def test_fusion_json_formatter_includes_exception_traceback() -> None:
    """JSON log formatting includes rendered exception tracebacks."""
    try:
        raise RuntimeError("coil current failed")
    except RuntimeError:
        record = logging.getLogger("scpn_fusion").makeRecord(
            "scpn_fusion",
            logging.ERROR,
            __file__,
            52,
            "solver failed",
            (),
            exc_info=sys.exc_info(),
        )

    payload = _as_object(FusionJSONFormatter().format(record))

    assert payload["message"] == "solver failed"
    assert "RuntimeError: coil current failed" in str(payload["exception"])


def test_setup_fusion_logging_emits_json_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """Structured setup emits machine-readable JSON records to stdout."""
    logger = logging.getLogger("scpn_fusion")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    try:
        setup_fusion_logging(level=logging.INFO, json_output=True)
        logger.info("hardening log", extra={"physics_context": {"q95": 3.2}})
        out = capsys.readouterr().out.strip().splitlines()
        assert out
        parsed = _as_object(out[-1])
        assert parsed["message"] == "hardening log"
        assert parsed["physics_context"]["q95"] == 3.2
    finally:
        _restore_logger(logger, old_handlers, old_level)


def test_setup_fusion_logging_replaces_existing_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Logging setup removes stale handlers before installing the configured sink."""
    logger = logging.getLogger("scpn_fusion")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    stale_handler = logging.StreamHandler()
    try:
        logger.addHandler(stale_handler)
        setup_fusion_logging(level=logging.WARNING, json_output=False)
        logger.warning("text log")
        out = capsys.readouterr().out.strip()

        assert stale_handler not in logger.handlers
        assert len(logger.handlers) == 1
        assert logger.level == logging.WARNING
        assert "WARNING in test_logging_config: text log" in out
    finally:
        _restore_logger(logger, old_handlers, old_level)


def test_setup_fusion_logging_writes_json_file(tmp_path: Path) -> None:
    """Logging setup writes structured records to an optional file sink."""
    logger = logging.getLogger("scpn_fusion")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    log_file = tmp_path / "fusion.jsonl"
    try:
        setup_fusion_logging(level=logging.INFO, json_output=True, log_file=str(log_file))
        logger.info("file log", extra={"physics_context": {"beta_n": 2.8}})

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        parsed = _as_object(lines[-1])
        assert parsed["message"] == "file log"
        assert parsed["physics_context"]["beta_n"] == 2.8
    finally:
        _restore_logger(logger, old_handlers, old_level)


def test_setup_fusion_logging_writes_text_file(tmp_path: Path) -> None:
    """Logging setup writes human-readable records to an optional text file."""
    logger = logging.getLogger("scpn_fusion")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    log_file = tmp_path / "fusion.log"
    try:
        setup_fusion_logging(level=logging.INFO, json_output=False, log_file=str(log_file))
        logger.info("plain file log")

        text = log_file.read_text(encoding="utf-8")
        assert "scpn_fusion" in text
        assert "INFO" in text
        assert "plain file log" in text
    finally:
        _restore_logger(logger, old_handlers, old_level)
