from __future__ import annotations

import json
import logging

from scpn_fusion.io.logging_config import FusionJSONFormatter, setup_fusion_logging


def test_fusion_json_formatter_includes_context_fields() -> None:
    record = logging.LogRecord(
        name="scpn_fusion",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="unit test message",
        args=(),
        exc_info=None,
    )
    record.physics_context = {"ip_ma": 8.7}  # type: ignore[attr-defined]
    payload = json.loads(FusionJSONFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "unit test message"
    assert payload["physics_context"]["ip_ma"] == 8.7


def test_setup_fusion_logging_emits_json_lines(capsys) -> None:
    logger = logging.getLogger("scpn_fusion")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    try:
        setup_fusion_logging(level=logging.INFO, json_output=True)
        logger.info("hardening log", extra={"physics_context": {"q95": 3.2}})
        out = capsys.readouterr().out.strip().splitlines()
        assert out
        parsed = json.loads(out[-1])
        assert parsed["message"] == "hardening log"
        assert parsed["physics_context"]["q95"] == 3.2
    finally:
        logger.handlers[:] = old_handlers
        logger.setLevel(old_level)
