# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Structured Logging Configuration
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

class FusionJSONFormatter(logging.Formatter):
    """
    JSON Formatter for SCPN Fusion Core.
    Encodes log records as structured machine-readable JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Include extra attributes if provided via 'extra' kwarg
        if hasattr(record, "physics_context"):
            log_data["physics_context"] = record.physics_context
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_fusion_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    log_file: str | None = None
) -> None:
    """
    Initializes structured logging for the entire SCPN Fusion Core.
    """
    root_logger = logging.getLogger("scpn_fusion")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(FusionJSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        ))
    root_logger.addHandler(console_handler)
    
    # Optional File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(FusionJSONFormatter() if json_output else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(file_handler)
        
    root_logger.info("Structured logging initialized", extra={"json_enabled": json_output})
