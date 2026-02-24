"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for ingestion by cloud logging tools."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload)


def get_logger(name: str, log_file: str = "artifacts/logs/run.log") -> logging.Logger:
    """Create a configured logger with file and stream handlers."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
