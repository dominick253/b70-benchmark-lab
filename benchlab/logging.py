"""
Logging infrastructure for BenchLab.

Structured logging with JSON support, log rotation, and
per-run log isolation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        extra_data = getattr(record, "extra_data", None)
        if extra_data is not None:
            log_entry["data"] = extra_data

        # Add exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class BenchLabLogger:
    """BenchLab structured logger."""

    def __init__(
        self,
        name: str = "benchlab",
        log_dir: str = "logs",
        verbose: bool = False,
        json_output: bool = False,
    ) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        if json_output:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        self.logger.addHandler(console_handler)

        # File handler with rotation
        log_file = self.log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """Log a message with optional structured data."""
        if kwargs:
            extra_data = kwargs
        else:
            extra_data = None

        record = self.logger.makeRecord(
            self.name,
            getattr(logging, level.upper(), logging.INFO),
            "(unknown)",
            0,
            message,
            (),
            None,
        )
        if extra_data is not None:
            record.__dict__["extra_data"] = extra_data
        self.logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.DEBUG, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.debug(message)

    def info(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.INFO, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.info(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.WARNING, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.ERROR, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.error(message)

    def critical(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.CRITICAL, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.critical(message)

    def exception(self, message: str, **kwargs: Any) -> None:
        if kwargs:
            record = self.logger.makeRecord(
                self.name, logging.ERROR, "(unknown)", 0, message, (), None,
            )
            record.__dict__["extra_data"] = kwargs
            self.logger.handle(record)
        else:
            self.logger.exception(message)


# Module-level convenience
_loggers: dict[str, BenchLabLogger] = {}


def get_logger(name: str = "benchlab", **kwargs: Any) -> BenchLabLogger:
    """Get or create a BenchLab logger."""
    if name not in _loggers:
        _loggers[name] = BenchLabLogger(name, **kwargs)
    return _loggers[name]
