"""
Structured logging setup for ExoScope.

Provides consistent log formatting across all modules,
with both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import get_config


def setup_logging(
    name: str = "exoscope",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with console + file handlers.

    Args:
        name: Logger name (typically module name).
        level: Override log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Override log file path.

    Returns:
        Configured logger instance.
    """
    config = get_config()

    log_level = level or config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format",
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    )
    log_file_path = log_file or config.get("logging.log_file", "logs/exoscope.log")

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create log file at {log_file_path}: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the exoscope namespace.

    Args:
        name: Module/component name (e.g. "data_ingestion.exoplanet").

    Returns:
        Logger instance.
    """
    parent = logging.getLogger("exoscope")
    if not parent.handlers:
        setup_logging()
    return logging.getLogger(f"exoscope.{name}")
