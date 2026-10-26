"""
Structured logging configuration with configurable log levels.
"""

import logging
import sys
from typing import Optional

_LOGGER_NAME = "defurnishing"
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Initializes and configures the package logger.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write log output to disk.

    Returns:
        Configured logging.Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """
    Retrieves the package logger. If not yet initialized, returns a default INFO logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.hasHandlers():
        return setup_logger("INFO")
    return logger
