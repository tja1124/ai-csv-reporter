"""Lightweight logging setup for the CSV reporter."""

import logging

from src.config import LOG_FILE, LOGS_DIR

LOGGER_NAME = "csv_reporter"


def setup_logging() -> logging.Logger:
    """
    Configure application logging to logs/app.log.

    Returns:
        The configured application logger.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the shared application logger."""
    return logging.getLogger(LOGGER_NAME)
