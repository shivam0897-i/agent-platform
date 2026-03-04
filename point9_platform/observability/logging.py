"""
Logging Configuration
=====================

Provides clean, readable, color-coded logs with filtering for noisy warnings.

Usage:
    from point9_platform.observability import setup_logging, get_logger

    # Setup at app start
    setup_logging(level="INFO", agent_name="my_agent")

    # Get logger in any module
    logger = get_logger("tools")
    logger.info("Processing started")
"""

import logging
import sys
import warnings
from typing import Optional, List

# Suppress common Pydantic/LiteLLM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Log levels
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta

    # Components
    TIMESTAMP = "\033[90m"  # Gray
    NAME = "\033[34m"       # Blue


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and clean output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }

    def format(self, record: logging.LogRecord) -> str:
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Shorten logger name (use first part for known third-party loggers)
        name_parts = record.name.split(".")
        if name_parts[0] in ("uvicorn", "httpx", "httpcore"):
            name = name_parts[0][:15].ljust(15)
        else:
            name = name_parts[-1][:15].ljust(15)

        # Format level
        level = record.levelname[:5].ljust(5)

        # Build colored message
        parts = [
            f"{Colors.TIMESTAMP}{timestamp}{Colors.RESET}",
            f"{level_color}{level}{Colors.RESET}",
            f"{Colors.NAME}{name}{Colors.RESET}",
            f"{record.getMessage()}"
        ]

        return " │ ".join(parts)


class NoiseFilter(logging.Filter):
    """Filter out noisy log messages from third-party libraries."""

    DEFAULT_PATTERNS = [
        "pydantic",
        "Expected 10 fields",
        "StreamingChoices",
        "PydanticSerializationUnexpectedValue",
        "yaml_file",
        "model_config",
    ]

    def __init__(self, patterns: List[str] = None):
        super().__init__()
        self.patterns = patterns or self.DEFAULT_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern in message for pattern in self.patterns)


# Module-level state
_agent_name: str = "point9_agent"
_configured: bool = False


def setup_logging(
    level: str = "INFO",
    agent_name: str = "point9_agent",
    filter_noise: bool = True,
    include_uvicorn: bool = True
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        agent_name: Name prefix for agent loggers
        filter_noise: Filter out noisy third-party warnings
        include_uvicorn: Include uvicorn access logs
    """
    global _agent_name, _configured
    _agent_name = agent_name
    _configured = True

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    if filter_noise:
        handler.addFilter(NoiseFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = [handler]

    # Configure specific loggers
    loggers_to_configure = [
        agent_name,
        "point9_platform",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []
        logger.addHandler(handler)
        logger.propagate = False

    # Suppress noisy loggers
    noisy_loggers = [
        "pydantic",
        "pydantic_settings",
        "httpx",
        "httpcore",
        "litellm",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Configure uvicorn
    if include_uvicorn:
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.handlers = [handler]
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    # Log startup
    logger = logging.getLogger(agent_name)
    logger.info("Logging configured: level=%s, noise_filter=%s", level, filter_noise)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Module/component name (e.g., "tools", "graph")

    Returns:
        Logger instance prefixed with agent name
    """
    return logging.getLogger(f"{_agent_name}.{name}")
