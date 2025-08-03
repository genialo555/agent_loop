"""General-purpose helpers (tokenisation, logging setup, etc.)."""

import logging
import structlog
from typing import Optional


def create_logger(name: str, level: Optional[str] = None) -> structlog.BoundLogger:
    """Create a structured logger with consistent configuration.
    
    Args:
        name: Logger name, typically __name__
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured structured logger
        
    Example:
        >>> logger = create_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    # Configure structlog if not already configured
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level or "INFO", logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)
