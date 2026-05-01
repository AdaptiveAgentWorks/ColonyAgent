from loguru import logger
import sys


def setup_logger(log_dir: str = "logs", level: str = "INFO"):
    """Configure logger: console + file."""
    logger.remove()  # Remove default
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        f"{log_dir}/{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="7 days",
        level=level,
    )
    return logger


# Default logger instance
log = logger
