
import logging
import os
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_filename: str = None,
    level: int = logging.INFO
):
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Auto-generate filename with today's date if not provided
    if log_filename is None:
        today = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"app_{today}.log"

    log_path = os.path.join(log_dir, log_filename)

    # This format shows: time | level | which file | message
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Remove any existing handlers to avoid duplicate log entries
    # (important when running in Streamlit which reloads frequently)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Handler 1: Write to log FILE
            logging.FileHandler(log_path, encoding="utf-8"),
            # Handler 2: Print to TERMINAL
            logging.StreamHandler()
        ]
    )

    # Confirm logging is working
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Application started")
    logger.info(f"Log file: {log_path}")
    logger.info("="*60)


def ensure_directories():
    """Create all required project directories if they don't exist."""
    dirs = ["data", "models", "logs", "notebooks"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)