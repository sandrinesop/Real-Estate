import logging
import os


def setup_logging(log_dir: str = "logs",
                  log_file: str = "real_estate.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.getLogger(__name__).info(
        f"Logging initialized. Log: {log_path}"
    )


def ensure_directories():
    """Create all required project directories."""
    for d in ["data", "models", "logs", "notebooks"]:
        os.makedirs(d, exist_ok=True)