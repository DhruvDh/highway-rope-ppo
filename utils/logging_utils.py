# utils/logging_utils.py
import os
import logging
import sys
from datetime import datetime

# Directory for storing artifacts and logs
ARTIFACTS_DIR = os.path.join("artifacts", "highway-ppo")
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")


def ensure_artifacts_dir(custom_path=None):
    """Create the artifacts directory if it doesn't exist."""
    artifacts_dir = custom_path or ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def setup_master_logger(log_level=logging.INFO):
    """
    Create and configure the master logger.
    Writes to a timestamped 'master.log' in LOGS_DIR and to stdout.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    # Use datetime with milliseconds and include process ID to ensure unique filenames
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # trim microseconds to milliseconds
    pid = os.getpid()
    master_log_path = os.path.join(LOGS_DIR, f"{timestamp}_{pid}_master.log")

    logger = logging.getLogger("master_logger")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing

    # File handler
    fh = logging.FileHandler(master_log_path)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(ch)

    logger.info(f"Master logger initialized. Log file: {master_log_path}")
    return logger


def setup_experiment_logger(
    experiment_id, log_level=logging.INFO, console_level=logging.WARNING
):
    """
    Create and configure a per-experiment logger.
    Writes detailed logs to timestamped 'experiment_<id>.log' and warnings+ to stdout.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    # Use datetime with milliseconds and include process ID to ensure unique filenames
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    pid = os.getpid()
    logger_name = f"experiment_{experiment_id}"
    exp_log_path = os.path.join(LOGS_DIR, f"{timestamp}_{pid}_{logger_name}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.handlers = []

    fh = logging.FileHandler(exp_log_path)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(ch)

    logger.info(
        f"Experiment logger initialized for experiment_{experiment_id}. Log file: {exp_log_path}"
    )
    return logger


def setup_logger(experiment_name="", log_level=logging.INFO):
    """Legacy setup function for backward compatibility."""
    if experiment_name:
        return setup_experiment_logger(experiment_name, log_level)
    else:
        return setup_master_logger(log_level)
