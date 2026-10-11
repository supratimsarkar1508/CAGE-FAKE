"""
CAGE-Fake: Core Utilities
Logging, serialization, reproducibility, and caching helpers.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any
import numpy as np
import joblib

def setup_logger(name: str = "cage_fake", log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def set_seed(seed: int = 42) -> None:
    """Enforces deterministic random seed across standard library, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def save_artifact(obj: Any, file_path: Path) -> None:
    """Persists a Python object to disk using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)

def load_artifact(file_path: Path) -> Any:
    """Loads a serialized artifact from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found at {file_path}")
    return joblib.load(file_path)
