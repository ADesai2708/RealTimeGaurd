"""
utils.py

Shared utilities for the RealTimeGuard ML pipeline.
"""

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =====================================================
# Logging
# =====================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger that writes to stdout (UTF-8).

    Forces UTF-8 encoding on Windows where stdout often defaults to
    cp1252, which cannot encode box-drawing or arrow characters.

    Parameters
    ----------
    name : str
        Logger name (usually __name__ of the calling module).
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers on repeated imports
        return logger

    logger.setLevel(level)

    # Force UTF-8 so Unicode chars (─ →) work on Windows cp1252 terminals
    try:
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", line_buffering=True
        )
    except AttributeError:
        # stdout may not have .buffer in some environments (e.g. pytest capture)
        utf8_stream = sys.stdout

    handler = logging.StreamHandler(utf8_stream)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# =====================================================
# JSON Helpers
# =====================================================

class _NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that converts numpy scalar types to
    native Python types so they can be serialized without errors.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: Path | str, indent: int = 4) -> None:
    """
    Save a dictionary as a JSON file, safely handling numpy types.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : Path | str
        Destination file path.
    indent : int
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, cls=_NumpyEncoder)


def load_json(path: Path | str) -> dict:
    """
    Load and return a JSON file as a dictionary.

    Parameters
    ----------
    path : Path | str
        Source file path.

    Returns
    -------
    dict
    """
    with open(Path(path), "r") as f:
        return json.load(f)


# =====================================================
# Feature Helpers
# =====================================================

def get_feature_list(df: pd.DataFrame, target_col: str = "isFraud") -> list[str]:
    """
    Return the ordered list of feature column names,
    excluding the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed training DataFrame.
    target_col : str
        Name of the label column to exclude.

    Returns
    -------
    list[str]
        Ordered feature names.
    """
    return [col for col in df.columns if col != target_col]
