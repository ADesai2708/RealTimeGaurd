"""
config.py

Central configuration for the RealTimeGuard ML pipeline.
"""

from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

# Create directories automatically
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# =====================================================
# Dataset
# =====================================================

DATASET_NAME = "paysim.csv"

DATA_PATH = RAW_DATA_DIR / DATASET_NAME

# =====================================================
# Training
# =====================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

DEVELOPMENT_MODE = True

NORMAL_SAMPLE_SIZE = 500000

# =====================================================
# LightGBM Parameters
# =====================================================

MODEL_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# =====================================================
# Model Metadata
# =====================================================

MODEL_VERSION = "1.0.0"