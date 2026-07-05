# conftest.py
# Ensures the ml-service root is on sys.path so that
# `from src.xxx import yyy` works in all test files.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
