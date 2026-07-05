"""
model.py

Creates machine learning models for RealTimeGuard.
"""

from lightgbm import LGBMClassifier

from src.config import MODEL_PARAMS


def build_model(scale_pos_weight: float):
    """
    Create and configure the LightGBM classifier.

    Parameters
    ----------
    scale_pos_weight : float
        Class weight for handling imbalanced data.

    Returns
    -------
    LGBMClassifier
    """

    params = MODEL_PARAMS.copy()

    params["scale_pos_weight"] = scale_pos_weight

    model = LGBMClassifier(**params)

    return model