"""
Model Loader – Loads all saved artifacts for inference.
"""

import json
import os
import joblib
import config

def load_model():
    """Load the trained model."""
    return joblib.load(config.MODEL_PATH)


def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(config.SCALER_PATH)


def load_encoders():
    """Load the label encoders dict for categorical features."""
    return joblib.load(config.ENCODER_PATH)


def load_target_encoder():
    """Load the target label encoder (if it exists)."""
    if config.TARGET_ENCODER_PATH.exists():
        return joblib.load(config.TARGET_ENCODER_PATH)
    return None


def load_metadata() -> dict:
    """Load the metadata JSON."""
    with open(config.METADATA_PATH, "r") as f:
        return json.load(f)


def load_all():
    """Convenience – load everything at once."""
    return {
        "model": load_model(),
        "scaler": load_scaler(),
        "encoders": load_encoders(),
        "target_encoder": load_target_encoder(),
        "metadata": load_metadata(),
    }
