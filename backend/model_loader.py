"""
Model Loader – Loads all saved artifacts for inference.
"""

import json
import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def _path(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)


def load_model():
    """Load the trained model."""
    return joblib.load(_path("model.pkl"))


def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(_path("scaler.pkl"))


def load_encoders():
    """Load the label encoders dict for categorical features."""
    return joblib.load(_path("encoder.pkl"))


def load_target_encoder():
    """Load the target label encoder (if it exists)."""
    path = _path("target_encoder.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_metadata() -> dict:
    """Load the metadata JSON."""
    with open(_path("metadata.json"), "r") as f:
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
