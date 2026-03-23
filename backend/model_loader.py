"""
Model Loader – Loads all saved artifacts for inference.
"""

import json
import os
import joblib
import config

try:
    import onnxruntime as ort
except ImportError:
    ort = None

def load_onnx_session(path):
    """Load an ONNX session if onnxruntime is available and file exists."""
    if ort and os.path.exists(path):
        return ort.InferenceSession(str(path))
    return None


def load_model():
    """Load the trained model (ONNX preferred, fallback to pkl)."""
    onx_path = config.MODEL_DIR / "model.onnx"
    if onx_path.exists() and ort:
        return load_onnx_session(onx_path)
    return joblib.load(config.MODEL_PATH)


def load_scaler():
    """Load the fitted StandardScaler (ONNX preferred, fallback to pkl)."""
    onx_path = config.MODEL_DIR / "scaler.onnx"
    if onx_path.exists() and ort:
        return load_onnx_session(onx_path)
    return joblib.load(config.SCALER_PATH)


def load_imputer():
    """Load the KNNImputer (ONNX preferred, fallback to pkl)."""
    onx_path = config.MODEL_DIR / "imputer.onnx"
    if onx_path.exists() and ort:
        return load_onnx_session(onx_path)
    
    imputer_path = config.MODEL_DIR / "imputer.pkl"
    if imputer_path.exists():
        return joblib.load(imputer_path)
    return None


def load_encoders():
    """Load the label encoders dict (JSON preferred for Vercel)."""
    json_path = config.MODEL_DIR / "encoder_map.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return joblib.load(config.ENCODER_PATH)


def load_target_encoder_classes():
    """Load the target label encoder classes (JSON preferred)."""
    json_path = config.MODEL_DIR / "target_encoder_classes.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
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
        "imputer": load_imputer(),
        "encoders": load_encoders(),
        "target_encoder_classes": load_target_encoder_classes(),
        "metadata": load_metadata(),
    }
