import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Model configuration
MODEL_DIR = BASE_DIR / os.environ.get("MODEL_DIR", "model")
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# API configuration
API_PORT = int(os.environ.get("PORT", 5000))
API_HOST = os.environ.get("HOST", "0.0.0.0")
API_DEBUG = os.environ.get("DEBUG", "True").lower() == "true"

# Frontend configuration
# Use the environment variable if provided, otherwise default to local backend
DEFAULT_API_URL = f"http://localhost:{API_PORT}"
API_URL = os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")

# Visualizations path (used by frontend)
CHARTS_DIR = MODEL_DIR
