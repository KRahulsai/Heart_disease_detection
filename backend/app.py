"""
Heart Disease Detection – Flask Backend API
============================================
Endpoints:
    POST /predict        → accepts JSON features, returns prediction
    GET  /health         → health check
    GET  /metadata       → returns model metadata (features, classes, etc.)
"""

import os
import sys

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request
from flask_cors import CORS
from backend.predict import predict
from backend.model_loader import load_metadata
import config

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok", 
        "message": "Heart Disease Detection API",
        "endpoints": ["/api/health", "/api/metadata", "/api/predict"]
    })


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Heart Disease Detection API is running"})


@app.route("/metadata", methods=["GET"])
@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Return model metadata so the frontend can auto-generate input fields."""
    try:
        meta = load_metadata()
        return jsonify({"status": "ok", "metadata": meta})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """
    Accept JSON body with feature key-value pairs.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON body provided"}), 400

        result = predict(data)
        return jsonify({"status": "ok", **result})

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 422
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  🚀 Backend running on http://localhost:{config.API_PORT}\n")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
