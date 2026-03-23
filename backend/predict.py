"""
Prediction module – Accepts a dict of feature values,
applies the same preprocessing used during training,
and returns the prediction + probability.
"""

import numpy as np
from backend.model_loader import load_all

# Lazy-loaded global cache
_artifacts: dict | None = None


def _get_artifacts() -> dict:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_all()
    return _artifacts


def validate_input(input_data: dict) -> dict:
    """Check if the provided medical details are sufficient for a reliable prediction."""
    # List of key medical indicators (excluding age, sex)
    medical_features = [
        'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
        'fasting_blood_sugar', 'max_heart_rate_achieved', 'exercise_induced_angina', 
        'st_depression', 'num_major_vessels', 'thalassemia',
        # Handle original/short feature names just in case
        'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca', 'thal'
    ]
    
    meaningful_count = 0
    total_medical = 0
    
    for feat in medical_features:
        if feat in input_data:
            total_medical += 1
            val = input_data[feat]
            # Convert to float for evaluation
            try:
                num_val = float(val)
                # We consider 0 or empty as not meaningful for this validation layer
                if num_val != 0:
                    meaningful_count += 1
            except:
                pass
                
    # Threshold: If ONLY age and sex are entered (0 meaningful medical features)
    if total_medical > 0 and meaningful_count == 0:
        return {
            "is_valid": False,
            "reason": "predict_zero"
        }
        
    return {"is_valid": True}


def predict(input_data: dict) -> dict:
    """
    Parameters
    ----------
    input_data : dict
        Keys = feature names, values = raw (unprocessed) feature values.

    Returns
    -------
    dict with keys: prediction, probability, label
    """
    arts = _get_artifacts()
    model = arts["model"]
    scaler = arts["scaler"]
    encoders = arts["encoders"]
    target_encoder = arts["target_encoder"]
    meta = arts["metadata"]

    feature_names = meta["feature_names"]
    cat_features = meta["categorical_features"]

    validation = validate_input(input_data)
    if not validation["is_valid"]:
        if validation.get("reason") == "predict_zero":
            return {
                "prediction": 0,
                "probability": [1.0, 0.0],
                "label": "0"  # Assuming target classes typically start with "0"
            }

    # Build feature vector in the correct order
    row = []
    for feat in feature_names:
        val = input_data.get(feat)
        if val is None:
            raise ValueError(f"Missing feature: '{feat}'")

        if feat in cat_features:
            le = encoders.get(feat)
            if le is None:
                raise ValueError(f"No encoder found for categorical feature '{feat}'")
            str_val = str(val)
            if str_val not in le.classes_:
                raise ValueError(
                    f"Unknown category '{str_val}' for feature '{feat}'. "
                    f"Valid: {list(le.classes_)}"
                )
            val = le.transform([str_val])[0]
        else:
            val = float(val)

        row.append(val)

    # Scale
    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0].tolist()

    # Decode label if target encoder exists
    label = str(pred)
    if target_encoder is not None:
        label = target_encoder.inverse_transform([int(pred)])[0]
    else:
        target_classes = meta.get("target_classes", [])
        if target_classes:
            idx = int(pred)
            if idx < len(target_classes):
                label = target_classes[idx]

    return {
        "prediction": int(pred),
        "label": str(label),
        "probability": proba,
    }
