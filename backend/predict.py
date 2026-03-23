import numpy as np
from backend.model_loader import load_all

# Lazy-loaded global cache
_artifacts: dict | None = None


def _get_artifacts() -> dict:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_all()
    return _artifacts


RENAMING_MAP = {
    'chest_pain_type': 'cp',
    'resting_blood_pressure': 'trestbps',
    'cholesterol': 'chol',
    'fasting_blood_sugar': 'fbs',
    'resting_ecg': 'restecg',
    'max_heart_rate': 'thalach',
    'exercise_induced_angina': 'exang',
    'st_depression': 'oldpeak',
    'st_slope': 'slope',
    'num_major_vessels': 'ca',
    'thalassemia': 'thal'
}


def get_age_group(age: float) -> str:
    if age <= 35: return 'Young'
    if age <= 50: return 'Middle'
    if age <= 65: return 'Senior'
    return 'Elderly'


def get_bp_category(bp: float) -> str:
    if bp <= 120: return 'Normal'
    if bp <= 140: return 'Elevated'
    return 'High'


def feature_engineer(data: dict) -> dict:
    """Apply identical feature engineering using pure Python/numpy."""
    # Ensure numeric for logic
    age = float(data.get('age', 0))
    
    # 1. Age groups
    data['age_group'] = get_age_group(age)
    
    # 2. Cholesterol-age ratio
    chol = float(data.get('chol', 0))
    data['chol_age_ratio'] = chol / (age + 1)
        
    # 3. Blood Pressure categories
    trestbps = float(data.get('trestbps', 0))
    data['bp_category'] = get_bp_category(trestbps)

    # 4. Interaction features
    thalach = float(data.get('thalach', 0))
    oldpeak = float(data.get('oldpeak', 0))
    data['hr_st_interaction'] = thalach * oldpeak

    return data


def predict(input_data: dict) -> dict:
    """
    Apply advanced features, KNN imputation, and prediction (Pandas-free).
    """
    arts = _get_artifacts()
    model = arts["model"]
    scaler = arts["scaler"]
    imputer = arts.get("imputer")
    encoders = arts["encoders"]
    meta = arts["metadata"]

    # 1. Rename and Clean Input
    data = {RENAMING_MAP.get(k, k): v for k, v in input_data.items()}
    
    # 2. Feature Engineering
    data = feature_engineer(data)
    
    # 3. Handle Categoricals & Vectorize
    feature_names = meta["feature_names"]
    cat_features = meta["categorical_features"]
    
    row = []
    for feat in feature_names:
        val = data.get(feat)
        
        if feat in cat_features:
            classes = encoders.get(feat, [])
            str_val = str(val) if val is not None else "Missing"
            
            if str_val in classes:
                val = float(classes.index(str_val))
            else:
                # Fallback to 'Missing' if it exists, else 0
                if "Missing" in classes:
                    val = float(classes.index("Missing"))
                elif classes:
                    val = 0.0
                else:
                    val = 0.0
        else:
            # Numeric conversion
            try:
                val = float(val) if val is not None else np.nan
            except:
                val = np.nan
            
            # 4. Zeros-as-missing for specific medical features (consistency with training)
            cols_maybe_missing = ['trestbps', 'chol', 'thalach', 'oldpeak']
            if feat in cols_maybe_missing and val == 0:
                val = np.nan
        
        row.append(val)

    # 5. Impute & Scale
    X = np.array(row, dtype=np.float32).reshape(1, -1)
    
    # -- Imputation --
    if imputer:
        if hasattr(imputer, "run"): # ONNX Session
            X_imputed = imputer.run(None, {imputer.get_inputs()[0].name: X})[0]
        else: # Falling back to pkl if available
            X_imputed = imputer.transform(X)
    else:
        # Fallback to zero if imputer missing
        mask = np.isnan(X)
        X[mask] = 0
        X_imputed = X

    # -- Scaling --
    if hasattr(scaler, "run"): # ONNX Session
        X_scaled = scaler.run(None, {scaler.get_inputs()[0].name: X_imputed.astype(np.float32)})[0]
    else:
        X_scaled = scaler.transform(X_imputed)

    # 6. Predict
    if hasattr(model, "run"): # ONNX Session
        input_name = model.get_inputs()[0].name
        # CatBoost ONNX usually returns [labels, probabilities]
        outputs = model.run(None, {input_name: X_scaled.astype(np.float32)})
        pred = int(outputs[0][0])
        
        # Handle probabilities (can be list of arrays or list of dicts/ZipMap)
        proba = None
        if len(outputs) > 1:
            raw_proba = outputs[1]
            if isinstance(raw_proba, list) and len(raw_proba) > 0:
                if isinstance(raw_proba[0], dict):
                    # ZipMap case: [{'0': prob0, '1': prob1}]
                    proba = [raw_proba[0].get(k, raw_proba[0].get(int(k), 0)) for k in sorted(raw_proba[0].keys(), key=lambda x: int(x))]
                elif hasattr(raw_proba[0], "tolist"):
                    proba = raw_proba[0].tolist()
                else:
                    proba = raw_proba[0]
    else:
        pred = model.predict(X_scaled)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0].tolist()

    # 7. Decode label
    label = str(pred)
    target_classes = arts.get("target_encoder_classes")
    if target_classes and int(pred) < len(target_classes):
        label = str(target_classes[int(pred)])
    else:
        # Fallback to metadata classes
        meta_classes = meta.get("target_classes", [])
        if meta_classes and int(pred) < len(meta_classes):
            label = str(meta_classes[int(pred)])

    return {
        "prediction": int(pred),
        "label": label,
        "probability": proba,
    }
