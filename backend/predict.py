import numpy as np
import pandas as pd
from backend.model_loader import load_all

# Lazy-loaded global cache
_artifacts: dict | None = None


def _get_artifacts() -> dict:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_all()
        # Add imputer to artifacts if not already there (model_loader needs update too)
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


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply identical feature engineering as used in training."""
    df = df.copy()
    
    # Use short names for logic
    age = df['age'].iloc[0]
    
    # 1. Age groups (bins)
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elderly']).astype(str)
    
    # 2. Cholesterol-age ratio
    if 'chol' in df.columns:
        df['chol_age_ratio'] = df['chol'] / (df['age'] + 1)
        
    # 3. Blood Pressure categories
    if 'trestbps' in df.columns:
        df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 250], labels=['Normal', 'Elevated', 'High']).astype(str)

    # 4. Interaction features
    if 'thalach' in df.columns and 'oldpeak' in df.columns:
        df['hr_st_interaction'] = df['thalach'] * df['oldpeak']

    return df


def predict(input_data: dict) -> dict:
    """
    Apply advanced features, KNN imputation, and prediction.
    """
    arts = _get_artifacts()
    model = arts["model"]
    scaler = arts["scaler"]
    imputer = arts.get("imputer")
    encoders = arts["encoders"]
    target_encoder = arts["target_encoder"]
    meta = arts["metadata"]

    # 1. Convert input to DataFrame and Rename
    df_input = pd.DataFrame([input_data])
    df_input = df_input.rename(columns=RENAMING_MAP)
    
    # Ensure numeric types for engineering
    for col in meta["numerical_features"]:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    # 2. Apply Feature Engineering
    df_feat = feature_engineer(df_input)
    
    # 3. Encode categoricals
    cat_features = meta["categorical_features"]
    for col in cat_features:
        if col in df_feat.columns:
            le = encoders.get(col)
            if le:
                val = str(df_feat[col].iloc[0])
                if val not in le.classes_:
                    # Fallback for unknown categories if needed, or handle missing
                    df_feat[col] = le.transform([le.classes_[0]])[0] 
                else:
                    df_feat[col] = le.transform([val])[0]

    # 4. Align columns with training
    feature_names = meta["feature_names"]
    X_input = df_feat[feature_names].copy()
    
    # Handle zeros-as-missing for medical features
    cols_maybe_missing = ['resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'trestbps', 'chol', 'thalach']
    for col in cols_maybe_missing:
        if col in X_input.columns:
            X_input[col] = X_input[col].replace(0, np.nan)

    # 5. Impute
    if imputer:
        X_imputed = imputer.transform(X_input)
    else:
        X_imputed = X_input.fillna(0).values

    # 6. Scale
    X_scaled = scaler.transform(X_imputed)

    # 7. Predict
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0].tolist()

    # Decode label
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
