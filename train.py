"""
Heart Disease Detection - Training Pipeline
=============================================
Dynamically analyzes any heart disease CSV dataset, trains multiple
classification models, selects the best one, and saves all artifacts.

Usage:
    python train.py --data data/heart.csv
    python train.py --data data/heart.csv --target target
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. DATASET LOADING & TARGET DETECTION
# --------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV dataset with basic validation."""
    if not os.path.exists(filepath):
        sys.exit(f"[ERROR] File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"\n{'='*60}")
    print(f"  Dataset loaded: {filepath}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"{'='*60}\n")
    return df


TARGET_HINTS = [
    "target", "label", "class", "output", "result", "diagnosis",
    "heart_disease", "heartdisease", "condition", "num", "goal",
]


def detect_target(df: pd.DataFrame, user_target: str | None = None) -> str:
    """Auto-detect the target column or use the one specified by user."""
    if user_target:
        if user_target not in df.columns:
            sys.exit(f"[ERROR] Specified target '{user_target}' not in columns: {list(df.columns)}")
        return user_target

    # Heuristic 1: name matching
    for col in df.columns:
        if col.strip().lower().replace(" ", "_") in TARGET_HINTS:
            print(f"  [OK] Auto-detected target column: '{col}'")
            return col

    # Heuristic 2: last column with few unique values (likely binary/multiclass)
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 10:
        print(f"  [OK] Using last column as target: '{last_col}' ({df[last_col].nunique()} classes)")
        return last_col

    # Cannot determine - list columns for the user
    print("\n  [WARN] Could not auto-detect target column.")
    print(f"  Available columns: {list(df.columns)}")
    sys.exit("  Re-run with --target <column_name>")


# --------------------------------------------------
# 2. EDA & PREPROCESSING
# --------------------------------------------------

def eda_summary(df: pd.DataFrame, target: str):
    """Print an EDA summary and save visualizations."""
    print("\n-- Exploratory Data Analysis --\n")
    print(df.describe(include="all").to_string())
    print(f"\nTarget distribution:\n{df[target].value_counts().to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    if df.isnull().sum().sum() == 0:
        print("  No missing values found.\n")

    # Save correlation heatmap for numeric features
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(config.MODEL_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()
        print(f"  [SAVED] {config.MODEL_DIR}/correlation_heatmap.png")

    # Target distribution bar chart
    plt.figure(figsize=(6, 4))
    df[target].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"])
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(config.MODEL_DIR / "target_distribution.png", dpi=150)
    plt.close()
    print(f"  [SAVED] {config.MODEL_DIR}/target_distribution.png\n")


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply advanced feature engineering to the heart disease dataset."""
    df = df.copy()
    
    # 1. Age groups (bins)
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elderly']).astype(str)
    
    # 2. Cholesterol-age ratio
    if 'cholesterol' in df.columns:
        df['chol_age_ratio'] = df['cholesterol'] / (df['age'] + 1)
    elif 'chol' in df.columns:
        df['chol_age_ratio'] = df['chol'] / (df['age'] + 1)
        
    # 3. Blood Pressure categories
    bp_col = 'resting_blood_pressure' if 'resting_blood_pressure' in df.columns else 'trestbps'
    if bp_col in df.columns:
        # standard medical thresholds (approximate)
        df['bp_category'] = pd.cut(df[bp_col], bins=[0, 120, 140, 250], labels=['Normal', 'Elevated', 'High']).astype(str)

    # 4. Interaction features (Combine key medical indicators)
    hr_col = 'max_heart_rate' if 'max_heart_rate' in df.columns else 'thalach'
    st_col = 'st_depression' if 'st_depression' in df.columns else 'oldpeak'
    if hr_col in df.columns and st_col in df.columns:
        df['hr_st_interaction'] = df[hr_col] * df[st_col]

    return df


def preprocess(df: pd.DataFrame, target: str):
    """
    Apply feature engineering, handle missing values with KNN,
    encode categoricals, scale, and balance with SMOTE.
    """
    df = feature_engineer(df)

    # Separate features and target
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # -- Encode target if non-numeric --
    target_encoder = None
    if y.dtype == "object" or y.dtype.name == "category":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target)
        print(f"  Encoded target classes: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    # -- Identify feature types --
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # -- Encode categorical features --
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Handle potential NaNs in categorical before encoding
        X[col] = X[col].fillna("Missing")
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # -- Treat zeros as missing for specific medical features (pre-imputation) --
    cols_maybe_missing = ['resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'trestbps', 'chol', 'thalach']
    for col in cols_maybe_missing:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)

    # -- Advanced Imputation (KNN) --
    print("  Applying KNNImputer...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # -- Feature Selection (SelectKBest) --
    print("  Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k='all') # Keep all for now but calculate scores
    X_selected = selector.fit_transform(X_imputed, y)
    
    # -- Scale --
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # -- Train-test split --
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # -- Handle Class Imbalance (SMOTE) --
    print("  Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n  Original train size: {X_train.shape[0]}  |  Resampled train size: {X_train_resampled.shape[0]}")
    print(f"  Test size: {X_test.shape[0]}")

    # -- Metadata for inference --
    feature_names = list(X.columns)
    metadata = {
        "feature_names": feature_names,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_column": target,
        "target_classes": list(map(str, target_encoder.classes_)) if target_encoder else sorted(y.unique().tolist()),
        "cat_encodings": {col: list(le.classes_) for col, le in label_encoders.items()},
    }

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler, imputer, label_encoders, target_encoder, metadata


# --------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# --------------------------------------------------

MODELS = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {"n_estimators": [100, 300, 500], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "params": {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]}
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(random_state=42, verbose=-1, importance_type='gain'),
        "params": {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [31, 62], "class_weight": ['balanced', None]}
    },
    "CatBoost": {
        "model": CatBoostClassifier(random_state=42, verbose=0),
        "params": {"iterations": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 8]}
    }
}


def train_and_evaluate(X_train, X_test, y_train, y_test, metadata):
    """Train all models using RandomizedSearchCV, evaluate, and build a Stacking Ensemble."""
    results = {}
    best_estimators = []
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print(f"\n{'='*60}")
    print("  Training & Hyperparameter Tuning (RandomizedSearchCV)")
    print(f"{'='*60}\n")

    for name, config in MODELS.items():
        print(f"  >> Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=20, # 50 might be too slow for this machine, using 20 for safety
            cv=skf,
            n_jobs=-1,
            scoring="f1",
            random_state=42
        )
        search.fit(X_train, y_train)
        
        model = search.best_estimator_
        best_estimators.append((name.lower().replace(" ", "_"), model))
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {"accuracy": acc, "f1": f1, "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0)}
        
        print(f"     Best Params: {search.best_params_}")
        print(f"     Accuracy  : {acc:.4f} | F1-score: {f1:.4f}")

    # -- Stacking Ensemble --
    print(f"\n  >> Building Stacking Ensemble...")
    stacking_model = StackingClassifier(
        estimators=best_estimators,
        final_estimator=LogisticRegression(),
        cv=skf
    )
    stacking_model.fit(X_train, y_train)
    
    y_pred_stack = stacking_model.predict(X_test)
    f1_stack = f1_score(y_test, y_pred_stack, average="binary", zero_division=0)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    
    results["Stacking Ensemble"] = {
        "accuracy": acc_stack, 
        "f1": f1_stack, 
        "precision": precision_score(y_test, y_pred_stack, zero_division=0), 
        "recall": recall_score(y_test, y_pred_stack, zero_division=0)
    }
    
    print(f"     Stacking accuracy: {acc_stack:.4f} | F1-score: {f1_stack:.4f}")

    # Find overall best
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_f1 = results[best_name]["f1"]
    
    if best_name == "Stacking Ensemble":
        best_model = stacking_model
    else:
        # Re-find the single estimator
        best_model = next(est for name_key, est in best_estimators if name_key == best_name.lower().replace(" ", "_"))

    # Comparison chart
    names = list(results.keys())
    f1s = [results[n]["f1"] for n in names]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=names, y=f1s, palette="viridis")
    plt.title("Model Comparison (F1 Score)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model/model_comparison.png", dpi=150)
    plt.close()

    print(f"\n  [BEST] Best Model: {best_name} (F1={best_f1:.4f})\n")
    return best_model, best_name, results


# --------------------------------------------------
# 4. SAVE ARTIFACTS
# --------------------------------------------------

def save_artifacts(model, scaler, imputer, label_encoders, target_encoder, metadata, best_name, results):
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(imputer, "model/imputer.pkl")
    joblib.dump(label_encoders, "model/encoder.pkl")
    if target_encoder:
        joblib.dump(target_encoder, "model/target_encoder.pkl")

    metadata["best_model_name"] = best_name
    metadata["results"] = {k: {m: round(float(v), 4) for m, v in vals.items()} for k, vals in results.items()}
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("  [SAVED] Artifacts:")
    print("     model/model.pkl")
    print("     model/scaler.pkl")
    print("     model/imputer.pkl")
    print("     model/encoder.pkl")
    if target_encoder:
        print("     model/target_encoder.pkl")
    print("     model/metadata.json")
    print("     model/model_comparison.png")
    print()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Classifier - Training Pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default=None, help="Name of the target column (auto-detected if omitted)")
    args = parser.parse_args()

    df = load_dataset(args.data)
    target = detect_target(df, args.target)

    eda_summary(df, target)
    X_train, X_test, y_train, y_test, scaler, imputer, label_encoders, target_encoder, metadata = preprocess(df, target)
    best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, metadata)
    save_artifacts(best_model, scaler, imputer, label_encoders, target_encoder, metadata, best_name, results)

    print("  [DONE] Training complete! You can now start the backend & frontend.\n")


if __name__ == "__main__":
    main()
