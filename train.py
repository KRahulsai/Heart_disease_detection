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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

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
    os.makedirs("model", exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("model/correlation_heatmap.png", dpi=150)
        plt.close()
        print("  [SAVED] model/correlation_heatmap.png")

    # Target distribution bar chart
    plt.figure(figsize=(6, 4))
    df[target].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"])
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("model/target_distribution.png", dpi=150)
    plt.close()
    print("  [SAVED] model/target_distribution.png\n")


def preprocess(df: pd.DataFrame, target: str):
    """
    Handle missing values, encode categoricals, scale numericals.
    Returns X_train, X_test, y_train, y_test, scaler, encoders, metadata.
    """
    df = df.copy()

    # Separate features and target
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # Identify feature types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  Numerical features ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}\n")

    # -- Treat zeros as missing for specific medical features --
    cols_zero_as_missing = ['resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'trestbps', 'chol', 'thalach']
    for col in cols_zero_as_missing:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)

    # -- Missing values (Imputation) --
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  Filled missing/zeros in '{col}' with median={median_val:.2f}")
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"  Filled missing in '{col}' with mode='{mode_val}'")

    # -- Encode target if non-numeric --
    target_encoder = None
    if y.dtype == "object" or y.dtype.name == "category":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target)
        print(f"  Encoded target classes: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    # -- Encode categorical features --
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  Label-encoded '{col}' -> {list(le.classes_)}")

    # -- Ensure all columns are numeric --
    feature_names = list(X.columns)

    # -- Scale --
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -- Train-test split --
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # -- Metadata for inference --
    metadata = {
        "feature_names": feature_names,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_column": target,
        "target_classes": list(map(str, target_encoder.classes_)) if target_encoder else sorted(y.unique().tolist()),
        "cat_encodings": {col: list(le.classes_) for col, le in label_encoders.items()},
    }

    return X_train, X_test, y_train, y_test, scaler, label_encoders, target_encoder, metadata


# --------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# --------------------------------------------------

MODELS = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    }
}


def train_and_evaluate(X_train, X_test, y_train, y_test, metadata):
    """Train all models using GridSearchCV, evaluate, select best, save comparison chart."""
    results = {}
    best_model = None
    best_score = -1
    best_name = ""

    print(f"\n{'='*60}")
    print("  Training & Hyperparameter Tuning (GridSearchCV)")
    print(f"{'='*60}\n")

    for name, config in MODELS.items():
        print(f"  >> Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,
            n_jobs=-1,
            scoring="f1_weighted"
        )
        grid_search.fit(X_train, y_train)
        
        # Get the best model from grid search
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        print(f"     Best Params: {grid_search.best_params_}")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     F1-score  : {f1:.4f}")
        print(f"     {'-'*40}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("xgboost", "xgb")
        plt.savefig(f"model/cm_{safe_name}.png", dpi=150)
        plt.close()

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    # Comparison bar chart
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3498db")
    ax.bar(x + width / 2, f1s, width, label="F1 Score", color="#e74c3c")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("model/model_comparison.png", dpi=150)
    plt.close()

    print(f"\n  [BEST] Best Model: {best_name} (F1={best_score:.4f})\n")
    return best_model, best_name, results


# --------------------------------------------------
# 4. SAVE ARTIFACTS
# --------------------------------------------------

def save_artifacts(model, scaler, label_encoders, target_encoder, metadata, best_name, results):
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(label_encoders, "model/encoder.pkl")
    if target_encoder:
        joblib.dump(target_encoder, "model/target_encoder.pkl")

    metadata["best_model_name"] = best_name
    metadata["results"] = {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in results.items()}
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("  [SAVED] Artifacts:")
    print("     model/model.pkl")
    print("     model/scaler.pkl")
    print("     model/encoder.pkl")
    if target_encoder:
        print("     model/target_encoder.pkl")
    print("     model/metadata.json")
    print("     model/model_comparison.png")
    print("     model/correlation_heatmap.png")
    print("     model/target_distribution.png")
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
    X_train, X_test, y_train, y_test, scaler, label_encoders, target_encoder, metadata = preprocess(df, target)
    best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, metadata)
    save_artifacts(best_model, scaler, label_encoders, target_encoder, metadata, best_name, results)

    print("  [DONE] Training complete! You can now start the backend & frontend.\n")


if __name__ == "__main__":
    main()
